"""Delta Rule Gated Linear Attention — tests composition interference with the
retrieval-and-correction mechanism from real GatedDeltaNet.

Key difference from the simplified variant (hybrid_attention.py / l2_norm_attention.py):

Simplified (prior experiments):
  S_t = g_t * S_{t-1} + k_t v_t^T               (naive additive accumulation)

Delta rule (this experiment):
  kv_mem_t = S_{t-1} @ k_t                        (retrieve what state knows about k_t)
  delta_t = (v_t - kv_mem_t) * beta_t             (correction: only new information)
  S_t = g_t * S_{t-1} + k_t * delta_t^T           (store correction, not raw value)

The delta rule means that when domains are composed:
  - Domain B's keys retrieve domain A's stored associations (kv_mem)
  - Domain B computes corrections against A's stored values
  - This is active cross-domain interference through shared state memory

Additional mechanisms from real GatedDeltaNet:
  - Per-dimension beta gating (learned update strength, not just scalar gate)
  - SiLU output gating (gated RMSNorm on output)
  - L2 QK normalization (proven to eliminate instability in prior experiment)

References:
  - torch_recurrent_gated_delta_rule in qwen3_5_transformers.py (HF Transformers)
  - Yang et al. 2024 (GatedDeltaNet)
  - exp_l2_norm_composition_stability (L2 norm eliminates instability)
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm
from ..capsule_moe.capsule_moe import CapsulePool
from ..l2_norm_attention.l2_norm_attention import l2norm
from ..hybrid_attention.hybrid_attention import CausalSelfAttention


class DeltaRuleGatedLinearAttention(nn.Module):
    """Gated linear attention with delta rule and L2-normalized Q/K.

    State update (per head, per timestep):
      g_t = exp(-A * softplus(a_t + dt_bias))    -- decay (log-space, from real GDN)
      beta_t = sigmoid(W_beta @ x_t)             -- per-head update strength
      q_t = l2norm(W_q @ x_t)                    -- L2-normalized queries
      k_t = l2norm(W_k @ x_t)                    -- L2-normalized keys
      v_t = W_v @ x_t                            -- values
      z_t = W_z @ x_t                            -- output gate (SiLU gating)

      # Delta rule: retrieve-and-correct
      kv_mem_t = (S_{t-1} * k_t[:, None]).sum(dim=-2)   -- retrieve stored assoc.
      delta_t = (v_t - kv_mem_t) * beta_t[:, None]       -- correction
      S_t = g_t[:, None, None] * S_{t-1} + k_t[:, None] * delta_t[None, :]

      # Output with SiLU gating
      o_t = (S_t * q_t[:, None]).sum(dim=-2)
      o_t = RMSNorm(o_t) * SiLU(z_t)                     -- gated output

    At micro scale (T=32) we materialize the full attention matrix rather than
    using chunk-based recurrence. This is mathematically equivalent but simpler.
    However, the delta rule makes the materialized form more complex because
    the effective values depend on previous state (not just raw v_t).

    For T=32 at micro scale, we use an explicit sequential recurrence over
    timesteps. This is O(T * d_h^2) per head -- acceptable at T=32, d_h=16.
    """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)

        # Per-head decay parameters (matching real GatedDeltaNet)
        # g = exp(-A * softplus(a + dt_bias))
        self.w_a = nn.Linear(n_embd, n_head, bias=False)    # input-dependent component
        self.dt_bias = mx.ones((n_head,))                     # learnable bias
        self.A_log = mx.log(mx.random.uniform(shape=(n_head,)) * 16)  # log(A)

        # Per-head beta (update strength for delta rule)
        self.w_beta = nn.Linear(n_embd, n_head, bias=False)

        # Output gate (SiLU gating, matching real GatedDeltaNet)
        self.w_z = nn.Linear(n_embd, n_embd, bias=False)
        self.out_norm = RMSNorm(self.head_dim)

        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        h = self.n_head
        d = self.head_dim

        q = self.wq(x).reshape(B, T, h, d)  # (B, T, h, d)
        k = self.wk(x).reshape(B, T, h, d)  # (B, T, h, d)
        v = self.wv(x).reshape(B, T, h, d)  # (B, T, h, d)

        # L2 normalize Q and K (proven to eliminate instability)
        q = l2norm(q, dim=-1)  # ||q||_2 = 1 per head
        k = l2norm(k, dim=-1)  # ||k||_2 = 1 per head

        # Decay gate: g = exp(-A * softplus(a + dt_bias))
        a = self.w_a(x)  # (B, T, h)
        A = mx.exp(self.A_log)  # (h,)
        g = mx.exp(-A * nn.softplus(a + self.dt_bias))  # (B, T, h), in (0, 1)

        # Beta: per-head update strength for delta rule
        beta = mx.sigmoid(self.w_beta(x))  # (B, T, h)

        # Output gate
        z = self.w_z(x).reshape(B, T, h, d)  # (B, T, h, d)

        # Sequential recurrence with delta rule
        # State shape: (B, h, d, d) -- one d x d matrix per head
        S = mx.zeros((B, h, d, d))

        outputs = []
        for t in range(T):
            q_t = q[:, t, :, :]   # (B, h, d)
            k_t = k[:, t, :, :]   # (B, h, d)
            v_t = v[:, t, :, :]   # (B, h, d)
            g_t = g[:, t, :]      # (B, h)
            beta_t = beta[:, t, :] # (B, h)

            # Decay state
            S = g_t[:, :, None, None] * S  # (B, h, d, d)

            # Retrieve: what does the state already know about this key?
            # kv_mem = S^T @ k = (k @ S)  -- since S is (d_k, d_v), kv_mem is (d_v,)
            # In real impl: (S * k[:, :, :, None]).sum(dim=-2)
            kv_mem = (S * k_t[:, :, :, None]).sum(axis=-2)  # (B, h, d_v)

            # Delta: correction = (v - retrieved) * beta
            delta = (v_t - kv_mem) * beta_t[:, :, None]  # (B, h, d)

            # Update state: S += k * delta^T
            S = S + k_t[:, :, :, None] * delta[:, :, None, :]  # (B, h, d, d)

            # Output: o = S^T @ q = (q @ S) -- same as (S * q).sum(dim=-2)
            o_t = (S * q_t[:, :, :, None]).sum(axis=-2)  # (B, h, d)
            outputs.append(o_t)

        # Stack outputs: (B, T, h, d)
        out = mx.stack(outputs, axis=1)  # (B, T, h, d)

        # Gated output: RMSNorm(out) * SiLU(z)
        out = self.out_norm(out) * nn.silu(z)

        # Merge heads and project
        out = out.reshape(B, T, C)  # (B, T, C)
        return self.wo(out)


class DeltaRuleHybridCapsuleBlock(nn.Module):
    """Transformer block with delta rule linear attention or full attention."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False,
                 layer_type: str = "full"):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(n_embd)
        if layer_type == "linear":
            self.attn = DeltaRuleGatedLinearAttention(n_embd, n_head)
        else:
            self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = CapsulePool(n_embd, n_groups, n_capsules_per_group,
                                        top_k_groups, uniform_routing)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("delta_rule_hybrid_capsule_moe", parent="l2_norm_hybrid_capsule_moe")
class DeltaRuleHybridCapsuleMoEGPT(nn.Module):
    """GPT with delta rule gated linear attention and CapsulePool MLP.

    Same architecture as L2NormHybridCapsuleMoEGPT but with:
      1. Delta rule state update (retrieve-and-correct, not naive additive)
      2. Per-dimension beta gating (learned update strength)
      3. SiLU output gating (gated RMSNorm on attention output)
      4. L2 QK normalization (kept from L2 norm experiment)

    Default 4-layer config with 3:1 linear:full pattern:
      Layer 0: delta rule linear attention + capsule pool
      Layer 1: delta rule linear attention + capsule pool
      Layer 2: delta rule linear attention + capsule pool
      Layer 3: full attention + capsule pool
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False,
                 layer_types: list[str] | None = None):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)

        if layer_types is None:
            layer_types = ["linear"] * (n_layer - 1) + ["full"]

        if len(layer_types) != n_layer:
            raise ValueError(f"len(layer_types)={len(layer_types)} != n_layer={n_layer}")

        self.layer_types = layer_types
        self.layers = [
            DeltaRuleHybridCapsuleBlock(
                n_embd, n_head, n_groups, n_capsules_per_group,
                top_k_groups, uniform_routing, layer_type=lt
            )
            for lt in layer_types
        ]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def aux_loss(self) -> mx.array:
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass
