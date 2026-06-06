"""Hybrid Attention CapsuleMoE — tests composition with linear + full attention mix.

Architecture overview:
  4 layers total with a 3:1 linear:full pattern: [linear, linear, linear, full]
  This mirrors Qwen3.5's approach scaled down to micro dimensions.

Linear attention layers use a gated linear recurrence (simplified GatedDeltaNet):
  - Input-dependent forget gate: g_t = sigmoid(W_g @ x_t)
  - Key/value linear projections (no softmax, no quadratic cost)
  - Recurrent state update: S_t = g_t * S_{t-1} + k_t^T v_t
  - Output: o_t = q_t @ S_t

Full attention layers use standard causal self-attention (same as base GPT).

The MLP is replaced by CapsulePool (same as capsule_moe) in all layers.
The only difference from CapsuleMoEGPT is the attention mechanism in 3/4 layers.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm
from ..capsule_moe.capsule_moe import CapsulePool, CapsuleGroup


# =============================================================================
# Gated Linear Attention (simplified GatedDeltaNet for micro scale)
# =============================================================================

class GatedLinearAttention(nn.Module):
    """Simplified GatedDeltaNet: gated linear recurrence replacing softmax attention.

    Key mechanism:
      q_t = W_q @ x_t                        (B, T, D) -> (B, T, D)
      k_t = W_k @ x_t                        (B, T, D) -> (B, T, D)
      v_t = W_v @ x_t                        (B, T, D) -> (B, T, D)
      g_t = sigmoid(W_g @ x_t)               (B, T, D) -> (B, T, 1 per head)
      S_t = g_t * S_{t-1} + k_t^T @ v_t      recurrent state (D_h, D_h) per head
      o_t = q_t @ S_t                        output per head

    Multi-head: n_head independent recurrences, each with head_dim = D / n_head.

    At micro scale (T=32), we unroll the recurrence explicitly. This is
    mathematically equivalent to the chunkwise implementation used at scale
    but simpler to implement and debug.
    """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        # Per-head forget gate (scalar gate per head per timestep)
        self.wg = nn.Linear(n_embd, n_head, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        h = self.n_head
        d = self.head_dim

        q = self.wq(x).reshape(B, T, h, d)  # (B, T, h, d)
        k = self.wk(x).reshape(B, T, h, d)  # (B, T, h, d)
        v = self.wv(x).reshape(B, T, h, d)  # (B, T, h, d)

        # Per-head scalar forget gate
        g = mx.sigmoid(self.wg(x))  # (B, T, h)

        # Build the causal linear attention via materialized attention matrix
        # For micro T=32, this is cheap and avoids explicit Python-level recurrence.
        #
        # The gated linear recurrence S_t = g_t * S_{t-1} + k_t v_t^T produces:
        #   o_t = q_t @ S_t = sum_{s<=t} [prod_{u=s+1}^{t} g_u] * (q_t @ k_s) * v_s
        #
        # We compute the effective attention weights:
        #   A[t,s] = (q_t @ k_s) * prod_{u=s+1}^{t} g_u    for s <= t
        #
        # The cumulative gate product is: G[t,s] = prod_{u=s+1}^{t} g_u
        # Which equals exp(sum_{u=s+1}^{t} log(g_u + eps))

        # Compute log-cumsum for gate products
        log_g = mx.log(g + 1e-6)  # (B, T, h)
        # Cumulative sum of log gates: cum[t] = sum_{u=0}^{t} log_g[u]
        cum_log_g = mx.cumsum(log_g, axis=1)  # (B, T, h)

        # G[t,s] = exp(cum_log_g[t] - cum_log_g[s]) for the gate product from s+1 to t
        # Shape manipulation for broadcasting: (B, T_q, 1, h) - (B, 1, T_k, h)
        cum_q = cum_log_g[:, :, None, :]  # (B, T, 1, h)
        cum_k = cum_log_g[:, None, :, :]  # (B, 1, T, h)
        log_decay = cum_q - cum_k  # (B, T, T, h)

        # Causal mask: only attend to s <= t
        causal_mask = mx.triu(mx.full((T, T), float("-inf")), k=1)  # (T, T)
        causal_mask = causal_mask[:, :, None]  # (T, T, 1) for broadcasting over h

        # Clamp log_decay to prevent overflow, apply causal mask
        log_decay = mx.clip(log_decay, a_min=-20.0, a_max=20.0)
        gate_weights = mx.exp(log_decay + causal_mask)  # (B, T, T, h)

        # QK similarity: q_t @ k_s for all (t, s) pairs
        # q: (B, T, h, d) -> (B, h, T, d)
        # k: (B, T, h, d) -> (B, h, d, T)
        q_t = q.transpose(0, 2, 1, 3)  # (B, h, T, d)
        k_t = k.transpose(0, 2, 3, 1)  # (B, h, d, T)
        qk = q_t @ k_t  # (B, h, T, T) -- no 1/sqrt(d) scaling for linear attention

        # gate_weights is (B, T, T, h) -> need (B, h, T, T)
        gate_weights = gate_weights.transpose(0, 3, 1, 2)  # (B, h, T, T)

        # Combined attention: element-wise product of QK scores and gate decay
        attn = qk * gate_weights  # (B, h, T, T)

        # Apply to values
        v_t = v.transpose(0, 2, 1, 3)  # (B, h, T, d)
        out = (attn @ v_t).transpose(0, 2, 1, 3).reshape(B, T, C)  # (B, T, C)

        return self.wo(out)


# =============================================================================
# Standard Causal Self-Attention (copied from base GPT for independence)
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


# =============================================================================
# Hybrid Block: selects attention type based on layer_type
# =============================================================================

class HybridCapsuleBlock(nn.Module):
    """Transformer block with configurable attention (linear or full) and CapsulePool MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False,
                 layer_type: str = "full"):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(n_embd)
        if layer_type == "linear":
            self.attn = GatedLinearAttention(n_embd, n_head)
        else:
            self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = CapsulePool(n_embd, n_groups, n_capsules_per_group,
                                        top_k_groups, uniform_routing)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


# =============================================================================
# Hybrid Capsule MoE GPT
# =============================================================================

@register("hybrid_capsule_moe", parent="capsule_moe")
class HybridCapsuleMoEGPT(nn.Module):
    """GPT with hybrid attention (linear + full) and CapsulePool MLP.

    Default 4-layer config with 3:1 linear:full pattern:
      Layer 0: linear attention + capsule pool
      Layer 1: linear attention + capsule pool
      Layer 2: linear attention + capsule pool
      Layer 3: full attention + capsule pool

    This mirrors Qwen3.5's architecture at micro scale.
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

        # Default 3:1 linear:full pattern
        if layer_types is None:
            layer_types = ["linear"] * (n_layer - 1) + ["full"]

        if len(layer_types) != n_layer:
            raise ValueError(f"len(layer_types)={len(layer_types)} != n_layer={n_layer}")

        self.layer_types = layer_types
        self.layers = [
            HybridCapsuleBlock(
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


# Also register a full-attention-only variant for controlled comparison
@register("full_attn_capsule_moe", parent="capsule_moe")
class FullAttnCapsuleMoEGPT(HybridCapsuleMoEGPT):
    """All-full-attention CapsuleMoE (control condition).

    Identical architecture to HybridCapsuleMoEGPT but with all full attention
    layers. This ensures the comparison is fair: same model class, same code
    paths, only the attention type differs.
    """

    def __init__(self, **kwargs):
        kwargs["layer_types"] = ["full"] * kwargs.get("n_layer", 4)
        super().__init__(**kwargs)
