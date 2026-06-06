"""Full GatedDeltaNet Composition Stack — all components combined.

This is the final integration test before declaring GatedDeltaNet composition-safe.
Combines ALL components from real GatedDeltaNet that were validated piecemeal:

From prior experiments (individually validated):
  1. L2 QK normalization (exp_l2_norm: 0/25 failures)
  2. Delta rule retrieval-and-correction (exp_delta_rule: 0.74x ratio, +0.39% gap)
  3. Gated linear recurrence (exp_hybrid_attention: 0.59x ratio)

New components in this experiment (untested for composition):
  4. Causal conv1d preprocessing on Q, K, V (local mixing before recurrence)
  5. Per-dimension beta gating (not just per-head scalar)
  6. SiLU output gating (already in delta rule exp, but combined here with conv1d)

The question: do these components interact to create emergent interference
not present when tested individually?

Architecture:
  4 layers, 3:1 linear:full pattern (matching Qwen3.5)
  Linear layers: full GatedDeltaNet (all 6 components)
  Full layer: standard causal self-attention
  MLP: CapsulePool in all layers

References:
  - Qwen3.5-0.8B (production GatedDeltaNet with all components)
  - qwen3_5_transformers.py: torch_recurrent_gated_delta_rule
  - experiments/models/delta_rule_attention/ (delta rule baseline)
  - experiments/models/l2_norm_attention/ (L2 norm baseline)
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm
from ..capsule_moe.capsule_moe import CapsulePool
from ..l2_norm_attention.l2_norm_attention import l2norm
from ..hybrid_attention.hybrid_attention import CausalSelfAttention


class CausalConv1d(nn.Module):
    """Causal 1D convolution matching GatedDeltaNet's preprocessing.

    Applies depthwise causal convolution to each channel independently.
    This provides local temporal mixing before the recurrence, allowing
    each position to incorporate information from kernel_size-1 previous
    positions.

    At micro scale (T=32), even kernel_size=4 covers 12.5% of the sequence.
    At macro scale (T=4096), kernel_size=4 covers <0.1% -- purely local.

    Implementation: standard conv1d with left-padding for causal alignment.
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        # Depthwise conv: each channel has its own kernel
        # Weight shape: (channels, 1, kernel_size) for depthwise
        # In MLX conv1d: weight shape is (out_channels, kernel_size, in_channels)
        # For depthwise we implement as grouped conv or manual per-channel.
        # Simpler: use a single Linear-like approach over the kernel window.
        # For micro scale, we implement as direct matmul over padded input.
        self.weight = mx.random.normal((channels, kernel_size)) * (kernel_size ** -0.5)

    def __call__(self, x):
        """x: (B, T, C) -> (B, T, C)"""
        B, T, C = x.shape
        # Left-pad for causal alignment: [pad zeros, x]
        padded = mx.concatenate([mx.zeros((B, self.pad, C)), x], axis=1)  # (B, T+pad, C)

        # Gather windows: for each t, collect [t, t+1, ..., t+kernel_size-1]
        # from padded, which corresponds to original positions [t-pad, ..., t]
        # This is causal: position t only sees positions <= t.
        windows = mx.stack(
            [padded[:, t:t + T, :] for t in range(self.kernel_size)],
            axis=-1,
        )  # (B, T, C, kernel_size)

        # Apply per-channel kernel: sum over kernel_size dimension
        out = (windows * self.weight[None, None, :, :]).sum(axis=-1)  # (B, T, C)
        return out


class FullGatedDeltaNetAttention(nn.Module):
    """Full GatedDeltaNet attention with all components.

    Complete mechanism per head per timestep:
      1. Project: q, k, v, a, beta, z = linear(x)
      2. Conv1d: q, k, v = causal_conv1d(q, k, v)     [NEW: local mixing]
      3. L2 norm: q, k = l2norm(q), l2norm(k)          [proven stable]
      4. Decay: g_t = exp(-A * softplus(a_t + dt_bias))
      5. Beta: beta_t = sigmoid(W_beta @ x_t)          [per-dim, not per-head]
      6. Delta rule:
           kv_mem = S^T @ k_t                           [retrieve]
           delta = (v_t - kv_mem) * beta_t              [correct]
           S = g * S + k * delta^T                      [update]
      7. Output: o = S^T @ q
      8. Gate: out = RMSNorm(o) * SiLU(z)               [output gating]
      9. Project: output = W_o @ concat(out)

    Key difference from delta_rule_attention.py:
      - Conv1d on Q, K, V before L2 norm (step 2)
      - Per-dimension beta (shape: B, T, h, d) not per-head (B, T, h)
    """

    def __init__(self, n_embd: int, n_head: int, conv_kernel: int = 4):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)

        # Conv1d preprocessing (causal, per-channel)
        self.conv_q = CausalConv1d(n_embd, kernel_size=conv_kernel)
        self.conv_k = CausalConv1d(n_embd, kernel_size=conv_kernel)
        self.conv_v = CausalConv1d(n_embd, kernel_size=conv_kernel)

        # Per-head decay parameters (matching real GatedDeltaNet)
        self.w_a = nn.Linear(n_embd, n_head, bias=False)
        self.dt_bias = mx.ones((n_head,))
        self.A_log = mx.log(mx.random.uniform(shape=(n_head,)) * 16)

        # Per-dimension beta (key difference from delta_rule_attention.py)
        # Real GatedDeltaNet: beta has shape (B, T, h, d_h), not (B, T, h)
        self.w_beta = nn.Linear(n_embd, n_embd, bias=False)  # projects to h*d

        # Output gate (SiLU gating)
        self.w_z = nn.Linear(n_embd, n_embd, bias=False)
        self.out_norm = RMSNorm(self.head_dim)

        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        h = self.n_head
        d = self.head_dim

        # Step 1: Linear projections
        q = self.wq(x)  # (B, T, C)
        k = self.wk(x)
        v = self.wv(x)

        # Step 2: Causal conv1d preprocessing (local temporal mixing)
        q = nn.silu(self.conv_q(q))  # SiLU activation after conv (matching real GDN)
        k = nn.silu(self.conv_k(k))
        v = nn.silu(self.conv_v(v))

        # Reshape to per-head
        q = q.reshape(B, T, h, d)
        k = k.reshape(B, T, h, d)
        v = v.reshape(B, T, h, d)

        # Step 3: L2 normalize Q and K (proven to eliminate instability)
        q = l2norm(q, dim=-1)
        k = l2norm(k, dim=-1)

        # Step 4: Decay gate
        a = self.w_a(x)  # (B, T, h)
        A = mx.exp(self.A_log)
        g = mx.exp(-A * nn.softplus(a + self.dt_bias))  # (B, T, h)

        # Step 5: Per-dimension beta (sigmoid, shape: B, T, h, d)
        beta = mx.sigmoid(self.w_beta(x).reshape(B, T, h, d))  # per-dim, not per-head

        # Step 6: Output gate input
        z = self.w_z(x).reshape(B, T, h, d)

        # Sequential recurrence with delta rule
        S = mx.zeros((B, h, d, d))

        outputs = []
        for t in range(T):
            q_t = q[:, t, :, :]   # (B, h, d)
            k_t = k[:, t, :, :]
            v_t = v[:, t, :, :]
            g_t = g[:, t, :]      # (B, h)
            beta_t = beta[:, t, :, :]  # (B, h, d) -- per-dimension

            # Decay state
            S = g_t[:, :, None, None] * S

            # Retrieve: what does state know about this key?
            kv_mem = (S * k_t[:, :, :, None]).sum(axis=-2)  # (B, h, d)

            # Delta correction with per-dimension beta
            delta = (v_t - kv_mem) * beta_t  # (B, h, d) -- per-dim scaling

            # Update state
            S = S + k_t[:, :, :, None] * delta[:, :, None, :]  # (B, h, d, d)

            # Output
            o_t = (S * q_t[:, :, :, None]).sum(axis=-2)
            outputs.append(o_t)

        # Stack outputs
        out = mx.stack(outputs, axis=1)  # (B, T, h, d)

        # Step 8: Gated output
        out = self.out_norm(out) * nn.silu(z)

        # Merge heads and project
        out = out.reshape(B, T, C)
        return self.wo(out)


class FullGDNStackBlock(nn.Module):
    """Transformer block with full GatedDeltaNet or full attention + CapsulePool."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False,
                 layer_type: str = "full", conv_kernel: int = 4):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(n_embd)
        if layer_type == "linear":
            self.attn = FullGatedDeltaNetAttention(n_embd, n_head, conv_kernel)
        else:
            self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = CapsulePool(n_embd, n_groups, n_capsules_per_group,
                                        top_k_groups, uniform_routing)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("full_gdn_stack_capsule_moe", parent="delta_rule_hybrid_capsule_moe")
class FullGDNStackCapsuleMoEGPT(nn.Module):
    """GPT with full GatedDeltaNet stack and CapsulePool MLP.

    Extends delta_rule_hybrid_capsule_moe with:
      1. Causal conv1d preprocessing on Q, K, V
      2. Per-dimension beta gating (not per-head scalar)

    All other components inherited from delta rule experiment:
      3. L2 QK normalization
      4. Delta rule retrieval-and-correction
      5. Parameterized decay gate
      6. SiLU output gating

    Default 4-layer config with 3:1 linear:full pattern:
      Layer 0: full GatedDeltaNet + capsule pool
      Layer 1: full GatedDeltaNet + capsule pool
      Layer 2: full GatedDeltaNet + capsule pool
      Layer 3: full attention + capsule pool
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False,
                 layer_types: list[str] | None = None,
                 conv_kernel: int = 4):
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
            FullGDNStackBlock(
                n_embd, n_head, n_groups, n_capsules_per_group,
                top_k_groups, uniform_routing, layer_type=lt,
                conv_kernel=conv_kernel,
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
