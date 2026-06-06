"""L2-normalized hybrid attention CapsuleMoE.

Extends the hybrid attention model by adding L2 normalization to Q and K
in the gated linear attention layers. This matches real GatedDeltaNet
(Qwen3.5 uses use_qk_l2norm_in_kernel=True).

The hypothesis: unnormalized QK products cause ~20% catastrophic failure
rate during composition. L2 normalization bounds QK product magnitudes
to [-1, 1] per element, eliminating the numerical instability.

Only change from hybrid_attention.py: l2norm(q) and l2norm(k) before
the QK product computation. Everything else is identical.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm
from ..capsule_moe.capsule_moe import CapsulePool
from ..hybrid_attention.hybrid_attention import (
    CausalSelfAttention,
    HybridCapsuleBlock,
    HybridCapsuleMoEGPT,
)


def l2norm(x, dim=-1, eps=1e-6):
    """L2 normalization matching the FLA library / GatedDeltaNet convention.

    x / sqrt(sum(x^2, dim) + eps)

    This bounds the norm of each vector along `dim` to 1, so the dot product
    of two L2-normalized vectors is bounded to [-1, 1] per element.
    """
    inv_norm = mx.rsqrt((x * x).sum(axis=dim, keepdims=True) + eps)
    return x * inv_norm


class L2NormGatedLinearAttention(nn.Module):
    """Gated linear attention with L2-normalized Q and K.

    Identical to GatedLinearAttention from hybrid_attention.py except:
      q = l2norm(W_q @ x)   -- unit-norm queries
      k = l2norm(W_k @ x)   -- unit-norm keys

    This bounds QK product magnitudes, preventing the numerical instability
    that causes catastrophic composition failure in ~20% of random inits.
    """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wg = nn.Linear(n_embd, n_head, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        h = self.n_head
        d = self.head_dim

        q = self.wq(x).reshape(B, T, h, d)  # (B, T, h, d)
        k = self.wk(x).reshape(B, T, h, d)  # (B, T, h, d)
        v = self.wv(x).reshape(B, T, h, d)  # (B, T, h, d)

        # L2 normalize Q and K along head dimension (the key change)
        q = l2norm(q, dim=-1)  # ||q||_2 = 1 per head
        k = l2norm(k, dim=-1)  # ||k||_2 = 1 per head

        # Per-head scalar forget gate
        g = mx.sigmoid(self.wg(x))  # (B, T, h)

        # Build the causal linear attention via materialized attention matrix
        # (same as hybrid_attention.py from this point)
        log_g = mx.log(g + 1e-6)  # (B, T, h)
        cum_log_g = mx.cumsum(log_g, axis=1)  # (B, T, h)

        cum_q = cum_log_g[:, :, None, :]  # (B, T, 1, h)
        cum_k = cum_log_g[:, None, :, :]  # (B, 1, T, h)
        log_decay = cum_q - cum_k  # (B, T, T, h)

        causal_mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        causal_mask = causal_mask[:, :, None]  # (T, T, 1)

        log_decay = mx.clip(log_decay, a_min=-20.0, a_max=20.0)
        gate_weights = mx.exp(log_decay + causal_mask)  # (B, T, T, h)

        # QK similarity -- now bounded because q and k are L2-normalized
        q_t = q.transpose(0, 2, 1, 3)  # (B, h, T, d)
        k_t = k.transpose(0, 2, 3, 1)  # (B, h, d, T)
        qk = q_t @ k_t  # (B, h, T, T) -- elements in [-1, 1]

        gate_weights = gate_weights.transpose(0, 3, 1, 2)  # (B, h, T, T)

        attn = qk * gate_weights  # (B, h, T, T)

        v_t = v.transpose(0, 2, 1, 3)  # (B, h, T, d)
        out = (attn @ v_t).transpose(0, 2, 1, 3).reshape(B, T, C)  # (B, T, C)

        return self.wo(out)


class L2NormHybridCapsuleBlock(nn.Module):
    """Transformer block with L2-normalized linear attention or full attention."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False,
                 layer_type: str = "full"):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(n_embd)
        if layer_type == "linear":
            self.attn = L2NormGatedLinearAttention(n_embd, n_head)
        else:
            self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = CapsulePool(n_embd, n_groups, n_capsules_per_group,
                                        top_k_groups, uniform_routing)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("l2_norm_hybrid_capsule_moe", parent="hybrid_capsule_moe")
class L2NormHybridCapsuleMoEGPT(nn.Module):
    """GPT with L2-normalized hybrid attention and CapsulePool MLP.

    Same architecture as HybridCapsuleMoEGPT but with L2 normalization
    applied to Q and K in linear attention layers. This matches real
    GatedDeltaNet's use_qk_l2norm_in_kernel=True.

    Default 4-layer config with 3:1 linear:full pattern:
      Layer 0: L2-norm linear attention + capsule pool
      Layer 1: L2-norm linear attention + capsule pool
      Layer 2: L2-norm linear attention + capsule pool
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
            L2NormHybridCapsuleBlock(
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
