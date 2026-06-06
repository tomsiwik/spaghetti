"""L2-normalized hybrid attention with value norm tracking.

Extends L2NormHybridCapsuleMoEGPT to record per-layer, per-head value norms
during forward passes. This enables monitoring whether value norms grow
unbounded during composition training, which would break the state
boundedness argument despite L2-normalized QK products.

The model is functionally identical to L2NormHybridCapsuleMoEGPT.
The only addition is instrumentation that records value norms when
tracking is enabled.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm
from ..capsule_moe.capsule_moe import CapsulePool
from ..hybrid_attention.hybrid_attention import CausalSelfAttention
from ..l2_norm_attention.l2_norm_attention import l2norm


class ValueTrackingGatedLinearAttention(nn.Module):
    """Gated linear attention with L2-normalized Q/K and value norm recording.

    Identical to L2NormGatedLinearAttention except it records value norms
    into self._value_norms when self._track_norms is True.
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
        self._track_norms = False
        self._value_norms = None  # will be (h,) mean over B,T

    def __call__(self, x):
        B, T, C = x.shape
        h = self.n_head
        d = self.head_dim

        q = self.wq(x).reshape(B, T, h, d)
        k = self.wk(x).reshape(B, T, h, d)
        v = self.wv(x).reshape(B, T, h, d)

        # L2 normalize Q and K
        q = l2norm(q, dim=-1)
        k = l2norm(k, dim=-1)

        # Record value norms if tracking is enabled
        if self._track_norms:
            # ||v||_2 per head, averaged over batch and time
            # v shape: (B, T, h, d) -> norm over d -> (B, T, h) -> mean over B,T -> (h,)
            v_norms = mx.sqrt((v * v).sum(axis=-1) + 1e-12)  # (B, T, h)
            self._value_norms = mx.mean(v_norms, axis=(0, 1))  # (h,)

        # Per-head scalar forget gate
        g = mx.sigmoid(self.wg(x))  # (B, T, h)

        # Build causal linear attention via materialized attention matrix
        log_g = mx.log(g + 1e-6)
        cum_log_g = mx.cumsum(log_g, axis=1)

        cum_q = cum_log_g[:, :, None, :]
        cum_k = cum_log_g[:, None, :, :]
        log_decay = cum_q - cum_k

        causal_mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        causal_mask = causal_mask[:, :, None]

        log_decay = mx.clip(log_decay, a_min=-20.0, a_max=20.0)
        gate_weights = mx.exp(log_decay + causal_mask)

        q_t = q.transpose(0, 2, 1, 3)
        k_t = k.transpose(0, 2, 3, 1)
        qk = q_t @ k_t

        gate_weights = gate_weights.transpose(0, 3, 1, 2)
        attn = qk * gate_weights

        v_t = v.transpose(0, 2, 1, 3)
        out = (attn @ v_t).transpose(0, 2, 1, 3).reshape(B, T, C)

        return self.wo(out)


class ValueTrackingHybridBlock(nn.Module):
    """Transformer block with value norm tracking in linear attention layers."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False,
                 layer_type: str = "full"):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(n_embd)
        if layer_type == "linear":
            self.attn = ValueTrackingGatedLinearAttention(n_embd, n_head)
        else:
            self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = CapsulePool(n_embd, n_groups, n_capsules_per_group,
                                        top_k_groups, uniform_routing)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("value_norm_tracking_moe", parent="l2_norm_hybrid_capsule_moe")
class L2NormValueTrackingGPT(nn.Module):
    """GPT with L2-normalized hybrid attention and value norm instrumentation.

    Functionally identical to L2NormHybridCapsuleMoEGPT. The only addition
    is the ability to enable value norm tracking for diagnostic purposes.

    Usage:
        model.enable_tracking()
        logits = model(tokens)
        norms = model.get_value_norms()  # dict: {layer_idx: array(h,)}
        model.disable_tracking()
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
            ValueTrackingHybridBlock(
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

    def enable_tracking(self):
        """Enable value norm tracking in all linear attention layers."""
        for layer in self.layers:
            if layer.layer_type == "linear":
                layer.attn._track_norms = True

    def disable_tracking(self):
        """Disable value norm tracking."""
        for layer in self.layers:
            if layer.layer_type == "linear":
                layer.attn._track_norms = False
                layer.attn._value_norms = None

    def get_value_norms(self) -> dict[int, list[float]]:
        """Return value norms from the last forward pass.

        Returns:
            dict mapping layer_idx -> list of per-head mean value norms.
            Only linear attention layers are included.
        """
        result = {}
        for i, layer in enumerate(self.layers):
            if layer.layer_type == "linear" and layer.attn._value_norms is not None:
                norms = layer.attn._value_norms
                mx.eval(norms)
                result[i] = norms.tolist()
        return result
