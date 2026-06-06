"""Shared/Unique Decomposition — always-on shared groups + routed unique residuals."""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class DecomposedCapsulePool(nn.Module):
    """Capsule pool with shared (always-on) groups + unique (routed) groups.

    Shared groups are always active with uniform weighting (1/n_shared).
    Unique groups are selected via softmax top-k routing.
    Output = shared_out + unique_out.
    """

    def __init__(self, n_embd: int, n_shared: int = 4, n_unique: int = 8,
                 n_capsules_per_group: int = 64, top_k_unique: int = 2,
                 uniform_unique: bool = False):
        super().__init__()
        self.n_shared = n_shared
        self.n_unique = n_unique
        self.top_k_unique = top_k_unique
        self.uniform_unique = uniform_unique

        self.shared_groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                              for _ in range(n_shared)]
        self.unique_groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                              for _ in range(n_unique)]

        # Router only over unique groups
        self.router = nn.Linear(n_embd, n_unique, bias=False)
        self._gate_probs = None

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        # Shared: always active, uniform weight
        w_shared = 1.0 / self.n_shared
        shared_out = mx.zeros_like(x)
        for group in self.shared_groups:
            shared_out = shared_out + w_shared * group(x)

        if self.n_unique == 0:
            return shared_out

        if self.uniform_unique:
            w_unique = 1.0 / self.n_unique
            unique_out = mx.zeros_like(x)
            for group in self.unique_groups:
                unique_out = unique_out + w_unique * group(x)
            self._gate_probs = mx.full((*x.shape[:-1], self.n_unique), w_unique)
            return shared_out + unique_out

        # Learned routing over unique groups
        scores = self.router(x)  # (B, T, n_unique)
        probs = mx.softmax(scores, axis=-1)
        self._gate_probs = probs

        # Top-k selection
        top_vals = mx.topk(scores, self.top_k_unique, axis=-1)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        unique_out = mx.zeros_like(x)
        for i, group in enumerate(self.unique_groups):
            w = masked_probs[..., i:i+1]
            unique_out = unique_out + w * group(x)

        return shared_out + unique_out

    def balance_loss(self) -> mx.array:
        if self._gate_probs is None or self.n_unique == 0:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))
        return self.n_unique * mx.sum(mean_probs * mean_probs)


class DecomposedBlock(nn.Module):
    """Transformer block with DecomposedCapsulePool replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_shared: int = 4, n_unique: int = 8,
                 n_capsules_per_group: int = 64, top_k_unique: int = 2,
                 uniform_unique: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.pool = DecomposedCapsulePool(
            n_embd, n_shared, n_unique, n_capsules_per_group,
            top_k_unique, uniform_unique)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.pool(self.norm2(x))
        return x


@register("procrustes_decomp", parent="capsule_moe")
class ProcrustesDecompGPT(nn.Module):
    """GPT with shared/unique decomposed capsule pools.

    Architecture:
    - Shared capsule groups: always active, uniform weighting
    - Unique capsule groups: routed via softmax top-k
    - Output = shared_contribution + unique_contribution
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_shared: int = 4, n_unique: int = 8,
                 n_capsules_per_group: int = 64, top_k_unique: int = 2,
                 uniform_unique: bool = False):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [DecomposedBlock(n_embd, n_head, n_shared, n_unique,
                                        n_capsules_per_group, top_k_unique,
                                        uniform_unique)
                       for _ in range(n_layer)]
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
            total = total + layer.pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass
