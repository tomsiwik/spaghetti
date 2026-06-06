"""Capsule MoE — rank-1 non-linear capsules in routable groups replace MLP."""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention


class CapsuleGroup(nn.Module):
    """A group of rank-1 non-linear capsules.

    Each capsule is (a_i, b_i) producing: b_i * ReLU(a_i^T x).
    In matrix form: B @ ReLU(A @ x), where A is (n_capsules, d) and B is (d, n_capsules).

    This is mathematically equivalent to a two-layer MLP with hidden dim = n_capsules,
    but conceptualized as a pool of independent rank-1 experts.
    """

    def __init__(self, n_embd: int, n_capsules: int):
        super().__init__()
        # A: (n_capsules, n_embd) -- rows are detector vectors a_i^T
        self.A = nn.Linear(n_embd, n_capsules, bias=False)
        # B: (n_embd, n_capsules) -- columns are expansion vectors b_i
        # Implemented as Linear(n_capsules, n_embd) so B.weight is (n_embd, n_capsules)
        self.B = nn.Linear(n_capsules, n_embd, bias=False)

    def __call__(self, x):
        """x: (..., d) -> output: (..., d)"""
        h = nn.relu(self.A(x))   # (..., n_capsules) -- activations, sparse via ReLU
        return self.B(h)         # (..., d) -- expand back


class CapsulePool(nn.Module):
    """Pool of capsule groups with top-k group routing.

    Level 1: Learned router selects top-k_g groups per token.
    Level 2: ReLU activation sparsity within selected groups (free).
    """

    def __init__(self, n_embd: int, n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False):
        super().__init__()
        self.n_groups = n_groups
        self.top_k_groups = top_k_groups
        self.uniform_routing = uniform_routing

        # Group router: token -> group scores
        self.router = nn.Linear(n_embd, n_groups, bias=False)

        # Capsule groups
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]

        self._gate_probs = None  # stored for balance loss

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        if self.uniform_routing:
            # Ablation: all groups weighted equally (1/G), no learned routing
            w = 1.0 / self.n_groups
            out = mx.zeros_like(x)
            for group in self.groups:
                out = out + w * group(x)
            self._gate_probs = mx.full((*x.shape[:-1], self.n_groups), w)
            return out

        # Level 1: Group routing
        scores = self.router(x)  # (B, T, G)
        probs = mx.softmax(scores, axis=-1)
        self._gate_probs = probs

        # Top-k group selection
        top_vals = mx.topk(scores, self.top_k_groups, axis=-1)  # (B, T, k_g)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)    # (B, T, 1)
        mask = (scores >= threshold).astype(mx.float32)
        masked_probs = probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1, keepdims=True) + 1e-8)

        # Run all groups, weight by masked probs (at small G this is cheaper than scatter)
        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = masked_probs[..., i:i+1]  # (B, T, 1)
            out = out + w * group(x)      # Level 2 sparsity is implicit in ReLU

        return out

    def balance_loss(self) -> mx.array:
        """L = G * sum(mean_prob_g^2). Minimized at uniform 1/G."""
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))  # (G,)
        return self.n_groups * mx.sum(mean_probs * mean_probs)


class CapsuleBlock(nn.Module):
    """Transformer block with CapsulePool replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = CapsulePool(n_embd, n_groups, n_capsules_per_group,
                                        top_k_groups, uniform_routing)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("capsule_moe", parent="moe")
class CapsuleMoEGPT(nn.Module):
    """GPT with CapsulePool replacing MLP in each block.

    Architecture:
    - Token + position embeddings (same as GPT)
    - N transformer blocks, each with:
      - Causal self-attention (same as GPT)
      - CapsulePool: P capsules in G groups, top-k_g group routing
    - Language model head (same as GPT)

    Default config: d=64, G=4, P/G=64 (P=256 total capsules).
    At P=4d, parameter count matches dense GPT exactly.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, uniform_routing: bool = False):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [CapsuleBlock(n_embd, n_head, n_groups,
                                    n_capsules_per_group, top_k_groups,
                                    uniform_routing)
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
            total = total + layer.capsule_pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass


@register("capsule_moe_uniform", parent="capsule_moe")
class CapsuleMoEUniformGPT(CapsuleMoEGPT):
    """Capsule MoE ablation: uniform routing (w_g = 1/G, no learned router)."""

    def __init__(self, **kwargs):
        kwargs["uniform_routing"] = True
        super().__init__(**kwargs)
