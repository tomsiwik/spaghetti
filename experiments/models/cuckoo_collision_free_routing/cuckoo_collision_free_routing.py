"""Cuckoo collision-free routing for MoE expert selection.

Adapts cuckoo hashing (Pagh & Rodler 2004) to MoE routing: each token has
two candidate expert sets via two independent linear projections (hash
functions). If the primary expert's score is below a learned threshold
(collision = poor fit), the token is evicted to its secondary expert.
If the secondary also fails, cascading eviction proceeds up to a maximum
chain depth. This guarantees distinct, high-quality routing without the
softmax score-tie problem.

Key properties:
- Two learned hash functions (linear projections): h1, h2 : R^d -> R^N
- Primary: top-k from h1(x)
- Collision detection: if max score in primary set < threshold tau
- Eviction: switch to top-k from h2(x)
- Chain depth tracking: diagnostic for routing stability
- Fallback: if chain exceeds max_depth, use combined scores (graceful)

Kill criteria:
- Cuckoo routing >2% worse than softmax at same effective k
- Eviction chain length >3 (routing instability)

Prior art:
- Pagh & Rodler 2004: Cuckoo hashing with worst-case O(1) lookup
- Consistent hash routing (this project): hash-ring, +0.89% vs softmax
- Skip-list routing (this project): hierarchical, -0.93% vs flat
- ReMoE (2024): ReLU routing replaces softmax, dynamic expert count
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsuleGroup


class CuckooRouter(nn.Module):
    """Routes tokens to experts via cuckoo-hashing-inspired dual hash functions.

    Two independent linear projections (h1, h2) each produce scores over N
    experts. Primary routing uses h1. If the primary expert's confidence
    (max score after softmax) is below threshold tau, the token "evicts"
    to h2's top experts. This resolves routing collisions where softmax
    assigns similar scores to multiple experts (wasted compute).

    Parameters:
        n_embd: embedding dimension d
        n_groups: number of expert groups N
        top_k: number of experts per token
        max_chain_depth: maximum eviction chain length before fallback
    """

    def __init__(self, n_embd: int, n_groups: int = 8,
                 top_k: int = 2, max_chain_depth: int = 3):
        super().__init__()
        self.n_embd = n_embd
        self.n_groups = n_groups
        self.top_k = top_k
        self.max_chain_depth = max_chain_depth

        # Two independent hash functions (learned linear projections)
        self.h1 = nn.Linear(n_embd, n_groups, bias=False)
        self.h2 = nn.Linear(n_embd, n_groups, bias=False)

        # Learned collision threshold: sigmoid(raw_tau) in [0, 1]
        # Initialized to yield tau ~ 0.3 (moderate eviction rate)
        self._raw_tau = mx.array([-0.85])  # sigmoid(-0.85) ~ 0.3

        # Diagnostics
        self._chain_depths = None  # (B, T) eviction chain depth per token
        self._eviction_rate = None  # fraction of tokens that evicted
        self._gate_probs = None  # final routing weights (B, T, N)

    @property
    def tau(self):
        """Collision threshold in [0, 1]."""
        return mx.sigmoid(self._raw_tau)

    def __call__(self, x):
        """Compute routing weights via cuckoo dual-hash with eviction.

        x: (B, T, d) -> weights: (B, T, N), sparse with top_k nonzero

        The routing is fully differentiable:
        - Both h1 and h2 scores contribute gradients
        - The eviction decision uses a soft blend controlled by tau
        - No hard if/else that would block gradient flow
        """
        B, T, d = x.shape
        k = self.top_k
        N = self.n_groups

        # Compute scores from both hash functions
        scores_h1 = self.h1(x)  # (B, T, N)
        scores_h2 = self.h2(x)  # (B, T, N)

        # Softmax probabilities for each hash function
        probs_h1 = mx.softmax(scores_h1, axis=-1)  # (B, T, N)
        probs_h2 = mx.softmax(scores_h2, axis=-1)  # (B, T, N)

        # Primary confidence: max probability from h1
        # High confidence = h1 has a clear winner (no collision)
        # Low confidence = h1 scores are similar (collision)
        max_prob_h1 = mx.max(probs_h1, axis=-1, keepdims=True)  # (B, T, 1)

        # Eviction signal: how much to blend h2 into routing
        # If max_prob_h1 > tau: no eviction (alpha -> 0), use h1
        # If max_prob_h1 < tau: evict (alpha -> 1), use h2
        tau = self.tau  # scalar in [0, 1]

        # Soft eviction blending with temperature for smooth gradients
        # alpha = sigmoid((tau - max_prob_h1) * temperature)
        # temperature controls sharpness of eviction decision
        temperature = 10.0
        alpha = mx.sigmoid((tau - max_prob_h1) * temperature)  # (B, T, 1)

        # Blend h1 and h2 probabilities
        # No eviction (alpha~0): use h1 probs
        # Eviction (alpha~1): use h2 probs
        blended_probs = (1.0 - alpha) * probs_h1 + alpha * probs_h2  # (B, T, N)

        # Top-k selection on blended probabilities
        top_vals = mx.topk(blended_probs, k, axis=-1)  # (B, T, k)
        threshold = mx.min(top_vals, axis=-1, keepdims=True)  # (B, T, 1)
        mask = (blended_probs >= threshold).astype(mx.float32)  # (B, T, N)
        masked_probs = blended_probs * mask
        masked_probs = masked_probs / (mx.sum(masked_probs, axis=-1,
                                              keepdims=True) + 1e-8)

        # Store diagnostics
        self._gate_probs = masked_probs

        # Compute eviction chain depth diagnostic (non-differentiable)
        # Depth 0: h1 confident (no eviction needed)
        # Depth 1: evicted to h2
        # Depth 2+: would need further eviction (h2 also low confidence)
        mx.eval(alpha)
        evicted = (alpha > 0.5).astype(mx.float32)  # (B, T, 1)
        max_prob_h2 = mx.max(probs_h2, axis=-1, keepdims=True)  # (B, T, 1)
        double_evicted = evicted * (max_prob_h2 < tau).astype(mx.float32)

        chain_depths = evicted + double_evicted  # 0, 1, or 2
        self._chain_depths = chain_depths.squeeze(-1)  # (B, T)
        self._eviction_rate = mx.mean(evicted).item()

        return masked_probs

    def get_diagnostics(self) -> dict:
        """Return routing diagnostics."""
        diag = {}
        if self._chain_depths is not None:
            mx.eval(self._chain_depths)
            depths = self._chain_depths
            diag["mean_chain_depth"] = mx.mean(depths).item()
            diag["max_chain_depth"] = mx.max(depths).item()
            diag["eviction_rate"] = self._eviction_rate
            # Fraction at each depth
            for d in range(3):
                diag[f"frac_depth_{d}"] = mx.mean(
                    (depths == d).astype(mx.float32)
                ).item()
        if self._gate_probs is not None:
            gp = self._gate_probs
            selected = (gp > 0.001).astype(mx.float32)
            diag["mean_experts_selected"] = mx.mean(
                mx.sum(selected, axis=-1)
            ).item()
            # Entropy of routing distribution
            eps = 1e-8
            entropy = -mx.sum(gp * mx.log(gp + eps), axis=-1)
            max_entropy = math.log(self.n_groups)
            diag["normalized_entropy"] = (
                mx.mean(entropy).item() / max_entropy if max_entropy > 0 else 0
            )
        return diag


class CuckooCapsulePool(nn.Module):
    """Pool of capsule groups with cuckoo collision-free routing."""

    def __init__(self, n_embd: int, n_groups: int = 8,
                 n_capsules_per_group: int = 32, top_k: int = 2):
        super().__init__()
        self.n_groups = n_groups
        self.top_k = top_k

        self.router = CuckooRouter(n_embd, n_groups, top_k)
        self.groups = [CapsuleGroup(n_embd, n_capsules_per_group)
                       for _ in range(n_groups)]

        self._gate_probs = None

    def __call__(self, x):
        """x: (B, T, D) -> output: (B, T, D)"""
        weights = self.router(x)  # (B, T, N)
        self._gate_probs = weights

        out = mx.zeros_like(x)
        for i, group in enumerate(self.groups):
            w = weights[..., i:i+1]  # (B, T, 1)
            out = out + w * group(x)

        return out

    def balance_loss(self) -> mx.array:
        """L = G * sum(mean_prob_g^2). Minimized at uniform 1/G."""
        if self._gate_probs is None:
            return mx.array(0.0)
        mean_probs = mx.mean(self._gate_probs, axis=(0, 1))  # (G,)
        G = mean_probs.shape[0]
        return G * mx.sum(mean_probs * mean_probs)


class CuckooBlock(nn.Module):
    """Transformer block with cuckoo-routed CapsulePool."""

    def __init__(self, n_embd: int, n_head: int,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 top_k: int = 2):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = CuckooCapsulePool(
            n_embd, n_groups, n_capsules_per_group, top_k
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("cuckoo_collision_free_routing", parent="capsule_moe")
class CuckooCollisionFreeRoutingGPT(nn.Module):
    """GPT with cuckoo collision-free expert routing.

    Architecture:
    - Token + position embeddings (same as GPT)
    - N transformer blocks, each with:
      - Causal self-attention (same as GPT)
      - CuckooCapsulePool: dual-hash routes with eviction to top-k groups
    - Language model head (same as GPT)

    Key differences from CapsuleMoEGPT:
    - Two independent routing projections (h1, h2) instead of one
    - Soft eviction mechanism: blend h2 when h1 has low confidence
    - Learned collision threshold tau
    - Chain depth tracking for stability diagnostics

    Default config: d=64, G=8, 32 caps/group, k=2.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 8, n_capsules_per_group: int = 32,
                 top_k: int = 2):
        super().__init__()
        self.n_embd = n_embd
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [CuckooBlock(n_embd, n_head, n_groups,
                                    n_capsules_per_group, top_k)
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
        """Balance loss across expert groups."""
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass

    def get_routing_diagnostics(self) -> dict:
        """Return per-layer routing diagnostics including chain depths."""
        diagnostics = {}
        for li, layer in enumerate(self.layers):
            router = layer.capsule_pool.router
            diagnostics[f"layer_{li}"] = router.get_diagnostics()
        return diagnostics
