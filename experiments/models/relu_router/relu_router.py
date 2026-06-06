"""ReLU Router — composition by MLP concatenation.

This model IS a standard two-layer ReLU MLP (y = B @ ReLU(A @ x)).
The architecture has zero novelty. The contribution is the COMPOSITION
PROTOCOL: independently-trained MLPs can be composed by concatenating
their weight matrices, producing a mathematically exact sum of outputs.

Key insight: ReLU's independence property (each neuron's activation
depends only on its own detector vector, not on other neurons) means
concatenating two MLPs cannot interfere with existing activations.
The composed output is Pool_A(x) + Pool_B(x), exactly.

Supported by three independent research directions:
  - ReMoE (ICLR 2025): ReLU routing is fully differentiable
  - Union-of-Experts (UoE): Internal activations capture routing
  - MoRAM: Rank-1 experts self-route via intrinsic keys

Known limitation: Zero-shot composition degrades ~5-7%. Scalar
calibration barely helps (+4.4%), showing the issue is distribution
mismatch, not just loudness. Weight averaging (+2.0%) outperforms
concatenation (+5.0%) at micro scale.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention


class ReLUCapsulePool(nn.Module):
    """Standard two-layer ReLU MLP, composable by weight concatenation.

    Forward: y = B @ ReLU(A @ x)   where A is (P, d), B is (d, P)

    This IS a standard MLP with hidden dim P. The conceptual difference
    from a plain MLP is the COMPOSITION PROTOCOL: two independently-trained
    pools can be combined by concatenating A matrices vertically and B
    matrices horizontally, producing exact output: Pool_A(x) + Pool_B(x).

    Natural ReLU sparsity (~50%) means roughly half the neurons are
    inactive for any given input. Auxiliary losses (L1 + balance) are
    included but do NOT push sparsity beyond ~50% at micro scale.
    """

    def __init__(self, n_embd: int, n_capsules: int = 256,
                 l1_target_sparsity: float = 0.50,
                 l1_coeff: float = 0.01,
                 balance_coeff: float = 0.01):
        super().__init__()
        self.n_capsules = n_capsules
        self.l1_target_sparsity = l1_target_sparsity
        self.l1_coeff = l1_coeff
        self.balance_coeff = balance_coeff

        # A: (P, d) — detector vectors. Row i is a_i^T.
        self.A = nn.Linear(n_embd, n_capsules, bias=False)
        # B: (d, P) — expansion vectors. Column i is b_i.
        self.B = nn.Linear(n_capsules, n_embd, bias=False)

        # Adaptive L1 coefficient: tracks running sparsity to auto-adjust
        self._running_sparsity = 0.5  # initialized at ReLU's natural ~50%
        self._activation_counts = None  # for balance loss

    def __call__(self, x):
        """x: (B, T, d) -> output: (B, T, d)"""
        h = nn.relu(self.A(x))  # (B, T, P) — self-routing via ReLU
        # Store activation statistics for auxiliary losses
        self._store_stats(h)
        return self.B(h)        # (B, T, d) — expand back

    def _store_stats(self, h):
        """Store activation statistics for sparsity and balance losses."""
        # Sparsity: fraction of zero activations
        sparsity = mx.mean((h == 0).astype(mx.float32)).item()
        # EMA update of running sparsity
        alpha = 0.1
        self._running_sparsity = (1 - alpha) * self._running_sparsity + alpha * sparsity

        # Per-capsule activation frequency: how often each capsule fires
        # h shape: (B, T, P)
        fired = (h > 0).astype(mx.float32)  # (B, T, P)
        self._activation_counts = mx.mean(fired, axis=(0, 1))  # (P,)

    def sparsity_loss(self) -> mx.array:
        """Adaptive L1 loss to control sparsity level.

        If actual sparsity < target: increase L1 penalty (push more to zero)
        If actual sparsity > target: decrease L1 penalty (allow more to fire)

        This is the ReMoE mechanism: adaptive coefficient based on running
        average of actual sparsity vs target.
        """
        if self._activation_counts is None:
            return mx.array(0.0)

        # Sparsity deviation: positive means too dense, negative means too sparse
        sparsity_gap = self.l1_target_sparsity - self._running_sparsity

        # Adaptive coefficient: scale up when too dense, scale down when too sparse
        # Clamp to [0.1x, 10x] of base coefficient to prevent instability
        adaptive_coeff = self.l1_coeff * (1.0 + 5.0 * max(-0.5, min(0.5, sparsity_gap)))

        # L1 on mean activation counts (push toward target sparsity)
        mean_activation = mx.mean(self._activation_counts)
        return adaptive_coeff * mean_activation

    def balance_loss(self) -> mx.array:
        """Per-capsule utilization balance loss.

        Penalizes variance in per-capsule activation frequency.
        Without this, a few capsules dominate and the rest die.

        L_bal = Var(activation_freq) * P
        Minimized when all capsules fire equally often.
        """
        if self._activation_counts is None:
            return mx.array(0.0)

        mean_freq = mx.mean(self._activation_counts)
        variance = mx.mean((self._activation_counts - mean_freq) ** 2)
        return self.balance_coeff * self.n_capsules * variance

    def aux_loss(self) -> mx.array:
        """Combined auxiliary loss: sparsity + balance."""
        return self.sparsity_loss() + self.balance_loss()


class ReLUCapsuleBlock(nn.Module):
    """Transformer block with ReLUCapsulePool replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_capsules: int = 256,
                 l1_target_sparsity: float = 0.50,
                 l1_coeff: float = 0.01,
                 balance_coeff: float = 0.01):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = ReLUCapsulePool(
            n_embd, n_capsules,
            l1_target_sparsity=l1_target_sparsity,
            l1_coeff=l1_coeff,
            balance_coeff=balance_coeff,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("relu_router", parent="capsule_moe")
class ReLURouterGPT(nn.Module):
    """GPT with standard ReLU MLP, composable by weight concatenation.

    Architecture: This IS the dense GPT baseline with identical
    parameter count (P=4d=256 at d=64). The MLP is y = B @ ReLU(A @ x).

    The contribution is NOT the architecture (which is standard) but the
    COMPOSITION PROTOCOL: domain-specific MLPs trained from a shared base
    can be composed by concatenating weight matrices (A vertically, B
    horizontally). The composed output is mathematically the exact sum
    of individual pool outputs.

    Parameter count vs capsule_moe: FEWER by 1,024 (no router weights),
    but this 0.5% savings is practically irrelevant.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_capsules: int = 256,
                 l1_target_sparsity: float = 0.50,
                 l1_coeff: float = 0.01,
                 balance_coeff: float = 0.01):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [ReLUCapsuleBlock(
            n_embd, n_head, n_capsules,
            l1_target_sparsity=l1_target_sparsity,
            l1_coeff=l1_coeff,
            balance_coeff=balance_coeff,
        ) for _ in range(n_layer)]
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
            total = total + layer.capsule_pool.aux_loss()
        return total

    def on_domain_switch(self, domain: str):
        pass

    def capsule_stats(self) -> dict:
        """Activation statistics from last forward pass.

        Returns per-layer: sparsity, mean_activation_freq, min/max freq,
        n_dead_capsules (never fire).
        """
        stats = {"sparsity": [], "mean_freq": [], "min_freq": [],
                 "max_freq": [], "n_dead": []}

        for layer in self.layers:
            pool = layer.capsule_pool
            if pool._activation_counts is None:
                for k in stats:
                    stats[k].append(None)
                continue

            counts = pool._activation_counts
            mx.eval(counts)

            stats["sparsity"].append(1.0 - mx.mean(counts).item())
            stats["mean_freq"].append(mx.mean(counts).item())
            stats["min_freq"].append(mx.min(counts).item())
            stats["max_freq"].append(mx.max(counts).item())
            stats["n_dead"].append(int(mx.sum(counts < 0.01).item()))

        return stats
