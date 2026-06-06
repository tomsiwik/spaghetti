"""SiLU Capsule — SiLU activation variant for capsule composition.

Identical to ReLU Router except activation: nn.silu instead of nn.relu.

Motivation: Qwen2.5-Coder uses SwiGLU (SiLU-gated), yet our capsules use
ReLU. Macro training showed 0% dead capsules — is that because pretrained
hidden states are distributed for SiLU, not ReLU? This experiment isolates
the activation function as a variable.

Key difference from ReLU:
  - ReLU: hard sparsity, exact zeros, truly dead neurons possible
  - SiLU: smooth, min ≈ -0.278 at x ≈ -1.28, never exactly zero
  - "Dead" capsules become "near-dead" (effective sparsity)

Composition identity still holds: B @ SiLU(A @ x) with B=0 → output=0,
so zero-init guarantees composition identity regardless of activation.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention


# SiLU (Swish) is available as nn.silu in MLX


class SiLUCapsulePool(nn.Module):
    """Two-layer SiLU MLP, composable by weight concatenation.

    Forward: y = B @ SiLU(A @ x)   where A is (P, d), B is (d, P)

    Same composition protocol as ReLU Router: concatenate A vertically
    and B horizontally. Identity holds because B is zero-initialized.

    Unlike ReLU, SiLU never produces exact zeros, so:
      - "Dead" is redefined as mean_activation < threshold
      - "Sparsity" is effective sparsity: fraction with |a| < threshold
    """

    NEAR_ZERO_THRESHOLD = 0.01  # for effective sparsity and near-dead detection

    def __init__(self, n_embd: int, n_capsules: int = 256,
                 l1_target_sparsity: float = 0.50,
                 l1_coeff: float = 0.01,
                 balance_coeff: float = 0.01):
        super().__init__()
        self.n_capsules = n_capsules
        self.l1_target_sparsity = l1_target_sparsity
        self.l1_coeff = l1_coeff
        self.balance_coeff = balance_coeff

        # A: (P, d) — detector vectors
        self.A = nn.Linear(n_embd, n_capsules, bias=False)
        # B: (d, P) — expansion vectors
        self.B = nn.Linear(n_capsules, n_embd, bias=False)

        # Adaptive L1 coefficient: tracks running sparsity
        self._running_sparsity = 0.3  # SiLU has lower natural sparsity than ReLU
        self._activation_counts = None
        self._last_activations = None  # for effective sparsity reporting

    def __call__(self, x):
        """x: (B, T, d) -> output: (B, T, d)"""
        h = nn.silu(self.A(x))  # (B, T, P) — smooth self-routing via SiLU
        self._store_stats(h)
        return self.B(h)        # (B, T, d) — expand back

    def _store_stats(self, h):
        """Store activation statistics for sparsity and balance losses."""
        self._last_activations = h

        # Effective sparsity: fraction of activations with |a| < threshold
        near_zero = (mx.abs(h) < self.NEAR_ZERO_THRESHOLD).astype(mx.float32)
        sparsity = mx.mean(near_zero).item()
        alpha = 0.1
        self._running_sparsity = (1 - alpha) * self._running_sparsity + alpha * sparsity

        # Per-capsule mean absolute activation (replaces binary fire count)
        # Higher = more active, lower = near-dead
        self._activation_counts = mx.mean(mx.abs(h), axis=(0, 1))  # (P,)

    def sparsity_loss(self) -> mx.array:
        """Adaptive L1 loss — same mechanism as ReLU Router."""
        if self._activation_counts is None:
            return mx.array(0.0)

        sparsity_gap = self.l1_target_sparsity - self._running_sparsity
        adaptive_coeff = self.l1_coeff * (1.0 + 5.0 * max(-0.5, min(0.5, sparsity_gap)))
        mean_activation = mx.mean(self._activation_counts)
        return adaptive_coeff * mean_activation

    def balance_loss(self) -> mx.array:
        """Per-capsule utilization balance loss (same as ReLU Router)."""
        if self._activation_counts is None:
            return mx.array(0.0)

        mean_act = mx.mean(self._activation_counts)
        variance = mx.mean((self._activation_counts - mean_act) ** 2)
        return self.balance_coeff * self.n_capsules * variance

    def aux_loss(self) -> mx.array:
        """Combined auxiliary loss: sparsity + balance."""
        return self.sparsity_loss() + self.balance_loss()


class SiLUCapsuleBlock(nn.Module):
    """Transformer block with SiLUCapsulePool replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_capsules: int = 256,
                 l1_target_sparsity: float = 0.50,
                 l1_coeff: float = 0.01,
                 balance_coeff: float = 0.01):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = SiLUCapsulePool(
            n_embd, n_capsules,
            l1_target_sparsity=l1_target_sparsity,
            l1_coeff=l1_coeff,
            balance_coeff=balance_coeff,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("silu_capsule", parent="relu_router")
class SiLUCapsuleGPT(nn.Module):
    """GPT with SiLU MLP, composable by weight concatenation.

    Identical to ReLURouterGPT except:
      - Activation: nn.silu instead of nn.relu
      - Dead capsule metric: mean_activation < threshold (not count == 0)
      - Reports effective sparsity (|a| < threshold) alongside natural sparsity
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
        self.layers = [SiLUCapsuleBlock(
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

        Returns per-layer: effective_sparsity, mean_activation,
        min/max activation, n_near_dead (mean |a| < threshold).
        """
        stats = {"eff_sparsity": [], "mean_act": [], "min_act": [],
                 "max_act": [], "n_near_dead": []}

        for layer in self.layers:
            pool = layer.capsule_pool
            if pool._activation_counts is None:
                for k in stats:
                    stats[k].append(None)
                continue

            counts = pool._activation_counts  # mean |activation| per capsule
            mx.eval(counts)

            threshold = SiLUCapsulePool.NEAR_ZERO_THRESHOLD
            stats["mean_act"].append(mx.mean(counts).item())
            stats["min_act"].append(mx.min(counts).item())
            stats["max_act"].append(mx.max(counts).item())
            stats["n_near_dead"].append(int(mx.sum(counts < threshold).item()))

            # Effective sparsity from last activations
            if pool._last_activations is not None:
                h = pool._last_activations
                eff_sp = mx.mean((mx.abs(h) < threshold).astype(mx.float32)).item()
                stats["eff_sparsity"].append(eff_sp)
            else:
                stats["eff_sparsity"].append(pool._running_sparsity)

        return stats
