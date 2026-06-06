"""Training-time composition compatibility — auxiliary losses for composability.

Extends ReLURouterGPT with auxiliary losses designed to make independently-
trained capsule pools more compatible at composition time.

Two auxiliary losses:

1. Weight orthogonality loss (L_ortho):
   Penalizes capsule weight deltas that align with the base model's weight
   directions. Inspired by InfLoRA's orthogonality constraints for continual
   learning, adapted for our composition setting.

   L_ortho = sum_l ||delta_A_l @ W_base_A_l^T||_F^2 / (||delta_A_l||_F^2 * ||W_base_A_l||_F^2 + eps)

   This encourages deltas to lie in the orthogonal complement of the base
   weight subspace, reducing interference when multiple domain deltas are
   composed (concatenated) on the same base.

2. Output-norm matching loss (L_norm):
   Penalizes deviation of capsule pool output norms from the base model's
   output norm. Addresses the distribution mismatch that causes composition
   degradation.

   L_norm = sum_l (||pool_l(x)||_2 / ||x||_2 - target_ratio)^2

   target_ratio is measured from the base model's MLP output norms during
   pretraining (stored as a running average).
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..relu_router.relu_router import ReLUCapsulePool


class CompatCapsulePool(ReLUCapsulePool):
    """ReLU capsule pool with composition-compatibility auxiliary losses.

    Extends ReLUCapsulePool with:
    - Weight orthogonality loss (vs frozen base snapshot)
    - Output-norm matching loss (vs target ratio from base)
    """

    def __init__(self, n_embd: int, n_capsules: int = 256,
                 ortho_coeff: float = 0.1,
                 norm_coeff: float = 0.1,
                 target_norm_ratio: float = 1.0,
                 **kwargs):
        super().__init__(n_embd, n_capsules, **kwargs)
        self.ortho_coeff = ortho_coeff
        self.norm_coeff = norm_coeff
        self.target_norm_ratio = target_norm_ratio

        # Frozen base weight snapshots (set externally after pretraining)
        self._base_A_weight = None  # (P, d) — frozen
        self._base_B_weight = None  # (d, P) — frozen

        # Output norm from last forward pass
        self._last_output_norm = None
        self._last_input_norm = None

    def __call__(self, x):
        """x: (B, T, d) -> output: (B, T, d)"""
        # Track input norm for norm-matching loss
        self._last_input_norm = mx.sqrt(
            mx.mean(mx.sum(x * x, axis=-1)) + 1e-8
        )

        h = nn.relu(self.A(x))  # (B, T, P)
        self._store_stats(h)
        out = self.B(h)  # (B, T, d)

        # Track output norm
        self._last_output_norm = mx.sqrt(
            mx.mean(mx.sum(out * out, axis=-1)) + 1e-8
        )

        return out

    def set_base_snapshot(self, A_weight, B_weight):
        """Freeze base weight snapshot for orthogonality loss."""
        self._base_A_weight = mx.stop_gradient(A_weight)
        self._base_B_weight = mx.stop_gradient(B_weight)

    def ortho_loss(self) -> mx.array:
        """Weight orthogonality loss: penalize delta alignment with base weights.

        L = ||delta_A @ base_A^T||_F^2 / (||delta_A||_F^2 * ||base_A||_F^2 + eps)
          + ||delta_B @ base_B^T||_F^2 / (||delta_B||_F^2 * ||base_B||_F^2 + eps)

        Measures how much of the weight update lies in the span of the base
        weights. Minimized when delta is orthogonal to base. Normalized to
        [0, 1] range regardless of weight magnitudes.
        """
        if self._base_A_weight is None:
            return mx.array(0.0)

        delta_A = self.A.weight - self._base_A_weight  # (P, d)
        delta_B = self.B.weight - self._base_B_weight  # (d, P)

        # Frobenius inner product: ||delta @ base^T||_F^2
        # For A: delta_A is (P, d), base_A^T is (d, P) -> product is (P, P)
        # ||M||_F^2 = trace(M @ M^T) = sum of squared elements
        prod_A = delta_A @ self._base_A_weight.T  # (P, P)
        num_A = mx.sum(prod_A * prod_A)
        denom_A = mx.sum(delta_A * delta_A) * mx.sum(self._base_A_weight * self._base_A_weight) + 1e-8

        prod_B = delta_B @ self._base_B_weight.T  # (d, d)
        num_B = mx.sum(prod_B * prod_B)
        denom_B = mx.sum(delta_B * delta_B) * mx.sum(self._base_B_weight * self._base_B_weight) + 1e-8

        return self.ortho_coeff * (num_A / denom_A + num_B / denom_B)

    def norm_loss(self) -> mx.array:
        """Output-norm matching loss: keep pool output norm close to target.

        L = (||pool(x)||_2 / ||x||_2 - target_ratio)^2

        Ensures that fine-tuned pools don't produce much larger or smaller
        outputs than the base model's pools. When composing by concatenation,
        pools with matched norms produce balanced contributions.
        """
        if self._last_output_norm is None or self._last_input_norm is None:
            return mx.array(0.0)

        ratio = self._last_output_norm / (self._last_input_norm + 1e-8)
        return self.norm_coeff * (ratio - self.target_norm_ratio) ** 2

    def compat_aux_loss(self) -> mx.array:
        """Combined compatibility auxiliary loss."""
        return self.ortho_loss() + self.norm_loss()

    def aux_loss(self) -> mx.array:
        """Full auxiliary loss: sparsity + balance + compatibility."""
        return super().aux_loss() + self.compat_aux_loss()


class CompatCapsuleBlock(nn.Module):
    """Transformer block with CompatCapsulePool."""

    def __init__(self, n_embd: int, n_head: int,
                 n_capsules: int = 256,
                 ortho_coeff: float = 0.1,
                 norm_coeff: float = 0.1,
                 target_norm_ratio: float = 1.0,
                 **kwargs):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = CompatCapsulePool(
            n_embd, n_capsules,
            ortho_coeff=ortho_coeff,
            norm_coeff=norm_coeff,
            target_norm_ratio=target_norm_ratio,
            **kwargs,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("training_compat", parent="relu_router")
class TrainingCompatGPT(nn.Module):
    """GPT with composition-compatible capsule pools.

    Same architecture as ReLURouterGPT, but capsule pools include auxiliary
    losses designed to improve post-hoc composition quality.

    Usage:
    1. Pretrain normally (aux losses are inactive until base snapshot is set)
    2. Call snapshot_base() to freeze current weights as reference
    3. Fine-tune per domain (aux losses now active, pushing deltas orthogonal
       to base and matching output norms)
    4. Compose by concatenating A/B matrices (same protocol as relu_router)
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_capsules: int = 256,
                 ortho_coeff: float = 0.1,
                 norm_coeff: float = 0.1,
                 target_norm_ratio: float = 1.0,
                 **kwargs):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [CompatCapsuleBlock(
            n_embd, n_head, n_capsules,
            ortho_coeff=ortho_coeff,
            norm_coeff=norm_coeff,
            target_norm_ratio=target_norm_ratio,
            **kwargs,
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

    def snapshot_base(self):
        """Freeze current weights as reference for orthogonality loss.

        Call this AFTER pretraining, BEFORE domain fine-tuning.
        """
        for layer in self.layers:
            pool = layer.capsule_pool
            pool.set_base_snapshot(
                pool.A.weight.__array__() if hasattr(pool.A.weight, '__array__') else pool.A.weight,
                pool.B.weight.__array__() if hasattr(pool.B.weight, '__array__') else pool.B.weight,
            )

    def measure_norm_ratios(self, dataset, batch_size=32, n_batches=5):
        """Measure base model output-norm ratios for norm matching loss.

        Call this AFTER pretraining to set target_norm_ratio per layer.
        """
        import random
        rng = random.Random(0)
        ratios_per_layer = [[] for _ in self.layers]

        for _ in range(n_batches):
            inputs, _ = dataset.get_batch(batch_size, rng)
            B, T = inputs.shape
            pos = mx.arange(T)
            x = self.wte(inputs) + self.wpe(pos)
            x = self.norm0(x)

            for l_idx, layer in enumerate(self.layers):
                x_pre = layer.norm2(x + layer.attn(layer.norm1(x)))
                pool_out = layer.capsule_pool(x_pre)
                mx.eval(pool_out)

                in_norm = mx.sqrt(mx.mean(mx.sum(x_pre * x_pre, axis=-1)) + 1e-8).item()
                out_norm = mx.sqrt(mx.mean(mx.sum(pool_out * pool_out, axis=-1)) + 1e-8).item()
                ratios_per_layer[l_idx].append(out_norm / (in_norm + 1e-8))

                x = x + layer.attn(layer.norm1(x))
                x = x + layer.capsule_pool(layer.norm2(x))

        # Set target ratios
        for l_idx, layer in enumerate(self.layers):
            if ratios_per_layer[l_idx]:
                avg_ratio = sum(ratios_per_layer[l_idx]) / len(ratios_per_layer[l_idx])
                layer.capsule_pool.target_norm_ratio = avg_ratio

        return [sum(r) / len(r) if r else 1.0 for r in ratios_per_layer]
