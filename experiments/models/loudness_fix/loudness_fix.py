"""Loudness Fix -- interventions for zero-shot ReLU Router composition.

Problem: When composing independently-trained ReLU MLP pools by weight
concatenation, the output is Pool_A(x) + Pool_B(x). This sum is
mathematically exact, but Pool_A and Pool_B may produce outputs of
very different magnitudes due to independent training dynamics. The
downstream layers (attention, layer norm, lm_head) were calibrated for
the output distribution of a SINGLE pool during joint training, not
for the sum of two independently-trained pools. This is the "loudness
problem" -- one pool can shout over the other.

Three interventions tested:

1. RMSNormComposedGPT: Per-pool RMSNorm normalization at inference.
   Each pool's output is normalized to unit RMS before summing.
   True zero-shot: no calibration data needed.

2. MatchedMagnitudeGPT: Auxiliary loss during domain-specific fine-tuning
   that penalizes deviation from a target output norm (measured from the
   shared pretrained pool). Ensures pools produce matched magnitudes
   during training, enabling zero-shot composition.

3. Scalar calibration: Implemented in test_composition.py (imported from
   relu_router). Trains 1 scalar per pool per layer (8 params total for
   2 domains, 4 layers). Diagnostic: if scalar calibration matches full
   calibration, loudness is the SOLE issue.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..relu_router.relu_router import ReLUCapsulePool, ReLUCapsuleBlock, ReLURouterGPT


# ============================================================
# Intervention 1: Per-Pool RMSNorm Composition
# ============================================================

class RMSNormComposedPool(nn.Module):
    """Two independently-trained pools with per-pool RMSNorm before summing.

    Forward pass:
        y = RMSNorm(B_a @ ReLU(A_a @ x)) + RMSNorm(B_b @ ReLU(A_b @ x))

    This normalizes each pool's output to unit RMS independently, then sums.
    The normalization removes magnitude differences between pools while
    preserving direction (which encodes the actual information).

    The scale factor (target_rms) controls the magnitude of each pool's
    contribution. Default: target_rms = 1 / n_pools, so the sum of N
    normalized pools has roughly the same magnitude as a single pool.
    """

    def __init__(self, n_embd: int, n_capsules_per_pool: int,
                 n_pools: int = 2, target_rms: float = None):
        super().__init__()
        self.n_embd = n_embd
        self.n_capsules_per_pool = n_capsules_per_pool
        self.n_pools = n_pools
        # Default: scale so sum of N pools has ~same magnitude as 1 pool
        self.target_rms = target_rms if target_rms is not None else 1.0 / n_pools
        self.eps = 1e-5

        # Create separate pools for each domain
        self.pools = [
            ReLUCapsulePool(n_embd, n_capsules_per_pool)
            for _ in range(n_pools)
        ]

    def _rms_norm(self, x):
        """RMSNorm without learnable parameters: x / RMS(x) * target_rms."""
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(ms + self.eps) * self.target_rms

    def __call__(self, x):
        """x: (B, T, d) -> output: (B, T, d)

        Each pool processes x independently. Outputs are RMSNorm'd then summed.
        """
        result = mx.zeros_like(x)
        for pool in self.pools:
            pool_out = pool(x)
            result = result + self._rms_norm(pool_out)
        return result

    def aux_loss(self):
        total = mx.array(0.0)
        for pool in self.pools:
            total = total + pool.aux_loss()
        return total


class RMSNormComposedBlock(nn.Module):
    """Transformer block with RMSNormComposedPool replacing MLP."""

    def __init__(self, n_embd: int, n_head: int,
                 n_capsules_per_pool: int = 128,
                 n_pools: int = 2, target_rms: float = None):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = RMSNormComposedPool(
            n_embd, n_capsules_per_pool,
            n_pools=n_pools, target_rms=target_rms,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("rmsnorm_composed", parent="relu_router")
class RMSNormComposedGPT(nn.Module):
    """GPT with per-pool RMSNorm composition for zero-shot multi-domain.

    This model is NOT used during training. It is constructed at composition
    time from independently-trained ReLURouterGPT domain models. The pools
    are loaded from domain-specific models, and the per-pool RMSNorm
    normalizes their outputs before summing.

    The key property: this requires NO calibration data. The normalization
    is purely based on the output statistics of each pool for the current
    input, making it truly zero-shot.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_capsules_per_pool: int = 128,
                 n_pools: int = 2, target_rms: float = None):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [RMSNormComposedBlock(
            n_embd, n_head, n_capsules_per_pool,
            n_pools=n_pools, target_rms=target_rms,
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


# ============================================================
# Intervention 2: Matched-Magnitude Training
# ============================================================

class MatchedMagnitudePool(ReLUCapsulePool):
    """ReLUCapsulePool with an auxiliary loss that penalizes output norm deviation.

    During domain-specific fine-tuning, this loss ensures the pool's output
    RMS stays close to a target (measured from the pretrained base pool).
    This prevents magnitude drift during fine-tuning, enabling zero-shot
    composition: if all domain pools produce outputs of similar magnitude,
    their sum will not be dominated by one domain.

    L_mag = mag_coeff * (RMS(output) - target_rms)^2

    The target_rms is set from the pretrained model before fine-tuning.
    """

    def __init__(self, n_embd: int, n_capsules: int = 256,
                 mag_coeff: float = 1.0, target_rms: float = None,
                 l1_target_sparsity: float = 0.50,
                 l1_coeff: float = 0.01,
                 balance_coeff: float = 0.01):
        super().__init__(n_embd, n_capsules,
                         l1_target_sparsity=l1_target_sparsity,
                         l1_coeff=l1_coeff,
                         balance_coeff=balance_coeff)
        self.mag_coeff = mag_coeff
        self.target_rms = target_rms  # set before fine-tuning
        self._last_output_rms = None

    def __call__(self, x):
        """x: (B, T, d) -> output: (B, T, d). Also stores output RMS."""
        h = nn.relu(self.A(x))
        self._store_stats(h)
        out = self.B(h)
        # Store output RMS for magnitude loss
        self._last_output_rms = mx.sqrt(
            mx.mean(out * out) + 1e-8
        )
        return out

    def magnitude_loss(self) -> mx.array:
        """Penalize deviation of output RMS from target.

        Returns 0 if target_rms is not set (during pretraining).
        """
        if self.target_rms is None or self._last_output_rms is None:
            return mx.array(0.0)
        diff = self._last_output_rms - self.target_rms
        return self.mag_coeff * diff * diff

    def aux_loss(self) -> mx.array:
        """Combined: sparsity + balance + magnitude matching."""
        return self.sparsity_loss() + self.balance_loss() + self.magnitude_loss()


class MatchedMagnitudeBlock(nn.Module):
    """Transformer block with MatchedMagnitudePool."""

    def __init__(self, n_embd: int, n_head: int,
                 n_capsules: int = 256,
                 mag_coeff: float = 1.0,
                 target_rms: float = None):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.capsule_pool = MatchedMagnitudePool(
            n_embd, n_capsules,
            mag_coeff=mag_coeff,
            target_rms=target_rms,
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.capsule_pool(self.norm2(x))
        return x


@register("matched_magnitude", parent="relu_router")
class MatchedMagnitudeGPT(nn.Module):
    """GPT with matched-magnitude training for zero-shot composition.

    During pretraining: behaves exactly like ReLURouterGPT.
    Before domain fine-tuning: call set_target_rms() to record the
    pretrained pool's output magnitude as the target.
    During fine-tuning: auxiliary loss penalizes deviation from target,
    keeping output magnitudes matched across domains.
    At composition: concatenate pools from different domains. Because
    magnitudes are matched, zero-shot composition should work.
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_capsules: int = 256,
                 mag_coeff: float = 1.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [MatchedMagnitudeBlock(
            n_embd, n_head, n_capsules,
            mag_coeff=mag_coeff,
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

    def set_target_rms(self, dataset, batch_size=32, n_batches=5, seed=42):
        """Measure output RMS of each layer's pool and set as target.

        Call this AFTER pretraining, BEFORE domain fine-tuning.
        Runs a few forward passes to estimate the output RMS distribution.
        """
        import random
        rng = random.Random(seed)

        # Collect RMS per layer
        layer_rms = [[] for _ in self.layers]

        for _ in range(n_batches):
            inputs, _ = dataset.get_batch(batch_size, rng)
            _ = self(inputs)
            mx.eval(self.parameters())

            for l_idx, layer in enumerate(self.layers):
                rms = layer.capsule_pool._last_output_rms
                if rms is not None:
                    layer_rms[l_idx].append(rms.item())

        # Set target RMS as the mean observed RMS
        targets = []
        for l_idx, layer in enumerate(self.layers):
            if layer_rms[l_idx]:
                target = sum(layer_rms[l_idx]) / len(layer_rms[l_idx])
            else:
                target = 1.0  # fallback
            layer.capsule_pool.target_rms = target
            targets.append(target)

        return targets

    def capsule_stats(self) -> dict:
        """Activation statistics from last forward pass."""
        stats = {"sparsity": [], "mean_freq": [], "min_freq": [],
                 "max_freq": [], "n_dead": [], "output_rms": [],
                 "target_rms": []}

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
            stats["output_rms"].append(
                pool._last_output_rms.item() if pool._last_output_rms is not None else None)
            stats["target_rms"].append(pool.target_rms)

        return stats


# ============================================================
# Composition Utilities
# ============================================================

def compose_with_rmsnorm(base_model, domain_models, vocab_size,
                          n_capsules_per_pool, target_rms=None, n_embd=64,
                          n_head=4, n_layer=4, block_size=32):
    """Create an RMSNormComposedGPT from domain-specific ReLURouterGPT models.

    Loads the shared parameters (attention, embeddings) from base_model and
    the per-domain capsule weights from domain_models. The per-pool RMSNorm
    normalizes their outputs at inference time.
    """
    n_pools = len(domain_models)

    composed = RMSNormComposedGPT(
        vocab_size=vocab_size, block_size=block_size,
        n_embd=n_embd, n_head=n_head, n_layer=n_layer,
        n_capsules_per_pool=n_capsules_per_pool,
        n_pools=n_pools, target_rms=target_rms,
    )

    # Copy shared parameters from base model
    base_params = dict(nn.utils.tree_flatten(base_model.parameters()))
    shared_weights = [(k, v) for k, v in base_params.items()
                      if "capsule_pool" not in k]
    composed.load_weights(shared_weights, strict=False)

    # Copy domain-specific capsule weights into each pool
    for layer_idx in range(n_layer):
        for d_idx, dm in enumerate(domain_models):
            src_pool = dm.layers[layer_idx].capsule_pool
            tgt_pool = composed.layers[layer_idx].capsule_pool.pools[d_idx]
            tgt_pool.A.load_weights([("weight", src_pool.A.weight)])
            tgt_pool.B.load_weights([("weight", src_pool.B.weight)])

    mx.eval(composed.parameters())
    return composed


def compose_matched_magnitude(base_model, domain_models, vocab_size,
                               n_capsules_total, n_embd=64, n_head=4,
                               n_layer=4, block_size=32):
    """Compose MatchedMagnitudeGPT models by standard weight concatenation.

    This uses the same concatenation as the plain relu_router composition.
    The matched-magnitude training should have ensured pools produce similar
    output magnitudes, making the concatenated sum well-behaved.
    """
    from ..relu_router.test_composition import compose_relu_models
    return compose_relu_models(base_model, domain_models)
