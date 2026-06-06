"""Pruning Controls -- validate Exp 9 findings with two missing controls.

Control 1 (Pre-composition death rate):
  Profile single-domain models BEFORE composition to measure baseline
  death rate. Distinguishes training-induced ReLU death from composition-
  induced distribution shift.

Control 2 (Random pruning baseline):
  Prune the same fraction of capsules at random (not targeting dead ones)
  and measure degradation. Validates that targeted dead-capsule profiling
  adds real value over naive pruning.

The model class is a thin wrapper for lineage tracking. All forward pass
logic is inherited from ReLURouterGPT via DeadCapsulePruningGPT.
"""

import random as stdlib_random

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool
from ..dead_capsule_pruning.dead_capsule_pruning import (
    profile_activations,
    identify_dead_capsules,
    prune_model,
)


def profile_single_domain(model: ReLURouterGPT,
                           own_domain_dataset,
                           cross_domain_dataset,
                           n_batches: int = 20,
                           batch_size: int = 32,
                           seed: int = 0) -> dict:
    """Profile a single-domain model on both own and cross-domain data.

    Returns per-layer statistics showing how many capsules are dead
    on own-domain data vs cross-domain data.

    Args:
        model: A single-domain ReLURouterGPT (P capsules, not composed).
        own_domain_dataset: Dataset from the domain this model was trained on.
        cross_domain_dataset: Dataset from the other domain.
        n_batches: Number of profiling batches.
        batch_size: Profiling batch size.
        seed: RNG seed.

    Returns:
        Dict with per-layer frequency arrays and death statistics.
    """
    # Profile on own domain
    freqs_own = profile_activations(
        model, own_domain_dataset,
        n_batches=n_batches, batch_size=batch_size, seed=seed,
    )

    # Profile on cross domain
    freqs_cross = profile_activations(
        model, cross_domain_dataset,
        n_batches=n_batches, batch_size=batch_size, seed=seed,
    )

    # Compute statistics
    stats = {
        "per_layer": [],
        "freqs_own": freqs_own,
        "freqs_cross": freqs_cross,
    }

    for l_idx in range(len(freqs_own)):
        fo = freqs_own[l_idx]
        fc = freqs_cross[l_idx]
        mx.eval(fo, fc)

        fo_list = fo.tolist()
        fc_list = fc.tolist()
        P = len(fo_list)

        dead_own = sum(1 for f in fo_list if f == 0.0)
        dead_cross = sum(1 for f in fc_list if f == 0.0)
        dead_both = sum(1 for a, b in zip(fo_list, fc_list)
                        if a == 0.0 and b == 0.0)
        alive_own_dead_cross = sum(1 for a, b in zip(fo_list, fc_list)
                                    if a > 0.0 and b == 0.0)

        layer_stats = {
            "layer": l_idx,
            "n_capsules": P,
            "dead_own": dead_own,
            "dead_cross": dead_cross,
            "dead_both": dead_both,
            "alive_own_dead_cross": alive_own_dead_cross,
            "pct_dead_own": dead_own / P * 100,
            "pct_dead_cross": dead_cross / P * 100,
            "pct_dead_both": dead_both / P * 100,
            "pct_alive_own_dead_cross": alive_own_dead_cross / P * 100,
        }
        stats["per_layer"].append(layer_stats)

    # Aggregate across layers
    total_P = sum(s["n_capsules"] for s in stats["per_layer"])
    total_dead_own = sum(s["dead_own"] for s in stats["per_layer"])
    total_dead_cross = sum(s["dead_cross"] for s in stats["per_layer"])
    total_dead_both = sum(s["dead_both"] for s in stats["per_layer"])

    stats["aggregate"] = {
        "total_capsules": total_P,
        "pct_dead_own": total_dead_own / total_P * 100,
        "pct_dead_cross": total_dead_cross / total_P * 100,
        "pct_dead_both": total_dead_both / total_P * 100,
    }

    return stats


def random_prune_model(model: ReLURouterGPT,
                        target_prune_rate: float,
                        seed: int = 0,
                        verbose: bool = True) -> dict:
    """Prune capsules at random (uniform, not targeting dead ones).

    For each layer, randomly selects capsules to prune such that
    the overall prune rate matches the target. The random selection
    is independent of activation frequency.

    Args:
        model: ReLURouterGPT to prune in-place.
        target_prune_rate: Fraction of capsules to prune (0 to 1).
        seed: RNG seed for reproducible random selection.
        verbose: Print per-layer statistics.

    Returns:
        Dict with pruning statistics.
    """
    rng = stdlib_random.Random(seed)
    stats = {
        "per_layer": [],
        "total_before": 0,
        "total_after": 0,
        "total_pruned": 0,
    }

    alive_masks = []
    for l_idx, layer in enumerate(model.layers):
        pool = layer.capsule_pool
        P = pool.n_capsules

        # Number to prune in this layer (proportional)
        n_prune = int(round(P * target_prune_rate))
        n_prune = min(n_prune, P - 1)  # Keep at least 1

        # Random selection of capsules to KEEP
        all_indices = list(range(P))
        prune_set = set(rng.sample(all_indices, n_prune))

        # Build alive mask (True = keep, False = prune)
        mask = mx.array([i not in prune_set for i in range(P)])
        alive_masks.append(mask)

        n_alive = P - n_prune
        layer_stats = {
            "layer": l_idx,
            "P_before": P,
            "P_after": n_alive,
            "n_pruned": n_prune,
            "pct_pruned": n_prune / P * 100,
        }
        stats["per_layer"].append(layer_stats)
        stats["total_before"] += P
        stats["total_after"] += n_alive
        stats["total_pruned"] += n_prune

    total_pct = (stats["total_pruned"] / stats["total_before"] * 100
                 if stats["total_before"] > 0 else 0)
    stats["pct_pruned"] = total_pct

    # Apply the pruning using the same prune_model function
    prune_model(model, alive_masks, verbose=verbose)

    return stats


@register("pruning_controls", parent="dead_capsule_pruning")
class PruningControlsGPT(ReLURouterGPT):
    """ReLURouterGPT used for pruning control experiments.

    This model IS a ReLURouterGPT. The experiment applies pruning
    controls (pre-composition profiling and random pruning baseline)
    as post-processing steps. No architectural changes.

    Registered for lineage tracking only.
    """
    pass
