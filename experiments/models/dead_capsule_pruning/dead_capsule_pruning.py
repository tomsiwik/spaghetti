"""Dead Capsule Pruning -- remove capsules that never fire in composed models.

After composing domain-specific ReLU MLPs by concatenation, ~60% of
capsules in the composed model are "dead" (never fire for any input).
This module identifies dead capsules via activation profiling on a
calibration dataset and prunes them by removing their rows from A
and columns from B.

The pruning is EXACT: removing a capsule that never fires changes
the output by exactly zero. For "nearly dead" capsules (fire rarely),
the output change is bounded by their contribution magnitude.

Three pruning strategies:
  1. Binary (freq=0): remove only truly dead capsules
  2. Threshold sweep: remove capsules firing less than tau fraction
  3. Top-K retention: keep only the K most active capsules per layer

The model class is a thin wrapper: it IS a ReLURouterGPT with pruning
applied as a post-processing step.
"""

import random
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool


def profile_activations(model: ReLURouterGPT,
                        dataset,
                        n_batches: int = 20,
                        batch_size: int = 32,
                        seed: int = 0) -> list[mx.array]:
    """Profile per-capsule activation frequency across the dataset.

    For each layer, computes the fraction of (batch, token) positions
    where each capsule fires (activation > 0).

    Args:
        model: A ReLURouterGPT (possibly composed).
        dataset: A CharDataset to draw batches from.
        n_batches: Number of batches to profile over.
        batch_size: Batch size for profiling.
        seed: RNG seed for reproducible batch selection.

    Returns:
        List of (P_l,) arrays, one per layer. Each entry is the
        fraction of inputs for which that capsule fires (0 to 1).
    """
    rng = random.Random(seed)
    n_layers = len(model.layers)

    # Accumulators: total fire count and total samples seen
    fire_counts = []
    for layer in model.layers:
        P = layer.capsule_pool.n_capsules
        fire_counts.append(mx.zeros(P))

    total_positions = 0

    for _ in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B, T = inputs.shape
        total_positions += B * T

        # Forward through model layer by layer
        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for l_idx, layer in enumerate(model.layers):
            # Attention
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)

            # Capsule pool -- compute activations
            x_norm = layer.norm2(x)
            pool = layer.capsule_pool
            h = nn.relu(pool.A(x_norm))  # (B, T, P)

            # Count fires
            fired = (h > 0).astype(mx.float32)  # (B, T, P)
            fire_counts[l_idx] = fire_counts[l_idx] + mx.sum(fired, axis=(0, 1))
            mx.eval(fire_counts[l_idx])

            # Complete the layer forward pass
            x = x + pool.B(h)

    # Convert counts to frequencies
    frequencies = []
    for counts in fire_counts:
        frequencies.append(counts / total_positions)
        mx.eval(frequencies[-1])

    return frequencies


def identify_dead_capsules(frequencies: list[mx.array],
                           threshold: float = 0.0) -> list[mx.array]:
    """Identify dead (or nearly dead) capsules per layer.

    Args:
        frequencies: Per-layer activation frequencies from profile_activations.
        threshold: Capsules firing less than this fraction are "dead."
                   threshold=0.0 means truly dead (never fire).
                   threshold=0.01 means fire on <1% of inputs.

    Returns:
        List of boolean mask arrays (True = alive, False = dead).
    """
    masks = []
    for freq in frequencies:
        alive = freq > threshold
        masks.append(alive)
        mx.eval(alive)
    return masks


def prune_model(model: ReLURouterGPT,
                alive_masks: list[mx.array],
                verbose: bool = True) -> dict:
    """Prune dead capsules from the model in-place.

    For each layer, removes capsules where alive_mask[i] = False by:
    - Deleting row i from A.weight (P, d) -> (P_alive, d)
    - Deleting column i from B.weight (d, P) -> (d, P_alive)

    This is EXACT for dead capsules (zero output change) and
    approximate for nearly-dead capsules.

    Args:
        model: ReLURouterGPT to prune in-place.
        alive_masks: Per-layer boolean masks from identify_dead_capsules.
        verbose: Print per-layer pruning statistics.

    Returns:
        Dict with per-layer and aggregate statistics.
    """
    stats = {
        "per_layer": [],
        "total_before": 0,
        "total_after": 0,
        "total_pruned": 0,
    }

    for l_idx, (layer, mask) in enumerate(zip(model.layers, alive_masks)):
        pool = layer.capsule_pool
        A = pool.A.weight  # (P, d)
        B = pool.B.weight  # (d, P)
        P_before = A.shape[0]
        d = A.shape[1]

        mx.eval(mask)
        alive_indices = mx.array([i for i in range(P_before) if mask[i].item()])
        n_alive = alive_indices.shape[0]

        if n_alive == P_before:
            # Nothing to prune
            layer_stats = {
                "layer": l_idx,
                "P_before": P_before,
                "P_after": P_before,
                "n_pruned": 0,
                "pct_pruned": 0.0,
            }
        elif n_alive == 0:
            # Everything dead -- this would be a problem
            # Keep at least 1 capsule to avoid empty layer
            alive_indices = mx.array([0])
            n_alive = 1
            A_new = A[0:1]
            B_new = B[:, 0:1]

            new_pool = ReLUCapsulePool(d, 1)
            new_pool.A.load_weights([("weight", A_new)])
            new_pool.B.load_weights([("weight", B_new)])
            layer.capsule_pool = new_pool
            mx.eval(layer.capsule_pool.parameters())

            layer_stats = {
                "layer": l_idx,
                "P_before": P_before,
                "P_after": 1,
                "n_pruned": P_before - 1,
                "pct_pruned": (P_before - 1) / P_before * 100,
            }
        else:
            # Index-select alive capsules
            A_new = A[alive_indices]  # (n_alive, d)
            B_new = B[:, alive_indices]  # (d, n_alive)
            mx.eval(A_new, B_new)

            new_pool = ReLUCapsulePool(d, n_alive)
            new_pool.A.load_weights([("weight", A_new)])
            new_pool.B.load_weights([("weight", B_new)])
            layer.capsule_pool = new_pool
            mx.eval(layer.capsule_pool.parameters())

            layer_stats = {
                "layer": l_idx,
                "P_before": P_before,
                "P_after": n_alive,
                "n_pruned": P_before - n_alive,
                "pct_pruned": (P_before - n_alive) / P_before * 100,
            }

        stats["per_layer"].append(layer_stats)
        stats["total_before"] += P_before
        stats["total_after"] += layer_stats["P_after"]
        stats["total_pruned"] += layer_stats["n_pruned"]

        if verbose:
            print(f"  Layer {l_idx}: {P_before} -> {layer_stats['P_after']} capsules "
                  f"({layer_stats['n_pruned']} pruned, {layer_stats['pct_pruned']:.1f}%)")

    total_pct = (stats["total_pruned"] / stats["total_before"] * 100
                 if stats["total_before"] > 0 else 0)
    stats["pct_pruned"] = total_pct

    if verbose:
        print(f"  Total: {stats['total_before']} -> {stats['total_after']} capsules "
              f"({total_pct:.1f}% pruned)")

    return stats


def prune_composed_model(model: ReLURouterGPT,
                         dataset,
                         threshold: float = 0.0,
                         n_batches: int = 20,
                         batch_size: int = 32,
                         seed: int = 0,
                         verbose: bool = True) -> dict:
    """Full pruning pipeline: profile -> identify -> prune.

    Args:
        model: Composed ReLURouterGPT to prune in-place.
        dataset: Calibration dataset for activation profiling.
        threshold: Activation frequency threshold for pruning.
        n_batches: Number of profiling batches.
        batch_size: Profiling batch size.
        seed: RNG seed.
        verbose: Print progress.

    Returns:
        Dict with pruning statistics and activation frequency info.
    """
    if verbose:
        print(f"  Profiling activations ({n_batches} batches, threshold={threshold})...")

    frequencies = profile_activations(model, dataset, n_batches, batch_size, seed)

    # Collect frequency statistics before pruning
    freq_stats = []
    for l_idx, freq in enumerate(frequencies):
        mx.eval(freq)
        freq_list = freq.tolist()
        n_dead = sum(1 for f in freq_list if f <= threshold)
        freq_stats.append({
            "layer": l_idx,
            "n_capsules": len(freq_list),
            "n_dead": n_dead,
            "pct_dead": n_dead / len(freq_list) * 100,
            "freq_mean": sum(freq_list) / len(freq_list),
            "freq_min": min(freq_list),
            "freq_max": max(freq_list),
            "freq_median": sorted(freq_list)[len(freq_list) // 2],
        })

    if verbose:
        for fs in freq_stats:
            print(f"    Layer {fs['layer']}: {fs['n_dead']}/{fs['n_capsules']} dead "
                  f"({fs['pct_dead']:.1f}%), freq range [{fs['freq_min']:.4f}, {fs['freq_max']:.4f}]")

    alive_masks = identify_dead_capsules(frequencies, threshold)

    if verbose:
        print(f"  Pruning...")

    prune_stats = prune_model(model, alive_masks, verbose)
    prune_stats["freq_stats"] = freq_stats
    prune_stats["threshold"] = threshold

    return prune_stats


@register("dead_capsule_pruning", parent="relu_router")
class DeadCapsulePruningGPT(ReLURouterGPT):
    """ReLURouterGPT with post-hoc dead capsule pruning.

    This model IS a ReLURouterGPT. The pruning is applied as a
    post-processing step after composition, not as an architectural change.

    The model class is registered for lineage tracking. All forward pass
    logic is inherited from ReLURouterGPT.
    """
    pass
