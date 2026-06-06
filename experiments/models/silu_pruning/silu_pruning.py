"""SiLU Pruning -- magnitude-threshold pruning for SiLU capsule MLPs.

ReLU produces exact zeros, enabling lossless dead capsule pruning (Exp 9:
57% pruned, 0% quality loss). SiLU(x) = x * sigmoid(x) never hits exact
zero -- its minimum is ~-0.278 at x ~ -1.28. This means the binary
dead/alive classification from ReLU pruning does not transfer.

This experiment answers: can we define a magnitude threshold below which
SiLU capsules are "functionally dead" and prune them without quality loss?

The approach:
  1. Profile mean absolute activation per capsule over calibration data
  2. Capsules with mean|activation| < tau are "functionally dead"
  3. Prune by removing their rows from A and columns from B
  4. Unlike ReLU (exact zero), SiLU pruning is APPROXIMATE:
     the error is bounded by tau * ||b_i||

Kill criterion: magnitude-threshold pruning degrades quality >5% vs unpruned.
"""

import copy
import random
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..silu_capsule.silu_capsule import SiLUCapsuleGPT, SiLUCapsulePool


def profile_silu_activations(model: SiLUCapsuleGPT,
                              dataset,
                              n_batches: int = 20,
                              batch_size: int = 32,
                              seed: int = 0) -> list[dict]:
    """Profile per-capsule activation magnitudes across the dataset.

    For each layer, computes:
      - mean_abs: mean absolute activation per capsule (P,)
      - max_abs: max absolute activation per capsule (P,)
      - freq_above_tau: fraction of positions where |activation| > tau

    Args:
        model: A SiLUCapsuleGPT (possibly composed).
        dataset: A CharDataset to draw batches from.
        n_batches: Number of batches to profile over.
        batch_size: Batch size for profiling.
        seed: RNG seed for reproducible batch selection.

    Returns:
        List of dicts (one per layer) with activation statistics.
    """
    rng = random.Random(seed)
    n_layers = len(model.layers)

    # Accumulators
    sum_abs = [mx.zeros(layer.capsule_pool.n_capsules) for layer in model.layers]
    sum_sq = [mx.zeros(layer.capsule_pool.n_capsules) for layer in model.layers]
    max_abs = [mx.zeros(layer.capsule_pool.n_capsules) for layer in model.layers]
    total_positions = 0

    # Threshold sweep: count positions above each threshold
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    freq_above = {tau: [mx.zeros(layer.capsule_pool.n_capsules)
                        for layer in model.layers]
                  for tau in thresholds}

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
            h = nn.silu(pool.A(x_norm))  # (B, T, P)

            # Accumulate statistics
            abs_h = mx.abs(h)  # (B, T, P)
            sum_abs[l_idx] = sum_abs[l_idx] + mx.sum(abs_h, axis=(0, 1))
            sum_sq[l_idx] = sum_sq[l_idx] + mx.sum(h * h, axis=(0, 1))
            max_abs[l_idx] = mx.maximum(max_abs[l_idx],
                                         mx.max(abs_h, axis=(0, 1)))

            # Frequency above thresholds
            for tau in thresholds:
                above = (abs_h > tau).astype(mx.float32)
                freq_above[tau][l_idx] = freq_above[tau][l_idx] + mx.sum(above, axis=(0, 1))

            mx.eval(sum_abs[l_idx], sum_sq[l_idx], max_abs[l_idx])
            for tau in thresholds:
                mx.eval(freq_above[tau][l_idx])

            # Complete the layer forward pass
            x = x + pool.B(h)

    # Compute final statistics
    results = []
    for l_idx in range(n_layers):
        P = model.layers[l_idx].capsule_pool.n_capsules
        mean_abs_arr = sum_abs[l_idx] / total_positions
        mean_sq_arr = sum_sq[l_idx] / total_positions
        std_arr = mx.sqrt(mx.maximum(mean_sq_arr - (sum_abs[l_idx] / total_positions) ** 2,
                                      mx.zeros(P)))
        mx.eval(mean_abs_arr, max_abs[l_idx], std_arr)

        freq_dict = {}
        for tau in thresholds:
            f = freq_above[tau][l_idx] / total_positions
            mx.eval(f)
            freq_dict[tau] = f

        results.append({
            "mean_abs": mean_abs_arr,
            "max_abs": max_abs[l_idx],
            "std_activation": std_arr,
            "freq_above": freq_dict,
            "n_capsules": P,
        })

    return results


def identify_prunable_capsules(profiles: list[dict],
                                threshold: float = 0.01,
                                method: str = "mean_abs") -> list[mx.array]:
    """Identify capsules to prune based on activation magnitude.

    Args:
        profiles: Per-layer profiles from profile_silu_activations.
        threshold: Magnitude threshold. Capsules below this are prunable.
        method: "mean_abs" (mean absolute activation) or
                "max_abs" (maximum absolute activation across all inputs).

    Returns:
        List of boolean mask arrays (True = alive, False = prune).
    """
    masks = []
    for prof in profiles:
        if method == "mean_abs":
            metric = prof["mean_abs"]
        elif method == "max_abs":
            metric = prof["max_abs"]
        else:
            raise ValueError(f"Unknown method: {method}")

        alive = metric > threshold
        mx.eval(alive)
        masks.append(alive)
    return masks


def prune_silu_model(model: SiLUCapsuleGPT,
                      alive_masks: list[mx.array],
                      verbose: bool = True) -> dict:
    """Prune capsules from the model in-place.

    Unlike ReLU pruning (exact zero change), SiLU pruning is APPROXIMATE.
    The error per pruned capsule i is bounded by:
        ||delta_y|| <= mean(|SiLU(a_i^T x)|) * ||b_i||

    For capsules with mean_abs < tau, this error is bounded by tau * ||b_i||.

    Args:
        model: SiLUCapsuleGPT to prune in-place.
        alive_masks: Per-layer boolean masks from identify_prunable_capsules.
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
            layer_stats = {
                "layer": l_idx,
                "P_before": P_before,
                "P_after": P_before,
                "n_pruned": 0,
                "pct_pruned": 0.0,
            }
        elif n_alive == 0:
            # Keep at least 1 capsule
            alive_indices = mx.array([0])
            n_alive = 1
            A_new = A[0:1]
            B_new = B[:, 0:1]

            new_pool = SiLUCapsulePool(d, 1)
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
            A_new = A[alive_indices]  # (n_alive, d)
            B_new = B[:, alive_indices]  # (d, n_alive)
            mx.eval(A_new, B_new)

            new_pool = SiLUCapsulePool(d, n_alive)
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


def full_silu_pruning_pipeline(model: SiLUCapsuleGPT,
                                dataset,
                                threshold: float = 0.01,
                                method: str = "mean_abs",
                                n_batches: int = 20,
                                batch_size: int = 32,
                                seed: int = 0,
                                verbose: bool = True) -> dict:
    """Full pipeline: profile -> identify -> prune.

    Args:
        model: SiLUCapsuleGPT to prune in-place.
        dataset: Calibration dataset for activation profiling.
        threshold: Magnitude threshold for pruning.
        method: "mean_abs" or "max_abs".
        n_batches: Number of profiling batches.
        batch_size: Profiling batch size.
        seed: RNG seed.
        verbose: Print progress.

    Returns:
        Dict with pruning statistics and activation profiles.
    """
    if verbose:
        print(f"  Profiling SiLU activations ({n_batches} batches, method={method}, tau={threshold})...")

    profiles = profile_silu_activations(model, dataset, n_batches, batch_size, seed)

    # Report activation distribution before pruning
    if verbose:
        for l_idx, prof in enumerate(profiles):
            mean_abs = prof["mean_abs"]
            mx.eval(mean_abs)
            mean_vals = mean_abs.tolist()
            n_below = sum(1 for v in mean_vals if v <= threshold)
            print(f"    Layer {l_idx}: {n_below}/{len(mean_vals)} below tau={threshold:.4f}, "
                  f"mean_abs range [{min(mean_vals):.6f}, {max(mean_vals):.6f}]")

    alive_masks = identify_prunable_capsules(profiles, threshold, method)

    if verbose:
        print(f"  Pruning...")

    prune_stats = prune_silu_model(model, alive_masks, verbose)
    prune_stats["profiles"] = [{
        "mean_abs_min": float(mx.min(p["mean_abs"]).item()),
        "mean_abs_max": float(mx.max(p["mean_abs"]).item()),
        "mean_abs_median": float(sorted(p["mean_abs"].tolist())[p["n_capsules"] // 2]),
        "max_abs_min": float(mx.min(p["max_abs"]).item()),
        "max_abs_max": float(mx.max(p["max_abs"]).item()),
        "n_capsules": p["n_capsules"],
    } for p in profiles]
    prune_stats["threshold"] = threshold
    prune_stats["method"] = method

    return prune_stats


@register("silu_pruning", parent="silu_capsule")
class SiLUPruningGPT(SiLUCapsuleGPT):
    """SiLUCapsuleGPT with post-hoc magnitude-threshold pruning.

    This model IS a SiLUCapsuleGPT. The pruning is applied as a
    post-processing step after training or composition, not as an
    architectural change.

    The model class is registered for lineage tracking. All forward pass
    logic is inherited from SiLUCapsuleGPT.
    """
    pass
