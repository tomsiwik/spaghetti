"""Activation-Based Behavioral Deduplication.

Exp 8 (capsule_dedup) tested weight-cosine deduplication and found only 1.9%
redundancy at cos>0.95. This was KILLED because shared knowledge is distributed,
not concentrated in similar-weight capsules.

This experiment tests whether ACTIVATION-BASED analysis finds functional
duplicates that weight-cosine misses. Key insight: ReLU creates many-to-one
mappings -- different weight vectors can produce identical activation patterns
on the actual data distribution.

Three behavioral similarity metrics:
  1. Co-activation Jaccard: J(i,j) = |fire_i AND fire_j| / |fire_i OR fire_j|
  2. Output correlation: Pearson(b_i*h_i, b_j*h_j) across positions
  3. Activation-conditioned output cosine: cos(b_i*h_i, b_j*h_j) where both fire

The model class is a thin wrapper: it IS a ReLURouterGPT with behavioral
deduplication applied as a post-processing step.
"""

import random
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool
from ..capsule_dedup.capsule_dedup import (
    cosine_similarity_matrix,
    find_redundant_clusters,
    merge_capsules,
)


# ---------------------------------------------------------------------------
# 1. Behavioral profiling: collect per-capsule activation vectors
# ---------------------------------------------------------------------------

def profile_behavioral(model: ReLURouterGPT,
                       dataset,
                       n_batches: int = 20,
                       batch_size: int = 32,
                       seed: int = 0) -> list[dict]:
    """Profile per-capsule behavioral statistics across the dataset.

    For each layer, collects:
      - co_fire: (P, P) matrix counting co-activation events
      - fire_count: (P,) count of positions where each capsule fires
      - output_sum: (P, d) sum of output contributions b_i * h_i
      - output_sq_sum: (P, d) sum of squared output contributions
      - output_cross: (P, P) pairwise dot products of output contributions
      - total_positions: int

    Args:
        model: A ReLURouterGPT (possibly composed).
        dataset: A CharDataset to draw batches from.
        n_batches: Number of batches to profile over.
        batch_size: Batch size for profiling.
        seed: RNG seed for reproducible batch selection.

    Returns:
        List of dicts, one per layer, with accumulated statistics.
    """
    rng = random.Random(seed)
    n_layers = len(model.layers)

    # Initialize accumulators
    layer_stats = []
    for layer in model.layers:
        P = layer.capsule_pool.n_capsules
        d = layer.capsule_pool.A.weight.shape[1]
        layer_stats.append({
            "co_fire": mx.zeros((P, P)),       # (P, P)
            "fire_count": mx.zeros((P,)),       # (P,)
            "output_dot": mx.zeros((P, P)),     # (P, P) pairwise output dots
            "total_positions": 0,
            "P": P,
            "d": d,
        })

    for batch_idx in range(n_batches):
        inputs, _ = dataset.get_batch(batch_size, rng)
        B_size, T = inputs.shape
        n_pos = B_size * T

        # Forward through model layer by layer
        pos = mx.arange(T)
        x = model.wte(inputs) + model.wpe(pos)
        x = model.norm0(x)

        for l_idx, layer in enumerate(model.layers):
            # Attention
            x_norm = layer.norm1(x)
            x = x + layer.attn(x_norm)

            # Capsule pool -- compute activations
            x_norm2 = layer.norm2(x)
            pool = layer.capsule_pool
            h = nn.relu(pool.A(x_norm2))  # (B, T, P)

            # Binary fire mask
            fired = (h > 0).astype(mx.float32)  # (B, T, P)

            # Flatten to (N, P) where N = B*T
            fired_flat = fired.reshape(-1, layer_stats[l_idx]["P"])  # (N, P)
            h_flat = h.reshape(-1, layer_stats[l_idx]["P"])          # (N, P)

            # Co-activation: fired^T @ fired gives (P, P) co-fire counts
            co = fired_flat.T @ fired_flat  # (P, P)
            layer_stats[l_idx]["co_fire"] = layer_stats[l_idx]["co_fire"] + co

            # Fire counts
            fc = mx.sum(fired_flat, axis=0)  # (P,)
            layer_stats[l_idx]["fire_count"] = layer_stats[l_idx]["fire_count"] + fc

            # Output contributions: each capsule's output is b_i * h_i
            # For output correlation, compute pairwise: sum over positions of
            # (h_i * h_j) * (b_i . b_j) but that factorizes wrong.
            # Instead: output_i = h_i * b_i^T (scalar * d-vector)
            # Dot(output_i, output_j) = h_i * h_j * (b_i . b_j)
            # Sum over positions: sum(h_i * h_j) * (b_i . b_j)
            # = (h^T @ h)[i,j] * (B^T @ B)[i,j]  -- BUT h^T@h is element-wise
            # Actually: for position n, output_n_i = h_n_i * b_i (d-dim vector)
            # dot(output_n_i, output_n_j) = h_n_i * h_n_j * (b_i . b_j)
            # sum over n: (sum_n h_n_i * h_n_j) * (b_i . b_j)
            # = (h^T @ h)[i,j] * (b_i . b_j)

            # h^T @ h gives activation magnitude co-occurrence
            hh = h_flat.T @ h_flat  # (P, P)
            layer_stats[l_idx]["output_dot"] = layer_stats[l_idx]["output_dot"] + hh

            layer_stats[l_idx]["total_positions"] += n_pos

            # Complete the layer forward pass
            x = x + pool.B(h)

        # Eval periodically to avoid memory buildup
        if batch_idx % 5 == 0:
            for ls in layer_stats:
                mx.eval(ls["co_fire"], ls["fire_count"], ls["output_dot"])

    # Final eval
    for ls in layer_stats:
        mx.eval(ls["co_fire"], ls["fire_count"], ls["output_dot"])

    return layer_stats


# ---------------------------------------------------------------------------
# 2. Compute behavioral similarity matrices
# ---------------------------------------------------------------------------

def compute_jaccard_matrix(stats: dict) -> mx.array:
    """Compute pairwise co-activation Jaccard similarity.

    J(i,j) = |fire_i AND fire_j| / |fire_i OR fire_j|
           = co_fire[i,j] / (fire_count[i] + fire_count[j] - co_fire[i,j])

    Args:
        stats: Per-layer stats dict from profile_behavioral.

    Returns:
        (P, P) Jaccard similarity matrix.
    """
    co = stats["co_fire"]           # (P, P) co-fire counts
    fc = stats["fire_count"]        # (P,) individual fire counts

    # Union = fire_i + fire_j - intersection
    # fc_i + fc_j as outer sum
    fc_sum = fc[:, None] + fc[None, :]  # (P, P)
    union = fc_sum - co                  # (P, P)

    # Avoid division by zero (both dead capsules)
    J = co / (union + 1e-8)
    mx.eval(J)
    return J


def compute_output_correlation_matrix(stats: dict, B_weight: mx.array) -> mx.array:
    """Compute pairwise output contribution correlation.

    For each pair (i,j):
      sum_n dot(output_n_i, output_n_j) = (sum_n h_n_i * h_n_j) * (b_i . b_j)
                                         = output_dot[i,j] * b_dot[i,j]

    Normalize by geometric mean of self-correlations to get correlation.

    Args:
        stats: Per-layer stats dict from profile_behavioral.
        B_weight: (d, P) expansion matrix.

    Returns:
        (P, P) output correlation matrix in [-1, 1].
    """
    hh = stats["output_dot"]  # (P, P) sum of h_i * h_j over positions

    # b_i . b_j for all pairs
    # B_weight is (d, P), columns are b_i
    b_dot = B_weight.T @ B_weight  # (P, P)

    # Combined: contribution_dot[i,j] = hh[i,j] * b_dot[i,j]
    contrib_dot = hh * b_dot  # (P, P) element-wise

    # Normalize: corr[i,j] = contrib_dot[i,j] / sqrt(contrib_dot[i,i] * contrib_dot[j,j])
    diag = mx.diag(contrib_dot)  # (P,)
    denom = mx.sqrt(mx.abs(diag[:, None] * diag[None, :]) + 1e-12)
    corr = contrib_dot / denom

    mx.eval(corr)
    return corr


def compute_conditioned_output_cosine(stats: dict, B_weight: mx.array) -> mx.array:
    """Compute output cosine conditioned on both capsules firing.

    For pair (i,j), consider only positions where BOTH fire.
    The output cosine = (b_i . b_j) * sign, where sign depends on
    whether the activations are positively correlated.

    For rank-1 capsules, this simplifies to:
      cos_out(i,j) = (b_i . b_j) / (||b_i|| * ||b_j||)
    because conditioning on both firing removes the gating effect.

    Note: This is actually just the weight-space b-vector cosine!
    The conditioning on co-activation is relevant only when
    combined with the Jaccard (which measures HOW OFTEN they co-fire).

    Args:
        stats: Per-layer stats dict.
        B_weight: (d, P) expansion matrix.

    Returns:
        (P, P) cosine similarity of b vectors.
    """
    # b vectors: columns of B_weight
    B_t = B_weight.T  # (P, d)
    norms = mx.sqrt(mx.sum(B_t * B_t, axis=1, keepdims=True) + 1e-8)
    B_normed = B_t / norms
    cos_b = B_normed @ B_normed.T  # (P, P)
    mx.eval(cos_b)
    return cos_b


# ---------------------------------------------------------------------------
# 3. Find behaviorally redundant pairs (above Jaccard threshold)
# ---------------------------------------------------------------------------

def find_behavioral_redundant(J: mx.array,
                              corr: mx.array,
                              cos_b: mx.array,
                              jaccard_threshold: float = 0.7,
                              output_corr_threshold: float = 0.5,
                              pool_sizes: list[int] | None = None,
                              exclude_dead: mx.array | None = None,
                              ) -> list[dict]:
    """Find pairs that are behaviorally redundant.

    A pair (i,j) is behaviorally redundant if:
      1. J(i,j) > jaccard_threshold  (they fire on similar inputs)
      2. corr(i,j) > output_corr_threshold  (their outputs are similar)
      3. Neither is dead (freq > 0)

    Args:
        J: (P, P) Jaccard similarity matrix.
        corr: (P, P) output correlation matrix.
        cos_b: (P, P) b-vector cosine similarity.
        jaccard_threshold: minimum Jaccard for co-activation redundancy.
        output_corr_threshold: minimum output correlation.
        pool_sizes: for cross-pool filtering.
        exclude_dead: (P,) boolean mask, True = alive.

    Returns:
        List of dicts with pair info, sorted by Jaccard descending.
    """
    mx.eval(J, corr, cos_b)
    P = J.shape[0]
    J_np = J.tolist()
    corr_np = corr.tolist()
    cos_b_np = cos_b.tolist()

    if exclude_dead is not None:
        mx.eval(exclude_dead)
        alive = exclude_dead.tolist()
    else:
        alive = [True] * P

    pairs = []
    for i in range(P):
        if not alive[i]:
            continue
        for j in range(i + 1, P):
            if not alive[j]:
                continue

            # Cross-pool filter
            if pool_sizes is not None:
                pool_i = _get_pool_index(i, pool_sizes)
                pool_j = _get_pool_index(j, pool_sizes)
                if pool_i == pool_j:
                    continue

            jac = J_np[i][j]
            out_corr = corr_np[i][j]
            b_cos = cos_b_np[i][j]

            if jac > jaccard_threshold and out_corr > output_corr_threshold:
                pairs.append({
                    "i": i, "j": j,
                    "jaccard": jac,
                    "output_corr": out_corr,
                    "b_cosine": b_cos,
                })

    pairs.sort(key=lambda p: p["jaccard"], reverse=True)
    return pairs


def _get_pool_index(capsule_idx: int, pool_sizes: list[int]) -> int:
    """Determine which pool a capsule belongs to."""
    cumsum = 0
    for pool_idx, size in enumerate(pool_sizes):
        cumsum += size
        if capsule_idx < cumsum:
            return pool_idx
    return len(pool_sizes) - 1


# ---------------------------------------------------------------------------
# 4. Compare behavioral vs weight-cosine redundancy
# ---------------------------------------------------------------------------

def compare_behavioral_vs_weight(model: ReLURouterGPT,
                                 dataset,
                                 n_batches: int = 20,
                                 batch_size: int = 32,
                                 seed: int = 0,
                                 jaccard_threshold: float = 0.7,
                                 output_corr_threshold: float = 0.5,
                                 weight_cos_threshold: float = 0.95,
                                 pool_sizes: list[int] | None = None,
                                 verbose: bool = True) -> dict:
    """Run full behavioral vs weight-cosine comparison.

    Returns:
        Dict with per-layer and aggregate comparison statistics.
    """
    # Profile behavioral
    if verbose:
        print("  Profiling behavioral statistics...")
    layer_stats = profile_behavioral(model, dataset, n_batches, batch_size, seed)

    results = {
        "per_layer": [],
        "total_weight_redundant": 0,
        "total_behavioral_redundant": 0,
        "total_behavioral_only": 0,  # found by behavioral but NOT weight
        "total_both": 0,             # found by both
        "total_weight_only": 0,      # found by weight but NOT behavioral
        "total_alive_capsules": 0,
        "total_capsules": 0,
    }

    for l_idx, layer in enumerate(model.layers):
        pool = layer.capsule_pool
        A = pool.A.weight  # (P, d)
        B = pool.B.weight  # (d, P)
        P = A.shape[0]
        stats = layer_stats[l_idx]

        # Dead capsule mask
        freq = stats["fire_count"] / max(stats["total_positions"], 1)
        alive_mask = freq > 0
        mx.eval(alive_mask)
        n_alive = int(mx.sum(alive_mask.astype(mx.float32)).item())

        # Weight-cosine similarity
        S_weight = cosine_similarity_matrix(A)
        mx.eval(S_weight)

        # Behavioral metrics
        J = compute_jaccard_matrix(stats)
        corr = compute_output_correlation_matrix(stats, B)
        cos_b = compute_conditioned_output_cosine(stats, B)

        # Find weight-cosine redundant pairs (among alive capsules)
        weight_pairs = set()
        S_w = S_weight.tolist()
        alive_list = alive_mask.tolist() if hasattr(alive_mask, 'tolist') else [True] * P
        for i in range(P):
            if not alive_list[i]:
                continue
            for j in range(i + 1, P):
                if not alive_list[j]:
                    continue
                if pool_sizes is not None:
                    pi = _get_pool_index(i, pool_sizes)
                    pj = _get_pool_index(j, pool_sizes)
                    if pi == pj:
                        continue
                if S_w[i][j] > weight_cos_threshold:
                    weight_pairs.add((i, j))

        # Find behavioral redundant pairs
        behavioral = find_behavioral_redundant(
            J, corr, cos_b,
            jaccard_threshold=jaccard_threshold,
            output_corr_threshold=output_corr_threshold,
            pool_sizes=pool_sizes,
            exclude_dead=alive_mask,
        )
        behavioral_pairs = set((p["i"], p["j"]) for p in behavioral)

        # Set comparison
        both = weight_pairs & behavioral_pairs
        behavioral_only = behavioral_pairs - weight_pairs
        weight_only = weight_pairs - behavioral_pairs

        layer_result = {
            "layer": l_idx,
            "P": P,
            "n_alive": n_alive,
            "n_dead": P - n_alive,
            "pct_dead": (P - n_alive) / P * 100,
            "n_weight_redundant_pairs": len(weight_pairs),
            "n_behavioral_redundant_pairs": len(behavioral_pairs),
            "n_both": len(both),
            "n_behavioral_only": len(behavioral_only),
            "n_weight_only": len(weight_only),
            "behavioral_pairs": behavioral,
            "weight_pairs": weight_pairs,  # store for correct set-difference in _count_unique_capsules_in_pairs
            # Jaccard distribution among alive capsules
            "jaccard_stats": _summarize_matrix(J, alive_mask, pool_sizes),
            "output_corr_stats": _summarize_matrix(corr, alive_mask, pool_sizes),
        }

        results["per_layer"].append(layer_result)
        results["total_weight_redundant"] += len(weight_pairs)
        results["total_behavioral_redundant"] += len(behavioral_pairs)
        results["total_behavioral_only"] += len(behavioral_only)
        results["total_both"] += len(both)
        results["total_weight_only"] += len(weight_only)
        results["total_alive_capsules"] += n_alive
        results["total_capsules"] += P

        if verbose:
            print(f"  Layer {l_idx}: P={P}, alive={n_alive}, dead={P - n_alive}")
            print(f"    Weight-cos pairs (>{weight_cos_threshold}): {len(weight_pairs)}")
            print(f"    Behavioral pairs (J>{jaccard_threshold}, corr>{output_corr_threshold}): {len(behavioral_pairs)}")
            print(f"    Behavioral-only (new finds): {len(behavioral_only)}")
            print(f"    Both methods: {len(both)}")
            print(f"    Jaccard cross-pool: mean={layer_result['jaccard_stats']['cross_mean']:.4f}, "
                  f"max={layer_result['jaccard_stats']['cross_max']:.4f}")
            if behavioral:
                top = behavioral[0]
                print(f"    Top behavioral pair: ({top['i']},{top['j']}) "
                      f"J={top['jaccard']:.3f} corr={top['output_corr']:.3f} "
                      f"b_cos={top['b_cosine']:.3f}")

    # Compute behavioral redundancy rate (percentage of alive capsule pairs)
    total_alive = results["total_alive_capsules"]
    # Total possible cross-pool alive pairs (approximate)
    results["behavioral_redundancy_pct"] = (
        results["total_behavioral_redundant"] / max(total_alive, 1) * 100
    )
    results["behavioral_only_pct"] = (
        results["total_behavioral_only"] / max(total_alive, 1) * 100
    )

    # Kill criterion: behavioral_only as % of total capsules
    results["behavioral_only_capsule_pct"] = _count_unique_capsules_in_pairs(
        results, "behavioral_only"
    ) / max(results["total_capsules"], 1) * 100

    if verbose:
        print(f"\n  AGGREGATE:")
        print(f"    Total capsules: {results['total_capsules']}")
        print(f"    Total alive: {results['total_alive_capsules']}")
        print(f"    Weight-cosine redundant pairs: {results['total_weight_redundant']}")
        print(f"    Behavioral redundant pairs: {results['total_behavioral_redundant']}")
        print(f"    Behavioral-ONLY (new finds): {results['total_behavioral_only']}")
        print(f"    Unique capsules in behavioral-only pairs: "
              f"{_count_unique_capsules_in_pairs(results, 'behavioral_only')}")
        print(f"    Behavioral-only capsule %: {results['behavioral_only_capsule_pct']:.1f}%")

    return results


def _summarize_matrix(M: mx.array, alive_mask: mx.array,
                      pool_sizes: list[int] | None) -> dict:
    """Summarize a similarity matrix for alive, cross-pool capsules."""
    mx.eval(M, alive_mask)
    M_np = M.tolist()
    alive = alive_mask.tolist() if hasattr(alive_mask, 'tolist') else [True] * len(M_np)
    P = len(M_np)

    cross_vals = []
    all_vals = []

    for i in range(P):
        if not alive[i]:
            continue
        for j in range(i + 1, P):
            if not alive[j]:
                continue
            val = M_np[i][j]
            all_vals.append(val)
            if pool_sizes is not None:
                pi = _get_pool_index(i, pool_sizes)
                pj = _get_pool_index(j, pool_sizes)
                if pi != pj:
                    cross_vals.append(val)

    return {
        "cross_mean": sum(cross_vals) / len(cross_vals) if cross_vals else 0.0,
        "cross_max": max(cross_vals) if cross_vals else 0.0,
        "cross_min": min(cross_vals) if cross_vals else 0.0,
        "all_mean": sum(all_vals) / len(all_vals) if all_vals else 0.0,
        "all_max": max(all_vals) if all_vals else 0.0,
        "n_cross": len(cross_vals),
        "n_all": len(all_vals),
    }


def _count_unique_capsules_in_pairs(results: dict, pair_type: str) -> int:
    """Count unique capsule indices across all layers for a given pair type."""
    unique = set()
    for lr in results["per_layer"]:
        if pair_type == "behavioral_only":
            # Compute the actual set difference: behavioral pairs minus weight pairs
            behavioral_set = set((p["i"], p["j"]) for p in lr["behavioral_pairs"])
            weight_set = lr.get("weight_pairs", set())
            behavioral_only_set = behavioral_set - weight_set
            for i, j in behavioral_only_set:
                unique.add((lr["layer"], i))
                unique.add((lr["layer"], j))
        elif pair_type == "behavioral_redundant":
            for p in lr["behavioral_pairs"]:
                unique.add((lr["layer"], p["i"]))
                unique.add((lr["layer"], p["j"]))
    return len(unique)


# ---------------------------------------------------------------------------
# 5. Behavioral deduplication with merging + quality test
# ---------------------------------------------------------------------------

def behavioral_deduplicate(model: ReLURouterGPT,
                           dataset,
                           jaccard_threshold: float = 0.7,
                           output_corr_threshold: float = 0.5,
                           n_batches: int = 20,
                           batch_size: int = 32,
                           seed: int = 0,
                           pool_sizes: list[int] | None = None,
                           verbose: bool = True) -> dict:
    """Apply behavioral deduplication: profile, find redundant, merge.

    Uses the same merging rule as capsule_dedup (a-average, b-sum)
    but selects merge candidates by behavioral similarity instead
    of weight cosine.

    Returns:
        Dict with profiling stats and merge statistics.
    """
    layer_stats = profile_behavioral(model, dataset, n_batches, batch_size, seed)

    total_before = 0
    total_after = 0
    total_merged = 0
    per_layer = []

    for l_idx, layer in enumerate(model.layers):
        pool = layer.capsule_pool
        A = pool.A.weight  # (P, d)
        B = pool.B.weight  # (d, P)
        P = A.shape[0]
        d = A.shape[1]
        stats = layer_stats[l_idx]

        # Dead capsule mask
        freq = stats["fire_count"] / max(stats["total_positions"], 1)
        alive_mask = freq > 0

        # Behavioral metrics
        J = compute_jaccard_matrix(stats)
        corr = compute_output_correlation_matrix(stats, B)
        cos_b = compute_conditioned_output_cosine(stats, B)

        # Find redundant pairs
        pairs = find_behavioral_redundant(
            J, corr, cos_b,
            jaccard_threshold=jaccard_threshold,
            output_corr_threshold=output_corr_threshold,
            pool_sizes=pool_sizes,
            exclude_dead=alive_mask,
        )

        # Convert pairs to clusters (greedy: each capsule can only merge once)
        used = set()
        clusters = []
        for p in pairs:
            i, j = p["i"], p["j"]
            if i not in used and j not in used:
                clusters.append([i, j])
                used.add(i)
                used.add(j)

        if clusters:
            A_new, B_new = merge_capsules(A, B, clusters)
            mx.eval(A_new, B_new)
            P_after = A_new.shape[0]

            new_pool = ReLUCapsulePool(d, P_after)
            new_pool.A.load_weights([("weight", A_new)])
            new_pool.B.load_weights([("weight", B_new)])
            layer.capsule_pool = new_pool
            mx.eval(layer.capsule_pool.parameters())
        else:
            P_after = P

        n_merged = P - P_after
        per_layer.append({
            "layer": l_idx,
            "P_before": P,
            "P_after": P_after,
            "n_clusters": len(clusters),
            "n_merged": n_merged,
            "n_pairs_found": len(pairs),
        })
        total_before += P
        total_after += P_after
        total_merged += n_merged

        if verbose:
            print(f"  Layer {l_idx}: {P} -> {P_after} ({n_merged} merged from {len(clusters)} clusters)")

    pct = total_merged / total_before * 100 if total_before > 0 else 0
    result = {
        "per_layer": per_layer,
        "total_before": total_before,
        "total_after": total_after,
        "total_merged": total_merged,
        "pct_merged": pct,
    }

    if verbose:
        print(f"  Total: {total_before} -> {total_after} ({pct:.1f}% merged)")

    return result


@register("behavioral_dedup", parent="capsule_dedup")
class BehavioralDedupGPT(ReLURouterGPT):
    """ReLURouterGPT with post-hoc activation-based behavioral deduplication.

    This model IS a ReLURouterGPT. The behavioral deduplication is applied
    as a post-processing step after composition, not as an architectural change.

    Registered for lineage tracking. All forward pass logic inherited.
    """
    pass
