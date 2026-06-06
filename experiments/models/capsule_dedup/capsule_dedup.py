"""Capsule Deduplication — merge redundant capsules across composed pools.

After composing domain-specific ReLU MLPs by concatenation, identify
capsules with near-identical a_i detector vectors (cosine similarity
> threshold) and merge them to reduce parameter count.

Merging rule (for rank-1 capsules):
  a_merged = mean(a_i for matched capsules)
  b_merged = sum(b_i for matched capsules)

Summing b preserves the full-strength additive output that downstream
layers expect. Averaging a preserves the activation region.

The model class is a thin wrapper: it IS a ReLURouterGPT with a
deduplication post-processing step applied to the composed weights.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool


def cosine_similarity_matrix(A: mx.array) -> mx.array:
    """Compute pairwise cosine similarity for rows of A.

    Args:
        A: (P, d) matrix where each row is a detector vector.

    Returns:
        S: (P, P) matrix where S[i,j] = cos(a_i, a_j).
    """
    # Normalize rows to unit length
    norms = mx.sqrt(mx.sum(A * A, axis=1, keepdims=True) + 1e-8)  # (P, 1)
    A_normed = A / norms  # (P, d)
    # Pairwise cosine = dot products of unit vectors
    S = A_normed @ A_normed.T  # (P, P)
    return S


def find_redundant_clusters(S: mx.array, threshold: float,
                            pool_sizes: list[int] | None = None) -> list[list[int]]:
    """Find clusters of redundant capsules using connected components.

    Args:
        S: (P_total, P_total) cosine similarity matrix.
        threshold: minimum cosine similarity to consider redundant.
        pool_sizes: optional list of pool sizes for cross-pool-only mode.
                   If None, all pairs are considered.

    Returns:
        List of clusters. Each cluster is a list of capsule indices
        that should be merged. Singletons are not returned.
    """
    mx.eval(S)
    S_np = S.tolist()
    P = len(S_np)

    # Build adjacency using union-find for connected components
    parent = list(range(P))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(P):
        for j in range(i + 1, P):
            if S_np[i][j] > threshold:
                # If pool_sizes specified, only merge cross-pool pairs
                if pool_sizes is not None:
                    pool_i = _get_pool_index(i, pool_sizes)
                    pool_j = _get_pool_index(j, pool_sizes)
                    if pool_i == pool_j:
                        continue  # skip within-pool pairs
                union(i, j)

    # Collect clusters
    clusters = {}
    for i in range(P):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    # Return only non-singleton clusters
    return [c for c in clusters.values() if len(c) > 1]


def find_all_redundant_clusters(S: mx.array, threshold: float) -> list[list[int]]:
    """Find clusters considering ALL pairs (cross-pool and within-pool)."""
    return find_redundant_clusters(S, threshold, pool_sizes=None)


def _get_pool_index(capsule_idx: int, pool_sizes: list[int]) -> int:
    """Determine which pool a capsule belongs to."""
    cumsum = 0
    for pool_idx, size in enumerate(pool_sizes):
        cumsum += size
        if capsule_idx < cumsum:
            return pool_idx
    return len(pool_sizes) - 1


def merge_capsules(A: mx.array, B: mx.array,
                   clusters: list[list[int]]) -> tuple[mx.array, mx.array]:
    """Merge redundant capsules: average a, sum b.

    Args:
        A: (P_total, d) detector matrix.
        B: (d, P_total) expansion matrix.
        clusters: list of clusters to merge.

    Returns:
        A_new: (P_new, d) where P_new = P_total - sum(len(c)-1 for c in clusters)
        B_new: (d, P_new)
    """
    P_total = A.shape[0]
    merged_indices = set()
    new_a_rows = []
    new_b_cols = []

    for cluster in clusters:
        # Average detector vectors
        a_cluster = mx.stack([A[i] for i in cluster])  # (k, d)
        a_merged = mx.mean(a_cluster, axis=0)  # (d,)

        # Sum expansion vectors
        b_cluster = mx.stack([B[:, i] for i in cluster])  # (k, d)
        b_merged = mx.sum(b_cluster, axis=0)  # (d,)

        new_a_rows.append(a_merged)
        new_b_cols.append(b_merged)
        merged_indices.update(cluster)

    # Collect surviving (unmerged) capsules
    surviving_a = []
    surviving_b = []
    for i in range(P_total):
        if i not in merged_indices:
            surviving_a.append(A[i])
            surviving_b.append(B[:, i])

    # Combine surviving + merged
    all_a = surviving_a + new_a_rows
    all_b = surviving_b + new_b_cols

    A_new = mx.stack(all_a)  # (P_new, d)
    B_new = mx.stack(all_b, axis=1)  # (d, P_new)

    return A_new, B_new


def deduplicate_composed_model(composed_model: ReLURouterGPT,
                                threshold: float = 0.95,
                                pool_sizes: list[int] | None = None,
                                cross_pool_only: bool = True,
                                verbose: bool = True) -> dict:
    """Deduplicate capsules in a composed ReLURouterGPT model.

    Modifies the model in-place by replacing A and B weight matrices
    with deduplicated versions. Returns statistics about what was merged.

    Args:
        composed_model: A composed ReLURouterGPT (with concatenated pools).
        threshold: cosine similarity threshold for merging.
        pool_sizes: sizes of each domain pool (e.g., [128, 128]).
                   Required if cross_pool_only=True.
        cross_pool_only: if True, only merge capsules from different pools.
        verbose: print per-layer statistics.

    Returns:
        Dict with per-layer and aggregate statistics.
    """
    stats = {"per_layer": [], "total_before": 0, "total_after": 0,
             "total_merged_clusters": 0, "total_capsules_removed": 0}

    for l_idx, layer in enumerate(composed_model.layers):
        pool = layer.capsule_pool
        A = pool.A.weight  # (P_total, d)
        B = pool.B.weight  # (d, P_total)
        P_before = A.shape[0]

        # Compute cosine similarity
        S = cosine_similarity_matrix(A)
        mx.eval(S)

        # Find redundant clusters
        if cross_pool_only and pool_sizes is not None:
            clusters = find_redundant_clusters(S, threshold, pool_sizes)
        else:
            clusters = find_all_redundant_clusters(S, threshold)

        if clusters:
            A_new, B_new = merge_capsules(A, B, clusters)
            mx.eval(A_new, B_new)
            P_after = A_new.shape[0]

            # Create new pool with reduced capsule count
            n_embd = A_new.shape[1]
            new_pool = ReLUCapsulePool(n_embd, P_after)
            new_pool.A.load_weights([("weight", A_new)])
            new_pool.B.load_weights([("weight", B_new)])
            layer.capsule_pool = new_pool
            mx.eval(layer.capsule_pool.parameters())
        else:
            P_after = P_before

        n_clusters = len(clusters)
        n_removed = P_before - P_after

        layer_stats = {
            "layer": l_idx,
            "P_before": P_before,
            "P_after": P_after,
            "n_clusters": n_clusters,
            "n_removed": n_removed,
            "cluster_sizes": [len(c) for c in clusters],
        }

        # Compute similarity statistics
        S_vals = S.tolist()
        cross_sims = []
        if pool_sizes is not None and len(pool_sizes) == 2:
            p0 = pool_sizes[0]
            for i in range(p0):
                for j in range(p0, P_before):
                    cross_sims.append(S_vals[i][j])
            if cross_sims:
                layer_stats["cross_sim_mean"] = sum(cross_sims) / len(cross_sims)
                layer_stats["cross_sim_max"] = max(cross_sims)
                layer_stats["cross_sim_above_threshold"] = sum(
                    1 for s in cross_sims if s > threshold
                ) / len(cross_sims)

        stats["per_layer"].append(layer_stats)
        stats["total_before"] += P_before
        stats["total_after"] += P_after
        stats["total_merged_clusters"] += n_clusters
        stats["total_capsules_removed"] += n_removed

        if verbose:
            print(f"  Layer {l_idx}: {P_before} -> {P_after} capsules "
                  f"({n_clusters} clusters, {n_removed} removed)")
            if "cross_sim_mean" in layer_stats:
                print(f"    Cross-pool sim: mean={layer_stats['cross_sim_mean']:.4f}, "
                      f"max={layer_stats['cross_sim_max']:.4f}, "
                      f">{threshold}: {layer_stats['cross_sim_above_threshold']:.1%}")

    pct_removed = (stats["total_capsules_removed"] /
                   stats["total_before"] * 100 if stats["total_before"] > 0 else 0)
    stats["pct_capsules_removed"] = pct_removed

    if verbose:
        print(f"  Total: {stats['total_before']} -> {stats['total_after']} capsules "
              f"({pct_removed:.1f}% reduction)")

    return stats


@register("capsule_dedup", parent="relu_router")
class CapsuleDedupGPT(ReLURouterGPT):
    """ReLURouterGPT with post-hoc capsule deduplication.

    This model IS a ReLURouterGPT. The deduplication is applied as a
    post-processing step after composition, not as an architectural change.

    The model class is registered for lineage tracking. All forward pass
    logic is inherited from ReLURouterGPT.
    """
    pass
