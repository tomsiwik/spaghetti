"""Union-Find Expert Merging: dynamic library compression.

Expert count grows monotonically when composing domain-specific capsule pools.
This experiment applies the union-find (disjoint-set) data structure with path
compression (Tarjan 1975) to iteratively merge similar experts, compressing the
library over time while preserving model quality.

The key insight is that behavioral similarity (co-activation Jaccard from
behavioral_dedup) identifies functionally redundant capsules that weight-cosine
misses. Union-find provides an efficient structure for iterative transitive
merging: if A~B and B~C, all three merge into one canonical expert.

Algorithm:
  1. Profile behavioral similarity (Jaccard + output correlation)
  2. For each pair above threshold, call union(i, j)
  3. Path compression ensures find(i) returns the canonical representative in
     near-constant amortized time
  4. Group all capsules by their canonical representative -> clusters
  5. Merge weights within each cluster (a-average, b-sum)
  6. Rebuild capsule pool with reduced size

Connection to behavioral_dedup: behavioral_dedup uses greedy pairing (each
capsule merges at most once). Union-find enables TRANSITIVE closure: if capsule
A is similar to B, and B to C, all three merge. This should find more
compression than greedy pairing at the same threshold.
"""

import random
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..relu_router.relu_router import ReLURouterGPT, ReLUCapsulePool
from ..behavioral_dedup.behavioral_dedup import (
    profile_behavioral,
    compute_jaccard_matrix,
    compute_output_correlation_matrix,
    compute_conditioned_output_cosine,
)
from ..capsule_dedup.capsule_dedup import merge_capsules


# ---------------------------------------------------------------------------
# 1. Union-Find data structure with path compression and union by rank
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint-set with path compression and union by rank (Tarjan 1975).

    Operations:
      find(x)    -> canonical representative of x's set, O(alpha(n)) amortized
      union(x,y) -> merge sets containing x and y
      clusters() -> dict mapping representative -> list of members

    Path compression: during find(x), all nodes on the path to root are
    directly linked to root. This amortizes future lookups.

    Union by rank: smaller tree is attached under root of larger tree,
    keeping tree depth O(log n) without path compression, O(alpha(n)) with.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n = n

    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if x and y were in different sets."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def clusters(self) -> dict[int, list[int]]:
        """Return dict mapping canonical representative -> members."""
        groups: dict[int, list[int]] = {}
        for i in range(self.n):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups

    def n_components(self) -> int:
        """Number of distinct sets."""
        return len(set(self.find(i) for i in range(self.n)))


# ---------------------------------------------------------------------------
# 2. Build union-find from similarity matrices
# ---------------------------------------------------------------------------

def build_union_find_from_similarity(
    J: mx.array,
    corr: mx.array,
    alive_mask: mx.array,
    jaccard_threshold: float = 0.3,
    output_corr_threshold: float = 0.3,
) -> UnionFind:
    """Build union-find by unioning all pairs above similarity thresholds.

    This is the key difference from behavioral_dedup's greedy pairing:
    union-find captures TRANSITIVE similarity. If A~B and B~C, all three
    are merged, even if A and C are not directly similar.

    Args:
        J: (P, P) Jaccard similarity matrix.
        corr: (P, P) output correlation matrix.
        alive_mask: (P,) boolean, True = capsule is alive (fires > 0).
        jaccard_threshold: minimum Jaccard for union.
        output_corr_threshold: minimum output correlation for union.

    Returns:
        UnionFind with all above-threshold pairs unioned.
    """
    mx.eval(J, corr, alive_mask)
    P = J.shape[0]
    J_np = J.tolist()
    corr_np = corr.tolist()
    alive = alive_mask.tolist() if hasattr(alive_mask, 'tolist') else [True] * P

    uf = UnionFind(P)
    n_unions = 0

    for i in range(P):
        if not alive[i]:
            continue
        for j in range(i + 1, P):
            if not alive[j]:
                continue
            if J_np[i][j] >= jaccard_threshold and corr_np[i][j] >= output_corr_threshold:
                if uf.union(i, j):
                    n_unions += 1

    return uf


# ---------------------------------------------------------------------------
# 3. Union-Find merging: profile -> union -> merge clusters -> rebuild pool
# ---------------------------------------------------------------------------

def union_find_merge(
    model: ReLURouterGPT,
    dataset,
    jaccard_threshold: float = 0.3,
    output_corr_threshold: float = 0.3,
    n_batches: int = 20,
    batch_size: int = 32,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Apply union-find based expert merging to a composed ReLURouterGPT.

    Protocol:
      1. Profile behavioral similarity (reuse behavioral_dedup)
      2. Build union-find with transitive closure
      3. Extract non-singleton clusters
      4. Merge weights: a-average, b-sum (reuse capsule_dedup)
      5. Rebuild capsule pools with reduced expert count

    Args:
        model: A composed ReLURouterGPT.
        dataset: CharDataset for profiling.
        jaccard_threshold: Jaccard threshold for union.
        output_corr_threshold: Output correlation threshold for union.
        n_batches: Profiling batches.
        batch_size: Profiling batch size.
        seed: RNG seed.
        verbose: Print progress.

    Returns:
        Dict with per-layer and aggregate merge statistics.
    """
    if verbose:
        print("  Profiling behavioral statistics...")
    layer_stats = profile_behavioral(model, dataset, n_batches, batch_size, seed)

    total_before = 0
    total_after = 0
    total_merged = 0
    total_clusters = 0
    per_layer = []

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

        # Compute similarity matrices
        J = compute_jaccard_matrix(stats)
        corr = compute_output_correlation_matrix(stats, B)

        # Build union-find with transitive closure
        uf = build_union_find_from_similarity(
            J, corr, alive_mask,
            jaccard_threshold=jaccard_threshold,
            output_corr_threshold=output_corr_threshold,
        )

        # Extract non-singleton clusters (these are what we merge)
        all_clusters = uf.clusters()
        merge_clusters = [members for members in all_clusters.values()
                         if len(members) > 1]

        # Also count dead capsules (we remove them too)
        alive_list = alive_mask.tolist() if hasattr(alive_mask, 'tolist') else [True] * P
        dead_indices = [i for i in range(P) if not alive_list[i]]
        n_dead = len(dead_indices)

        if merge_clusters:
            A_new, B_new = merge_capsules(A, B, merge_clusters)
            mx.eval(A_new, B_new)
            P_after = A_new.shape[0]

            new_pool = ReLUCapsulePool(A_new.shape[1], P_after)
            new_pool.A.load_weights([("weight", A_new)])
            new_pool.B.load_weights([("weight", B_new)])
            layer.capsule_pool = new_pool
            mx.eval(layer.capsule_pool.parameters())
        else:
            P_after = P

        n_merged = P - P_after
        n_nontrivial = len(merge_clusters)
        max_cluster = max((len(c) for c in merge_clusters), default=0)

        per_layer.append({
            "layer": l_idx,
            "P_before": P,
            "P_after": P_after,
            "n_merged": n_merged,
            "n_clusters": n_nontrivial,
            "max_cluster_size": max_cluster,
            "cluster_sizes": sorted([len(c) for c in merge_clusters], reverse=True),
            "n_dead": n_dead,
            "n_components_uf": uf.n_components(),
        })
        total_before += P
        total_after += P_after
        total_merged += n_merged
        total_clusters += n_nontrivial

        if verbose:
            print(f"  Layer {l_idx}: {P} -> {P_after} "
                  f"({n_merged} merged from {n_nontrivial} clusters, "
                  f"max_cluster={max_cluster}, dead={n_dead})")

    pct = total_merged / total_before * 100 if total_before > 0 else 0
    result = {
        "per_layer": per_layer,
        "total_before": total_before,
        "total_after": total_after,
        "total_merged": total_merged,
        "total_clusters": total_clusters,
        "pct_merged": pct,
        "jaccard_threshold": jaccard_threshold,
        "output_corr_threshold": output_corr_threshold,
    }

    if verbose:
        print(f"\n  AGGREGATE: {total_before} -> {total_after} "
              f"({pct:.1f}% reduction, {total_clusters} clusters)")

    return result


# ---------------------------------------------------------------------------
# 4. Threshold sweep: find Pareto frontier of compression vs quality
# ---------------------------------------------------------------------------

def threshold_sweep(
    model: ReLURouterGPT,
    dataset,
    val_dataset,
    thresholds: list[tuple[float, float]] | None = None,
    n_batches: int = 20,
    batch_size: int = 32,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """Sweep Jaccard/correlation thresholds and measure compression vs quality.

    For each threshold pair, we:
      1. Clone the model
      2. Apply union-find merging
      3. Evaluate val loss
      4. Record compression ratio and quality delta

    Args:
        model: The composed, pre-merge model.
        dataset: Training data for profiling.
        val_dataset: Validation data for quality measurement.
        thresholds: List of (jaccard_threshold, output_corr_threshold) pairs.
        n_batches: Profiling batch count.
        batch_size: Batch size.
        seed: RNG seed.
        verbose: Print progress.

    Returns:
        List of result dicts, one per threshold pair.
    """
    from experiments.train import evaluate
    import copy

    if thresholds is None:
        thresholds = [
            (0.2, 0.2),   # aggressive
            (0.3, 0.3),   # moderate
            (0.4, 0.3),   # moderate-conservative
            (0.5, 0.4),   # conservative
            (0.7, 0.5),   # very conservative (behavioral_dedup default)
        ]

    # Baseline quality
    baseline_loss = evaluate(model, val_dataset, batch_size)
    if verbose:
        print(f"\n  Baseline val loss: {baseline_loss:.4f}")

    results = []
    for jac_t, corr_t in thresholds:
        if verbose:
            print(f"\n  --- Threshold: J>{jac_t}, corr>{corr_t} ---")

        # Deep copy model for each threshold test
        model_copy = copy.deepcopy(model)
        mx.eval(model_copy.parameters())

        merge_result = union_find_merge(
            model_copy, dataset,
            jaccard_threshold=jac_t,
            output_corr_threshold=corr_t,
            n_batches=n_batches,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
        )

        # Evaluate quality after merging
        post_loss = evaluate(model_copy, val_dataset, batch_size)
        delta_pct = (post_loss - baseline_loss) / baseline_loss * 100

        result = {
            "jaccard_threshold": jac_t,
            "output_corr_threshold": corr_t,
            "baseline_loss": baseline_loss,
            "post_merge_loss": post_loss,
            "delta_pct": delta_pct,
            "pct_merged": merge_result["pct_merged"],
            "total_before": merge_result["total_before"],
            "total_after": merge_result["total_after"],
            "total_merged": merge_result["total_merged"],
            "total_clusters": merge_result["total_clusters"],
            "per_layer": merge_result["per_layer"],
        }
        results.append(result)

        if verbose:
            verdict = "PASS" if delta_pct <= 3.0 and merge_result["pct_merged"] >= 20.0 else "FAIL"
            print(f"  Result: {post_loss:.4f} ({delta_pct:+.2f}%), "
                  f"merged {merge_result['pct_merged']:.1f}% -> {verdict}")

    return results


# ---------------------------------------------------------------------------
# 5. Compare union-find vs greedy pairing (behavioral_dedup baseline)
# ---------------------------------------------------------------------------

def compare_uf_vs_greedy(
    model: ReLURouterGPT,
    dataset,
    val_dataset,
    jaccard_threshold: float = 0.3,
    output_corr_threshold: float = 0.3,
    n_batches: int = 20,
    batch_size: int = 32,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Head-to-head comparison: union-find transitive vs greedy pairing.

    Both methods use identical similarity thresholds and profiling data.
    The only difference is the clustering strategy:
      - Greedy: each capsule merges at most once (behavioral_dedup)
      - Union-find: transitive closure merges entire connected components

    Args:
        model: Composed, pre-merge model.
        dataset: Training data for profiling.
        val_dataset: Validation data.
        jaccard_threshold: Shared threshold.
        output_corr_threshold: Shared threshold.

    Returns:
        Dict with both methods' results.
    """
    from experiments.train import evaluate
    from ..behavioral_dedup.behavioral_dedup import behavioral_deduplicate
    import copy

    baseline_loss = evaluate(model, val_dataset, batch_size)

    # Union-find method
    model_uf = copy.deepcopy(model)
    mx.eval(model_uf.parameters())
    uf_result = union_find_merge(
        model_uf, dataset,
        jaccard_threshold=jaccard_threshold,
        output_corr_threshold=output_corr_threshold,
        n_batches=n_batches, batch_size=batch_size, seed=seed,
        verbose=verbose,
    )
    uf_loss = evaluate(model_uf, val_dataset, batch_size)

    # Greedy method (behavioral_dedup)
    model_greedy = copy.deepcopy(model)
    mx.eval(model_greedy.parameters())
    greedy_result = behavioral_deduplicate(
        model_greedy, dataset,
        jaccard_threshold=jaccard_threshold,
        output_corr_threshold=output_corr_threshold,
        n_batches=n_batches, batch_size=batch_size, seed=seed,
        verbose=verbose,
    )
    greedy_loss = evaluate(model_greedy, val_dataset, batch_size)

    return {
        "baseline_loss": baseline_loss,
        "union_find": {
            "post_loss": uf_loss,
            "delta_pct": (uf_loss - baseline_loss) / baseline_loss * 100,
            "pct_merged": uf_result["pct_merged"],
            "total_merged": uf_result["total_merged"],
            "total_clusters": uf_result["total_clusters"],
        },
        "greedy": {
            "post_loss": greedy_loss,
            "delta_pct": (greedy_loss - baseline_loss) / baseline_loss * 100,
            "pct_merged": greedy_result["pct_merged"],
            "total_merged": greedy_result["total_merged"],
        },
    }


# ---------------------------------------------------------------------------
# 6. Model registration
# ---------------------------------------------------------------------------

@register("union_find_merge", parent="behavioral_dedup")
class UnionFindMergeGPT(ReLURouterGPT):
    """ReLURouterGPT with post-hoc union-find-based expert merging.

    This model IS a ReLURouterGPT. The union-find merging is applied as a
    post-processing step after composition, not as an architectural change.

    The key improvement over behavioral_dedup's greedy pairing is TRANSITIVE
    closure: if A~B and B~C, the union-find approach merges all three, while
    greedy pairing only merges one pair.
    """
    pass
