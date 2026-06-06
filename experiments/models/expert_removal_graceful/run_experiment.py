"""Expert Removal Graceful: Weight-space simulation of expert removal.

Tests whether removing an expert from a Gram-Schmidt-composed merged model
breaks remaining experts. Three strategies compared:

(a) Naive subtraction: W_merged - delta_k' (subtract GS-orthogonalized delta)
(b) GS recomputation: re-orthogonalize remaining N-1 from scratch
(c) "Never added" baseline: compose N-1 from original deltas via GS

If naive subtraction matches the "never added" baseline (thanks to
near-orthogonality), cascade recomputation is unnecessary and removal is O(1).

Quality proxy: reconstruction error = ||W_method - W_baseline||_F / ||W_baseline||_F
where W_baseline is the "never added" ground truth.

Kill criteria:
  K1: removing expert causes >3% regression on remaining expert quality
  K2: GS cascade recomputation takes >10 min at N=50
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================================
# Synthetic LoRA Expert Generation
# ============================================================================

def generate_lora_expert(d: int, r: int, rng: np.random.RandomState) -> dict:
    """Generate a single synthetic LoRA expert.

    Returns dict with A (d_in, r) and B (r, d_out) matrices.
    The weight delta is dW = A @ B, shape (d, d).

    We simulate a single linear layer for simplicity. The math generalizes
    to multiple layers (the flattened delta is just longer).
    """
    A = rng.randn(d, r) / np.sqrt(d)
    B = rng.randn(r, d) / np.sqrt(r)
    return {"A": A, "B": B, "dW": A @ B}


def generate_expert_set(N: int, d: int, r: int,
                         cluster_structure: dict | None = None,
                         seed: int = 42) -> list[dict]:
    """Generate N synthetic LoRA experts with controlled cosine structure.

    Args:
        N: number of experts
        d: model dimension
        r: LoRA rank
        cluster_structure: if None, purely random (near-orthogonal).
            If dict with keys:
                n_clusters: number of clusters
                within_cos: target within-cluster cosine similarity
            Creates clustered experts where within-cluster experts share
            a common subspace component.
        seed: random seed

    Returns:
        list of N expert dicts, each with keys "A", "B", "dW"
    """
    rng = np.random.RandomState(seed)

    if cluster_structure is None:
        # Purely random experts -- near-orthogonal by Johnson-Lindenstrauss
        return [generate_lora_expert(d, r, rng) for _ in range(N)]

    n_clusters = cluster_structure["n_clusters"]
    within_cos = cluster_structure["within_cos"]

    # Assign experts to clusters round-robin
    cluster_assignments = [i % n_clusters for i in range(N)]

    # Generate a shared direction per cluster
    cluster_shared = {}
    for c in range(n_clusters):
        shared_A = rng.randn(d, r) / np.sqrt(d)
        shared_B = rng.randn(r, d) / np.sqrt(r)
        shared_dW = shared_A @ shared_B
        shared_dW = shared_dW / np.linalg.norm(shared_dW)  # normalize
        cluster_shared[c] = shared_dW

    # For each expert: mix shared + unique components
    # cos(expert_i, expert_j) ~ alpha^2 if same cluster, ~0 if different
    alpha = np.sqrt(within_cos)
    beta = np.sqrt(1.0 - within_cos)

    experts = []
    for i in range(N):
        c = cluster_assignments[i]
        unique = generate_lora_expert(d, r, rng)
        unique_dW = unique["dW"]
        unique_dW = unique_dW / np.linalg.norm(unique_dW)  # normalize

        # Remove shared component from unique to make them independent
        shared = cluster_shared[c]
        proj = np.sum(unique_dW * shared) * shared
        unique_dW = unique_dW - proj
        unique_norm = np.linalg.norm(unique_dW)
        if unique_norm > 1e-12:
            unique_dW = unique_dW / unique_norm

        # Mix
        dW = alpha * shared + beta * unique_dW
        # Scale to reasonable magnitude
        dW = dW * rng.uniform(0.8, 1.2) * 0.01

        experts.append({"dW": dW, "cluster": c})

    return experts


def flatten_dW(expert: dict) -> np.ndarray:
    """Flatten expert delta to 1D vector."""
    return expert["dW"].flatten()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


# ============================================================================
# Gram-Schmidt Orthogonalization (pure numpy, no MLX)
# ============================================================================

def gram_schmidt(deltas: list[np.ndarray]) -> list[np.ndarray]:
    """Apply Gram-Schmidt orthogonalization to a list of flattened delta vectors.

    Returns orthogonalized vectors in the same order.
    """
    ortho = []
    for k, v in enumerate(deltas):
        v = v.copy()
        for e in ortho:
            dot_ve = np.dot(v, e)
            dot_ee = np.dot(e, e)
            if dot_ee > 1e-12:
                v = v - (dot_ve / dot_ee) * e
        ortho.append(v)
    return ortho


def merge_with_gs(deltas: list[np.ndarray]) -> np.ndarray:
    """Gram-Schmidt orthogonalize then sum all deltas.

    Returns the merged delta vector (sum of orthogonalized deltas).
    """
    ortho = gram_schmidt(deltas)
    return sum(ortho)


# ============================================================================
# Expert Removal Strategies
# ============================================================================

def naive_removal(ortho_deltas: list[np.ndarray], merged: np.ndarray,
                  remove_idx: int) -> np.ndarray:
    """Strategy (a): Naive subtraction.

    W_new = W_merged - delta_k'
    where delta_k' is the GS-orthogonalized version of expert k.

    This is O(1) -- just subtract the stored orthogonalized delta.
    """
    return merged - ortho_deltas[remove_idx]


def gs_recompute(deltas: list[np.ndarray], remove_idx: int) -> np.ndarray:
    """Strategy (b): GS recomputation.

    Remove expert k from the list, then re-orthogonalize the remaining
    N-1 experts from scratch. This is O(N^2 * D) where D is delta dimension.
    """
    remaining = [d for i, d in enumerate(deltas) if i != remove_idx]
    return merge_with_gs(remaining)


def never_added(deltas: list[np.ndarray], remove_idx: int) -> np.ndarray:
    """Strategy (c): "Never added" baseline.

    Compose N-1 experts from scratch via GS, as if expert k was never present.
    This is identical to gs_recompute -- they are the same operation.
    The distinction is conceptual: this IS the ground truth we compare against.
    """
    remaining = [d for i, d in enumerate(deltas) if i != remove_idx]
    return merge_with_gs(remaining)


# ============================================================================
# Quality Metrics
# ============================================================================

def reconstruction_error(w_method: np.ndarray, w_baseline: np.ndarray) -> float:
    """Relative reconstruction error between method and baseline.

    Returns ||w_method - w_baseline||_F / ||w_baseline||_F as percentage.
    """
    diff_norm = np.linalg.norm(w_method - w_baseline)
    base_norm = np.linalg.norm(w_baseline)
    if base_norm < 1e-12:
        return 0.0
    return float(diff_norm / base_norm) * 100.0


def per_expert_quality(w_merged: np.ndarray, expert_deltas: list[np.ndarray],
                       active_indices: list[int]) -> dict:
    """Measure how well each active expert's contribution is preserved.

    For each expert i in active_indices, compute:
      projection_score = |<w_merged, delta_i>| / (||w_merged|| * ||delta_i||)

    This measures cosine alignment: how much of expert i's direction is present
    in the merged model. Higher = better preservation.
    """
    scores = {}
    for i in active_indices:
        d = expert_deltas[i]
        cos = cosine_sim(w_merged, d)
        scores[i] = abs(cos)
    return scores


# ============================================================================
# Main Experiment
# ============================================================================

def run_single_removal(N: int, d: int, r: int, remove_idx: int,
                        cluster_structure: dict | None,
                        seed: int) -> dict:
    """Run a single expert removal experiment.

    Returns dict with all metrics.
    """
    # Generate experts
    experts = generate_expert_set(N, d, r, cluster_structure, seed)
    deltas = [flatten_dW(e) for e in experts]

    # Measure pairwise cosines
    cos_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            c = cosine_sim(deltas[i], deltas[j])
            cos_matrix[i, j] = c
            cos_matrix[j, i] = c

    mean_cos = np.mean(np.abs(cos_matrix[np.triu_indices(N, k=1)]))
    max_cos = np.max(np.abs(cos_matrix[np.triu_indices(N, k=1)]))

    # GS orthogonalize all N experts
    t0 = time.time()
    ortho_all = gram_schmidt(deltas)
    gs_time_all = time.time() - t0

    merged_all = sum(ortho_all)

    # Strategy (a): Naive subtraction
    t0 = time.time()
    w_naive = naive_removal(ortho_all, merged_all, remove_idx)
    naive_time = time.time() - t0

    # Strategy (b)/(c): GS recompute (= "never added" baseline)
    t0 = time.time()
    w_recompute = gs_recompute(deltas, remove_idx)
    recompute_time = time.time() - t0

    # Compute reconstruction error: naive vs recompute (ground truth)
    recon_error = reconstruction_error(w_naive, w_recompute)

    # Per-expert quality assessment
    active_indices = [i for i in range(N) if i != remove_idx]

    # Quality with naive removal
    q_naive = per_expert_quality(w_naive, deltas, active_indices)
    # Quality with recompute (ground truth)
    q_recompute = per_expert_quality(w_recompute, deltas, active_indices)
    # Quality with all N experts (before removal)
    q_before = per_expert_quality(merged_all, deltas, list(range(N)))

    # Per-expert quality regression: (q_naive - q_recompute) / q_recompute
    per_expert_regression = {}
    for i in active_indices:
        if q_recompute[i] > 1e-12:
            regression_pct = (q_naive[i] - q_recompute[i]) / q_recompute[i] * 100
        else:
            regression_pct = 0.0
        per_expert_regression[i] = regression_pct

    mean_regression = np.mean(list(per_expert_regression.values()))
    max_regression = np.max(np.abs(list(per_expert_regression.values())))

    # Signal retention of removed expert
    removed_orig_norm = np.linalg.norm(deltas[remove_idx])
    removed_ortho_norm = np.linalg.norm(ortho_all[remove_idx])
    signal_retention = removed_ortho_norm / removed_orig_norm if removed_orig_norm > 1e-12 else 0.0

    return {
        "N": N,
        "d": d,
        "r": r,
        "remove_idx": remove_idx,
        "seed": seed,
        "cluster_structure": str(cluster_structure),
        "mean_cos": mean_cos,
        "max_cos": max_cos,
        "gs_time_all": gs_time_all,
        "naive_time": naive_time,
        "recompute_time": recompute_time,
        "recon_error_pct": recon_error,
        "mean_regression_pct": mean_regression,
        "max_regression_pct": max_regression,
        "signal_retention": signal_retention,
        "per_expert_regression": per_expert_regression,
        "q_naive": q_naive,
        "q_recompute": q_recompute,
    }


def run_full_experiment():
    """Run the complete expert removal graceful experiment."""
    print("=" * 70)
    print("  EXPERIMENT: Expert Removal Graceful (Weight-Space)")
    print("  Kill: >3% PPL regression OR GS recompute >10 min at N=50")
    print("=" * 70)

    d = 896  # Qwen 0.5B dimension
    r = 16   # production LoRA rank
    seeds = [42, 123, 777]

    all_results = []

    # ================================================================
    # TEST 1: Near-orthogonal experts (random, no clustering)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 1: Near-orthogonal experts (random LoRA at d=896, r=16)")
    print("=" * 70)

    N_values = [10, 20, 50]

    print(f"\n{'N':>4} {'Seed':>6} {'MeanCos':>9} {'MaxCos':>9} "
          f"{'Recon%':>9} {'MeanReg%':>10} {'MaxReg%':>9} "
          f"{'NaiveT':>8} {'GSrecT':>8} {'SigRet':>8}")
    print("-" * 100)

    for N in N_values:
        for seed in seeds:
            # Remove middle expert
            remove_idx = N // 2
            r_dict = run_single_removal(N, d, r, remove_idx,
                                         cluster_structure=None, seed=seed)
            all_results.append({**r_dict, "test": "orthogonal",
                                "config": f"N={N}_mid"})
            print(f"{N:>4} {seed:>6} {r_dict['mean_cos']:>9.6f} "
                  f"{r_dict['max_cos']:>9.6f} "
                  f"{r_dict['recon_error_pct']:>9.4f} "
                  f"{r_dict['mean_regression_pct']:>+10.4f} "
                  f"{r_dict['max_regression_pct']:>9.4f} "
                  f"{r_dict['naive_time']*1000:>7.1f}ms "
                  f"{r_dict['recompute_time']*1000:>7.1f}ms "
                  f"{r_dict['signal_retention']:>8.4f}")

    # ================================================================
    # TEST 2: Clustered experts (within-cluster cos ~ 0.3)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 2: Clustered experts (3 clusters, within-cos=0.3)")
    print("=" * 70)

    cluster_cfg = {"n_clusters": 3, "within_cos": 0.3}

    print(f"\n{'N':>4} {'Seed':>6} {'MeanCos':>9} {'MaxCos':>9} "
          f"{'Recon%':>9} {'MeanReg%':>10} {'MaxReg%':>9} "
          f"{'NaiveT':>8} {'GSrecT':>8} {'SigRet':>8}")
    print("-" * 100)

    for N in N_values:
        for seed in seeds:
            remove_idx = N // 2
            r_dict = run_single_removal(N, d, r, remove_idx,
                                         cluster_structure=cluster_cfg, seed=seed)
            all_results.append({**r_dict, "test": "clustered_0.3",
                                "config": f"N={N}_mid"})
            print(f"{N:>4} {seed:>6} {r_dict['mean_cos']:>9.6f} "
                  f"{r_dict['max_cos']:>9.6f} "
                  f"{r_dict['recon_error_pct']:>9.4f} "
                  f"{r_dict['mean_regression_pct']:>+10.4f} "
                  f"{r_dict['max_regression_pct']:>9.4f} "
                  f"{r_dict['naive_time']*1000:>7.1f}ms "
                  f"{r_dict['recompute_time']*1000:>7.1f}ms "
                  f"{r_dict['signal_retention']:>8.4f}")

    # ================================================================
    # TEST 3: High-overlap clustered experts (within-cos=0.5)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 3: High-overlap clusters (3 clusters, within-cos=0.5)")
    print("=" * 70)

    cluster_cfg_high = {"n_clusters": 3, "within_cos": 0.5}

    print(f"\n{'N':>4} {'Seed':>6} {'MeanCos':>9} {'MaxCos':>9} "
          f"{'Recon%':>9} {'MeanReg%':>10} {'MaxReg%':>9} "
          f"{'NaiveT':>8} {'GSrecT':>8} {'SigRet':>8}")
    print("-" * 100)

    for N in N_values:
        for seed in seeds:
            remove_idx = N // 2
            r_dict = run_single_removal(N, d, r, remove_idx,
                                         cluster_structure=cluster_cfg_high,
                                         seed=seed)
            all_results.append({**r_dict, "test": "clustered_0.5",
                                "config": f"N={N}_mid"})
            print(f"{N:>4} {seed:>6} {r_dict['mean_cos']:>9.6f} "
                  f"{r_dict['max_cos']:>9.6f} "
                  f"{r_dict['recon_error_pct']:>9.4f} "
                  f"{r_dict['mean_regression_pct']:>+10.4f} "
                  f"{r_dict['max_regression_pct']:>9.4f} "
                  f"{r_dict['naive_time']*1000:>7.1f}ms "
                  f"{r_dict['recompute_time']*1000:>7.1f}ms "
                  f"{r_dict['signal_retention']:>8.4f}")

    # ================================================================
    # TEST 4: Removal position sensitivity (N=20, clustered)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 4: Removal position sensitivity (N=20, clustered 0.3)")
    print("=" * 70)

    N = 20
    cluster_cfg = {"n_clusters": 3, "within_cos": 0.3}
    positions = [0, N // 4, N // 2, 3 * N // 4, N - 1]

    print(f"\n{'Pos':>4} {'Seed':>6} {'Recon%':>9} {'MeanReg%':>10} "
          f"{'MaxReg%':>9} {'SigRet':>8}")
    print("-" * 55)

    for pos in positions:
        for seed in seeds:
            r_dict = run_single_removal(N, d, r, pos,
                                         cluster_structure=cluster_cfg, seed=seed)
            all_results.append({**r_dict, "test": "position",
                                "config": f"pos={pos}"})
            print(f"{pos:>4} {seed:>6} {r_dict['recon_error_pct']:>9.4f} "
                  f"{r_dict['mean_regression_pct']:>+10.4f} "
                  f"{r_dict['max_regression_pct']:>9.4f} "
                  f"{r_dict['signal_retention']:>8.4f}")

    # ================================================================
    # TEST 5: Timing at scale (N=50, 100, 200)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 5: GS recomputation timing at scale")
    print("=" * 70)

    timing_N_values = [10, 20, 50, 100, 200]

    print(f"\n{'N':>5} {'GS_all(s)':>10} {'GS_recomp(s)':>13} {'Naive(ms)':>10} "
          f"{'Recon%':>9}")
    print("-" * 55)

    for N in timing_N_values:
        seed = 42
        remove_idx = N // 2
        r_dict = run_single_removal(N, d, r, remove_idx,
                                     cluster_structure=None, seed=seed)
        all_results.append({**r_dict, "test": "timing",
                            "config": f"N={N}"})
        print(f"{N:>5} {r_dict['gs_time_all']:>10.3f} "
              f"{r_dict['recompute_time']:>13.3f} "
              f"{r_dict['naive_time']*1000:>10.3f} "
              f"{r_dict['recon_error_pct']:>9.6f}")

    # ================================================================
    # TEST 6: Multiple sequential removals (stress test)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 6: Sequential removal (remove 1, 3, 5 experts from N=20)")
    print("=" * 70)

    N = 20
    cluster_cfg = {"n_clusters": 3, "within_cos": 0.3}
    seed = 42

    experts = generate_expert_set(N, d, r, cluster_cfg, seed)
    deltas = [flatten_dW(e) for e in experts]

    # Ground truth: full GS merge of all N
    ortho_all = gram_schmidt(deltas)
    merged_all = sum(ortho_all)

    print(f"\n{'#Removed':>9} {'Method':>12} {'Recon%':>9} {'Time(ms)':>10}")
    print("-" * 45)

    for n_remove in [1, 3, 5]:
        remove_indices = list(range(0, n_remove))  # remove first n_remove
        remaining_indices = [i for i in range(N) if i not in remove_indices]
        remaining_deltas = [deltas[i] for i in remaining_indices]

        # Ground truth: GS of remaining from scratch
        t0 = time.time()
        w_gt = merge_with_gs(remaining_deltas)
        gt_time = time.time() - t0

        # Naive: sequential subtraction
        t0 = time.time()
        w_naive = merged_all.copy()
        for idx in remove_indices:
            w_naive = w_naive - ortho_all[idx]
        naive_time = time.time() - t0

        recon = reconstruction_error(w_naive, w_gt)

        print(f"{n_remove:>9} {'naive':>12} {recon:>9.4f} "
              f"{naive_time*1000:>10.3f}")
        print(f"{n_remove:>9} {'GS_recomp':>12} {0.0:>9.4f} "
              f"{gt_time*1000:>10.3f}")

    # ================================================================
    # AGGREGATE RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    # K1: >3% regression
    # Use reconstruction error as PPL proxy
    ortho_results = [r for r in all_results if r["test"] == "orthogonal"]
    clustered_03 = [r for r in all_results if r["test"] == "clustered_0.3"]
    clustered_05 = [r for r in all_results if r["test"] == "clustered_0.5"]

    for label, results in [("Near-orthogonal", ortho_results),
                            ("Clustered cos=0.3", clustered_03),
                            ("Clustered cos=0.5", clustered_05)]:
        if not results:
            continue
        recon_errors = [r["recon_error_pct"] for r in results]
        max_recon = max(recon_errors)
        mean_recon = np.mean(recon_errors)
        max_regressions = [r["max_regression_pct"] for r in results]
        worst_reg = max(max_regressions)

        print(f"\n  {label}:")
        print(f"    Reconstruction error: mean={mean_recon:.4f}%, max={max_recon:.4f}%")
        print(f"    Max per-expert regression: {worst_reg:.4f}%")

    # K1 assessment
    all_recon = [r["recon_error_pct"] for r in all_results
                 if r["test"] in ("orthogonal", "clustered_0.3", "clustered_0.5")]
    max_recon_overall = max(all_recon) if all_recon else 0
    k1_pass = max_recon_overall < 3.0

    print(f"\n  K1 (recon error < 3%): {'PASS' if k1_pass else 'KILL'}")
    print(f"      Worst reconstruction error: {max_recon_overall:.4f}%")
    print(f"      Threshold: 3.0%")

    # K2 assessment: timing at N=50
    timing_n50 = [r for r in all_results
                  if r["test"] == "timing" and r["N"] == 50]
    if timing_n50:
        recompute_time_n50 = timing_n50[0]["recompute_time"]
        k2_pass = recompute_time_n50 < 600  # 10 min
        print(f"\n  K2 (GS recompute < 10 min at N=50): {'PASS' if k2_pass else 'KILL'}")
        print(f"      Recompute time at N=50: {recompute_time_n50:.3f}s")
        print(f"      Threshold: 600s")
    else:
        k2_pass = True

    # Bonus: Is naive removal sufficient?
    print("\n" + "-" * 70)
    print("  DECISION: Is naive subtraction sufficient?")
    print("-" * 70)

    for label, results in [("Near-orthogonal", ortho_results),
                            ("Clustered cos=0.3", clustered_03),
                            ("Clustered cos=0.5", clustered_05)]:
        if not results:
            continue
        recon_errors = [r["recon_error_pct"] for r in results]
        max_recon = max(recon_errors)
        speedup_factors = [r["recompute_time"] / max(r["naive_time"], 1e-9)
                           for r in results]
        mean_speedup = np.mean(speedup_factors)
        print(f"\n  {label}:")
        print(f"    Max reconstruction error: {max_recon:.4f}%")
        print(f"    Mean speedup (naive vs recompute): {mean_speedup:.0f}x")
        if max_recon < 1.0:
            print(f"    -> Naive subtraction IS sufficient (<1% error)")
        elif max_recon < 3.0:
            print(f"    -> Naive subtraction is MARGINAL (1-3% error)")
        else:
            print(f"    -> Naive subtraction is INSUFFICIENT (>3% error)")

    # Save results
    results_path = Path(__file__).parent / "results.json"
    serializable = []
    for r in all_results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                sr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sr[k] = float(v)
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            elif isinstance(v, dict):
                sr[k] = {str(kk): float(vv) if isinstance(vv, (float, np.floating)) else vv
                         for kk, vv in v.items()}
            else:
                sr[k] = v
        serializable.append(sr)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    return all_results, k1_pass, k2_pass


if __name__ == "__main__":
    t0 = time.time()
    results, k1, k2 = run_full_experiment()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Overall: {'PASS' if (k1 and k2) else 'KILL'}")
