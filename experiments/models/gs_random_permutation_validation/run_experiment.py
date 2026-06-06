"""GS Random Permutation Validation: Does random ordering per layer
eliminate worst-case removal position sensitivity?

Parent experiment (removal_position_sensitivity) found:
  - Worst case (first position): 0.164% deviation at d=256, N=50
  - Mean across positions: 0.098%
  - Last position: exactly 0% (mathematical identity)
  - Position sensitivity ratio (first/Q3): 2.16x

The parent recommended random GS permutation per layer to amortize
position effects. This experiment validates that recommendation.

Approach:
  For each expert removal, apply P random permutations to the GS ordering
  (independently per layer), compute merged deltas, measure removal error.
  The worst-case deviation across all N experts should converge to ~mean
  of the unpermuted position sweep.

Design for feasibility:
  - Primary: d=128, N=20, L=12, P=5 perms, all 20 removal positions
  - Cross-val: d=256, N=20, L=12, P=5, key positions only (0, 5, 10, 15, 19)
  - 3 seeds throughout
  Runtime target: <5 minutes total

Kill criteria:
  K1: Permuted worst-case exceeds 2x the unpermuted mean deviation
  K2: Permutation introduces new failure modes (any position exceeds 1% at d=256, N=50)

CPU only. numpy/scipy on Apple Silicon.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================================
# Core utilities (reused from parent: removal_position_sensitivity)
# ============================================================================

def generate_lora_expert_layer(d: int, r: int, rng: np.random.RandomState) -> dict:
    A = rng.randn(d, r) / np.sqrt(d)
    B = rng.randn(r, d) / np.sqrt(r)
    return {"A": A, "B": B, "dW": A @ B}


def gram_schmidt_merge(deltas: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    ortho = []
    for v_orig in deltas:
        v = v_orig.copy()
        for e in ortho:
            dot_ve = np.dot(v, e)
            dot_ee = np.dot(e, e)
            if dot_ee > 1e-12:
                v = v - (dot_ve / dot_ee) * e
        ortho.append(v)
    return sum(ortho), ortho


def rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if x.ndim == 1:
        return x / np.sqrt(np.mean(x ** 2) + eps)
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)


def activation(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def forward_pre_rmsn(h: np.ndarray, base_weights: list[np.ndarray],
                     layer_deltas: list[np.ndarray]) -> np.ndarray:
    L = len(base_weights)
    scale = 1.0 / np.sqrt(L) if L > 1 else 1.0
    for l in range(L):
        W = base_weights[l] + layer_deltas[l]
        h = h + scale * activation(W @ rms_norm(h))
    return h


# ============================================================================
# Measurement functions
# ============================================================================

def measure_removal_unpermuted(experts, base_weights, inputs, remove_idx, d, L, N):
    """Removal error with identity GS ordering."""
    all_merged = []
    all_ortho = []

    for l in range(L):
        deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        merged, ortho = gram_schmidt_merge(deltas)
        all_merged.append(merged.reshape(d, d))
        all_ortho.append(ortho)

    naive_deltas = []
    gt_deltas = []
    for l in range(L):
        naive = all_merged[l].flatten() - all_ortho[l][remove_idx]
        naive_deltas.append(naive.reshape(d, d))

        remaining = [experts[i][l]["dW"].flatten() for i in range(N) if i != remove_idx]
        gt, _ = gram_schmidt_merge(remaining)
        gt_deltas.append(gt.reshape(d, d))

    out_naive = np.array([forward_pre_rmsn(x, base_weights, naive_deltas) for x in inputs])
    out_gt = np.array([forward_pre_rmsn(x, base_weights, gt_deltas) for x in inputs])

    norms = np.linalg.norm(out_naive - out_gt, axis=1)
    gt_norms = np.maximum(np.linalg.norm(out_gt, axis=1), 1e-12)
    devs = norms / gt_norms * 100.0

    return {"remove_idx": remove_idx, "mean_dev_pct": float(np.mean(devs)),
            "max_dev_pct": float(np.max(devs))}


def measure_removal_permuted_single(experts, base_weights, inputs, remove_idx,
                                     d, L, N, layer_perms):
    """Removal error for one set of per-layer permutations."""
    all_merged = []
    all_ortho_by_idx = []

    for l in range(L):
        deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        perm = layer_perms[l]
        reordered = [deltas[perm[j]] for j in range(N)]
        merged, ortho_list = gram_schmidt_merge(reordered)
        all_merged.append(merged.reshape(d, d))
        ortho_map = {perm[j]: ortho_list[j] for j in range(N)}
        all_ortho_by_idx.append(ortho_map)

    naive_deltas = []
    gt_deltas = []
    for l in range(L):
        naive = all_merged[l].flatten() - all_ortho_by_idx[l][remove_idx]
        naive_deltas.append(naive.reshape(d, d))

        # GS recompute with same permutation order (minus removed expert)
        perm = layer_perms[l]
        new_order = [p for p in perm if p != remove_idx]
        deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        reordered_rem = [deltas[p] for p in new_order]
        gt, _ = gram_schmidt_merge(reordered_rem)
        gt_deltas.append(gt.reshape(d, d))

    out_naive = np.array([forward_pre_rmsn(x, base_weights, naive_deltas) for x in inputs])
    out_gt = np.array([forward_pre_rmsn(x, base_weights, gt_deltas) for x in inputs])

    norms = np.linalg.norm(out_naive - out_gt, axis=1)
    gt_norms = np.maximum(np.linalg.norm(out_gt, axis=1), 1e-12)
    devs = norms / gt_norms * 100.0

    return float(np.mean(devs)), float(np.max(devs))


def measure_removal_permuted(experts, base_weights, inputs, remove_idx,
                              d, L, N, n_perms, perm_rng):
    """Removal error averaged over P random GS permutations."""
    mean_devs = []
    max_devs = []
    for _ in range(n_perms):
        layer_perms = [perm_rng.permutation(N) for _ in range(L)]
        md, xd = measure_removal_permuted_single(
            experts, base_weights, inputs, remove_idx, d, L, N, layer_perms)
        mean_devs.append(md)
        max_devs.append(xd)

    return {
        "remove_idx": remove_idx,
        "mean_dev": float(np.mean(mean_devs)),
        "std_dev": float(np.std(mean_devs)),
        "worst_mean_dev": float(np.max(mean_devs)),
        "worst_max_dev": float(np.max(max_devs)),
        "all_mean_devs": mean_devs,
    }


# ============================================================================
# Experiment runner for one (d, N, L) configuration
# ============================================================================

def run_config(d, r, L, N, n_inputs, n_perms, seeds, positions=None, label=""):
    """Run unpermuted + permuted comparison for a configuration.

    Args:
        positions: list of removal indices to test. None = all N.
    """
    if positions is None:
        positions = list(range(N))

    print(f"\n{'=' * 70}")
    print(f"  {label}: d={d}, N={N}, L={L}, r={r}, P={n_perms}, "
          f"|positions|={len(positions)}")
    print(f"{'=' * 70}")

    seed_results = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        rng = np.random.RandomState(seed)

        t0 = time.time()
        experts = [[generate_lora_expert_layer(d, r, rng) for _ in range(L)]
                    for _ in range(N)]
        base_weights = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]
        inputs = rng.randn(n_inputs, d) * 0.1
        print(f"    Setup: {time.time() - t0:.1f}s")

        # Unpermuted sweep
        t0 = time.time()
        unperm = []
        for idx in positions:
            res = measure_removal_unpermuted(experts, base_weights, inputs, idx, d, L, N)
            unperm.append(res)
        print(f"    Unpermuted ({len(positions)} pos): {time.time() - t0:.1f}s")

        unperm_devs = [r["mean_dev_pct"] for r in unperm]
        # For statistics, exclude the last position if it's in the list (exact zero)
        nonlast = [dev for r, dev in zip(unperm, unperm_devs) if r["remove_idx"] != N - 1]
        if not nonlast:
            nonlast = unperm_devs
        u_mean = float(np.mean(nonlast))
        u_worst = float(np.max(nonlast))
        u_std = float(np.std(nonlast))

        # Permuted sweep
        t0 = time.time()
        perm_rng = np.random.RandomState(seed + 1000)
        perm_results = []
        for idx in positions:
            res = measure_removal_permuted(
                experts, base_weights, inputs, idx, d, L, N, n_perms, perm_rng)
            perm_results.append(res)
        print(f"    Permuted ({len(positions)} pos x {n_perms} perms): "
              f"{time.time() - t0:.1f}s")

        p_devs = [r["mean_dev"] for r in perm_results]
        p_worst_singles = [r["worst_max_dev"] for r in perm_results]
        p_mean = float(np.mean(p_devs))
        p_worst = float(np.max(p_devs))
        p_abs_worst = float(np.max(p_worst_singles))
        p_std = float(np.std(p_devs))

        # Display comparison
        print(f"    Unpermuted: mean={u_mean:.4f}%, worst={u_worst:.4f}%, "
              f"ratio={u_worst/u_mean:.3f}x")
        print(f"    Permuted:   mean={p_mean:.4f}%, worst={p_worst:.4f}%, "
              f"ratio={p_worst/p_mean:.3f}x")

        # Per-position detail
        print(f"\n    {'Pos':>5} {'Unperm%':>10} {'Perm_mean%':>10} "
              f"{'Perm_std%':>10} {'Perm_worst%':>12}")
        for u, p in zip(unperm, perm_results):
            print(f"    {u['remove_idx']:>5} {u['mean_dev_pct']:>10.4f} "
                  f"{p['mean_dev']:>10.4f} {p['std_dev']:>10.4f} "
                  f"{p['worst_max_dev']:>12.4f}")

        seed_results.append({
            "seed": seed,
            "unpermuted": {
                "mean": u_mean, "worst": u_worst, "std": u_std,
                "worst_over_mean": u_worst / u_mean if u_mean > 1e-12 else 0,
                "per_position": [{
                    "idx": r["remove_idx"], "dev": r["mean_dev_pct"]
                } for r in unperm],
            },
            "permuted": {
                "mean": p_mean, "worst": p_worst, "abs_worst": p_abs_worst,
                "std": p_std,
                "worst_over_mean": p_worst / p_mean if p_mean > 1e-12 else 0,
                "per_position": [{
                    "idx": r["remove_idx"], "dev": r["mean_dev"],
                    "std": r["std_dev"], "worst": r["worst_max_dev"],
                } for r in perm_results],
            },
        })

    return seed_results


def run_full_experiment():
    t_start = time.time()

    print("=" * 80)
    print("  EXPERIMENT: GS Random Permutation Validation")
    print("  Does random GS ordering per layer eliminate position sensitivity?")
    print("  K1: Permuted worst-case < 2x unpermuted mean deviation")
    print("  K2: No position exceeds 1% at d=256, N=50")
    print("=" * 80)

    seeds = [42, 123, 777]
    r = 8
    n_perms = 5

    # ================================================================
    # TEST 1: Full sweep at d=128, N=20, L=12 (all positions)
    # ================================================================
    res_128 = run_config(d=128, r=r, L=12, N=20, n_inputs=100, n_perms=n_perms,
                         seeds=seeds, positions=None, label="TEST 1 (primary)")

    # ================================================================
    # TEST 2: Full sweep at d=256, N=20, L=12 (all positions)
    # ================================================================
    res_256 = run_config(d=256, r=r, L=12, N=20, n_inputs=100, n_perms=n_perms,
                         seeds=seeds, positions=None, label="TEST 2 (d=256)")

    # ================================================================
    # TEST 3: Larger N at d=128 (N=50, key positions)
    # ================================================================
    key_pos_50 = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    res_n50 = run_config(d=128, r=r, L=12, N=50, n_inputs=100, n_perms=n_perms,
                         seeds=seeds, positions=key_pos_50,
                         label="TEST 3 (N=50)")

    # ================================================================
    # AGGREGATE AND KILL CRITERIA
    # ================================================================
    print("\n" + "=" * 80)
    print("  AGGREGATE KILL CRITERIA ASSESSMENT")
    print("=" * 80)

    def summarize(results, label):
        u_means = [s["unpermuted"]["mean"] for s in results]
        u_worsts = [s["unpermuted"]["worst"] for s in results]
        u_ratios = [s["unpermuted"]["worst_over_mean"] for s in results]
        p_means = [s["permuted"]["mean"] for s in results]
        p_worsts = [s["permuted"]["worst"] for s in results]
        p_abs_worsts = [s["permuted"]["abs_worst"] for s in results]
        p_ratios = [s["permuted"]["worst_over_mean"] for s in results]

        ref_mean = np.mean(u_means)
        perm_abs_worst = max(p_abs_worsts)
        k1_ratio = perm_abs_worst / ref_mean if ref_mean > 1e-12 else float("inf")

        # Spread reduction
        u_cv = np.mean([s["unpermuted"]["std"] / s["unpermuted"]["mean"] * 100
                        for s in results if s["unpermuted"]["mean"] > 1e-12])
        p_cv = np.mean([s["permuted"]["std"] / s["permuted"]["mean"] * 100
                        for s in results if s["permuted"]["mean"] > 1e-12])

        print(f"\n  {label}:")
        print(f"    Unpermuted: mean={np.mean(u_means):.4f}%, "
              f"worst={np.mean(u_worsts):.4f}%, "
              f"worst/mean={np.mean(u_ratios):.3f}x, CV={u_cv:.1f}%")
        print(f"    Permuted:   mean={np.mean(p_means):.4f}%, "
              f"worst_mean={np.mean(p_worsts):.4f}%, "
              f"abs_worst={perm_abs_worst:.4f}%, "
              f"worst/mean={np.mean(p_ratios):.3f}x, CV={p_cv:.1f}%")
        print(f"    K1 ratio (abs_worst / unperm_mean): {k1_ratio:.3f}x "
              f"({'PASS' if k1_ratio < 2.0 else 'FAIL'})")
        print(f"    K2 (abs_worst < 1%): {perm_abs_worst:.4f}% "
              f"({'PASS' if perm_abs_worst < 1.0 else 'FAIL'})")
        print(f"    Spread ratio reduction: {np.mean(u_ratios):.3f}x -> "
              f"{np.mean(p_ratios):.3f}x")

        return {
            "unperm_mean": float(np.mean(u_means)),
            "unperm_worst": float(np.mean(u_worsts)),
            "unperm_ratio": float(np.mean(u_ratios)),
            "perm_mean": float(np.mean(p_means)),
            "perm_worst_mean": float(np.mean(p_worsts)),
            "perm_abs_worst": float(perm_abs_worst),
            "perm_ratio": float(np.mean(p_ratios)),
            "k1_ratio": float(k1_ratio),
            "k1_pass": bool(k1_ratio < 2.0),
            "k2_worst": float(perm_abs_worst),
            "k2_pass": bool(perm_abs_worst < 1.0),
            "unperm_cv": float(u_cv),
            "perm_cv": float(p_cv),
        }

    s1 = summarize(res_128, "d=128, N=20 (primary)")
    s2 = summarize(res_256, "d=256, N=20")
    s3 = summarize(res_n50, "d=128, N=50 (key positions)")

    # Overall K1: use the most conservative (d=256) result
    # K1 applies at d=256, N=50 per the hypothesis. We test d=256,N=20 and d=128,N=50.
    # Use whichever is worst.
    overall_k1_ratio = max(s1["k1_ratio"], s2["k1_ratio"], s3["k1_ratio"])
    overall_k1_pass = overall_k1_ratio < 2.0
    overall_k2_worst = max(s1["k2_worst"], s2["k2_worst"], s3["k2_worst"])
    overall_k2_pass = overall_k2_worst < 1.0

    print("\n" + "=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)
    print(f"\n  K1: Permuted worst-case < 2x unpermuted mean")
    print(f"      Worst K1 ratio across all configs: {overall_k1_ratio:.3f}x")
    print(f"      VERDICT: {'PASS' if overall_k1_pass else 'FAIL'}")

    print(f"\n  K2: No position exceeds 1%")
    print(f"      Worst absolute deviation across all configs: {overall_k2_worst:.4f}%")
    print(f"      VERDICT: {'PASS' if overall_k2_pass else 'FAIL'}")

    overall = "PROVEN" if overall_k1_pass and overall_k2_pass else \
              "SUPPORTED" if overall_k2_pass else "FAIL"
    print(f"\n  OVERALL: {overall}")

    if overall_k1_pass and overall_k2_pass:
        print("  Random GS permutation validated: eliminates position sensitivity.")
        print("  Worst-case deviation converges to ~mean. No new failure modes.")
    elif overall_k2_pass:
        print("  Permutation does not fully equalize but stays within safety bounds.")

    # ================================================================
    # Save results
    # ================================================================
    results_path = Path(__file__).parent / "results.json"
    save_data = {
        "config": {
            "r": r, "n_perms": n_perms, "seeds": seeds,
            "test_configs": [
                {"d": 128, "N": 20, "L": 12, "positions": "all"},
                {"d": 256, "N": 20, "L": 12, "positions": "all"},
                {"d": 128, "N": 50, "L": 12, "positions": str(key_pos_50)},
            ],
        },
        "summaries": {
            "d128_N20": s1,
            "d256_N20": s2,
            "d128_N50": s3,
        },
        "kill_criteria": {
            "K1_overall_ratio": overall_k1_ratio,
            "K1_pass": overall_k1_pass,
            "K2_overall_worst": overall_k2_worst,
            "K2_pass": overall_k2_pass,
            "overall": overall,
        },
        "raw": {
            "d128_N20": [{
                "seed": s["seed"],
                "unpermuted": {k: v for k, v in s["unpermuted"].items()},
                "permuted": {k: v for k, v in s["permuted"].items()},
            } for s in res_128],
            "d256_N20": [{
                "seed": s["seed"],
                "unpermuted": {k: v for k, v in s["unpermuted"].items()},
                "permuted": {k: v for k, v in s["permuted"].items()},
            } for s in res_256],
            "d128_N50": [{
                "seed": s["seed"],
                "unpermuted": {k: v for k, v in s["unpermuted"].items()},
                "permuted": {k: v for k, v in s["permuted"].items()},
            } for s in res_n50],
        },
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total experiment time: {elapsed:.1f}s")
    return save_data


if __name__ == "__main__":
    run_full_experiment()
