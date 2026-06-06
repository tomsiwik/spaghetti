"""Expert Removal Position Sensitivity: Does GS ordering affect removal safety?

The parent experiment (removal_safety_complete_bound) only removed the middle expert
(index N//2). Gram-Schmidt orthogonalization is order-dependent:
  - Expert 0 retains its full original direction
  - Expert N-1 has been maximally projected against all predecessors

This creates an asymmetry: removing expert 0 means all subsequent experts' GS
corrections referenced a now-missing vector. Removing expert N-1 only removes
the most-orthogonalized vector, which should be cleanest.

We sweep removal position across {0, N//4, N//2, 3N//4, N-1} and measure:
  1. Per-layer weight-space error (naive subtraction vs GS recompute)
  2. Output deviation through Pre-RMSNorm transformer

Kill criteria:
  K1: position-dependent deviation varies by <2x across all positions at N=50
  K2: worst-case position still within 2x of the mean bound

CPU only. numpy/scipy on Apple Silicon.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================================
# Core utilities (reused from parent: removal_safety_complete_bound)
# ============================================================================

def generate_lora_expert_layer(d: int, r: int, rng: np.random.RandomState) -> dict:
    """Generate a single LoRA expert for one layer."""
    A = rng.randn(d, r) / np.sqrt(d)
    B = rng.randn(r, d) / np.sqrt(r)
    return {"A": A, "B": B, "dW": A @ B}


def gram_schmidt_merge(deltas: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """GS-orthogonalize then sum flattened deltas.

    Returns (merged_sum, list_of_orthogonalized_deltas).
    The orthogonalized deltas are order-dependent: delta_i' depends on
    deltas 0..i-1.
    """
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
    """RMSNorm without learnable scale (Qwen/Llama style)."""
    if x.ndim == 1:
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return x / rms
    else:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return x / rms


def activation(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def forward_pre_rmsn(h: np.ndarray, base_weights: list[np.ndarray],
                     layer_deltas: list[np.ndarray]) -> np.ndarray:
    """Pre-RMSNorm transformer forward pass."""
    L = len(base_weights)
    scale = 1.0 / np.sqrt(L) if L > 1 else 1.0
    for l in range(L):
        W = base_weights[l] + layer_deltas[l]
        h = h + scale * activation(W @ rms_norm(h))
    return h


# ============================================================================
# Position sweep experiment
# ============================================================================

def measure_removal_at_position(
    experts: list[list[dict]],
    base_weights: list[np.ndarray],
    inputs: np.ndarray,
    remove_idx: int,
    d: int, L: int, N: int,
) -> dict:
    """Measure removal error for a specific position in the GS ordering.

    Same methodology as parent experiment but parameterized on remove_idx.
    """
    # GS merge all N experts per layer, compute naive removal vs GS recompute
    all_merged_deltas = []
    all_ortho_deltas = []
    per_layer_errors = []

    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        merged_flat, ortho_flat = gram_schmidt_merge(layer_deltas)
        all_merged_deltas.append(merged_flat.reshape(d, d))
        all_ortho_deltas.append(ortho_flat)

        # Naive removal vs GS recompute
        naive_flat = merged_flat - ortho_flat[remove_idx]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)

        diff_norm = np.linalg.norm(naive_flat - gt_flat)
        gt_norm = np.linalg.norm(gt_flat)
        recon_err = float(diff_norm / (gt_norm + 1e-12)) * 100.0
        per_layer_errors.append(recon_err)

    sum_per_layer = sum(per_layer_errors)
    mean_per_layer = float(np.mean(per_layer_errors))
    std_per_layer = float(np.std(per_layer_errors))

    # Forward pass: all experts (full model)
    outputs_all = np.array([forward_pre_rmsn(inp, base_weights, all_merged_deltas)
                            for inp in inputs])

    # Forward pass: naive removal
    naive_removed_deltas = []
    for l in range(L):
        merged = all_merged_deltas[l].flatten()
        removed = merged - all_ortho_deltas[l][remove_idx]
        naive_removed_deltas.append(removed.reshape(d, d))

    outputs_naive = np.array([forward_pre_rmsn(inp, base_weights, naive_removed_deltas)
                              for inp in inputs])

    # Forward pass: GS recompute (ground truth)
    gt_removed_deltas = []
    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)
        gt_removed_deltas.append(gt_flat.reshape(d, d))

    outputs_gt = np.array([forward_pre_rmsn(inp, base_weights, gt_removed_deltas)
                           for inp in inputs])

    # Compute metrics
    naive_vs_gt_norms = np.linalg.norm(outputs_naive - outputs_gt, axis=1)
    gt_norms = np.linalg.norm(outputs_gt, axis=1)
    safe_gt = np.maximum(gt_norms, 1e-12)
    relative_devs = naive_vs_gt_norms / safe_gt * 100.0

    mean_dev = float(np.mean(relative_devs))
    max_dev = float(np.max(relative_devs))
    median_dev = float(np.median(relative_devs))
    std_dev = float(np.std(relative_devs))

    amp_ratio = mean_dev / sum_per_layer if sum_per_layer > 1e-12 else 0.0

    # How much the expert contributed to the full output
    all_vs_gt_norms = np.linalg.norm(outputs_all - outputs_gt, axis=1)
    expert_signal = float(np.mean(all_vs_gt_norms / safe_gt * 100.0))

    # Measure the GS residual norm (how much of the original was retained after GS)
    # This quantifies the "orthogonalization quality" at this position
    gs_retention_ratios = []
    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        _, ortho_flat = gram_schmidt_merge(layer_deltas)
        orig_norm = np.linalg.norm(layer_deltas[remove_idx])
        ortho_norm = np.linalg.norm(ortho_flat[remove_idx])
        if orig_norm > 1e-12:
            gs_retention_ratios.append(ortho_norm / orig_norm)

    mean_gs_retention = float(np.mean(gs_retention_ratios))

    return {
        "remove_idx": remove_idx,
        "position_label": _position_label(remove_idx, N),
        "mean_per_layer_error_pct": mean_per_layer,
        "std_per_layer_error_pct": std_per_layer,
        "sum_per_layer_error_pct": sum_per_layer,
        "mean_output_dev_pct": mean_dev,
        "max_output_dev_pct": max_dev,
        "median_output_dev_pct": median_dev,
        "std_output_dev_pct": std_dev,
        "amplification_ratio": amp_ratio,
        "expert_signal_pct": expert_signal,
        "mean_gs_retention": mean_gs_retention,
        "per_layer_errors": per_layer_errors,
    }


def _position_label(idx: int, N: int) -> str:
    if idx == 0:
        return "first"
    elif idx == N - 1:
        return "last"
    elif idx == N // 4:
        return "Q1"
    elif idx == N // 2:
        return "middle"
    elif idx == 3 * N // 4:
        return "Q3"
    else:
        return f"idx_{idx}"


# ============================================================================
# Main experiment
# ============================================================================

def run_position_sweep(d: int, r: int, L: int, N: int, n_inputs: int,
                       seeds: list[int]) -> dict:
    """Sweep removal position across key indices."""
    positions = [0, N // 4, N // 2, 3 * N // 4, N - 1]
    position_labels = [_position_label(p, N) for p in positions]

    all_results = {}  # label -> list of per-seed results

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        rng = np.random.RandomState(seed)

        # Generate experts (same for all positions within a seed)
        print(f"    Generating {N} experts x {L} layers at d={d}, r={r}...")
        t0 = time.time()
        experts = []
        for i in range(N):
            layers = [generate_lora_expert_layer(d, r, rng) for _ in range(L)]
            experts.append(layers)

        base_weights = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]
        inputs = rng.randn(n_inputs, d) * 0.1
        print(f"    Generation: {time.time() - t0:.1f}s")

        for pos, label in zip(positions, position_labels):
            print(f"    Position {pos} ({label})...")
            t1 = time.time()
            result = measure_removal_at_position(
                experts, base_weights, inputs,
                remove_idx=pos, d=d, L=L, N=N,
            )
            result["seed"] = seed
            result["d"] = d
            result["r"] = r
            result["L"] = L
            result["N"] = N
            elapsed = time.time() - t1
            print(f"      dev={result['mean_output_dev_pct']:.4f}%, "
                  f"amp={result['amplification_ratio']:.4f}, "
                  f"gs_ret={result['mean_gs_retention']:.4f}, "
                  f"time={elapsed:.1f}s")

            if label not in all_results:
                all_results[label] = []
            all_results[label].append(result)

    return all_results


def run_full_experiment():
    t_start = time.time()

    print("=" * 80)
    print("  EXPERIMENT: Expert Removal Position Sensitivity")
    print("  Does GS ordering affect removal safety?")
    print("  K1: position deviation varies by <2x across all positions")
    print("  K2: worst-case position within 2x of mean bound")
    print("=" * 80)

    seeds = [42, 123, 777]

    # ================================================================
    # TEST 1: Position sweep at target scale (d=256, N=50, L=24)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 1: Position sweep at d=256, L=24, N=50")
    print("=" * 80)

    results = run_position_sweep(
        d=256, r=8, L=24, N=50, n_inputs=200, seeds=seeds
    )

    # ================================================================
    # Aggregate results
    # ================================================================
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    summary = {}
    print(f"\n  {'Position':>10} {'MeanDev%':>10} {'StdDev':>8} {'MaxDev%':>10} "
          f"{'AmpRatio':>10} {'GS_Ret':>8} {'SumEps%':>10}")
    print("  " + "-" * 70)

    for label in ["first", "Q1", "middle", "Q3", "last"]:
        if label not in results:
            continue
        runs = results[label]
        devs = [r["mean_output_dev_pct"] for r in runs]
        maxs = [r["max_output_dev_pct"] for r in runs]
        amps = [r["amplification_ratio"] for r in runs]
        rets = [r["mean_gs_retention"] for r in runs]
        seps = [r["sum_per_layer_error_pct"] for r in runs]

        summary[label] = {
            "mean_dev": float(np.mean(devs)),
            "std_dev": float(np.std(devs)),
            "max_dev": float(np.mean(maxs)),
            "mean_amp": float(np.mean(amps)),
            "mean_gs_retention": float(np.mean(rets)),
            "mean_sum_eps": float(np.mean(seps)),
            "remove_idx": runs[0]["remove_idx"],
        }

        s = summary[label]
        print(f"  {label:>10} {s['mean_dev']:>10.4f} {s['std_dev']:>8.4f} "
              f"{s['max_dev']:>10.4f} {s['mean_amp']:>10.4f} "
              f"{s['mean_gs_retention']:>8.4f} {s['mean_sum_eps']:>10.4f}")

    # ================================================================
    # TEST 2: Position sensitivity ratio
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 2: Position sensitivity analysis")
    print("=" * 80)

    all_devs = [summary[l]["mean_dev"] for l in summary]
    min_dev = min(all_devs)
    max_dev_val = max(all_devs)
    mean_dev = np.mean(all_devs)
    ratio_max_min = max_dev_val / min_dev if min_dev > 1e-12 else float("inf")
    ratio_max_mean = max_dev_val / mean_dev if mean_dev > 1e-12 else float("inf")

    print(f"\n  Min deviation:  {min_dev:.4f}% (position: "
          f"{[l for l in summary if summary[l]['mean_dev'] == min_dev][0]})")
    print(f"  Max deviation:  {max_dev_val:.4f}% (position: "
          f"{[l for l in summary if summary[l]['mean_dev'] == max_dev_val][0]})")
    print(f"  Mean deviation: {mean_dev:.4f}%")
    print(f"  Max/Min ratio:  {ratio_max_min:.3f}x")
    print(f"  Max/Mean ratio: {ratio_max_mean:.3f}x")
    print(f"  CV:             {np.std(all_devs) / mean_dev * 100:.1f}%")

    # ================================================================
    # TEST 3: GS retention vs position (explains the mechanism)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 3: GS retention ratio vs position")
    print("=" * 80)

    print(f"\n  GS retention = ||delta_k'|| / ||delta_k|| after orthogonalization")
    print(f"  Lower retention = more projection removed = more orthogonalized")
    print()
    for label in ["first", "Q1", "middle", "Q3", "last"]:
        if label not in summary:
            continue
        s = summary[label]
        print(f"  {label:>10}: retention={s['mean_gs_retention']:.4f}, "
              f"dev={s['mean_dev']:.4f}%, sum_eps={s['mean_sum_eps']:.4f}%")

    # Correlation between position index and deviation
    positions_idx = [summary[l]["remove_idx"] for l in summary]
    position_devs = [summary[l]["mean_dev"] for l in summary]
    if len(positions_idx) >= 3:
        slope, intercept, r_val, p_val, se = stats.linregress(
            positions_idx, position_devs)
        print(f"\n  Linear regression (position vs deviation):")
        print(f"    slope={slope:.6f}, R^2={r_val**2:.4f}, p={p_val:.4f}")

    # Correlation between GS retention and deviation
    position_rets = [summary[l]["mean_gs_retention"] for l in summary]
    if len(position_rets) >= 3:
        slope_r, intercept_r, r_val_r, p_val_r, se_r = stats.linregress(
            position_rets, position_devs)
        print(f"\n  Linear regression (GS retention vs deviation):")
        print(f"    slope={slope_r:.6f}, R^2={r_val_r**2:.4f}, p={p_val_r:.4f}")

    # ================================================================
    # TEST 4: Smaller scale for faster iteration (d=128)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 4: Cross-validation at d=128, L=24, N=50")
    print("=" * 80)

    results_128 = run_position_sweep(
        d=128, r=8, L=24, N=50, n_inputs=200, seeds=seeds
    )

    summary_128 = {}
    print(f"\n  {'Position':>10} {'MeanDev%':>10} {'StdDev':>8} {'GS_Ret':>8}")
    print("  " + "-" * 40)

    for label in ["first", "Q1", "middle", "Q3", "last"]:
        if label not in results_128:
            continue
        runs = results_128[label]
        devs = [r["mean_output_dev_pct"] for r in runs]
        rets = [r["mean_gs_retention"] for r in runs]
        summary_128[label] = {
            "mean_dev": float(np.mean(devs)),
            "std_dev": float(np.std(devs)),
            "mean_gs_retention": float(np.mean(rets)),
        }
        s = summary_128[label]
        print(f"  {label:>10} {s['mean_dev']:>10.4f} {s['std_dev']:>8.4f} "
              f"{s['mean_gs_retention']:>8.4f}")

    all_devs_128 = [summary_128[l]["mean_dev"] for l in summary_128]
    ratio_128 = max(all_devs_128) / min(all_devs_128) if min(all_devs_128) > 1e-12 else float("inf")
    print(f"\n  d=128 Max/Min ratio: {ratio_128:.3f}x")

    # ================================================================
    # TEST 5: Dense position sweep (every 5th index) at d=128
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 5: Dense position sweep at d=128, N=50 (single seed)")
    print("=" * 80)

    rng = np.random.RandomState(42)
    experts_dense = []
    for i in range(50):
        layers = [generate_lora_expert_layer(128, 8, rng) for _ in range(24)]
        experts_dense.append(layers)
    base_dense = [rng.randn(128, 128) / np.sqrt(128) for _ in range(24)]
    inputs_dense = rng.randn(200, 128) * 0.1

    dense_positions = list(range(0, 50, 5)) + [49]
    dense_results = []
    for pos in dense_positions:
        res = measure_removal_at_position(
            experts_dense, base_dense, inputs_dense,
            remove_idx=pos, d=128, L=24, N=50,
        )
        dense_results.append(res)
        print(f"  pos={pos:3d} ({res['position_label']:>6}): "
              f"dev={res['mean_output_dev_pct']:.4f}%, "
              f"gs_ret={res['mean_gs_retention']:.4f}")

    dense_devs = [r["mean_output_dev_pct"] for r in dense_results]
    dense_ratio = max(dense_devs) / min(dense_devs) if min(dense_devs) > 1e-12 else float("inf")
    print(f"\n  Dense sweep Max/Min ratio: {dense_ratio:.3f}x")

    # ================================================================
    # KILL CRITERIA ASSESSMENT
    # ================================================================
    print("\n" + "=" * 80)
    print("  KILL CRITERIA ASSESSMENT")
    print("=" * 80)

    # K1: position-dependent deviation varies by <2x
    k1_pass = ratio_max_min < 2.0
    print(f"\n  K1: Position deviation ratio < 2x")
    print(f"      d=256 Max/Min ratio: {ratio_max_min:.3f}x")
    print(f"      d=128 Max/Min ratio: {ratio_128:.3f}x")
    print(f"      Dense sweep ratio:   {dense_ratio:.3f}x")
    print(f"      VERDICT: {'PASS' if k1_pass else 'FAIL'} "
          f"(position sensitivity is {'low' if k1_pass else 'significant'})")

    # K2: worst-case position within 2x of mean bound
    k2_ratio = ratio_max_mean
    k2_pass = k2_ratio < 2.0
    print(f"\n  K2: Worst-case position within 2x of mean")
    print(f"      d=256 Max/Mean ratio: {k2_ratio:.3f}x")
    print(f"      VERDICT: {'PASS' if k2_pass else 'FAIL'}")

    overall = "PROVEN" if k1_pass and k2_pass else "SUPPORTED" if k1_pass or k2_pass else "FAIL"
    print(f"\n  OVERALL: {overall}")
    if k1_pass and k2_pass:
        print(f"    Position sensitivity exists but is small (<2x).")
        print(f"    The parent experiment's middle-only removal is representative.")
        print(f"    No position-specific safety concern for SOLE deployment.")
    elif not k1_pass:
        print(f"    Position sensitivity is significant (>{ratio_max_min:.1f}x).")
        print(f"    Early experts (low GS index) may need special handling.")
        print(f"    Recommendation: use random permutation before GS to amortize.")

    # ================================================================
    # Save results
    # ================================================================
    results_path = Path(__file__).parent / "results.json"

    # Convert per-seed results to serializable format
    def clean_results(res_dict):
        cleaned = {}
        for label, runs in res_dict.items():
            cleaned[label] = []
            for r in runs:
                r_clean = {k: v for k, v in r.items() if k != "per_layer_errors"}
                cleaned[label].append(r_clean)
        return cleaned

    save_data = {
        "d256_summary": summary,
        "d128_summary": summary_128,
        "d256_raw": clean_results(results),
        "d128_raw": clean_results(results_128),
        "dense_sweep": [{
            "position": r["remove_idx"],
            "label": r["position_label"],
            "mean_dev_pct": r["mean_output_dev_pct"],
            "gs_retention": r["mean_gs_retention"],
        } for r in dense_results],
        "kill_criteria": {
            "K1_max_min_ratio_d256": ratio_max_min,
            "K1_max_min_ratio_d128": ratio_128,
            "K1_max_min_ratio_dense": dense_ratio,
            "K1_pass": k1_pass,
            "K2_max_mean_ratio_d256": k2_ratio,
            "K2_pass": k2_pass,
            "overall": overall,
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
