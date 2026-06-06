"""Correlated Per-Layer Errors: Does sub-additivity survive adversarial correlation?

Parent experiment (multilayer_removal_cascade) proved:
  - amp_ratio = 0.25 at L=24 with INDEPENDENT per-layer errors
  - Sub-additivity driven by 3 mechanisms:
    1. Activation masking (~50% suppression per layer)
    2. Direction randomization (sqrt(L) vs L scaling)
    3. Spectral contraction

This experiment attacks Mechanism 2: if all layers' removal errors point in
the SAME direction (correlated expert), direction randomization fails.
The question is whether Mechanisms 1+3 alone are sufficient.

Kill criteria:
  K1: correlated errors amplify >2x vs independent errors at same cosine
      -> sub-additivity breaks, need cosine bound on correlation
  K2: amplification ratio still <1.0 even with maximally correlated errors
      -> sub-additivity is robust, correlation doesn't matter

CPU only. numpy/scipy on Apple Silicon.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================================
# Expert Generation with Controlled Inter-Layer Correlation
# ============================================================================

def generate_base_weights(L: int, d: int, seed: int) -> list[np.ndarray]:
    """Generate L random base weight matrices."""
    rng = np.random.RandomState(seed)
    return [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]


def generate_experts_with_correlation(
    N: int, L: int, d: int, r: int,
    inter_layer_corr: float,
    intra_layer_cos: float | None,
    seed: int
) -> list[list[dict]]:
    """Generate N experts with controlled inter-layer correlation.

    Args:
        N: number of experts
        L: number of layers
        d: dimension
        r: LoRA rank
        inter_layer_corr: correlation between the SAME expert's deltas across
            layers. 0.0 = independent (parent experiment baseline),
            1.0 = perfectly aligned (adversarial worst case).
        intra_layer_cos: cosine between DIFFERENT experts within a layer.
            None = random (near-orthogonal at high d).
        seed: random seed

    Returns: experts[expert_idx][layer_idx] = {"dW": (d, d)}
    """
    rng = np.random.RandomState(seed)

    experts = []
    for i in range(N):
        # Generate a "semantic direction" for this expert -- the direction
        # that is consistent across all layers when inter_layer_corr > 0
        expert_direction = rng.randn(d, d)
        expert_direction = expert_direction / np.linalg.norm(expert_direction)

        layers = []
        for l in range(L):
            # Generate a random per-layer component
            random_component = rng.randn(d, d)

            # Remove projection onto expert_direction (make orthogonal)
            proj = np.sum(random_component * expert_direction) * expert_direction
            random_component = random_component - proj
            rn = np.linalg.norm(random_component)
            if rn > 1e-12:
                random_component = random_component / rn

            # Mix: corr * shared_direction + sqrt(1-corr^2) * random
            alpha = inter_layer_corr
            beta = np.sqrt(1.0 - alpha ** 2)
            dW = alpha * expert_direction + beta * random_component

            # Scale to realistic LoRA magnitude
            # Match parent: A (d,r) ~ N(0, 1/d), B (r,d) ~ N(0, 1/r)
            # => ||A@B||_F ~ sqrt(d*d * r / (d*r)) = sqrt(d/1) ... simplify
            # Just use a consistent magnitude
            target_norm = np.sqrt(d) * r / d  # ~ r/sqrt(d)
            dW = dW * target_norm

            layers.append({"dW": dW})
        experts.append(layers)

    # If intra_layer_cos is specified, adjust within-layer structure
    if intra_layer_cos is not None and intra_layer_cos > 0.01:
        # Add a shared within-layer direction to increase cross-expert cosine
        for l in range(L):
            shared_dir = rng.randn(d, d)
            shared_dir = shared_dir / np.linalg.norm(shared_dir)

            alpha_intra = np.sqrt(intra_layer_cos)
            beta_intra = np.sqrt(1.0 - intra_layer_cos)

            for i in range(N):
                dW = experts[i][l]["dW"]
                norm_orig = np.linalg.norm(dW)

                # Project out shared component, re-add with controlled magnitude
                proj = np.sum(dW * shared_dir) * shared_dir
                unique = dW - proj
                un = np.linalg.norm(unique)
                if un > 1e-12:
                    unique = unique / un

                dW_new = alpha_intra * shared_dir + beta_intra * unique
                dW_new = dW_new * norm_orig  # preserve magnitude
                experts[i][l]["dW"] = dW_new

    return experts


# ============================================================================
# Core computation (reused from parent with minimal changes)
# ============================================================================

def gram_schmidt_merge(deltas: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """GS-orthogonalize then sum flattened deltas."""
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


def activation(x: np.ndarray, kind: str = "gelu") -> np.ndarray:
    if kind == "relu":
        return np.maximum(0, x)
    elif kind == "gelu":
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    elif kind == "linear":
        return x
    else:
        raise ValueError(f"Unknown activation: {kind}")


def forward_multilayer(h: np.ndarray, base_weights: list[np.ndarray],
                       layer_deltas: list[np.ndarray],
                       act: str = "gelu") -> np.ndarray:
    for l in range(len(base_weights)):
        W = base_weights[l] + layer_deltas[l]
        h = activation(W @ h, kind=act)
    return h


def compute_inter_layer_correlation(experts: list[list[dict]], L: int) -> dict:
    """Measure actual inter-layer correlation for each expert.

    For each expert, compute mean cosine between its delta at layer l
    and its delta at all other layers l'.
    """
    N = len(experts)
    per_expert_corr = []

    for i in range(N):
        cos_vals = []
        for l1 in range(L):
            for l2 in range(l1 + 1, L):
                d1 = experts[i][l1]["dW"].flatten()
                d2 = experts[i][l2]["dW"].flatten()
                cos = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12)
                cos_vals.append(cos)
        per_expert_corr.append(float(np.mean(cos_vals)))

    return {
        "mean_inter_layer_corr": float(np.mean(per_expert_corr)),
        "min_inter_layer_corr": float(np.min(per_expert_corr)),
        "max_inter_layer_corr": float(np.max(per_expert_corr)),
    }


def compute_intra_layer_cosines(experts: list[list[dict]], L: int) -> float:
    """Mean |cos| between different experts within layers."""
    N = len(experts)
    cos_vals = []
    for l in range(L):
        for i in range(N):
            for j in range(i + 1, N):
                di = experts[i][l]["dW"].flatten()
                dj = experts[j][l]["dW"].flatten()
                cos = abs(np.dot(di, dj) / (np.linalg.norm(di) * np.linalg.norm(dj) + 1e-12))
                cos_vals.append(cos)
    return float(np.mean(cos_vals))


# ============================================================================
# Core Experiment: Removal with Correlated Errors
# ============================================================================

def run_correlated_removal(
    N: int, L: int, d: int, r: int,
    inter_layer_corr: float,
    intra_layer_cos: float | None,
    act: str, n_inputs: int,
    remove_idx: int, seed: int
) -> dict:
    """Run one configuration and measure amplification."""
    rng = np.random.RandomState(seed + 9999)  # separate from expert gen

    # Generate experts with controlled correlation
    experts = generate_experts_with_correlation(
        N, L, d, r, inter_layer_corr, intra_layer_cos, seed
    )

    # Generate base weights
    base_weights = generate_base_weights(L, d, seed + 5555)

    # Generate test inputs
    inputs = rng.randn(n_inputs, d) * 0.1

    # Measure actual correlations
    inter_corr = compute_inter_layer_correlation(experts, L)
    intra_cos = compute_intra_layer_cosines(experts, L)

    # Step 1: Merge all N experts per layer via GS
    all_merged_deltas = []
    all_ortho_deltas = []
    per_layer_recon_errors = []

    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        merged_flat, ortho_flat = gram_schmidt_merge(layer_deltas)
        all_merged_deltas.append(merged_flat.reshape(d, d))
        all_ortho_deltas.append(ortho_flat)

        # Per-layer naive removal error
        naive_flat = merged_flat - ortho_flat[remove_idx]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)

        diff_norm = np.linalg.norm(naive_flat - gt_flat)
        gt_norm = np.linalg.norm(gt_flat)
        recon_err = float(diff_norm / (gt_norm + 1e-12)) * 100.0
        per_layer_recon_errors.append(recon_err)

    # Step 2: Forward pass with ALL experts
    outputs_all = np.array([
        forward_multilayer(inp, base_weights, all_merged_deltas, act)
        for inp in inputs
    ])

    # Step 3: Forward pass AFTER naive removal
    naive_removed_deltas = []
    for l in range(L):
        merged = all_merged_deltas[l].flatten()
        removed = merged - all_ortho_deltas[l][remove_idx]
        naive_removed_deltas.append(removed.reshape(d, d))

    outputs_naive = np.array([
        forward_multilayer(inp, base_weights, naive_removed_deltas, act)
        for inp in inputs
    ])

    # Step 4: Forward pass with GS recompute (ground truth removal)
    gt_removed_deltas = []
    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)
        gt_removed_deltas.append(gt_flat.reshape(d, d))

    outputs_gt = np.array([
        forward_multilayer(inp, base_weights, gt_removed_deltas, act)
        for inp in inputs
    ])

    # Metrics
    sum_per_layer = sum(per_layer_recon_errors)
    mean_per_layer = float(np.mean(per_layer_recon_errors))

    # Output deviation: naive vs GT
    naive_vs_gt_norms = np.linalg.norm(outputs_naive - outputs_gt, axis=1)
    gt_norms = np.linalg.norm(outputs_gt, axis=1)
    safe_gt = np.maximum(gt_norms, 1e-12)
    relative_devs = naive_vs_gt_norms / safe_gt * 100.0

    mean_output_dev = float(np.mean(relative_devs))
    max_output_dev = float(np.max(relative_devs))
    median_output_dev = float(np.median(relative_devs))

    # Amplification ratio
    if sum_per_layer > 1e-12:
        amplification_ratio = mean_output_dev / sum_per_layer
    else:
        amplification_ratio = 0.0

    # Expert signal
    all_vs_gt_norms = np.linalg.norm(outputs_all - outputs_gt, axis=1)
    expert_signal = float(np.mean(all_vs_gt_norms / safe_gt * 100.0))

    return {
        "N": N, "L": L, "d": d, "r": r,
        "inter_layer_corr_target": inter_layer_corr,
        "intra_layer_cos_target": intra_layer_cos,
        "activation": act,
        "remove_idx": remove_idx,
        "seed": seed,
        # Actual measured correlations
        "actual_inter_layer_corr": inter_corr["mean_inter_layer_corr"],
        "actual_intra_layer_cos": intra_cos,
        # Weight-space metrics
        "per_layer_recon_errors": [float(x) for x in per_layer_recon_errors],
        "sum_per_layer_error": sum_per_layer,
        "mean_per_layer_error": mean_per_layer,
        # Output-space metrics
        "mean_output_dev_pct": mean_output_dev,
        "max_output_dev_pct": max_output_dev,
        "median_output_dev_pct": median_output_dev,
        # Key ratio
        "amplification_ratio": amplification_ratio,
        "expert_signal_pct": expert_signal,
    }


# ============================================================================
# Experiment Runner
# ============================================================================

def run_full_experiment():
    t_start = time.time()

    print("=" * 78)
    print("  EXPERIMENT: Correlated Per-Layer Errors")
    print("  Kill: K1 corr_amp/indep_amp > 2x | K2 amp_ratio < 1.0 even at corr=1")
    print("=" * 78)

    d = 64
    r = 8
    N = 8
    n_inputs = 200
    seeds = [42, 123, 777]

    all_results = []

    # ================================================================
    # TEST 1: Inter-layer correlation sweep at fixed depth L=24
    # The critical test: vary correlation from 0 (independent) to 1 (aligned)
    # ================================================================
    print("\n" + "=" * 78)
    print("  TEST 1: Inter-layer correlation sweep (L=24, d=64, N=8)")
    print("  corr=0 is parent baseline; corr=1 is adversarial worst case")
    print("=" * 78)

    L = 24
    corr_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    print(f"\n{'Corr':>6} {'Seed':>6} {'ActCorr':>9} {'IntraCos':>10} "
          f"{'SumLayErr%':>12} {'MeanOutDev%':>13} {'AmpRatio':>10}")
    print("-" * 85)

    for corr in corr_values:
        for seed in seeds:
            res = run_correlated_removal(
                N, L, d, r, inter_layer_corr=corr,
                intra_layer_cos=None, act="gelu",
                n_inputs=n_inputs, remove_idx=N // 2, seed=seed
            )
            res["test"] = "corr_sweep"
            all_results.append(res)
            print(f"{corr:>6.1f} {seed:>6} {res['actual_inter_layer_corr']:>9.4f} "
                  f"{res['actual_intra_layer_cos']:>10.6f} "
                  f"{res['sum_per_layer_error']:>12.4f} "
                  f"{res['mean_output_dev_pct']:>13.4f} "
                  f"{res['amplification_ratio']:>10.4f}")

    # ================================================================
    # TEST 2: Correlation x Depth interaction
    # Does correlation matter more at greater depth?
    # ================================================================
    print("\n" + "=" * 78)
    print("  TEST 2: Correlation x Depth interaction")
    print("  Comparing corr=0 vs corr=1 at different depths")
    print("=" * 78)

    L_values = [1, 4, 8, 12, 24]

    print(f"\n{'L':>4} {'Corr':>6} {'Seed':>6} {'MeanOutDev%':>13} {'AmpRatio':>10}")
    print("-" * 50)

    for L in L_values:
        for corr in [0.0, 0.5, 1.0]:
            for seed in seeds:
                res = run_correlated_removal(
                    N, L, d, r, inter_layer_corr=corr,
                    intra_layer_cos=None, act="gelu",
                    n_inputs=n_inputs, remove_idx=N // 2, seed=seed
                )
                res["test"] = "corr_x_depth"
                all_results.append(res)
                print(f"{L:>4} {corr:>6.1f} {seed:>6} "
                      f"{res['mean_output_dev_pct']:>13.4f} "
                      f"{res['amplification_ratio']:>10.4f}")

    # ================================================================
    # TEST 3: Correlation x Dimension interaction
    # Does higher d rescue sub-additivity even with correlation?
    # ================================================================
    print("\n" + "=" * 78)
    print("  TEST 3: Correlation x Dimension (L=24, corr=1.0 vs 0.0)")
    print("=" * 78)

    d_values = [32, 64, 128, 256]
    L = 24

    print(f"\n{'d':>5} {'Corr':>6} {'Seed':>6} {'MeanOutDev%':>13} {'AmpRatio':>10}")
    print("-" * 50)

    for d_val in d_values:
        for corr in [0.0, 1.0]:
            for seed in seeds:
                res = run_correlated_removal(
                    N, L, d_val, r, inter_layer_corr=corr,
                    intra_layer_cos=None, act="gelu",
                    n_inputs=n_inputs, remove_idx=N // 2, seed=seed
                )
                res["test"] = "corr_x_dim"
                all_results.append(res)
                print(f"{d_val:>5} {corr:>6.1f} {seed:>6} "
                      f"{res['mean_output_dev_pct']:>13.4f} "
                      f"{res['amplification_ratio']:>10.4f}")

    # ================================================================
    # TEST 4: Correlation + intra-layer clustering (double adversarial)
    # Worst case: correlated across layers AND clustered within layer
    # ================================================================
    print("\n" + "=" * 78)
    print("  TEST 4: Double adversarial — correlated + clustered (L=24, d=64)")
    print("=" * 78)

    L = 24

    print(f"\n{'IntCorr':>8} {'IntraCos':>10} {'Seed':>6} "
          f"{'MeanOutDev%':>13} {'AmpRatio':>10}")
    print("-" * 55)

    for inter_corr in [0.0, 0.5, 1.0]:
        for intra_cos in [None, 0.3]:
            for seed in seeds:
                res = run_correlated_removal(
                    N, L, d, r, inter_layer_corr=inter_corr,
                    intra_layer_cos=intra_cos, act="gelu",
                    n_inputs=n_inputs, remove_idx=N // 2, seed=seed
                )
                res["test"] = "double_adversarial"
                all_results.append(res)
                intra_label = f"{intra_cos}" if intra_cos else "rand"
                print(f"{inter_corr:>8.1f} {intra_label:>10} {seed:>6} "
                      f"{res['mean_output_dev_pct']:>13.4f} "
                      f"{res['amplification_ratio']:>10.4f}")

    # ================================================================
    # TEST 5: Activation comparison under correlation
    # Does activation masking still help when errors are correlated?
    # ================================================================
    print("\n" + "=" * 78)
    print("  TEST 5: Activation comparison under max correlation (L=24, d=64)")
    print("=" * 78)

    L = 24
    activations = ["linear", "relu", "gelu"]

    print(f"\n{'Act':>8} {'Corr':>6} {'Seed':>6} {'MeanOutDev%':>13} {'AmpRatio':>10}")
    print("-" * 55)

    for act in activations:
        for corr in [0.0, 1.0]:
            for seed in seeds:
                res = run_correlated_removal(
                    N, L, d, r, inter_layer_corr=corr,
                    intra_layer_cos=None, act=act,
                    n_inputs=n_inputs, remove_idx=N // 2, seed=seed
                )
                res["test"] = "act_under_corr"
                all_results.append(res)
                print(f"{act:>8} {corr:>6.1f} {seed:>6} "
                      f"{res['mean_output_dev_pct']:>13.4f} "
                      f"{res['amplification_ratio']:>10.4f}")

    # ================================================================
    # AGGREGATE ANALYSIS
    # ================================================================
    print("\n" + "=" * 78)
    print("  AGGREGATE ANALYSIS")
    print("=" * 78)

    # Test 1 analysis: correlation sweep
    sweep = [r for r in all_results if r["test"] == "corr_sweep"]

    print("\n  Correlation Sweep Summary (L=24, d=64):")
    print(f"  {'Corr':>6} {'MeanOutDev%':>13} {'StdOutDev':>11} "
          f"{'MeanAmpRatio':>14} {'MaxOutDev%':>12}")
    print("  " + "-" * 65)

    corr_amp_ratios = {}  # corr -> list of amp_ratios
    for corr in corr_values:
        corr_results = [r for r in sweep if r["inter_layer_corr_target"] == corr]
        devs = [r["mean_output_dev_pct"] for r in corr_results]
        amps = [r["amplification_ratio"] for r in corr_results]
        maxdevs = [r["max_output_dev_pct"] for r in corr_results]
        corr_amp_ratios[corr] = amps
        print(f"  {corr:>6.1f} {np.mean(devs):>13.4f} {np.std(devs):>11.4f} "
              f"{np.mean(amps):>14.4f} {np.max(maxdevs):>12.4f}")

    # Key comparison: corr=1 vs corr=0
    amp_indep = np.mean(corr_amp_ratios.get(0.0, [0]))
    amp_corr = np.mean(corr_amp_ratios.get(1.0, [0]))
    if amp_indep > 1e-12:
        amplification_multiplier = amp_corr / amp_indep
    else:
        amplification_multiplier = float('inf')

    dev_indep = np.mean([r["mean_output_dev_pct"] for r in sweep
                         if r["inter_layer_corr_target"] == 0.0])
    dev_corr = np.mean([r["mean_output_dev_pct"] for r in sweep
                        if r["inter_layer_corr_target"] == 1.0])
    if dev_indep > 1e-12:
        dev_multiplier = dev_corr / dev_indep
    else:
        dev_multiplier = float('inf')

    print(f"\n  Correlation amplification:")
    print(f"    Independent (corr=0): amp_ratio = {amp_indep:.4f}, "
          f"mean_dev = {dev_indep:.4f}%")
    print(f"    Correlated (corr=1):  amp_ratio = {amp_corr:.4f}, "
          f"mean_dev = {dev_corr:.4f}%")
    print(f"    Amp ratio multiplier (corr/indep): {amplification_multiplier:.2f}x")
    print(f"    Output dev multiplier (corr/indep): {dev_multiplier:.2f}x")

    # Test 2 analysis: correlation x depth
    corr_x_depth = [r for r in all_results if r["test"] == "corr_x_depth"]

    print("\n  Correlation x Depth Summary:")
    print(f"  {'L':>4} {'Corr':>6} {'MeanAmpRatio':>14} {'MeanDev%':>10}")
    print("  " + "-" * 40)

    for L in L_values:
        for corr in [0.0, 0.5, 1.0]:
            res_subset = [r for r in corr_x_depth
                          if r["L"] == L and r["inter_layer_corr_target"] == corr]
            if res_subset:
                ma = np.mean([r["amplification_ratio"] for r in res_subset])
                md = np.mean([r["mean_output_dev_pct"] for r in res_subset])
                print(f"  {L:>4} {corr:>6.1f} {ma:>14.4f} {md:>10.4f}")

    # Test 3 analysis: correlation x dimension
    corr_x_dim = [r for r in all_results if r["test"] == "corr_x_dim"]

    print("\n  Dimension Scaling Under Correlation:")
    print(f"  {'d':>5} {'Corr':>6} {'MeanDev%':>10} {'MeanAmpRatio':>14} {'Dev_Ratio':>11}")
    print("  " + "-" * 55)

    for d_val in d_values:
        devs_0 = [r["mean_output_dev_pct"] for r in corr_x_dim
                  if r["d"] == d_val and r["inter_layer_corr_target"] == 0.0]
        devs_1 = [r["mean_output_dev_pct"] for r in corr_x_dim
                  if r["d"] == d_val and r["inter_layer_corr_target"] == 1.0]
        amps_0 = [r["amplification_ratio"] for r in corr_x_dim
                  if r["d"] == d_val and r["inter_layer_corr_target"] == 0.0]
        amps_1 = [r["amplification_ratio"] for r in corr_x_dim
                  if r["d"] == d_val and r["inter_layer_corr_target"] == 1.0]

        if devs_0 and devs_1:
            ratio = np.mean(devs_1) / (np.mean(devs_0) + 1e-12)
            print(f"  {d_val:>5} {'0.0':>6} {np.mean(devs_0):>10.4f} "
                  f"{np.mean(amps_0):>14.4f} {'(baseline)':>11}")
            print(f"  {d_val:>5} {'1.0':>6} {np.mean(devs_1):>10.4f} "
                  f"{np.mean(amps_1):>14.4f} {ratio:>11.2f}x")

    # Linear regression: does correlation predict amplification?
    all_corrs = [r["inter_layer_corr_target"] for r in sweep]
    all_amps = [r["amplification_ratio"] for r in sweep]
    if len(all_corrs) > 2:
        slope, intercept, r_val, p_val, se = stats.linregress(all_corrs, all_amps)
        print(f"\n  Regression: amp_ratio = {slope:.4f} * corr + {intercept:.4f}")
        print(f"    R^2 = {r_val**2:.4f}, p = {p_val:.6f}")

    # ================================================================
    # KILL CRITERIA ASSESSMENT
    # ================================================================
    print("\n" + "=" * 78)
    print("  KILL CRITERIA ASSESSMENT")
    print("=" * 78)

    # K1: correlated errors amplify >2x vs independent
    print(f"\n  K1: Correlated amp / Independent amp > 2x?")
    print(f"    Amp ratio multiplier: {amplification_multiplier:.2f}x")
    print(f"    Output dev multiplier: {dev_multiplier:.2f}x")

    k1_amp = amplification_multiplier > 2.0
    k1_dev = dev_multiplier > 2.0

    if k1_amp or k1_dev:
        print(f"    K1 TRIGGERED: sub-additivity breaks under correlation")
        # Find the correlation threshold where amp starts to exceed 2x baseline
        print(f"\n    Finding correlation threshold for 2x amplification:")
        for corr in corr_values:
            corr_results = [r for r in sweep if r["inter_layer_corr_target"] == corr]
            dev_at_corr = np.mean([r["mean_output_dev_pct"] for r in corr_results])
            ratio_at_corr = dev_at_corr / (dev_indep + 1e-12)
            marker = " <-- 2x threshold" if 1.8 <= ratio_at_corr <= 2.2 else ""
            exceeded = " ** EXCEEDED **" if ratio_at_corr > 2.0 else ""
            print(f"      corr={corr:.1f}: dev_ratio = {ratio_at_corr:.2f}x{marker}{exceeded}")
    else:
        print(f"    K1 NOT triggered: correlation does not break sub-additivity by 2x")

    # K2: amp_ratio still < 1.0 even with max correlation?
    amp_corr_max = np.max([r["amplification_ratio"] for r in sweep
                           if r["inter_layer_corr_target"] == 1.0])
    print(f"\n  K2: amp_ratio < 1.0 even at max correlation?")
    print(f"    Max amp_ratio at corr=1.0: {amp_corr_max:.4f}")

    if amp_corr_max < 1.0:
        print(f"    K2 TRIGGERED: sub-additivity ROBUST even under max correlation")
        print(f"    Mechanisms 1 (activation masking) + 3 (spectral contraction)")
        print(f"    are sufficient alone; direction randomization is bonus, not required")
    else:
        print(f"    K2 NOT triggered: max correlation pushes amp_ratio above 1.0")

    # Overall verdict
    print(f"\n  OVERALL VERDICT:")
    if not k1_amp and not k1_dev and amp_corr_max < 1.0:
        print(f"    Sub-additivity is ROBUST to correlation.")
        print(f"    No cosine bound on inter-layer correlation needed.")
        print(f"    K1: not triggered, K2: triggered (robust)")
    elif k1_amp or k1_dev:
        print(f"    Sub-additivity WEAKENED by correlation.")
        if amp_corr_max < 1.0:
            print(f"    BUT amp_ratio still < 1.0, so errors still dampen.")
            print(f"    K1: triggered ({amplification_multiplier:.1f}x), "
                  f"K2: triggered (still sub-additive)")
            print(f"    Practical impact: moderate. Correlated experts need monitoring")
            print(f"    but the system remains safe.")
        else:
            print(f"    amp_ratio > 1.0 at max correlation: errors AMPLIFY.")
            print(f"    K1: triggered, K2: not triggered")
            print(f"    Practical impact: HIGH. Need inter-layer correlation bound.")
    else:
        print(f"    K1: not triggered, K2: not triggered")
        print(f"    Intermediate result -- needs further analysis.")

    # ================================================================
    # Save results
    # ================================================================
    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")

    return all_results


if __name__ == "__main__":
    run_full_experiment()
