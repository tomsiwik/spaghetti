"""Alpha Residual Scaling Ablation: How much of alpha=0.022 comes from 1/sqrt(L)?

The adversarial review of removal_safety_complete_bound identified that all micro
experiments used 1/sqrt(L) residual scaling, which is NOT standard in production
architectures (Qwen2.5, Llama use unscaled residual connections, i.e., scale=1.0).

This experiment runs the complete bound pipeline with BOTH:
  - scale = 1/sqrt(L) (the original, non-standard setting)
  - scale = 1.0 (the production Qwen/Llama setting)

We measure the amplification ratio alpha for each and assess:
  K1: alpha_unscaled < 10x * alpha_scaled (i.e., alpha_unscaled < 0.22)
  K2: D = sum_eps * alpha_unscaled < 5% at d=256, L=24, N=50

If K1 fails, the safety bound from the parent experiment is an artifact of the
non-standard scaling and does not transfer to production architectures.

Additionally, we test a third variant that matches Qwen/Llama more closely:
  - Pre-RMSNorm with scale=1.0 BUT with properly scaled weight initialization
    (W ~ N(0, 1/d) already used, so this is the default)

CPU only. numpy/scipy on Apple Silicon.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================================
# Core utilities (from parent experiment)
# ============================================================================

def generate_lora_expert_layer(d: int, r: int, rng: np.random.RandomState) -> dict:
    """Generate a single LoRA expert for one layer."""
    A = rng.randn(d, r) / np.sqrt(d)
    B = rng.randn(r, d) / np.sqrt(r)
    return {"A": A, "B": B, "dW": A @ B}


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


# ============================================================================
# Forward passes with configurable residual scaling
# ============================================================================

def forward_pre_rmsn(h: np.ndarray, base_weights: list[np.ndarray],
                     layer_deltas: list[np.ndarray],
                     residual_scale: float = None) -> np.ndarray:
    """Pre-RMSNorm transformer: h_{l+1} = h_l + scale * sigma((W_l + Delta_l) @ RMSNorm(h_l)).

    Args:
        residual_scale: If None, uses 1/sqrt(L) (original). If a float, uses that value.
                       Set to 1.0 for production Qwen/Llama behavior.
    """
    L = len(base_weights)
    if residual_scale is None:
        scale = 1.0 / np.sqrt(L) if L > 1 else 1.0
    else:
        scale = residual_scale
    for l in range(L):
        W = base_weights[l] + layer_deltas[l]
        h = h + scale * activation(W @ rms_norm(h))
    return h


def forward_feedforward(h: np.ndarray, base_weights: list[np.ndarray],
                        layer_deltas: list[np.ndarray]) -> np.ndarray:
    """Pure feedforward (no residual, no norm): h_{l+1} = sigma((W_l + Delta_l) @ h_l)."""
    for l in range(len(base_weights)):
        W = base_weights[l] + layer_deltas[l]
        h = activation(W @ h)
    return h


# ============================================================================
# Core experiment: measure amplification ratio for a given architecture
# ============================================================================

def run_single_config(d: int, r: int, L: int, N: int,
                      n_inputs: int, remove_idx: int, seed: int,
                      residual_scale: float = None,
                      arch_name: str = "pre_rmsn_scaled") -> dict:
    """Run the complete bound experiment for one configuration.

    Args:
        residual_scale: None = 1/sqrt(L), 1.0 = production, or any float.
        arch_name: label for this architecture variant.
    """
    rng = np.random.RandomState(seed)
    t0 = time.time()

    # Step 1: Generate N experts at L layers
    experts = []
    for i in range(N):
        layers = [generate_lora_expert_layer(d, r, rng) for _ in range(L)]
        experts.append(layers)

    # Generate base weights (scaled by 1/sqrt(d) for stability)
    base_weights = [rng.randn(d, d) / np.sqrt(d) for _ in range(L)]

    # Generate test inputs
    inputs = rng.randn(n_inputs, d) * 0.1

    # Step 2: GS merge all N experts per layer
    all_merged_deltas = []
    all_ortho_deltas = []
    per_layer_errors = []

    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        merged_flat, ortho_flat = gram_schmidt_merge(layer_deltas)
        all_merged_deltas.append(merged_flat.reshape(d, d))
        all_ortho_deltas.append(ortho_flat)

        # Compute naive vs GS-recompute error for this layer
        naive_flat = merged_flat - ortho_flat[remove_idx]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)

        diff_norm = np.linalg.norm(naive_flat - gt_flat)
        gt_norm = np.linalg.norm(gt_flat)
        recon_err = float(diff_norm / (gt_norm + 1e-12)) * 100.0
        per_layer_errors.append(recon_err)

    sum_per_layer = sum(per_layer_errors)

    # Step 3: Forward pass with ALL experts
    forward_fn = lambda h, bw, ld: forward_pre_rmsn(h, bw, ld, residual_scale=residual_scale)

    outputs_all = np.array([forward_fn(inp, base_weights, all_merged_deltas)
                            for inp in inputs])

    # Step 4: Forward pass after naive removal
    naive_removed_deltas = []
    for l in range(L):
        merged = all_merged_deltas[l].flatten()
        removed = merged - all_ortho_deltas[l][remove_idx]
        naive_removed_deltas.append(removed.reshape(d, d))

    outputs_naive = np.array([forward_fn(inp, base_weights, naive_removed_deltas)
                              for inp in inputs])

    # Step 5: Forward pass with GS recompute (ground truth)
    gt_removed_deltas = []
    for l in range(L):
        layer_deltas = [experts[i][l]["dW"].flatten() for i in range(N)]
        remaining = [layer_deltas[i] for i in range(N) if i != remove_idx]
        gt_flat, _ = gram_schmidt_merge(remaining)
        gt_removed_deltas.append(gt_flat.reshape(d, d))

    outputs_gt = np.array([forward_fn(inp, base_weights, gt_removed_deltas)
                           for inp in inputs])

    # Step 6: Compute metrics
    naive_vs_gt_norms = np.linalg.norm(outputs_naive - outputs_gt, axis=1)
    gt_norms = np.linalg.norm(outputs_gt, axis=1)
    safe_gt = np.maximum(gt_norms, 1e-12)
    relative_devs = naive_vs_gt_norms / safe_gt * 100.0

    mean_dev = float(np.mean(relative_devs))
    max_dev = float(np.max(relative_devs))
    median_dev = float(np.median(relative_devs))
    std_dev = float(np.std(relative_devs))

    # Amplification ratio
    amp_ratio = mean_dev / sum_per_layer if sum_per_layer > 1e-12 else 0.0

    # Check for numerical issues (NaN/Inf in outputs)
    has_nan = bool(np.any(np.isnan(outputs_all)) or np.any(np.isnan(outputs_gt)))
    has_inf = bool(np.any(np.isinf(outputs_all)) or np.any(np.isinf(outputs_gt)))

    # Output magnitude (for diagnosing explosion)
    output_rms_all = float(np.sqrt(np.mean(outputs_all ** 2)))
    output_rms_gt = float(np.sqrt(np.mean(outputs_gt ** 2)))

    elapsed = time.time() - t0

    return {
        "arch_name": arch_name,
        "residual_scale": "1/sqrt(L)" if residual_scale is None else residual_scale,
        "effective_scale": 1.0 / np.sqrt(L) if residual_scale is None else residual_scale,
        "d": d, "r": r, "L": L, "N": N,
        "n_inputs": n_inputs, "remove_idx": remove_idx, "seed": seed,
        "sum_per_layer_error_pct": sum_per_layer,
        "mean_output_dev_pct": mean_dev,
        "max_output_dev_pct": max_dev,
        "median_output_dev_pct": median_dev,
        "std_output_dev_pct": std_dev,
        "amplification_ratio": amp_ratio,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "output_rms_all": output_rms_all,
        "output_rms_gt": output_rms_gt,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Scale sweep: run both variants across multiple dimensions
# ============================================================================

def run_scale_sweep(seeds: list[int]) -> dict:
    """Run the experiment at multiple scales for both scaled and unscaled variants."""

    configs = [
        {"d": 64, "r": 8, "L": 24, "N": 8, "n_inputs": 300, "label": "d64_N8"},
        {"d": 64, "r": 8, "L": 24, "N": 50, "n_inputs": 300, "label": "d64_N50"},
        {"d": 128, "r": 8, "L": 24, "N": 50, "n_inputs": 200, "label": "d128_N50"},
        {"d": 256, "r": 8, "L": 24, "N": 50, "n_inputs": 200, "label": "d256_N50"},
    ]

    variants = [
        {"residual_scale": None, "arch_name": "pre_rmsn_scaled",
         "desc": "Pre-RMSNorm + 1/sqrt(L) (original)"},
        {"residual_scale": 1.0, "arch_name": "pre_rmsn_unscaled",
         "desc": "Pre-RMSNorm + scale=1.0 (production Qwen/Llama)"},
    ]

    all_results = []

    for variant in variants:
        print(f"\n{'=' * 80}")
        print(f"  VARIANT: {variant['desc']}")
        print(f"{'=' * 80}")

        for cfg in configs:
            print(f"\n  Config: {cfg['label']} (d={cfg['d']}, N={cfg['N']}, L={cfg['L']})")
            print("  " + "-" * 60)

            for seed in seeds:
                res = run_single_config(
                    d=cfg["d"], r=cfg["r"], L=cfg["L"], N=cfg["N"],
                    n_inputs=cfg["n_inputs"],
                    remove_idx=cfg["N"] // 2,
                    seed=seed,
                    residual_scale=variant["residual_scale"],
                    arch_name=variant["arch_name"],
                )
                res["config_label"] = cfg["label"]
                all_results.append(res)

                status = ""
                if res["has_nan"]:
                    status = " [NaN!]"
                elif res["has_inf"]:
                    status = " [Inf!]"

                print(f"    seed={seed}: dev={res['mean_output_dev_pct']:.4f}%, "
                      f"amp={res['amplification_ratio']:.4f}, "
                      f"rms={res['output_rms_all']:.2e}{status}")

    return all_results


# ============================================================================
# Depth sweep: how does alpha vary with L for both variants?
# ============================================================================

def run_depth_sweep(seeds: list[int]) -> list[dict]:
    """Sweep L from 4 to 48 at d=64, N=8 for both variants."""
    depths = [4, 8, 12, 16, 24, 32, 48]
    d, r, N, n_inputs = 64, 8, 8, 300

    variants = [
        {"residual_scale": None, "arch_name": "scaled"},
        {"residual_scale": 1.0, "arch_name": "unscaled"},
    ]

    results = []
    for variant in variants:
        print(f"\n  Depth sweep ({variant['arch_name']}):")
        for L in depths:
            seed_amps = []
            seed_devs = []
            for seed in seeds:
                res = run_single_config(
                    d=d, r=r, L=L, N=N, n_inputs=n_inputs,
                    remove_idx=N // 2, seed=seed,
                    residual_scale=variant["residual_scale"],
                    arch_name=variant["arch_name"],
                )
                seed_amps.append(res["amplification_ratio"])
                seed_devs.append(res["mean_output_dev_pct"])
                results.append(res)

            mean_amp = np.mean(seed_amps)
            mean_dev = np.mean(seed_devs)
            status = ""
            if any(r_item.get("has_nan", False) for r_item in results[-len(seeds):]):
                status = " [NaN]"
            print(f"    L={L:3d}: amp={mean_amp:.4f}, dev={mean_dev:.4f}%{status}")

    return results


# ============================================================================
# Main experiment
# ============================================================================

def run_full_experiment():
    t_start = time.time()

    print("=" * 80)
    print("  EXPERIMENT: Alpha Residual Scaling Ablation")
    print("  Does 1/sqrt(L) artificially suppress amplification ratio?")
    print("  K1: alpha_unscaled < 10x * alpha_scaled (< 0.22)")
    print("  K2: D = sum_eps * alpha_unscaled < 5% at d=256, L=24, N=50")
    print("=" * 80)

    seeds = [42, 123, 777]

    # ================================================================
    # TEST 1: Scale sweep (both variants)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 1: Scale sweep — scaled vs unscaled across d and N")
    print("=" * 80)

    scale_results = run_scale_sweep(seeds)

    # ================================================================
    # TEST 2: Depth sweep (alpha vs L)
    # ================================================================
    print("\n" + "=" * 80)
    print("  TEST 2: Depth sweep — alpha vs L for both variants")
    print("=" * 80)

    depth_results = run_depth_sweep(seeds)

    # ================================================================
    # ANALYSIS: Compare variants
    # ================================================================
    print("\n" + "=" * 80)
    print("  ANALYSIS: Scaled vs Unscaled Comparison")
    print("=" * 80)

    # Group scale results by (config_label, arch_name)
    grouped = {}
    for r_item in scale_results:
        key = (r_item["config_label"], r_item["arch_name"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r_item)

    # Print comparison table
    print(f"\n  {'Config':>12} {'Variant':>20} {'MeanDev%':>10} {'AmpRatio':>10} "
          f"{'SumEps%':>10} {'OutputRMS':>12}")
    print("  " + "-" * 80)

    config_labels = ["d64_N8", "d64_N50", "d128_N50", "d256_N50"]
    arch_names = ["pre_rmsn_scaled", "pre_rmsn_unscaled"]

    comparison = {}
    for label in config_labels:
        for arch in arch_names:
            key = (label, arch)
            if key not in grouped:
                continue
            runs = grouped[key]
            devs = [r_item["mean_output_dev_pct"] for r_item in runs]
            amps = [r_item["amplification_ratio"] for r_item in runs]
            sum_eps = [r_item["sum_per_layer_error_pct"] for r_item in runs]
            rms_vals = [r_item["output_rms_all"] for r_item in runs]

            mean_dev = np.mean(devs)
            mean_amp = np.mean(amps)
            mean_sum_eps = np.mean(sum_eps)
            mean_rms = np.mean(rms_vals)
            has_nan = any(r_item["has_nan"] for r_item in runs)

            comparison[key] = {
                "mean_dev": mean_dev,
                "mean_amp": mean_amp,
                "mean_sum_eps": mean_sum_eps,
                "mean_rms": mean_rms,
                "has_nan": has_nan,
                "d": runs[0]["d"],
                "N": runs[0]["N"],
            }

            nan_flag = " NaN!" if has_nan else ""
            short_arch = "1/sqrt(L)" if "scaled" in arch and "un" not in arch else "scale=1.0"
            print(f"  {label:>12} {short_arch:>20} {mean_dev:>10.4f} {mean_amp:>10.4f} "
                  f"{mean_sum_eps:>10.2f} {mean_rms:>12.2e}{nan_flag}")

    # ================================================================
    # ANALYSIS: Alpha ratio (unscaled / scaled)
    # ================================================================
    print(f"\n  Alpha ratio (unscaled / scaled):")
    print(f"  {'Config':>12} {'alpha_scaled':>14} {'alpha_unscaled':>16} {'Ratio':>8} {'K1 (<10x)':>12}")
    print("  " + "-" * 70)

    alpha_ratios = []
    for label in config_labels:
        key_s = (label, "pre_rmsn_scaled")
        key_u = (label, "pre_rmsn_unscaled")
        if key_s not in comparison or key_u not in comparison:
            continue
        alpha_s = comparison[key_s]["mean_amp"]
        alpha_u = comparison[key_u]["mean_amp"]
        if comparison[key_u]["has_nan"]:
            ratio_str = "NaN"
            k1_str = "N/A"
        else:
            ratio = alpha_u / alpha_s if alpha_s > 1e-12 else float("inf")
            alpha_ratios.append(ratio)
            ratio_str = f"{ratio:.2f}x"
            k1_str = "PASS" if ratio < 10.0 else "FAIL"
        print(f"  {label:>12} {alpha_s:>14.4f} {alpha_u:>16.4f} {ratio_str:>8} {k1_str:>12}")

    # ================================================================
    # ANALYSIS: Combined bound at target scale (d=256, L=24, N=50)
    # ================================================================
    print(f"\n" + "=" * 80)
    print(f"  COMBINED BOUND: D = sum_eps * alpha at d=256, L=24, N=50")
    print(f"=" * 80)

    key_target_s = ("d256_N50", "pre_rmsn_scaled")
    key_target_u = ("d256_N50", "pre_rmsn_unscaled")

    if key_target_s in comparison and key_target_u in comparison:
        cs = comparison[key_target_s]
        cu = comparison[key_target_u]

        # Note: sum_eps is the same for both (weight-space error is scale-independent)
        # ... actually it IS the same because sum_eps is measured in weight space,
        # not forward-pass space. Let me verify by comparing.

        D_scaled = cs["mean_sum_eps"] * cs["mean_amp"]
        D_unscaled = cu["mean_sum_eps"] * cu["mean_amp"]

        print(f"\n  Scaled (1/sqrt(L)):")
        print(f"    sum_eps = {cs['mean_sum_eps']:.4f}%")
        print(f"    alpha   = {cs['mean_amp']:.4f}")
        print(f"    D_pred  = {D_scaled:.4f}%")
        print(f"    D_empir = {cs['mean_dev']:.4f}%")

        if not cu["has_nan"]:
            print(f"\n  Unscaled (production):")
            print(f"    sum_eps = {cu['mean_sum_eps']:.4f}%")
            print(f"    alpha   = {cu['mean_amp']:.4f}")
            print(f"    D_pred  = {D_unscaled:.4f}%")
            print(f"    D_empir = {cu['mean_dev']:.4f}%")

            print(f"\n  K2 Assessment: D_unscaled < 5%?")
            k2_pass = D_unscaled < 5.0 and cu["mean_dev"] < 5.0
            print(f"    D_pred  = {D_unscaled:.4f}% {'<' if D_unscaled < 5.0 else '>='} 5.0% "
                  f"-> {'PASS' if D_unscaled < 5.0 else 'FAIL'}")
            print(f"    D_empir = {cu['mean_dev']:.4f}% {'<' if cu['mean_dev'] < 5.0 else '>='} 5.0% "
                  f"-> {'PASS' if cu['mean_dev'] < 5.0 else 'FAIL'}")
        else:
            print(f"\n  Unscaled (production): NUMERICAL INSTABILITY (NaN/Inf)")
            print(f"    Forward pass diverges without 1/sqrt(L) scaling at L=24")
            k2_pass = False

    # ================================================================
    # ANALYSIS: Depth sweep comparison
    # ================================================================
    print(f"\n" + "=" * 80)
    print(f"  DEPTH SWEEP: Alpha vs L for both variants")
    print(f"=" * 80)

    depth_grouped = {}
    for r_item in depth_results:
        key = (r_item["L"], r_item["arch_name"])
        if key not in depth_grouped:
            depth_grouped[key] = []
        depth_grouped[key].append(r_item)

    depths_tested = sorted(set(r_item["L"] for r_item in depth_results))
    print(f"\n  {'L':>4} {'alpha_scaled':>14} {'alpha_unscaled':>16} {'ratio':>8} {'dev_scaled%':>12} {'dev_unscaled%':>14}")
    print("  " + "-" * 72)

    depth_comparison = []
    for L_val in depths_tested:
        key_s = (L_val, "scaled")
        key_u = (L_val, "unscaled")
        if key_s in depth_grouped and key_u in depth_grouped:
            amps_s = [r_item["amplification_ratio"] for r_item in depth_grouped[key_s]]
            amps_u = [r_item["amplification_ratio"] for r_item in depth_grouped[key_u]]
            devs_s = [r_item["mean_output_dev_pct"] for r_item in depth_grouped[key_s]]
            devs_u = [r_item["mean_output_dev_pct"] for r_item in depth_grouped[key_u]]
            has_nan_u = any(r_item["has_nan"] for r_item in depth_grouped[key_u])

            mean_amp_s = np.mean(amps_s)
            mean_amp_u = np.mean(amps_u)
            mean_dev_s = np.mean(devs_s)
            mean_dev_u = np.mean(devs_u)

            if has_nan_u:
                print(f"  {L_val:>4} {mean_amp_s:>14.4f} {'NaN':>16} {'N/A':>8} "
                      f"{mean_dev_s:>12.4f} {'NaN':>14}")
            else:
                ratio = mean_amp_u / mean_amp_s if mean_amp_s > 1e-12 else float("inf")
                depth_comparison.append({"L": L_val, "ratio": ratio,
                                        "alpha_s": mean_amp_s, "alpha_u": mean_amp_u})
                print(f"  {L_val:>4} {mean_amp_s:>14.4f} {mean_amp_u:>16.4f} {ratio:>8.2f}x "
                      f"{mean_dev_s:>12.4f} {mean_dev_u:>14.4f}")

    # ================================================================
    # ANALYSIS: What does 1/sqrt(L) actually contribute?
    # ================================================================
    print(f"\n" + "=" * 80)
    print(f"  DECOMPOSITION: Sources of dampening in Pre-RMSNorm")
    print(f"=" * 80)

    # The feedforward baseline (no residual, no norm) gives amp=0.25 at L=24
    # Pre-RMSNorm with 1/sqrt(L) gives amp=0.022
    # Pre-RMSNorm without 1/sqrt(L) gives amp=??? (this experiment)
    #
    # Total dampening = feedforward_amp / pre_rmsn_amp
    # 1/sqrt(L) contribution = alpha_unscaled / alpha_scaled
    # Architectural contribution (residual + RMSNorm) = feedforward_amp / alpha_unscaled

    ff_amp = 0.25  # from multilayer_removal_cascade at L=24

    if alpha_ratios:
        # Use the target config (d=256, N=50) if available, else average
        if key_target_u in comparison and not comparison[key_target_u]["has_nan"]:
            alpha_scaled = comparison[key_target_s]["mean_amp"]
            alpha_unscaled = comparison[key_target_u]["mean_amp"]
        else:
            # Use d=64 as fallback
            key_s_64 = ("d64_N8", "pre_rmsn_scaled")
            key_u_64 = ("d64_N8", "pre_rmsn_unscaled")
            alpha_scaled = comparison[key_s_64]["mean_amp"]
            alpha_unscaled = comparison[key_u_64]["mean_amp"]

        total_dampening = ff_amp / alpha_scaled
        sqrt_l_factor = alpha_unscaled / alpha_scaled
        arch_factor = ff_amp / alpha_unscaled

        print(f"\n  Feedforward baseline (L=24): alpha_ff = {ff_amp:.3f}")
        print(f"  Pre-RMSNorm + 1/sqrt(L):    alpha_s  = {alpha_scaled:.4f}")
        print(f"  Pre-RMSNorm (production):    alpha_u  = {alpha_unscaled:.4f}")
        print()
        print(f"  Total dampening vs FF:              {total_dampening:.1f}x")
        print(f"    = 1/sqrt(L) contribution:         {sqrt_l_factor:.2f}x")
        print(f"    x Arch contribution (resid+RMSNorm): {arch_factor:.2f}x")
        print(f"    (check: {sqrt_l_factor:.2f} x {arch_factor:.2f} = "
              f"{sqrt_l_factor * arch_factor:.1f}x vs {total_dampening:.1f}x)")

    # ================================================================
    # KILL CRITERIA
    # ================================================================
    print(f"\n" + "=" * 80)
    print(f"  KILL CRITERIA ASSESSMENT")
    print(f"=" * 80)

    # K1: alpha_unscaled < 10x * alpha_scaled
    if alpha_ratios:
        max_ratio = max(alpha_ratios)
        mean_ratio = np.mean(alpha_ratios)
        k1_pass = max_ratio < 10.0

        print(f"\n  K1: alpha_unscaled < 10x * alpha_scaled (i.e., < 0.22)")
        print(f"      Mean ratio across configs: {mean_ratio:.2f}x")
        print(f"      Max ratio across configs:  {max_ratio:.2f}x")
        print(f"      VERDICT: {'PASS' if k1_pass else 'FAIL'}")
        if not k1_pass:
            print(f"      The 1/sqrt(L) scaling accounts for >{max_ratio:.0f}x of the dampening.")
            print(f"      The alpha=0.022 result DOES NOT TRANSFER to production architectures.")
    else:
        k1_pass = False
        print(f"\n  K1: CANNOT ASSESS (no valid unscaled results)")

    # K2: D = sum_eps * alpha_unscaled < 5% at d=256, L=24, N=50
    if key_target_u in comparison and not comparison[key_target_u]["has_nan"]:
        cu = comparison[key_target_u]
        D_unscaled = cu["mean_sum_eps"] * cu["mean_amp"]
        k2_pass = cu["mean_dev"] < 5.0

        print(f"\n  K2: D = sum_eps * alpha < 5% at d=256, L=24, N=50 (unscaled)")
        print(f"      sum_eps = {cu['mean_sum_eps']:.4f}%")
        print(f"      alpha_unscaled = {cu['mean_amp']:.4f}")
        print(f"      D_predicted = {D_unscaled:.4f}%")
        print(f"      D_empirical = {cu['mean_dev']:.4f}%")
        print(f"      VERDICT: {'PASS' if k2_pass else 'FAIL'}")
    else:
        k2_pass = False
        print(f"\n  K2: CANNOT ASSESS (unscaled variant unstable at d=256, L=24, N=50)")
        # Check if smaller configs work
        for label in ["d64_N8", "d64_N50", "d128_N50"]:
            key_u = (label, "pre_rmsn_unscaled")
            if key_u in comparison and not comparison[key_u]["has_nan"]:
                cu = comparison[key_u]
                D = cu["mean_sum_eps"] * cu["mean_amp"]
                print(f"      {label}: D_empir={cu['mean_dev']:.4f}%, D_pred={D:.4f}%")

    overall = "PROVEN" if k1_pass and k2_pass else "SUPPORTED" if k1_pass or k2_pass else "KILLED"
    print(f"\n  OVERALL VERDICT: {overall}")

    if k1_pass:
        print(f"    The safety bound transfers to production architectures.")
        print(f"    Alpha_unscaled is {mean_ratio:.1f}x higher, but still within 10x of 0.022.")
    else:
        print(f"    WARNING: The 1/sqrt(L) scaling is a significant contributor to alpha=0.022.")
        print(f"    The production alpha is {mean_ratio:.1f}x higher.")
        if k2_pass:
            print(f"    However, the combined bound D still predicts <5%, so safety is maintained.")
        else:
            print(f"    The safety bound must be recalibrated for production architectures.")

    # ================================================================
    # Save results
    # ================================================================
    results_path = Path(__file__).parent / "results.json"
    save_data = {
        "scale_results": scale_results,
        "depth_results": depth_results,
        "comparison": {str(k): v for k, v in comparison.items()},
        "alpha_ratios": alpha_ratios,
        "depth_comparison": depth_comparison,
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
        "verdict": overall,
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total experiment time: {elapsed:.1f}s")

    return save_data


if __name__ == "__main__":
    run_full_experiment()
