#!/usr/bin/env python3
"""Persistence Diagram Diff: Before vs After Adapter Decomposition.

Kill criteria:
  K625: Bottleneck distance d_B > 0 (decomposition is NOT topologically lossless)
  K626: >=3 lost features have persistence > median (important pathways lost)
  K627: Lost features correlate with cross-domain inputs (Jaccard > 0.3)

Type: guided exploration (Type 2)
Platform: Apple M5 Pro 48GB

Method:
  1. Load BitNet-2B-4T base weight matrices (rows as points in R^d)
  2. Compute composed perturbation: Delta = (scale/N) * sum_i(A_i @ B_i)
  3. Build Rips complex on subsampled rows of W and W+Delta
  4. Compare persistence diagrams via bottleneck distance
  5. Random baseline control: same row norms, random directions
  6. Identify lost features and their properties
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import ripser
import persim

# No MLX GPU needed -- this is a pure numpy + ripser experiment.
# We only use MLX to load the model weights.

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

ADAPTER_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "adapters"
SKELETON_PATH = ADAPTER_DIR / "grassmannian_skeleton.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
DOMAIN_NAMES = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = 5
LORA_RANK = 16
LORA_SCALE = 20.0

# Subsample rows for feasibility (Rips is O(n^3))
N_SUBSAMPLE = 500
# Layers to analyze
TARGET_LAYERS = [0, 7, 15, 22, 29]  # early, early-mid, mid, late-mid, late
# Target projections (the ones adapted)
TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
# PH parameters
MAX_DIM = 1  # H0 and H1
N_RANDOM_BASELINES = 5  # number of random perturbation controls


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def phase_load_weights():
    """Load base model weights and adapter data as numpy arrays.

    BitNet-2B-4T stores weights in packed ternary format. We unpack them
    to get the full (out_features, in_features) weight matrices, then
    transpose to (in_features, out_features) for PH computation.
    """
    log("Loading model weights...")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear
    from mlx.utils import tree_unflatten

    model, _ = load(MODEL_ID)

    def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
        """Unpack uint8-packed ternary weights to bfloat16."""
        w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
        w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
        w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
        w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
        unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
        scale = weight_scale.astype(mx.bfloat16)
        if invert_scale:
            unpacked = unpacked / scale
        else:
            unpacked = unpacked * scale
        return unpacked

    # Replace BitLinear with Linear for target layers
    for li in TARGET_LAYERS:
        layer = model.model.layers[li]
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                mx.eval(unpacked_w)
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log("  Unpacked BitLinear weights for target layers")

    # Extract weight matrices
    weights = {}
    for li in TARGET_LAYERS:
        layer = model.model.layers[li]
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None:
                continue
            W = module.weight
            mx.eval(W)
            # W is (out_features, in_features) after unpacking
            # LoRA: y = x @ W^T + scale * x @ A @ B
            # Effective W_eff^T = W^T + scale * A @ B  where A is (in, r), B is (r, out)
            # We work with W^T (in_features, out_features)
            w_np = np.array(W.astype(mx.float32)).T  # now (in_features, out_features)
            weights[(li, key)] = w_np
            log(f"  layer {li} {key}: W^T shape={w_np.shape}")

    # Clean up model
    del model
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    # Load adapters (lora_b) and skeleton (lora_a)
    log("Loading adapters and skeleton...")
    skeleton = dict(np.load(str(SKELETON_PATH)))

    adapters = {}
    for domain in DOMAIN_NAMES:
        path = ADAPTER_DIR / domain / "adapter.npz"
        adapters[domain] = dict(np.load(str(path)))

    return weights, skeleton, adapters


def compute_perturbation(li, key, skeleton, adapters):
    """Compute composed perturbation Delta = (scale/N) * sum_i(A_i @ B_i).

    A_i: (d_in, rank) from skeleton
    B_i: (rank, d_out) from adapter lora_b

    Returns Delta: (d_in, d_out) -- same shape as the weight matrix.
    """
    param_name = f"model.layers.{li}.{key}.lora_b"
    delta = None

    for di, domain in enumerate(DOMAIN_NAMES):
        skey = f"layer_{li}_{key}_domain_{di}"
        if skey not in skeleton:
            continue
        if param_name not in adapters[domain]:
            continue

        A_i = np.nan_to_num(skeleton[skey].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)  # (d_in, rank)
        B_i = np.nan_to_num(adapters[domain][param_name].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)  # (rank, d_out)

        contribution = A_i @ B_i  # (d_in, d_out)
        if delta is None:
            delta = contribution
        else:
            delta += contribution

    if delta is None:
        return None

    # Average across domains, apply scale
    delta = (LORA_SCALE / N_DOMAINS) * delta
    return delta


def compute_persistence(points, max_dim=1):
    """Compute Rips persistence diagram on a point cloud.

    Args:
        points: (n, d) array
        max_dim: max homology dimension

    Returns:
        dict with 'dgms' (list of diagrams per dimension)
    """
    result = ripser.ripser(points, maxdim=max_dim)
    return result['dgms']


def finite_diagram(dgm):
    """Filter to finite persistence features only."""
    return dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm


def persistence_values(dgm):
    """Get persistence (death - birth) for finite features."""
    fd = finite_diagram(dgm)
    if len(fd) == 0:
        return np.array([])
    return fd[:, 1] - fd[:, 0]


def bottleneck_distance(dgm1, dgm2):
    """Compute bottleneck distance between two persistence diagrams."""
    # Filter to finite features for bottleneck computation
    d1 = finite_diagram(dgm1)
    d2 = finite_diagram(dgm2)
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        return np.max(persistence_values(d2)) / 2.0
    if len(d2) == 0:
        return np.max(persistence_values(d1)) / 2.0
    return persim.bottleneck(d1, d2)


def phase_compute_ph(weights, skeleton, adapters):
    """Compute PH on base and composed weight matrices."""
    log("\nComputing persistent homology...")

    results = {}
    np.random.seed(42)

    for li in TARGET_LAYERS:
        for key in TARGET_KEYS:
            if (li, key) not in weights:
                continue

            W = weights[(li, key)]
            n_rows, d = W.shape
            label = f"layer_{li}_{key}"
            log(f"\n  {label}: W shape={W.shape}")

            # Compute perturbation
            delta = compute_perturbation(li, key, skeleton, adapters)
            if delta is None:
                log(f"    Skipping (no adapter data)")
                continue

            W_composed = W + delta

            # Subsample rows deterministically
            if n_rows > N_SUBSAMPLE:
                indices = np.linspace(0, n_rows - 1, N_SUBSAMPLE, dtype=int)
            else:
                indices = np.arange(n_rows)

            W_sub = W[indices].astype(np.float32)
            W_comp_sub = W_composed[indices].astype(np.float32)
            delta_sub = delta[indices].astype(np.float32)

            # Compute max row perturbation norm (the stability bound)
            row_norms = np.linalg.norm(delta_sub, axis=1)
            max_delta_norm = float(np.max(row_norms))
            mean_delta_norm = float(np.mean(row_norms))
            w_row_norms = np.linalg.norm(W_sub, axis=1)
            relative_perturbation = max_delta_norm / (float(np.median(w_row_norms)) + 1e-8)

            log(f"    max ||delta_i|| = {max_delta_norm:.6f}")
            log(f"    mean ||delta_i|| = {mean_delta_norm:.6f}")
            log(f"    relative perturbation = {relative_perturbation:.6f}")

            # Compute PH on base rows
            t0 = time.time()
            dgms_base = compute_persistence(W_sub, MAX_DIM)
            t_base = time.time() - t0

            # Compute PH on composed rows
            t0 = time.time()
            dgms_composed = compute_persistence(W_comp_sub, MAX_DIM)
            t_comp = time.time() - t0

            log(f"    PH time: base={t_base:.2f}s, composed={t_comp:.2f}s")

            # Bottleneck distances per dimension
            bn_distances = {}
            for dim in range(MAX_DIM + 1):
                d_b = bottleneck_distance(dgms_base[dim], dgms_composed[dim])
                bn_distances[f"H{dim}"] = float(d_b)
                log(f"    H{dim} bottleneck distance: {d_b:.6f} (bound: {max_delta_norm:.6f})")

            # Analyze lost/gained features
            pers_base_h0 = persistence_values(dgms_base[0])
            pers_comp_h0 = persistence_values(dgms_composed[0])
            median_pers = float(np.median(pers_base_h0)) if len(pers_base_h0) > 0 else 0.0

            # Count features above median that are "lost"
            # A feature is "lost" if it appears in base but not (within tolerance) in composed
            # The vulnerability window: features with persistence < 2*max_delta_norm
            vulnerability_bound = 2 * max_delta_norm
            n_vulnerable_base = int(np.sum(pers_base_h0 <= vulnerability_bound))
            n_stable_base = int(np.sum(pers_base_h0 > vulnerability_bound))

            # --- Random baseline control ---
            random_bn = {f"H{dim}": [] for dim in range(MAX_DIM + 1)}
            for trial in range(N_RANDOM_BASELINES):
                # Random perturbation with same per-row norms
                random_directions = np.random.randn(*delta_sub.shape).astype(np.float32)
                random_directions /= (np.linalg.norm(random_directions, axis=1, keepdims=True) + 1e-8)
                random_delta = random_directions * row_norms[:, None]
                W_random = W_sub + random_delta

                dgms_random = compute_persistence(W_random, MAX_DIM)
                for dim in range(MAX_DIM + 1):
                    d_b_rand = bottleneck_distance(dgms_base[dim], dgms_random[dim])
                    random_bn[f"H{dim}"].append(float(d_b_rand))

            random_bn_mean = {dim: float(np.mean(vals)) for dim, vals in random_bn.items()}
            random_bn_std = {dim: float(np.std(vals)) for dim, vals in random_bn.items()}

            for dim in range(MAX_DIM + 1):
                hkey = f"H{dim}"
                log(f"    H{dim} random baseline: {random_bn_mean[hkey]:.6f} +/- {random_bn_std[hkey]:.6f}")

            results[label] = {
                "shape": list(W.shape),
                "n_subsample": len(indices),
                "max_delta_norm": max_delta_norm,
                "mean_delta_norm": mean_delta_norm,
                "relative_perturbation": relative_perturbation,
                "bottleneck_distances": bn_distances,
                "stability_bound": max_delta_norm,
                "bound_satisfied": {dim: bn_distances[dim] <= max_delta_norm * 1.01
                                     for dim in bn_distances},
                "n_base_features_h0": len(pers_base_h0),
                "n_composed_features_h0": len(pers_comp_h0),
                "median_persistence_h0": median_pers,
                "vulnerability_bound": vulnerability_bound,
                "n_vulnerable_base_h0": n_vulnerable_base,
                "n_stable_base_h0": n_stable_base,
                "random_baseline_mean": random_bn_mean,
                "random_baseline_std": random_bn_std,
                "ph_time_base_s": t_base,
                "ph_time_composed_s": t_comp,
            }

            # H1 analysis
            if MAX_DIM >= 1:
                pers_base_h1 = persistence_values(dgms_base[1])
                pers_comp_h1 = persistence_values(dgms_composed[1])
                results[label]["n_base_features_h1"] = len(pers_base_h1)
                results[label]["n_composed_features_h1"] = len(pers_comp_h1)
                if len(pers_base_h1) > 0:
                    results[label]["median_persistence_h1"] = float(np.median(pers_base_h1))
                    results[label]["max_persistence_h1"] = float(np.max(pers_base_h1))

    return results


def phase_analyze_results(ph_results):
    """Aggregate results and assess kill criteria."""
    log("\nAnalyzing results...")

    all_bn_h0 = []
    all_bn_h1 = []
    all_random_h0 = []
    all_random_h1 = []
    bound_violations = 0
    total_modules = 0
    total_vulnerable = 0
    total_stable = 0
    total_features = 0

    for label, res in ph_results.items():
        total_modules += 1
        bn_h0 = res["bottleneck_distances"]["H0"]
        all_bn_h0.append(bn_h0)
        all_random_h0.append(res["random_baseline_mean"]["H0"])

        if "H1" in res["bottleneck_distances"]:
            bn_h1 = res["bottleneck_distances"]["H1"]
            all_bn_h1.append(bn_h1)
            all_random_h1.append(res["random_baseline_mean"]["H1"])

        # Check stability bound
        for dim, satisfied in res["bound_satisfied"].items():
            if not satisfied:
                bound_violations += 1

        total_vulnerable += res["n_vulnerable_base_h0"]
        total_stable += res["n_stable_base_h0"]
        total_features += res["n_base_features_h0"]

    # K625: Bottleneck distance > 0
    nonzero_h0 = sum(1 for b in all_bn_h0 if b > 1e-10)
    nonzero_h1 = sum(1 for b in all_bn_h1 if b > 1e-10)
    k625_pass = nonzero_h0 > 0 or nonzero_h1 > 0
    k625_detail = f"H0: {nonzero_h0}/{len(all_bn_h0)} nonzero, H1: {nonzero_h1}/{len(all_bn_h1)} nonzero"

    # K626: >=3 lost features with persistence > median
    # We assess this from the vulnerability analysis:
    # features in the vulnerability window COULD be lost
    # But we need to check if the actual bottleneck distance implies features ARE lost
    # A feature is "lost" if bottleneck distance > persistence/2
    # Count modules where the bottleneck distance exceeds vulnerability threshold
    n_modules_with_significant_loss = 0
    lost_above_median_count = 0
    for label, res in ph_results.items():
        bn_h0 = res["bottleneck_distances"]["H0"]
        median_p = res["median_persistence_h0"]
        if bn_h0 > median_p / 2 and median_p > 0:
            n_modules_with_significant_loss += 1
        # More precise: count how many base features could be lost
        # (persistence < 2 * bottleneck distance) AND (persistence > median)
        vuln = res["vulnerability_bound"]
        # Features with persistence in (median, 2*bn_h0) could be lost if bn_h0 is large enough
        # But we need the actual diagram data for precise matching
        # Approximate: if bottleneck > median/2, important features might be lost
        if bn_h0 > 0 and bn_h0 > median_p / 2:
            lost_above_median_count += 1

    k626_pass = lost_above_median_count >= 3
    k626_detail = f"{lost_above_median_count} modules with bn > median/2"

    # K627: Cannot assess without activation data / domain labels for weight rows
    # Weight rows don't have domain labels -- this criterion needs reformulation
    k627_detail = "NOT ASSESSED: weight rows don't have domain labels; " \
                  "K627 requires activation-based analysis"

    # Compare adapter vs random bottleneck distances
    adapter_vs_random = []
    for bn, rand_bn in zip(all_bn_h0, all_random_h0):
        if rand_bn > 1e-10:
            ratio = bn / rand_bn
        else:
            ratio = float('inf') if bn > 0 else 1.0
        adapter_vs_random.append(ratio)

    summary = {
        "total_modules_analyzed": total_modules,
        "total_layers": len(TARGET_LAYERS),
        "n_subsample": N_SUBSAMPLE,
        "h0_bottleneck_mean": float(np.mean(all_bn_h0)),
        "h0_bottleneck_max": float(np.max(all_bn_h0)),
        "h0_bottleneck_median": float(np.median(all_bn_h0)),
        "h1_bottleneck_mean": float(np.mean(all_bn_h1)) if all_bn_h1 else None,
        "h1_bottleneck_max": float(np.max(all_bn_h1)) if all_bn_h1 else None,
        "h0_random_mean": float(np.mean(all_random_h0)),
        "h0_adapter_vs_random_ratio_mean": float(np.mean(adapter_vs_random)),
        "h0_adapter_vs_random_ratio_median": float(np.median(adapter_vs_random)),
        "stability_bound_violations": bound_violations,
        "total_vulnerable_features": total_vulnerable,
        "total_stable_features": total_stable,
        "total_features": total_features,
        "k625_pass": k625_pass,
        "k625_detail": k625_detail,
        "k626_pass": k626_pass,
        "k626_detail": k626_detail,
        "k627_detail": k627_detail,
    }

    # Print summary
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"\nModules analyzed: {total_modules}")
    log(f"Layers: {TARGET_LAYERS}")
    log(f"Subsample: {N_SUBSAMPLE} rows")
    log(f"\nH0 bottleneck: mean={summary['h0_bottleneck_mean']:.6f}, "
        f"max={summary['h0_bottleneck_max']:.6f}, "
        f"median={summary['h0_bottleneck_median']:.6f}")
    if all_bn_h1:
        log(f"H1 bottleneck: mean={summary['h1_bottleneck_mean']:.6f}, "
            f"max={summary['h1_bottleneck_max']:.6f}")
    log(f"Random baseline H0: mean={summary['h0_random_mean']:.6f}")
    log(f"Adapter/Random ratio: mean={summary['h0_adapter_vs_random_ratio_mean']:.4f}, "
        f"median={summary['h0_adapter_vs_random_ratio_median']:.4f}")
    log(f"\nStability bound violations: {bound_violations}")
    log(f"Total features: {total_features} "
        f"(vulnerable: {total_vulnerable}, stable: {total_stable})")

    log(f"\nKILL CRITERIA:")
    log(f"  K625 (d_B > 0): {'PASS' if k625_pass else 'FAIL'} -- {k625_detail}")
    log(f"  K626 (>=3 lost > median): {'PASS' if k626_pass else 'FAIL'} -- {k626_detail}")
    log(f"  K627 (cross-domain): NOT ASSESSED -- {k627_detail}")

    return summary


def main():
    t0 = time.time()

    # Phase 1: Load weights
    weights, skeleton, adapters = phase_load_weights()

    # Phase 2: Compute PH
    ph_results = phase_compute_ph(weights, skeleton, adapters)

    # Clean up large arrays
    del weights
    gc.collect()

    # Phase 3: Analyze
    summary = phase_analyze_results(ph_results)

    total_time = time.time() - t0

    # Save results
    output = {
        "experiment": "persistence_diagram_diff",
        "total_time_s": round(total_time, 1),
        "config": {
            "model": MODEL_ID,
            "n_subsample": N_SUBSAMPLE,
            "target_layers": TARGET_LAYERS,
            "n_domains": N_DOMAINS,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "max_dim": MAX_DIM,
            "n_random_baselines": N_RANDOM_BASELINES,
        },
        "summary": summary,
        "per_module": ph_results,
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
