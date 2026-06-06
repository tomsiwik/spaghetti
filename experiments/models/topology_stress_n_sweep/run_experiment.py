#!/usr/bin/env python3
"""Topology Stress Test: At what N does composition lose high-persistence features?

Kill criteria:
  K634: At least one N in {10,15,24,50} loses >=1 high-persistence feature (persistence > median)
  K635: d_B grows monotonically with N (Spearman rho > 0.8 across sweep points)

Type: guided exploration (Type 2)
Platform: Apple M5 Pro 48GB

Method:
  1. Load BitNet-2B-4T base weight matrices (subset of layers/projections)
  2. For each N in {5, 10, 15, 24, 50}:
     a. Compose N adapters (5 real + synthetic for N>5)
     b. Test both 1/N averaging and additive composition
     c. Compute PH on base and composed weight point clouds
     d. Measure bottleneck distance, lost features, perturbation norms
  3. Analyze scaling: does d_B grow monotonically with N?
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

ADAPTER_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "adapters"
SKELETON_PATH = ADAPTER_DIR / "grassmannian_skeleton.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
DOMAIN_NAMES = ["medical", "code", "math", "legal", "finance"]
LORA_RANK = 16
LORA_SCALE = 20.0

# Focused analysis: 2 layers (early + late), 3 key projections
TARGET_LAYERS = [0, 29]
TARGET_KEYS = ["self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj"]

# Sweep parameters
N_SWEEP = [5, 10, 15, 24, 50]
N_SUBSAMPLE = 500
MAX_DIM = 1
N_RANDOM_BASELINES = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def phase_load_weights():
    """Load base model weights and adapter data as numpy arrays."""
    log("Loading model weights...")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.bitlinear_layers import BitLinear
    from mlx.utils import tree_unflatten

    model, _ = load(MODEL_ID)

    def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
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
            w_np = np.array(W.astype(mx.float32)).T  # (in_features, out_features)
            weights[(li, key)] = w_np
            log(f"  layer {li} {key}: W^T shape={w_np.shape}")

    del model
    gc.collect()
    try:
        import mlx.core as mx2
        mx2.clear_cache()
    except Exception:
        pass

    log("Loading adapters and skeleton...")
    skeleton = dict(np.load(str(SKELETON_PATH)))
    adapters = {}
    for domain in DOMAIN_NAMES:
        path = ADAPTER_DIR / domain / "adapter.npz"
        adapters[domain] = dict(np.load(str(path)))

    return weights, skeleton, adapters


def generate_synthetic_adapters(real_adapters, skeleton, n_synthetic, rng):
    """Generate synthetic adapters matching real adapter statistics.

    Strategy: for each parameter key in the real adapters, compute the mean
    and std of the B-matrices across the 5 real domains, then sample new
    B-matrices from a Gaussian with matching statistics. This preserves the
    spectral profile approximately.
    """
    # Collect stats per key from real adapters
    key_stats = {}
    all_keys = set()
    for domain_data in real_adapters.values():
        all_keys.update(domain_data.keys())

    for param_key in all_keys:
        values = []
        for domain_data in real_adapters.values():
            if param_key in domain_data:
                values.append(domain_data[param_key])
        if not values:
            continue
        stacked = np.stack(values, axis=0)
        key_stats[param_key] = {
            "mean": np.mean(stacked, axis=0),
            "std": np.std(stacked, axis=0) + 1e-8,
            "shape": values[0].shape,
        }

    synthetic = []
    for i in range(n_synthetic):
        adapter = {}
        for param_key, stat in key_stats.items():
            # Sample from Gaussian matching per-element mean/std
            adapter[param_key] = rng.normal(
                loc=stat["mean"], scale=stat["std"]
            ).astype(np.float32)
        synthetic.append(adapter)

    # Also need synthetic skeleton entries for synthetic domains
    # Real skeleton keys: layer_{li}_{key}_domain_{di}
    skel_stats = {}
    for skey in skeleton:
        # Parse domain index from key
        parts = skey.rsplit("_domain_", 1)
        if len(parts) != 2:
            continue
        base_key = parts[0]
        if base_key not in skel_stats:
            skel_stats[base_key] = []
        skel_stats[base_key].append(skeleton[skey])

    synthetic_skeleton_entries = {}
    for base_key, real_values in skel_stats.items():
        stacked = np.stack(real_values, axis=0)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0) + 1e-8
        for i in range(n_synthetic):
            new_key = f"{base_key}_domain_{5 + i}"
            synthetic_skeleton_entries[new_key] = rng.normal(
                loc=mean, scale=std
            ).astype(np.float32)

    return synthetic, synthetic_skeleton_entries


def compute_perturbation_n(li, key, skeleton, all_adapters, all_domain_indices, n_domains,
                           scale, averaging=True):
    """Compute composed perturbation for N adapters.

    If averaging=True: Delta = (scale/N) * sum_i(A_i @ B_i)
    If averaging=False: Delta = scale * sum_i(A_i @ B_i)  (stress test)
    """
    param_name = f"model.layers.{li}.{key}.lora_b"
    delta = None
    n_active = 0

    for di, (adapter_data, skel_domain_idx) in enumerate(
        zip(all_adapters, all_domain_indices)
    ):
        skey = f"layer_{li}_{key}_domain_{skel_domain_idx}"
        if skey not in skeleton:
            continue
        if param_name not in adapter_data:
            continue

        A_i = np.nan_to_num(skeleton[skey].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        B_i = np.nan_to_num(adapter_data[param_name].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        contribution = A_i @ B_i
        if delta is None:
            delta = contribution.copy()
        else:
            delta += contribution
        n_active += 1

    if delta is None or n_active == 0:
        return None

    if averaging:
        delta = (scale / n_active) * delta
    else:
        delta = scale * delta

    return delta


def compute_persistence(points, max_dim=1):
    import ripser
    result = ripser.ripser(points, maxdim=max_dim)
    return result['dgms']


def finite_diagram(dgm):
    return dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm


def persistence_values(dgm):
    fd = finite_diagram(dgm)
    if len(fd) == 0:
        return np.array([])
    return fd[:, 1] - fd[:, 0]


def bottleneck_distance(dgm1, dgm2):
    import persim
    d1 = finite_diagram(dgm1)
    d2 = finite_diagram(dgm2)
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        return float(np.max(persistence_values(d2))) / 2.0
    if len(d2) == 0:
        return float(np.max(persistence_values(d1))) / 2.0
    return float(persim.bottleneck(d1, d2))


def count_lost_features(dgm_base, dgm_composed, threshold_persistence):
    """Count features in base with persistence > threshold that are 'lost'.

    A feature is considered lost if there is no matching feature in composed
    within reasonable tolerance. We use a greedy matching approach:
    for each base feature with high persistence, find the closest composed feature.
    If the closest is further than the feature's persistence / 2, it's lost.
    """
    pb = persistence_values(dgm_base)
    pc = persistence_values(dgm_composed)

    high_base = pb[pb > threshold_persistence]
    if len(high_base) == 0:
        return 0, 0

    if len(pc) == 0:
        return len(high_base), len(high_base)

    # Use bottleneck matching to determine actual lost count
    # A high-persistence feature is lost if it's matched to the diagonal
    # (i.e., its matched partner has persistence 0 or doesn't exist)
    # Approximation: compare sorted persistence values
    high_base_sorted = np.sort(high_base)[::-1]
    high_composed = pc[pc > threshold_persistence / 2]  # generous threshold
    high_composed_sorted = np.sort(high_composed)[::-1] if len(high_composed) > 0 else np.array([])

    # Count base features that have no approximate match in composed
    lost = 0
    matched_composed = set()
    for bp in high_base_sorted:
        found_match = False
        for ci, cp in enumerate(high_composed_sorted):
            if ci in matched_composed:
                continue
            if abs(bp - cp) < bp * 0.5:  # within 50% of base persistence
                matched_composed.add(ci)
                found_match = True
                break
        if not found_match:
            lost += 1

    return lost, len(high_base)


def phase_sweep(weights, skeleton, adapters):
    """Run the N-sweep experiment."""
    log("\nStarting N-sweep...")
    rng = np.random.default_rng(42)

    # Pre-generate synthetic adapters for max N
    max_synthetic = max(N_SWEEP) - 5
    log(f"Generating {max_synthetic} synthetic adapters...")
    synthetic_adapters, synthetic_skeleton = generate_synthetic_adapters(
        adapters, skeleton, max_synthetic, rng
    )
    log(f"  Generated {len(synthetic_adapters)} synthetic adapters")

    # Merge synthetic skeleton into main skeleton
    full_skeleton = dict(skeleton)
    full_skeleton.update(synthetic_skeleton)

    # Subsample indices (fixed across all N for fair comparison)
    subsample_indices = {}
    for (li, key), W in weights.items():
        n_rows = W.shape[0]
        if n_rows > N_SUBSAMPLE:
            subsample_indices[(li, key)] = np.linspace(0, n_rows - 1, N_SUBSAMPLE, dtype=int)
        else:
            subsample_indices[(li, key)] = np.arange(n_rows)

    # Compute base PH once (reuse across all N values)
    log("\nComputing base PH (reused across all N)...")
    base_ph = {}
    base_stats = {}
    for (li, key), W in weights.items():
        idx = subsample_indices[(li, key)]
        W_sub = W[idx].astype(np.float32)
        label = f"layer_{li}_{key}"

        t0 = time.time()
        dgms = compute_persistence(W_sub, MAX_DIM)
        t_ph = time.time() - t0

        pers_h0 = persistence_values(dgms[0])
        pers_h1 = persistence_values(dgms[1]) if MAX_DIM >= 1 else np.array([])

        base_ph[label] = dgms
        base_stats[label] = {
            "n_features_h0": len(pers_h0),
            "n_features_h1": len(pers_h1),
            "median_persistence_h0": float(np.median(pers_h0)) if len(pers_h0) > 0 else 0.0,
            "median_persistence_h1": float(np.median(pers_h1)) if len(pers_h1) > 0 else 0.0,
            "max_persistence_h0": float(np.max(pers_h0)) if len(pers_h0) > 0 else 0.0,
            "max_persistence_h1": float(np.max(pers_h1)) if len(pers_h1) > 0 else 0.0,
            "ph_time_s": t_ph,
        }
        log(f"  {label}: {len(pers_h0)} H0 features, {len(pers_h1)} H1 features, "
            f"median_pers_h0={base_stats[label]['median_persistence_h0']:.4f}, "
            f"time={t_ph:.1f}s")

    # Sweep N values
    sweep_results = {}
    for N in N_SWEEP:
        log(f"\n{'='*60}")
        log(f"N = {N}")
        log(f"{'='*60}")

        # Build adapter list: first 5 are real, rest are synthetic
        all_adapters_list = [adapters[d] for d in DOMAIN_NAMES]
        all_domain_indices = list(range(5))
        if N > 5:
            for i in range(N - 5):
                all_adapters_list.append(synthetic_adapters[i])
                all_domain_indices.append(5 + i)

        n_results = {}
        for (li, key), W in weights.items():
            label = f"layer_{li}_{key}"
            idx = subsample_indices[(li, key)]
            W_sub = W[idx].astype(np.float32)

            # Test both composition schemes
            for scheme_name, averaging in [("averaging", True), ("additive", False)]:
                delta = compute_perturbation_n(
                    li, key, full_skeleton, all_adapters_list, all_domain_indices,
                    N, LORA_SCALE, averaging=averaging
                )
                if delta is None:
                    log(f"  {label} ({scheme_name}): skipped (no adapter data)")
                    continue

                delta_sub = delta[idx].astype(np.float32)
                W_comp_sub = W_sub + delta_sub

                # Perturbation norms
                row_norms = np.linalg.norm(delta_sub, axis=1)
                max_delta = float(np.max(row_norms))
                mean_delta = float(np.mean(row_norms))

                # Compute PH on composed
                t0 = time.time()
                dgms_comp = compute_persistence(W_comp_sub, MAX_DIM)
                t_ph = time.time() - t0

                # Bottleneck distances
                bn = {}
                for dim in range(MAX_DIM + 1):
                    bn[f"H{dim}"] = bottleneck_distance(base_ph[label][dim], dgms_comp[dim])

                # Count lost high-persistence features
                median_h0 = base_stats[label]["median_persistence_h0"]
                median_h1 = base_stats[label]["median_persistence_h1"]

                lost_h0, total_high_h0 = count_lost_features(
                    base_ph[label][0], dgms_comp[0], median_h0
                )
                lost_h1, total_high_h1 = 0, 0
                if MAX_DIM >= 1:
                    lost_h1, total_high_h1 = count_lost_features(
                        base_ph[label][1], dgms_comp[1], median_h1
                    )

                # Vulnerability window analysis
                pers_base_h0 = persistence_values(base_ph[label][0])
                vuln_window = 2 * max_delta
                n_vulnerable = int(np.sum(pers_base_h0 <= vuln_window)) if len(pers_base_h0) > 0 else 0
                n_in_window = int(np.sum((pers_base_h0 > median_h0) & (pers_base_h0 <= vuln_window))) if len(pers_base_h0) > 0 else 0

                # Feature counts
                pers_comp_h0 = persistence_values(dgms_comp[0])
                pers_comp_h1 = persistence_values(dgms_comp[1]) if MAX_DIM >= 1 else np.array([])

                result_key = f"{label}_{scheme_name}"
                n_results[result_key] = {
                    "layer": li,
                    "key": key,
                    "scheme": scheme_name,
                    "n_adapters": N,
                    "max_delta_norm": max_delta,
                    "mean_delta_norm": mean_delta,
                    "bottleneck_h0": bn["H0"],
                    "bottleneck_h1": bn.get("H1", None),
                    "stability_bound_satisfied_h0": bn["H0"] <= max_delta * 1.01,
                    "lost_high_pers_h0": lost_h0,
                    "total_high_pers_h0": total_high_h0,
                    "lost_high_pers_h1": lost_h1,
                    "total_high_pers_h1": total_high_h1,
                    "vulnerability_window": vuln_window,
                    "n_vulnerable_features": n_vulnerable,
                    "n_high_in_vuln_window": n_in_window,
                    "n_features_h0_composed": len(pers_comp_h0),
                    "n_features_h1_composed": len(pers_comp_h1),
                    "ph_time_s": t_ph,
                }

                log(f"  {result_key}: max_delta={max_delta:.4f}, "
                    f"bn_H0={bn['H0']:.6f}, bn_H1={bn.get('H1', 0):.6f}, "
                    f"lost_h0={lost_h0}/{total_high_h0}, lost_h1={lost_h1}/{total_high_h1}, "
                    f"vuln_window={vuln_window:.4f}, time={t_ph:.1f}s")

        sweep_results[f"N={N}"] = n_results

    return sweep_results, base_stats


def phase_analyze(sweep_results, base_stats):
    """Analyze sweep results and assess kill criteria."""
    log("\n" + "=" * 60)
    log("ANALYSIS")
    log("=" * 60)

    # Aggregate by N for each scheme
    analysis = {"averaging": {}, "additive": {}}

    for n_key, n_results in sweep_results.items():
        N = int(n_key.split("=")[1])
        for result_key, res in n_results.items():
            scheme = res["scheme"]
            if N not in analysis[scheme]:
                analysis[scheme][N] = {
                    "max_deltas": [], "bn_h0": [], "bn_h1": [],
                    "lost_h0": 0, "total_high_h0": 0,
                    "lost_h1": 0, "total_high_h1": 0,
                    "n_high_in_vuln": 0,
                }
            a = analysis[scheme][N]
            a["max_deltas"].append(res["max_delta_norm"])
            a["bn_h0"].append(res["bottleneck_h0"])
            if res["bottleneck_h1"] is not None:
                a["bn_h1"].append(res["bottleneck_h1"])
            a["lost_h0"] += res["lost_high_pers_h0"]
            a["total_high_h0"] += res["total_high_pers_h0"]
            a["lost_h1"] += res["lost_high_pers_h1"]
            a["total_high_h1"] += res["total_high_pers_h1"]
            a["n_high_in_vuln"] += res["n_high_in_vuln_window"]

    # Print summary tables
    for scheme in ["averaging", "additive"]:
        log(f"\n--- Scheme: {scheme} ---")
        log(f"{'N':>4}  {'max_delta':>10}  {'mean_bn_h0':>12}  {'mean_bn_h1':>12}  "
            f"{'lost_h0':>8}  {'lost_h1':>8}  {'high_in_vuln':>12}")
        for N in sorted(analysis[scheme].keys()):
            a = analysis[scheme][N]
            md = float(np.mean(a["max_deltas"]))
            bh0 = float(np.mean(a["bn_h0"]))
            bh1 = float(np.mean(a["bn_h1"])) if a["bn_h1"] else 0.0
            log(f"{N:>4}  {md:>10.4f}  {bh0:>12.6f}  {bh1:>12.6f}  "
                f"{a['lost_h0']:>8}  {a['lost_h1']:>8}  {a['n_high_in_vuln']:>12}")

    # K634: At least one N in {10,15,24,50} loses >=1 high-persistence feature
    k634_pass = False
    k634_detail_parts = []
    for scheme in ["averaging", "additive"]:
        for N in [10, 15, 24, 50]:
            if N not in analysis[scheme]:
                continue
            a = analysis[scheme][N]
            total_lost = a["lost_h0"] + a["lost_h1"]
            if total_lost > 0:
                k634_pass = True
                k634_detail_parts.append(f"{scheme} N={N}: {a['lost_h0']} H0 + {a['lost_h1']} H1 lost")

    if not k634_pass:
        k634_detail = "No high-persistence features lost at any N in any scheme"
    else:
        k634_detail = "; ".join(k634_detail_parts)

    # K635: d_B grows monotonically with N (Spearman rho > 0.8)
    k635_results = {}
    for scheme in ["averaging", "additive"]:
        ns = sorted(analysis[scheme].keys())
        if len(ns) < 3:
            continue
        bn_h0_means = [float(np.mean(analysis[scheme][n]["bn_h0"])) for n in ns]
        if len(set(bn_h0_means)) > 1:
            rho, p = stats.spearmanr(ns, bn_h0_means)
        else:
            rho, p = 0.0, 1.0
        k635_results[scheme] = {"rho": float(rho), "p": float(p), "ns": ns, "bn_h0_means": bn_h0_means}
        log(f"\n{scheme}: Spearman rho(N, mean_bn_h0) = {rho:.4f} (p={p:.4f})")

    # K635 passes if either scheme shows rho > 0.8
    k635_pass = any(r["rho"] > 0.8 for r in k635_results.values())
    k635_detail = "; ".join(
        f"{s}: rho={r['rho']:.4f}, p={r['p']:.4f}" for s, r in k635_results.items()
    )

    # Additional: perturbation norm scaling
    log("\nPerturbation norm scaling:")
    for scheme in ["averaging", "additive"]:
        ns = sorted(analysis[scheme].keys())
        norms = [float(np.mean(analysis[scheme][n]["max_deltas"])) for n in ns]
        log(f"  {scheme}: N={ns}, norms={[f'{x:.4f}' for x in norms]}")

    summary = {
        "k634_pass": k634_pass,
        "k634_detail": k634_detail,
        "k635_pass": k635_pass,
        "k635_detail": k635_detail,
        "k635_rho_by_scheme": k635_results,
        "analysis_by_scheme": {},
        "base_stats": base_stats,
    }

    # Serialize analysis
    for scheme in ["averaging", "additive"]:
        scheme_data = {}
        for N in sorted(analysis[scheme].keys()):
            a = analysis[scheme][N]
            scheme_data[str(N)] = {
                "mean_max_delta": float(np.mean(a["max_deltas"])),
                "mean_bn_h0": float(np.mean(a["bn_h0"])),
                "mean_bn_h1": float(np.mean(a["bn_h1"])) if a["bn_h1"] else None,
                "max_bn_h0": float(np.max(a["bn_h0"])),
                "total_lost_h0": a["lost_h0"],
                "total_lost_h1": a["lost_h1"],
                "total_high_h0": a["total_high_h0"],
                "total_high_h1": a["total_high_h1"],
                "n_high_in_vulnerability_window": a["n_high_in_vuln"],
            }
        summary["analysis_by_scheme"][scheme] = scheme_data

    log(f"\nKILL CRITERIA:")
    log(f"  K634 (feature loss at N>5): {'PASS' if k634_pass else 'FAIL'} -- {k634_detail}")
    log(f"  K635 (monotonic d_B): {'PASS' if k635_pass else 'FAIL'} -- {k635_detail}")

    return summary


def main():
    t0 = time.time()

    # Phase 1: Load weights
    weights, skeleton, adapters = phase_load_weights()

    # Phase 2: N-sweep
    sweep_results, base_stats = phase_sweep(weights, skeleton, adapters)

    # Clean up
    del weights
    gc.collect()

    # Phase 3: Analyze
    summary = phase_analyze(sweep_results, base_stats)

    total_time = time.time() - t0

    # Save results
    output = {
        "experiment": "topology_stress_n_sweep",
        "total_time_s": round(total_time, 1),
        "config": {
            "model": MODEL_ID,
            "n_subsample": N_SUBSAMPLE,
            "target_layers": TARGET_LAYERS,
            "target_keys": TARGET_KEYS,
            "n_sweep": N_SWEEP,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "max_dim": MAX_DIM,
            "n_random_baselines": N_RANDOM_BASELINES,
        },
        "summary": summary,
        "per_n": sweep_results,
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")
    log(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
