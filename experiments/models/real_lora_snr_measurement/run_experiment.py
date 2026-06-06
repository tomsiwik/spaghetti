#!/usr/bin/env python3
"""
Real LoRA SNR Measurement

Measures the spectral profile (SNR, r_95, r_99) of real LoRA deltas from
pilot-50 trained adapters to determine whether adaptive rank selection
has practical benefit.

Usage: python run_experiment.py
"""

import json
import re
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

ADAPTER_DIR = Path(__file__).resolve().parent.parent.parent.parent / "adapters"
EXPERTS = ["bash", "math", "medical", "python", "sql"]
RANK = 16
RESULTS_FILE = Path(__file__).resolve().parent / "results.json"


def load_adapter(expert_name: str) -> dict:
    """Load all A/B weight pairs from a safetensors adapter."""
    path = ADAPTER_DIR / expert_name / "adapter_model.safetensors"
    tensors = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def parse_key(key: str):
    """Parse a safetensor key into (layer_idx, module_name, lora_type)."""
    m = re.match(
        r"base_model\.model\.model\.layers\.(\d+)\.(.*?)\.(lora_[AB])\.weight", key
    )
    if m:
        return int(m.group(1)), m.group(2), m.group(3)
    return None, None, None


def compute_spectral_stats(A: np.ndarray, B: np.ndarray) -> dict:
    """Compute SVD-based spectral statistics for a LoRA delta B@A."""
    # Delta = B @ A, shape [d_out, d_in], rank <= r
    delta = B @ A  # [d_out, d_in]

    # Full SVD is expensive for large matrices. Since rank <= 16,
    # we can compute SVD of the smaller r x r matrix.
    # But numpy's svd on the full matrix is fine for r=16 (it returns
    # at most min(d_out, d_in) singular values, but only 16 are non-zero).
    # For efficiency, compute SVD of the r x r Gram matrix.
    # S(B@A)^2 = eigenvalues of A^T B^T B A (r x r matrix)
    gram = A @ A.T  # [r, r]
    BtB = B.T @ B   # [r, r]
    M = BtB @ gram   # [r, r] -- eigenvalues of M = sigma^2 of delta

    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.sort(np.maximum(eigvals, 0))[::-1]  # descending, clamp negatives
    singular_values = np.sqrt(eigvals)

    # Filter near-zero singular values
    sv = singular_values[singular_values > 1e-10]
    r_eff = len(sv)

    if r_eff < 2:
        return {
            "snr": float("inf") if r_eff == 1 else 0.0,
            "r_95": 1,
            "r_99": 1,
            "rho": 1.0,
            "r_eff": r_eff,
            "sv_max": float(sv[0]) if r_eff > 0 else 0.0,
            "sv_min": float(sv[-1]) if r_eff > 0 else 0.0,
            "singular_values": sv.tolist(),
        }

    snr = float(sv[0] / sv[-1])

    # Cumulative variance
    sv_sq = sv ** 2
    cumvar = np.cumsum(sv_sq) / np.sum(sv_sq)
    r_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    r_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    rho = r_99 / r_95 if r_95 > 0 else 1.0

    return {
        "snr": snr,
        "r_95": r_95,
        "r_99": r_99,
        "rho": rho,
        "r_eff": r_eff,
        "sv_max": float(sv[0]),
        "sv_min": float(sv[-1]),
        "singular_values": sv.tolist(),
    }


def main():
    t0 = time.time()
    all_results = []
    expert_summaries = {}

    for expert in EXPERTS:
        print(f"Loading {expert}...")
        tensors = load_adapter(expert)

        # Group by (layer, module)
        pairs = {}
        for key in tensors:
            layer, module, lora_type = parse_key(key)
            if layer is None:
                continue
            if (layer, module) not in pairs:
                pairs[(layer, module)] = {}
            pairs[(layer, module)][lora_type] = tensors[key]

        expert_stats = []
        for (layer, module), mats in sorted(pairs.items()):
            A = mats["lora_A"]  # [r, d_in]
            B = mats["lora_B"]  # [d_out, r]
            stats = compute_spectral_stats(A, B)
            stats["expert"] = expert
            stats["layer"] = layer
            stats["module"] = module
            # Don't save full singular values in per-entry (too verbose)
            sv_list = stats.pop("singular_values")
            all_results.append(stats)
            expert_stats.append(stats)

        # Per-expert summary
        snrs = [s["snr"] for s in expert_stats if np.isfinite(s["snr"])]
        r95s = [s["r_95"] for s in expert_stats]
        r99s = [s["r_99"] for s in expert_stats]
        rhos = [s["rho"] for s in expert_stats]

        expert_summaries[expert] = {
            "n_entries": len(expert_stats),
            "snr_mean": float(np.mean(snrs)),
            "snr_median": float(np.median(snrs)),
            "snr_min": float(np.min(snrs)),
            "snr_max": float(np.max(snrs)),
            "snr_std": float(np.std(snrs)),
            "r95_mean": float(np.mean(r95s)),
            "r99_mean": float(np.mean(r99s)),
            "rho_mean": float(np.mean(rhos)),
            "rho_min": float(np.min(rhos)),
            "rho_max": float(np.max(rhos)),
            "rho_std": float(np.std(rhos)),
            "pct_snr_below_10": float(np.mean(np.array(snrs) < 10) * 100),
        }
        print(f"  {expert}: SNR mean={expert_summaries[expert]['snr_mean']:.1f}, "
              f"median={expert_summaries[expert]['snr_median']:.1f}, "
              f"min={expert_summaries[expert]['snr_min']:.1f}, "
              f"max={expert_summaries[expert]['snr_max']:.1f}, "
              f"r95={expert_summaries[expert]['r95_mean']:.1f}, "
              f"r99={expert_summaries[expert]['r99_mean']:.1f}, "
              f"rho={expert_summaries[expert]['rho_mean']:.2f}")

    # Per-module aggregation
    module_names = sorted(set(r["module"] for r in all_results))
    module_summaries = {}
    for mod in module_names:
        mod_snrs = [r["snr"] for r in all_results if r["module"] == mod and np.isfinite(r["snr"])]
        mod_rhos = [r["rho"] for r in all_results if r["module"] == mod]
        mod_r95 = [r["r_95"] for r in all_results if r["module"] == mod]
        mod_r99 = [r["r_99"] for r in all_results if r["module"] == mod]
        module_summaries[mod] = {
            "snr_mean": float(np.mean(mod_snrs)),
            "snr_median": float(np.median(mod_snrs)),
            "snr_min": float(np.min(mod_snrs)),
            "snr_max": float(np.max(mod_snrs)),
            "r95_mean": float(np.mean(mod_r95)),
            "r99_mean": float(np.mean(mod_r99)),
            "rho_mean": float(np.mean(mod_rhos)),
            "pct_snr_below_10": float(np.mean(np.array(mod_snrs) < 10) * 100),
        }

    # Per-layer aggregation
    layers = sorted(set(r["layer"] for r in all_results))
    layer_summaries = {}
    for layer in layers:
        layer_snrs = [r["snr"] for r in all_results if r["layer"] == layer and np.isfinite(r["snr"])]
        layer_rhos = [r["rho"] for r in all_results if r["layer"] == layer]
        layer_summaries[layer] = {
            "snr_mean": float(np.mean(layer_snrs)),
            "snr_median": float(np.median(layer_snrs)),
            "rho_mean": float(np.mean(layer_rhos)),
        }

    # Global aggregation
    all_snrs = [r["snr"] for r in all_results if np.isfinite(r["snr"])]
    all_rhos = [r["rho"] for r in all_results]
    all_r95 = [r["r_95"] for r in all_results]
    all_r99 = [r["r_99"] for r in all_results]

    # K1: Are there ANY entries with SNR < 10?
    pct_below_10 = float(np.mean(np.array(all_snrs) < 10) * 100)
    all_above_10 = pct_below_10 == 0.0

    # K2: Does rho vary more than 1.5x across experts?
    expert_rho_means = [expert_summaries[e]["rho_mean"] for e in EXPERTS]
    rho_range_ratio = max(expert_rho_means) / min(expert_rho_means) if min(expert_rho_means) > 0 else float("inf")
    rho_low_diversity = rho_range_ratio < 1.5

    # Also check per-entry rho diversity
    rho_entry_range = max(all_rhos) / min(all_rhos) if min(all_rhos) > 0 else float("inf")

    elapsed = time.time() - t0

    results = {
        "experiment": "real_lora_snr_measurement",
        "description": "Measure effective SNR of pilot-50 trained LoRA deltas via SVD",
        "elapsed_seconds": round(elapsed, 2),
        "n_experts": len(EXPERTS),
        "n_layers": len(layers),
        "n_modules": len(module_names),
        "n_entries": len(all_results),
        "global_stats": {
            "snr_mean": float(np.mean(all_snrs)),
            "snr_median": float(np.median(all_snrs)),
            "snr_min": float(np.min(all_snrs)),
            "snr_max": float(np.max(all_snrs)),
            "snr_std": float(np.std(all_snrs)),
            "snr_p10": float(np.percentile(all_snrs, 10)),
            "snr_p25": float(np.percentile(all_snrs, 25)),
            "snr_p75": float(np.percentile(all_snrs, 75)),
            "snr_p90": float(np.percentile(all_snrs, 90)),
            "pct_snr_below_10": pct_below_10,
            "pct_snr_below_5": float(np.mean(np.array(all_snrs) < 5) * 100),
            "r95_mean": float(np.mean(all_r95)),
            "r95_median": float(np.median(all_r95)),
            "r99_mean": float(np.mean(all_r99)),
            "r99_median": float(np.median(all_r99)),
            "rho_mean": float(np.mean(all_rhos)),
            "rho_min": float(np.min(all_rhos)),
            "rho_max": float(np.max(all_rhos)),
            "rho_std": float(np.std(all_rhos)),
        },
        "expert_summaries": expert_summaries,
        "module_summaries": module_summaries,
        "layer_summaries": {str(k): v for k, v in layer_summaries.items()},
        "kill_criteria": {
            "K1_all_snr_above_10": all_above_10,
            "K1_pct_below_10": pct_below_10,
            "K1_verdict": "KILLED (vacuous)" if all_above_10 else "SURVIVES",
            "K2_rho_range_ratio_across_experts": round(rho_range_ratio, 4),
            "K2_rho_entry_range": round(rho_entry_range, 4),
            "K2_low_diversity": rho_low_diversity,
            "K2_verdict": "KILLED (no diversity)" if rho_low_diversity else "SURVIVES",
        },
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Entries analyzed: {len(all_results)}")
    print(f"\nGlobal SNR: mean={np.mean(all_snrs):.1f}, median={np.median(all_snrs):.1f}, "
          f"min={np.min(all_snrs):.1f}, max={np.max(all_snrs):.1f}")
    print(f"Global r_95: mean={np.mean(all_r95):.1f}, median={np.median(all_r95):.0f}")
    print(f"Global r_99: mean={np.mean(all_r99):.1f}, median={np.median(all_r99):.0f}")
    print(f"Global rho: mean={np.mean(all_rhos):.2f}, min={np.min(all_rhos):.2f}, max={np.max(all_rhos):.2f}")
    print(f"\n% entries with SNR < 10: {pct_below_10:.1f}%")
    print(f"% entries with SNR < 5:  {float(np.mean(np.array(all_snrs) < 5) * 100):.1f}%")

    print(f"\n--- Per-Module SNR ---")
    for mod in module_names:
        s = module_summaries[mod]
        print(f"  {mod:30s}: SNR mean={s['snr_mean']:8.1f}, median={s['snr_median']:8.1f}, "
              f"r95={s['r95_mean']:.1f}, r99={s['r99_mean']:.1f}, rho={s['rho_mean']:.2f}")

    print(f"\n--- Per-Expert rho ---")
    for expert in EXPERTS:
        s = expert_summaries[expert]
        print(f"  {expert:10s}: rho mean={s['rho_mean']:.3f}, min={s['rho_min']:.3f}, max={s['rho_max']:.3f}")

    print(f"\n--- Kill Criteria ---")
    print(f"K1 (all SNR >= 10): {results['kill_criteria']['K1_verdict']}")
    print(f"   {pct_below_10:.1f}% of entries have SNR < 10")
    print(f"K2 (rho range < 1.5x): {results['kill_criteria']['K2_verdict']}")
    print(f"   Expert rho range ratio: {rho_range_ratio:.4f}")

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
