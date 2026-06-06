#!/usr/bin/env python3
"""
Effective-Delta Cosine Measurement Experiment

Compares two orthogonality metrics for LoRA adapters on BitNet-2B-4T:
  1. Raw parameter cosine: cos(concat(vec(A_i), vec(B_i)), concat(vec(A_j), vec(B_j)))
  2. Effective-delta cosine: cos(vec(DW_i), vec(DW_j)) where DW = B^T @ A^T

The Grassmannian skeleton guarantee operates on the effective delta, not raw params.
This experiment measures whether the current raw-param proxy (tools/orthogonality.py)
is conservative (overestimates interference) or dangerously loose.

Prior finding at toy scale (d=64): B-matrix cos 0.0298 -> delta cos 0.0017 (17x filter).
Expected: effective-delta cosine < raw parameter cosine at d=2560.

Kill criteria:
  K1: effective-delta cosine exceeds 0.05 for any adapter pair
  K2: effective-delta cosine > 5x raw parameter cosine for any pair

Data source: 5 trained adapters from bitnet_2b_real_composition (200 steps, r=16).

Platform: Apple Silicon, numpy only (no GPU needed). $0.
Expected runtime: <5 minutes.
"""

import json
import time
import sys
from pathlib import Path

import numpy as np

# ============================================================================
# Configuration
# ============================================================================
ADAPTER_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "adapters"
RESULTS_FILE = Path(__file__).parent / "results.json"

DOMAINS = ["medical", "code", "math", "legal", "creative"]

# Map directory names -- code adapter might be stored as "python"
DOMAIN_DIR_MAP = {
    "medical": "medical",
    "code": "python",  # stored as python/ in bitnet_2b_real_composition
    "math": "math",
    "legal": "legal",
    "creative": "creative",
}


# ============================================================================
# Helper functions
# ============================================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def load_adapter(domain: str) -> dict:
    """Load adapter weights from NPZ file."""
    dir_name = DOMAIN_DIR_MAP.get(domain, domain)
    path = ADAPTER_DIR / dir_name / "adapter.npz"
    if not path.exists():
        raise FileNotFoundError(f"Adapter not found: {path}")
    data = dict(np.load(str(path)))
    return data


def separate_ab(weights: dict):
    """Separate A and B matrices by layer and module.

    Returns:
        a_dict: {(layer, module): np.array}  -- lora_a matrices
        b_dict: {(layer, module): np.array}  -- lora_b matrices
    """
    a_dict = {}
    b_dict = {}
    for key, val in sorted(weights.items()):
        if "lora_a" in key:
            # Extract layer and module from key like
            # "model.layers.0.mlp.gate_proj.lora_a"
            parts = key.replace(".lora_a", "")
            a_dict[parts] = val
        elif "lora_b" in key:
            parts = key.replace(".lora_b", "")
            b_dict[parts] = val
    return a_dict, b_dict


def compute_raw_param_vector(weights: dict) -> np.ndarray:
    """Concatenate all lora_a and lora_b params into a single vector.
    This is what tools/orthogonality.py does."""
    parts = []
    for key in sorted(weights.keys()):
        if "lora_a" in key or "lora_b" in key:
            parts.append(weights[key].flatten())
    return np.concatenate(parts)


def compute_effective_delta_vector(a_dict: dict, b_dict: dict) -> np.ndarray:
    """Compute vec(B^T @ A^T) for each module and concatenate.

    For each module:
      lora_a: (d_in, r)
      lora_b: (r, d_out)
      effective delta DW = lora_b.T @ lora_a.T  -- shape (d_out, d_in)

    This represents the actual weight perturbation in the model.
    """
    parts = []
    for module_key in sorted(a_dict.keys()):
        if module_key not in b_dict:
            continue
        A = a_dict[module_key]  # (d_in, r)
        B = b_dict[module_key]  # (r, d_out)
        # DW = B^T @ A^T = (d_out, r) @ (r, d_in) = (d_out, d_in)
        DW = B.T @ A.T
        parts.append(DW.flatten())
    return np.concatenate(parts)


def compute_a_only_vector(a_dict: dict) -> np.ndarray:
    """Concatenate only lora_a parameters."""
    parts = []
    for key in sorted(a_dict.keys()):
        parts.append(a_dict[key].flatten())
    return np.concatenate(parts)


def compute_b_only_vector(b_dict: dict) -> np.ndarray:
    """Concatenate only lora_b parameters."""
    parts = []
    for key in sorted(b_dict.keys()):
        parts.append(b_dict[key].flatten())
    return np.concatenate(parts)


def compute_a_coherence(a_dict_i: dict, a_dict_j: dict) -> float:
    """Compute mean ||A_i^T A_j||_F across all modules.
    This is the coherence metric from the Grassmannian skeleton."""
    coherences = []
    for key in sorted(a_dict_i.keys()):
        if key not in a_dict_j:
            continue
        A_i = a_dict_i[key]  # (d_in, r)
        A_j = a_dict_j[key]  # (d_in, r)
        # A_i^T @ A_j: (r, d_in) @ (d_in, r) = (r, r)
        cross = A_i.T @ A_j
        coh = np.linalg.norm(cross, 'fro')
        coherences.append(coh)
    return float(np.mean(coherences))


def compute_b_condition_numbers(b_dict: dict) -> list:
    """Compute condition number of each B matrix."""
    kappas = []
    for key in sorted(b_dict.keys()):
        B = b_dict[key]  # (r, d_out)
        svs = np.linalg.svd(B, compute_uv=False)
        if svs[-1] > 1e-12:
            kappas.append(float(svs[0] / svs[-1]))
        else:
            kappas.append(float('inf'))
    return kappas


# ============================================================================
# Main experiment
# ============================================================================
def main():
    t_start = time.time()

    print("=" * 70)
    print("Effective-Delta Cosine Measurement")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: Load all adapters
    # ------------------------------------------------------------------
    print("\n[Phase 1] Loading adapters...")
    adapters = {}
    a_dicts = {}
    b_dicts = {}

    for domain in DOMAINS:
        weights = load_adapter(domain)
        adapters[domain] = weights
        a, b = separate_ab(weights)
        a_dicts[domain] = a
        b_dicts[domain] = b
        n_modules = len(a)
        # Show a sample shape
        sample_key = list(a.keys())[0]
        print(f"  {domain}: {n_modules} modules, "
              f"sample A shape {a[sample_key].shape}, "
              f"sample B shape {b[sample_key].shape}")

    # ------------------------------------------------------------------
    # Phase 2: Compute vectors
    # ------------------------------------------------------------------
    print("\n[Phase 2] Computing parameter and delta vectors...")
    raw_vectors = {}
    delta_vectors = {}
    a_vectors = {}
    b_vectors = {}

    for domain in DOMAINS:
        raw_vectors[domain] = compute_raw_param_vector(adapters[domain])
        delta_vectors[domain] = compute_effective_delta_vector(a_dicts[domain], b_dicts[domain])
        a_vectors[domain] = compute_a_only_vector(a_dicts[domain])
        b_vectors[domain] = compute_b_only_vector(b_dicts[domain])

    d_raw = len(raw_vectors[DOMAINS[0]])
    d_eff = len(delta_vectors[DOMAINS[0]])
    d_a = len(a_vectors[DOMAINS[0]])
    d_b = len(b_vectors[DOMAINS[0]])

    print(f"  Raw param vector dim:      D_raw = {d_raw:,}")
    print(f"  Effective delta vector dim: D_eff = {d_eff:,}")
    print(f"  A-only vector dim:         D_a   = {d_a:,}")
    print(f"  B-only vector dim:         D_b   = {d_b:,}")
    print(f"  D_eff / D_raw = {d_eff / d_raw:.1f}x")

    # ------------------------------------------------------------------
    # Phase 3: Compute all pairwise cosines
    # ------------------------------------------------------------------
    print("\n[Phase 3] Computing pairwise cosines...")

    pairs = []
    for i in range(len(DOMAINS)):
        for j in range(i + 1, len(DOMAINS)):
            d_i, d_j = DOMAINS[i], DOMAINS[j]

            cos_raw = abs(cosine_similarity(raw_vectors[d_i], raw_vectors[d_j]))
            cos_eff = abs(cosine_similarity(delta_vectors[d_i], delta_vectors[d_j]))
            cos_a = abs(cosine_similarity(a_vectors[d_i], a_vectors[d_j]))
            cos_b = abs(cosine_similarity(b_vectors[d_i], b_vectors[d_j]))
            a_coherence = compute_a_coherence(a_dicts[d_i], a_dicts[d_j])

            # Ratio: effective / raw
            ratio = cos_eff / cos_raw if cos_raw > 1e-12 else float('inf')

            pairs.append({
                "pair": f"{d_i}-{d_j}",
                "cos_raw": round(cos_raw, 8),
                "cos_effective": round(cos_eff, 8),
                "cos_a_only": round(cos_a, 8),
                "cos_b_only": round(cos_b, 8),
                "a_coherence_mean": round(a_coherence, 6),
                "ratio_eff_over_raw": round(ratio, 4),
            })

    # Print results table
    print(f"\n{'Pair':<20} {'|cos| raw':>12} {'|cos| eff':>12} {'|cos| A':>12} "
          f"{'|cos| B':>12} {'A coher':>10} {'eff/raw':>10}")
    print("-" * 90)
    for p in pairs:
        print(f"{p['pair']:<20} {p['cos_raw']:>12.8f} {p['cos_effective']:>12.8f} "
              f"{p['cos_a_only']:>12.8f} {p['cos_b_only']:>12.8f} "
              f"{p['a_coherence_mean']:>10.6f} {p['ratio_eff_over_raw']:>10.4f}")

    # ------------------------------------------------------------------
    # Phase 4: Aggregate statistics
    # ------------------------------------------------------------------
    print("\n[Phase 4] Aggregate statistics...")

    cos_raw_values = [p["cos_raw"] for p in pairs]
    cos_eff_values = [p["cos_effective"] for p in pairs]
    cos_a_values = [p["cos_a_only"] for p in pairs]
    cos_b_values = [p["cos_b_only"] for p in pairs]
    ratios = [p["ratio_eff_over_raw"] for p in pairs]

    stats = {
        "raw": {
            "mean": round(float(np.mean(cos_raw_values)), 8),
            "max": round(float(np.max(cos_raw_values)), 8),
            "min": round(float(np.min(cos_raw_values)), 8),
        },
        "effective": {
            "mean": round(float(np.mean(cos_eff_values)), 8),
            "max": round(float(np.max(cos_eff_values)), 8),
            "min": round(float(np.min(cos_eff_values)), 8),
        },
        "a_only": {
            "mean": round(float(np.mean(cos_a_values)), 8),
            "max": round(float(np.max(cos_a_values)), 8),
            "min": round(float(np.min(cos_a_values)), 8),
        },
        "b_only": {
            "mean": round(float(np.mean(cos_b_values)), 8),
            "max": round(float(np.max(cos_b_values)), 8),
            "min": round(float(np.min(cos_b_values)), 8),
        },
        "ratio_eff_over_raw": {
            "mean": round(float(np.mean(ratios)), 4),
            "max": round(float(np.max(ratios)), 4),
            "min": round(float(np.min(ratios)), 4),
        },
    }

    print(f"\n  Metric         Mean |cos|     Max |cos|     Min |cos|")
    print(f"  {'='*55}")
    print(f"  Raw param      {stats['raw']['mean']:.8f}  {stats['raw']['max']:.8f}  {stats['raw']['min']:.8f}")
    print(f"  Effective       {stats['effective']['mean']:.8f}  {stats['effective']['max']:.8f}  {stats['effective']['min']:.8f}")
    print(f"  A-only          {stats['a_only']['mean']:.8f}  {stats['a_only']['max']:.8f}  {stats['a_only']['min']:.8f}")
    print(f"  B-only          {stats['b_only']['mean']:.8f}  {stats['b_only']['max']:.8f}  {stats['b_only']['min']:.8f}")
    print(f"\n  Ratio (eff/raw):  mean={stats['ratio_eff_over_raw']['mean']:.4f}  "
          f"max={stats['ratio_eff_over_raw']['max']:.4f}  "
          f"min={stats['ratio_eff_over_raw']['min']:.4f}")

    # ------------------------------------------------------------------
    # Phase 5: B-matrix condition numbers
    # ------------------------------------------------------------------
    print("\n[Phase 5] B-matrix condition numbers...")
    b_kappas = {}
    for domain in DOMAINS:
        kappas = compute_b_condition_numbers(b_dicts[domain])
        b_kappas[domain] = {
            "mean": round(float(np.mean(kappas)), 2),
            "max": round(float(np.max(kappas)), 2),
            "median": round(float(np.median(kappas)), 2),
        }
        print(f"  {domain}: mean kappa={b_kappas[domain]['mean']:.1f}, "
              f"max={b_kappas[domain]['max']:.1f}, "
              f"median={b_kappas[domain]['median']:.1f}")

    # ------------------------------------------------------------------
    # Phase 6: Per-layer decomposition (sample a few layers)
    # ------------------------------------------------------------------
    print("\n[Phase 6] Per-layer effective-delta analysis (sample layers)...")
    sample_layers = [0, 14, 29]  # first, middle, last
    sample_module = "self_attn.q_proj"

    layer_analysis = []
    for layer_idx in sample_layers:
        module_key = f"model.layers.{layer_idx}.{sample_module}"
        if module_key not in a_dicts[DOMAINS[0]]:
            continue

        # Compute per-module effective delta cosines for this layer
        layer_pairs = []
        for i in range(len(DOMAINS)):
            for j in range(i + 1, len(DOMAINS)):
                A_i = a_dicts[DOMAINS[i]][module_key]
                A_j = a_dicts[DOMAINS[j]][module_key]
                B_i = b_dicts[DOMAINS[i]][module_key]
                B_j = b_dicts[DOMAINS[j]][module_key]

                DW_i = (B_i.T @ A_i.T).flatten()
                DW_j = (B_j.T @ A_j.T).flatten()
                cos_layer = abs(cosine_similarity(DW_i, DW_j))

                # A coherence for this specific module
                cross = A_i.T @ A_j
                a_coh = np.linalg.norm(cross, 'fro')

                layer_pairs.append({
                    "pair": f"{DOMAINS[i]}-{DOMAINS[j]}",
                    "cos_eff": cos_layer,
                    "a_coherence": a_coh,
                })

        mean_cos = np.mean([p["cos_eff"] for p in layer_pairs])
        max_cos = np.max([p["cos_eff"] for p in layer_pairs])
        mean_coh = np.mean([p["a_coherence"] for p in layer_pairs])

        layer_analysis.append({
            "layer": layer_idx,
            "module": sample_module,
            "mean_cos_eff": round(float(mean_cos), 8),
            "max_cos_eff": round(float(max_cos), 8),
            "mean_a_coherence": round(float(mean_coh), 6),
        })
        print(f"  Layer {layer_idx} ({sample_module}): "
              f"mean |cos_eff|={mean_cos:.8f}, "
              f"max={max_cos:.8f}, "
              f"mean A-coherence={mean_coh:.6f}")

    # ------------------------------------------------------------------
    # Phase 7: Kill criteria assessment
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    max_eff = stats["effective"]["max"]
    max_ratio = stats["ratio_eff_over_raw"]["max"]

    k1_pass = max_eff < 0.05
    k2_pass = max_ratio < 5.0

    print(f"\n  K1: max |cos_effective| = {max_eff:.8f} (threshold: 0.05)")
    print(f"      Margin: {0.05 / max_eff:.1f}x" if max_eff > 0 else "      Margin: inf")
    print(f"      Result: {'PASS' if k1_pass else 'FAIL'}")

    print(f"\n  K2: max ratio eff/raw = {max_ratio:.4f} (threshold: 5.0)")
    print(f"      Margin: {5.0 / max_ratio:.1f}x" if max_ratio > 0 else "      Margin: inf")
    print(f"      Result: {'PASS' if k2_pass else 'FAIL'}")

    overall = "PASS" if (k1_pass and k2_pass) else "FAIL"
    print(f"\n  Overall: {overall}")

    # Determine filtering ratio (how much does effective improve over raw)
    filtering_ratio = stats["raw"]["mean"] / stats["effective"]["mean"] if stats["effective"]["mean"] > 0 else float('inf')
    print(f"\n  Decorrelation filter: {filtering_ratio:.1f}x "
          f"(raw mean / eff mean)")
    print(f"  Prior toy-scale finding: 17x at d=64")
    print(f"  Expected: >= 17x at d=2560 (more dimensions = better filtering)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")

    results = {
        "experiment": "bitnet_effective_delta_cosine",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adapter_source": "bitnet_2b_real_composition (200 steps, r=16)",
        "domains": DOMAINS,
        "n_pairs": len(pairs),
        "dimensions": {
            "D_raw": d_raw,
            "D_effective": d_eff,
            "D_a": d_a,
            "D_b": d_b,
            "ratio_eff_over_raw_dim": round(d_eff / d_raw, 1),
        },
        "pairs": pairs,
        "statistics": stats,
        "b_condition_numbers": b_kappas,
        "layer_analysis": layer_analysis,
        "kill_criteria": {
            "K1_max_eff_cosine": {
                "threshold": 0.05,
                "measured": max_eff,
                "margin": round(0.05 / max_eff, 1) if max_eff > 0 else None,
                "result": "PASS" if k1_pass else "FAIL",
            },
            "K2_max_ratio": {
                "threshold": 5.0,
                "measured": max_ratio,
                "margin": round(5.0 / max_ratio, 1) if max_ratio > 0 else None,
                "result": "PASS" if k2_pass else "FAIL",
            },
        },
        "verdict": "SUPPORTED" if (k1_pass and k2_pass) else "KILLED",
        "decorrelation_filter": round(filtering_ratio, 1),
        "runtime_seconds": round(elapsed, 1),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
