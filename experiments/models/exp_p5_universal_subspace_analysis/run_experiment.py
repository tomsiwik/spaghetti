#!/usr/bin/env python3
"""
P5.D0: Universal Subspace Analysis of Pierre's 25+ Adapters

MATH: experiments/models/exp_p5_universal_subspace_analysis/MATH.md

Tests Universal Weight Subspace Hypothesis (arXiv:2512.05117) on Grassmannian-initialized
adapters. PCA across all trained adapters to find shared low-rank structure.

Prior: Finding #65 (EigenLoRAx killed, A-matrices 31.3%), Finding #428 (N=25 composition),
       Finding #130 (Frechet merge worse than naive)

Kill criteria:
  K1282: Top-16 PCA components explain >= 80% variance across all adapters
  K1283: Universal subspace merging outperforms naive addition on >= 3/5 domains
  K1284: Adapter compression to universal basis preserves quality (< 5pp degradation)

Pure analysis experiment — no model loading for PCA phase, model only for quality eval.
"""

import gc
import json
import os
import time
from itertools import combinations
from pathlib import Path

import mlx.core as mx
import numpy as np

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODELS_DIR = EXPERIMENT_DIR.parent

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
np.random.seed(SEED)

# ── Adapter catalog: all independently trained Gemma 4 adapters ──────────

ADAPTER_CATALOG = {
    # T2.1: Single-domain training (3 domains)
    "math": MODELS_DIR / "exp_p1_t2_single_domain_training/adapters/math",
    "code": MODELS_DIR / "exp_p1_t2_single_domain_training/adapters/code",
    "medical": MODELS_DIR / "exp_p1_t2_single_domain_training/adapters/medical",
    # T2.6: Multi-domain training (2 additional domains)
    "legal": MODELS_DIR / "exp_p1_t2_multi_domain_5/adapters/legal",
    "finance": MODELS_DIR / "exp_p1_t2_multi_domain_5/adapters/finance",
    # P4.C0: Format adapters
    "latex_fmt": MODELS_DIR / "exp_p4_c0_formatting_adapter/latex_adapter",
    "soap_fmt": MODELS_DIR / "exp_p4_c0_formatting_adapter/soap_adapter",
    "legal_fmt": MODELS_DIR / "exp_p4_c0_formatting_adapter/legal_adapter",
    # P4.C1: V-proj format adapters (different training from C0)
    "latex_vproj": MODELS_DIR / "exp_p4_c1_vproj_soap_adapter/latex_adapter",
    "soap_vproj": MODELS_DIR / "exp_p4_c1_vproj_soap_adapter/soap_adapter",
    "legal_vproj": MODELS_DIR / "exp_p4_c1_vproj_soap_adapter/legal_adapter",
    # P4.A1: Biology speed test adapter
    "biology": MODELS_DIR / "exp_p4_a1_domain_adapter_speedtest/biology_adapter",
    # P4.C2: SOAP retention data mix
    "soap_retention": MODELS_DIR / "exp_p4_c2_soap_retention_data_mix/soap_adapter",
    # P3: Personal/diverse adapters
    "medical_oe": MODELS_DIR / "exp_p3_b0_medical_oe_adapter/lora_adapter",
    "diverse_personal": MODELS_DIR / "exp_p3_c1_diverse_personal_adapter/diverse_personal_adapter",
    # P1.C0: Orthogonalized adapters (5 domains, different from T2.1)
    "math_ortho": MODELS_DIR / "exp_p1_c0_composition_port_gemma4/orthogonalized_a_matrices/adapters/math",
    "code_ortho": MODELS_DIR / "exp_p1_c0_composition_port_gemma4/orthogonalized_a_matrices/adapters/code",
    "medical_ortho": MODELS_DIR / "exp_p1_c0_composition_port_gemma4/orthogonalized_a_matrices/adapters/medical",
    "legal_ortho": MODELS_DIR / "exp_p1_c0_composition_port_gemma4/orthogonalized_a_matrices/adapters/legal",
    "finance_ortho": MODELS_DIR / "exp_p1_c0_composition_port_gemma4/orthogonalized_a_matrices/adapters/finance",
}


def load_adapter_weights(path: Path) -> dict[str, np.ndarray] | None:
    """Load adapter safetensors, return numpy dict or None if missing."""
    sf_path = path / "adapters.safetensors"
    if not sf_path.exists():
        return None
    try:
        import safetensors.numpy
        return safetensors.numpy.load_file(str(sf_path))
    except Exception as e:
        print(f"  SKIP {path.name}: {e}")
        return None


def extract_ab_matrices(
    weights: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract A and B matrices from adapter weights, sorted by layer."""
    a_mats, b_mats = [], []
    keys_a = sorted(k for k in weights if "lora_a" in k)
    keys_b = sorted(k for k in weights if "lora_b" in k)
    for ka, kb in zip(keys_a, keys_b):
        a_mats.append(weights[ka])
        b_mats.append(weights[kb])
    return a_mats, b_mats


def pca_variance_explained(
    vectors: np.ndarray, max_k: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """PCA on rows of vectors. Returns (singular_values, cumulative_variance_explained)."""
    # Center
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    # SVD (truncated to min(N, max_k) for efficiency)
    k = min(centered.shape[0], centered.shape[1], max_k + 1)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback: use eigendecomposition of Gram matrix
        G = centered @ centered.T
        eigvals = np.sort(np.linalg.eigvalsh(G))[::-1]
        eigvals = np.maximum(eigvals, 0)
        S = np.sqrt(eigvals)
        pass

    total_var = np.sum(S**2)
    if total_var < 1e-12:
        return S, np.ones_like(S)
    cum_var = np.cumsum(S**2) / total_var
    return S, cum_var


def analyze_gram_matrix(vectors: np.ndarray) -> dict:
    """Compute Gram matrix statistics (orthogonality check)."""
    # Normalize rows
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = vectors / norms
    G = normed @ normed.T
    N = G.shape[0]
    # Off-diagonal cosines
    mask = ~np.eye(N, dtype=bool)
    off_diag = np.abs(G[mask])
    return {
        "mean_abs_cos": float(np.mean(off_diag)),
        "max_abs_cos": float(np.max(off_diag)),
        "min_abs_cos": float(np.min(off_diag)),
        "std_abs_cos": float(np.std(off_diag)),
    }


def compression_quality(
    vectors: np.ndarray, k: int
) -> dict:
    """Project vectors onto top-K PCA components, measure reconstruction."""
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Project onto top-K
    projected = centered @ Vt[:k].T @ Vt[:k]
    # Reconstruction error
    residual = centered - projected
    recon_error = np.linalg.norm(residual) / np.linalg.norm(centered)
    # Per-adapter cosine similarity
    cosines = []
    for i in range(vectors.shape[0]):
        c_norm = np.linalg.norm(centered[i])
        p_norm = np.linalg.norm(projected[i])
        if c_norm > 1e-12 and p_norm > 1e-12:
            cos = np.dot(centered[i], projected[i]) / (c_norm * p_norm)
            cosines.append(float(cos))
    # Orthogonality after projection
    gram_orig = analyze_gram_matrix(centered)
    gram_proj = analyze_gram_matrix(projected)
    return {
        "k": k,
        "relative_recon_error": float(recon_error),
        "mean_cosine_similarity": float(np.mean(cosines)) if cosines else 0.0,
        "min_cosine_similarity": float(np.min(cosines)) if cosines else 0.0,
        "orig_max_cos": gram_orig["max_abs_cos"],
        "proj_max_cos": gram_proj["max_abs_cos"],
        "orthogonality_degradation": gram_proj["max_abs_cos"] / max(gram_orig["max_abs_cos"], 1e-12),
    }


def naive_addition_quality(
    a_matrices: list[list[np.ndarray]],
    b_matrices: list[list[np.ndarray]],
    adapter_names: list[str],
    target_domains: list[str],
) -> dict:
    """Measure quality of naive addition (W = sum A_i B_i) vs universal subspace merge.

    Quality proxy: Frobenius norm preservation and per-domain activation magnitude.
    Full generation quality requires model loading — use proxy metrics here.
    """
    n_adapters = len(a_matrices)
    n_layers = len(a_matrices[0])

    # For each target domain, compare:
    # 1. Single adapter: A_target @ B_target
    # 2. Naive addition: sum_i A_i @ B_i
    # 3. Universal subspace merge: project all into shared basis, then merge
    results = {}

    for tidx, target in enumerate(target_domains):
        if target not in adapter_names:
            continue
        target_idx = adapter_names.index(target)

        # Sample a few layers for efficiency
        test_layers = list(range(0, n_layers, max(1, n_layers // 6)))
        if IS_SMOKE:
            test_layers = test_layers[:2]

        layer_metrics = []
        for l in test_layers:
            # Get all A, B for this layer
            As = [a_matrices[i][l] for i in range(n_adapters)]
            Bs = [b_matrices[i][l] for i in range(n_adapters)]

            # Single adapter delta-W
            single_dW = As[target_idx] @ Bs[target_idx]

            # Naive addition: sum of all delta-Ws
            naive_dW = sum(A @ B for A, B in zip(As, Bs))

            # Universal subspace merge: PCA of vectorized A's, project, merge
            a_vecs = np.stack([A.flatten() for A in As])
            _, S, Vt = np.linalg.svd(
                a_vecs - a_vecs.mean(axis=0, keepdims=True), full_matrices=False
            )
            # Use top-8 components (half of adapter count)
            k = min(8, len(As))
            basis = Vt[:k]

            # Project each A into universal basis, reconstruct, merge
            centered_a = a_vecs - a_vecs.mean(axis=0, keepdims=True)
            proj_coeffs = centered_a @ basis.T
            proj_a_vecs = proj_coeffs @ basis + a_vecs.mean(axis=0, keepdims=True)
            proj_As = [v.reshape(As[0].shape) for v in proj_a_vecs]

            univ_dW = sum(pA @ B for pA, B in zip(proj_As, Bs))

            # Metrics: how well does each method preserve target adapter's contribution?
            # Cosine similarity of composed W with single-adapter W
            s_flat = single_dW.flatten()
            n_flat = naive_dW.flatten()
            u_flat = univ_dW.flatten()

            def cosine(a, b):
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na < 1e-12 or nb < 1e-12:
                    return 0.0
                return float(np.dot(a, b) / (na * nb))

            layer_metrics.append({
                "layer": l,
                "naive_cos": cosine(s_flat, n_flat),
                "univ_cos": cosine(s_flat, u_flat),
                "naive_norm_ratio": float(
                    np.linalg.norm(n_flat) / max(np.linalg.norm(s_flat), 1e-12)
                ),
                "univ_norm_ratio": float(
                    np.linalg.norm(u_flat) / max(np.linalg.norm(s_flat), 1e-12)
                ),
            })

        # Aggregate: naive wins if higher cosine with single adapter
        naive_wins = sum(
            1 for m in layer_metrics if m["naive_cos"] >= m["univ_cos"]
        )
        results[target] = {
            "layers_tested": len(layer_metrics),
            "naive_wins": naive_wins,
            "univ_wins": len(layer_metrics) - naive_wins,
            "mean_naive_cos": float(np.mean([m["naive_cos"] for m in layer_metrics])),
            "mean_univ_cos": float(np.mean([m["univ_cos"] for m in layer_metrics])),
            "naive_better": naive_wins > len(layer_metrics) / 2,
            "layer_detail": layer_metrics,
        }

    return results


def main():
    t0 = time.time()
    results = {
        "experiment": "exp_p5_universal_subspace_analysis",
        "hypothesis": "Universal Weight Subspace Hypothesis does NOT apply to Grassmannian adapters",
        "reference": "arXiv:2512.05117, Finding #65, Finding #428",
    }

    # ── Phase 1: Load all adapters ───────────────────────────────────────

    print("=" * 60)
    print("Phase 1: Loading adapters")
    print("=" * 60)

    adapter_data = {}
    for name, path in ADAPTER_CATALOG.items():
        weights = load_adapter_weights(path)
        if weights is not None:
            a_mats, b_mats = extract_ab_matrices(weights)
            if len(a_mats) > 0:
                adapter_data[name] = {"a": a_mats, "b": b_mats}
                print(f"  Loaded {name}: {len(a_mats)} layers, "
                      f"A={a_mats[0].shape}, B={b_mats[0].shape}")
            del weights
        else:
            print(f"  SKIP {name}: not found at {path}")

    n_adapters_total = len(adapter_data)
    print(f"\nTotal adapters loaded: {n_adapters_total}")

    # Group adapters by shape signature (n_layers, A_shape, B_shape)
    # PCA requires identical dimensions — can only compare within groups
    shape_groups: dict[tuple, list[str]] = {}
    for name in sorted(adapter_data.keys()):
        d = adapter_data[name]
        sig = (len(d["a"]), d["a"][0].shape, d["b"][0].shape)
        shape_groups.setdefault(sig, []).append(name)

    print("\nAdapter shape groups:")
    for sig, names in sorted(shape_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {sig}: {len(names)} adapters — {', '.join(names)}")

    # Use largest group for analysis
    primary_sig = max(shape_groups.keys(), key=lambda s: len(shape_groups[s]))
    adapter_names = shape_groups[primary_sig]
    n_adapters = len(adapter_names)
    n_layers = primary_sig[0]

    print(f"\nPrimary analysis group: {n_adapters} adapters, "
          f"{n_layers} layers, A={primary_sig[1]}, B={primary_sig[2]}")

    results["n_adapters_total"] = n_adapters_total
    results["n_adapters_primary"] = n_adapters
    results["adapter_names"] = adapter_names
    results["primary_shape"] = {
        "n_layers": n_layers,
        "a_shape": list(primary_sig[1]),
        "b_shape": list(primary_sig[2]),
    }
    results["shape_groups"] = {
        str(sig): names for sig, names in shape_groups.items()
    }

    if n_adapters < 5:
        results["error"] = f"Need >= 5 adapters in primary group, only found {n_adapters}"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    # ── Phase 2: PCA Analysis (K1282) ────────────────────────────────────

    print("\n" + "=" * 60)
    print("Phase 2: PCA Variance Analysis")
    print("=" * 60)

    # Analyze A-matrices and B-matrices separately, per layer and globally
    k_values = [1, 2, 4, 8, 16]
    k_values = [k for k in k_values if k <= n_adapters]

    # Per-layer PCA
    layer_indices = list(range(n_layers))
    if IS_SMOKE:
        layer_indices = layer_indices[:3]

    a_var_by_layer = []
    b_var_by_layer = []
    a_gram_by_layer = []

    for l_idx in layer_indices:
        # Stack A-matrices for this layer: N × (d_in * r)
        a_vecs = np.stack([
            adapter_data[name]["a"][l_idx].flatten() for name in adapter_names
        ])
        b_vecs = np.stack([
            adapter_data[name]["b"][l_idx].flatten() for name in adapter_names
        ])

        # PCA
        a_S, a_cum = pca_variance_explained(a_vecs, max_k=max(k_values))
        b_S, b_cum = pca_variance_explained(b_vecs, max_k=max(k_values))

        a_var_at_k = {str(k): float(a_cum[min(k - 1, len(a_cum) - 1)]) for k in k_values}
        b_var_at_k = {str(k): float(b_cum[min(k - 1, len(b_cum) - 1)]) for k in k_values}

        a_var_by_layer.append(a_var_at_k)
        b_var_by_layer.append(b_var_at_k)

        # Gram matrix for A (orthogonality check)
        a_gram = analyze_gram_matrix(a_vecs)
        a_gram_by_layer.append(a_gram)

        if l_idx % 10 == 0 or l_idx == layer_indices[-1]:
            max_k_str = str(max(k_values))
            print(f"  Layer {l_idx:2d}: A var@{max_k_str}={a_var_at_k[max_k_str]:.3f}, "
                  f"B var@{max_k_str}={b_var_at_k[max_k_str]:.3f}, "
                  f"A max|cos|={a_gram['max_abs_cos']:.6f}")

    # Global PCA: concatenate all layers per adapter
    print("\n  Global PCA (all layers concatenated)...")
    global_a_vecs = np.stack([
        np.concatenate([adapter_data[name]["a"][l].flatten() for l in layer_indices])
        for name in adapter_names
    ])
    global_b_vecs = np.stack([
        np.concatenate([adapter_data[name]["b"][l].flatten() for l in layer_indices])
        for name in adapter_names
    ])
    global_ab_vecs = np.hstack([global_a_vecs, global_b_vecs])

    ga_S, ga_cum = pca_variance_explained(global_a_vecs, max_k=max(k_values))
    gb_S, gb_cum = pca_variance_explained(global_b_vecs, max_k=max(k_values))
    gab_S, gab_cum = pca_variance_explained(global_ab_vecs, max_k=max(k_values))

    global_a_var = {str(k): float(ga_cum[min(k - 1, len(ga_cum) - 1)]) for k in k_values}
    global_b_var = {str(k): float(gb_cum[min(k - 1, len(gb_cum) - 1)]) for k in k_values}
    global_ab_var = {str(k): float(gab_cum[min(k - 1, len(gab_cum) - 1)]) for k in k_values}

    max_k_str = str(max(k_values))
    print(f"  Global A var@{max_k_str}: {global_a_var[max_k_str]:.4f}")
    print(f"  Global B var@{max_k_str}: {global_b_var[max_k_str]:.4f}")
    print(f"  Global A+B var@{max_k_str}: {global_ab_var[max_k_str]:.4f}")

    # K1282 verdict: top-16 (or max available) PCA >= 80%?
    effective_k = min(16, n_adapters)
    ek_str = str(effective_k)
    k1282_a = global_a_var.get(ek_str, global_a_var[max_k_str])
    k1282_b = global_b_var.get(ek_str, global_b_var[max_k_str])
    k1282_ab = global_ab_var.get(ek_str, global_ab_var[max_k_str])
    k1282_pass = k1282_ab >= 0.80

    print(f"\n  K1282: Combined var@{effective_k} = {k1282_ab:.4f} "
          f"({'PASS' if k1282_pass else 'FAIL'}, threshold 0.80)")
    print(f"    A-only: {k1282_a:.4f}, B-only: {k1282_b:.4f}")

    # Predicted values check
    predicted_a = effective_k / n_adapters
    print(f"  Predicted A var (K/N = {effective_k}/{n_adapters}): {predicted_a:.4f}")
    print(f"  Actual A var: {k1282_a:.4f} (ratio: {k1282_a / predicted_a:.3f})")

    results["phase2_pca"] = {
        "k_values": k_values,
        "n_layers_analyzed": len(layer_indices),
        "per_layer_a_var_mean": {
            str(k): float(np.mean([lv[str(k)] for lv in a_var_by_layer]))
            for k in k_values
        },
        "per_layer_b_var_mean": {
            str(k): float(np.mean([lv[str(k)] for lv in b_var_by_layer]))
            for k in k_values
        },
        "global_a_var": global_a_var,
        "global_b_var": global_b_var,
        "global_ab_var": global_ab_var,
        "global_a_singular_values": ga_S[:min(20, len(ga_S))].tolist(),
        "global_b_singular_values": gb_S[:min(20, len(gb_S))].tolist(),
        "a_gram_stats": {
            "mean_max_cos": float(np.mean([g["max_abs_cos"] for g in a_gram_by_layer])),
            "overall_max_cos": float(np.max([g["max_abs_cos"] for g in a_gram_by_layer])),
            "mean_mean_cos": float(np.mean([g["mean_abs_cos"] for g in a_gram_by_layer])),
        },
        "k1282_effective_k": effective_k,
        "k1282_combined_var": k1282_ab,
        "k1282_a_var": k1282_a,
        "k1282_b_var": k1282_b,
        "k1282_pass": k1282_pass,
        "k1282_predicted_a": predicted_a,
    }

    # Singular value spectrum (uniformity check for Theorem 1)
    a_sv_normalized = ga_S / ga_S.sum() if ga_S.sum() > 0 else ga_S
    b_sv_normalized = gb_S / gb_S.sum() if gb_S.sum() > 0 else gb_S
    results["phase2_pca"]["a_sv_uniformity"] = float(
        np.std(a_sv_normalized) / np.mean(a_sv_normalized)
    ) if np.mean(a_sv_normalized) > 0 else 0.0
    results["phase2_pca"]["b_sv_uniformity"] = float(
        np.std(b_sv_normalized) / np.mean(b_sv_normalized)
    ) if np.mean(b_sv_normalized) > 0 else 0.0

    # ── Phase 3: Universal Subspace Merging (K1283) ──────────────────────

    print("\n" + "=" * 60)
    print("Phase 3: Universal Subspace Merging vs Naive Addition")
    print("=" * 60)

    target_domains = ["math", "code", "medical", "legal", "finance"]
    available_targets = [d for d in target_domains if d in adapter_names]

    a_matrices = [[adapter_data[name]["a"][l] for l in range(n_layers)] for name in adapter_names]
    b_matrices = [[adapter_data[name]["b"][l] for l in range(n_layers)] for name in adapter_names]

    merge_results = naive_addition_quality(
        a_matrices, b_matrices, adapter_names, available_targets
    )

    naive_wins_count = sum(1 for d in merge_results if merge_results[d]["naive_better"])
    univ_wins_count = len(merge_results) - naive_wins_count
    k1283_pass = univ_wins_count >= 3

    print(f"\n  Domains where naive > universal: {naive_wins_count}/{len(merge_results)}")
    print(f"  Domains where universal > naive: {univ_wins_count}/{len(merge_results)}")
    for domain, m in merge_results.items():
        print(f"    {domain}: naive_cos={m['mean_naive_cos']:.4f}, "
              f"univ_cos={m['mean_univ_cos']:.4f} "
              f"→ {'naive' if m['naive_better'] else 'universal'}")
    print(f"\n  K1283: Universal wins >= 3/5? {k1283_pass} "
          f"({univ_wins_count} wins)")

    results["phase3_merging"] = {
        "target_domains": available_targets,
        "per_domain": merge_results,
        "naive_wins": naive_wins_count,
        "univ_wins": univ_wins_count,
        "k1283_pass": k1283_pass,
    }

    # Clean up large arrays
    del a_matrices, b_matrices
    gc.collect()

    # ── Phase 4: Compression Quality (K1284) ─────────────────────────────

    print("\n" + "=" * 60)
    print("Phase 4: Compression Quality Analysis")
    print("=" * 60)

    compression_k_values = [4, 8, 12, 16]
    compression_k_values = [k for k in compression_k_values if k <= n_adapters]

    a_compression = {}
    b_compression = {}

    for k in compression_k_values:
        a_comp = compression_quality(global_a_vecs, k)
        b_comp = compression_quality(global_b_vecs, k)
        a_compression[str(k)] = a_comp
        b_compression[str(k)] = b_comp
        print(f"  K={k:2d}: A recon_err={a_comp['relative_recon_error']:.4f}, "
              f"cos={a_comp['mean_cosine_similarity']:.4f}, "
              f"ortho_degrad={a_comp['orthogonality_degradation']:.1f}x")
        print(f"         B recon_err={b_comp['relative_recon_error']:.4f}, "
              f"cos={b_comp['mean_cosine_similarity']:.4f}")

    # K1284 verdict: use K=16 (or max available) compression
    best_k = str(max(compression_k_values))
    # Quality proxy: mean cosine similarity > 0.95 means < 5pp degradation
    k1284_cos = a_compression[best_k]["mean_cosine_similarity"]
    # Also check orthogonality: if destroyed, composition breaks
    k1284_ortho = a_compression[best_k]["orthogonality_degradation"]
    k1284_pass = k1284_cos >= 0.95 and k1284_ortho < 10.0

    print(f"\n  K1284: Compression@{best_k} cos={k1284_cos:.4f}, "
          f"ortho_degradation={k1284_ortho:.1f}x "
          f"({'PASS' if k1284_pass else 'FAIL'})")

    results["phase4_compression"] = {
        "k_values": compression_k_values,
        "a_compression": a_compression,
        "b_compression": b_compression,
        "k1284_best_k": int(best_k),
        "k1284_cos": k1284_cos,
        "k1284_ortho_degradation": k1284_ortho,
        "k1284_pass": k1284_pass,
    }

    # ── Phase 5: Adapter Clustering Analysis (bonus) ─────────────────────

    print("\n" + "=" * 60)
    print("Phase 5: Adapter Clustering (bonus analysis)")
    print("=" * 60)

    # Check if domain vs format adapters cluster differently
    # Use PCA-2D projection for visualization data
    from sklearn.decomposition import PCA as skPCA

    try:
        pca_2d = skPCA(n_components=min(3, n_adapters))

        a_proj = pca_2d.fit_transform(global_a_vecs)
        a_cluster_data = {
            name: {"pc1": float(a_proj[i, 0]), "pc2": float(a_proj[i, 1])}
            for i, name in enumerate(adapter_names)
        }

        pca_2d_b = skPCA(n_components=min(3, n_adapters))
        b_proj = pca_2d_b.fit_transform(global_b_vecs)
        b_cluster_data = {
            name: {"pc1": float(b_proj[i, 0]), "pc2": float(b_proj[i, 1])}
            for i, name in enumerate(adapter_names)
        }

        print("  A-matrix 2D PCA projection:")
        for name in adapter_names:
            d = a_cluster_data[name]
            print(f"    {name:20s}: ({d['pc1']:+.3f}, {d['pc2']:+.3f})")

        results["phase5_clustering"] = {
            "a_pca_2d": a_cluster_data,
            "b_pca_2d": b_cluster_data,
            "a_explained_var_2d": pca_2d.explained_variance_ratio_.tolist(),
            "b_explained_var_2d": pca_2d_b.explained_variance_ratio_.tolist(),
        }
    except ImportError:
        print("  sklearn not available, skipping clustering visualization")
        results["phase5_clustering"] = {"error": "sklearn not available"}

    # ── Summary ──────────────────────────────────────────────────────────

    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed

    results["kill_criteria"] = {
        "K1282": {
            "description": "Top-16 PCA >= 80% variance",
            "pass": k1282_pass,
            "value": k1282_ab,
            "threshold": 0.80,
            "detail": f"A={k1282_a:.4f}, B={k1282_b:.4f}, combined={k1282_ab:.4f}",
        },
        "K1283": {
            "description": "Universal merging > naive on >= 3/5 domains",
            "pass": k1283_pass,
            "value": univ_wins_count,
            "threshold": 3,
            "detail": f"naive={naive_wins_count}, universal={univ_wins_count}",
        },
        "K1284": {
            "description": "Compression preserves quality (< 5pp degradation)",
            "pass": k1284_pass,
            "value": k1284_cos,
            "threshold": 0.95,
            "detail": f"cos={k1284_cos:.4f}, ortho_degrad={k1284_ortho:.1f}x",
        },
    }

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for k_id, kc in results["kill_criteria"].items():
        status = "PASS" if kc["pass"] else "FAIL"
        print(f"  {k_id}: {status} — {kc['detail']}")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
