"""E2: Null-Space Composition Theorem — Grassmannian adapters in base model null space.

Measures what fraction of Grassmannian adapter subspace lies in the null space
of base model weight matrices. For v_proj (512×2560), the null space is at least
80% of R^2560 by rank deficiency alone. The question is whether effective rank
reduction pushes this even higher.

Phases:
1. Effective rank measurement (SVD per layer)
2. Projection residual (Grassmannian A onto row space of W)
3. Composition quality test (tau with/without null-space projection)
"""

import json
import os
import gc
import time
import sys

import mlx.core as mx
import mlx.nn as nn

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
RANK = 6
N_ADAPTERS = 5
EPS_THRESHOLDS = [0.01, 0.001]
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

if SMOKE_TEST:
    LAYER_INDICES = [0, 20, 41]
    N_PROMPTS = 3
else:
    LAYER_INDICES = list(range(42))
    N_PROMPTS = 10


def load_model():
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    return model, tokenizer


def dequantize_weight(qlinear):
    W = mx.dequantize(qlinear.weight, qlinear.scales, qlinear.biases,
                      qlinear.group_size, qlinear.bits)
    mx.eval(W)
    return W


def grassmannian_A(d_in, rank, n_adapters, seed=42):
    """Partition QR construction for N orthogonal A matrices."""
    key = mx.random.key(seed)
    W_random = mx.random.normal(key=key, shape=(d_in, n_adapters * rank))
    mx.eval(W_random)
    Q, _ = mx.linalg.qr(W_random, stream=mx.cpu)
    mx.eval(Q)
    A_matrices = []
    for i in range(n_adapters):
        A_i = Q[:, i * rank:(i + 1) * rank].T
        mx.eval(A_i)
        A_matrices.append(A_i)
    return A_matrices


def compute_effective_rank_and_null_frac(W, A_matrices, eps_thresholds):
    """Compute effective rank of W and null-space fraction for each A."""
    W_f32 = W.astype(mx.float32)
    mx.eval(W_f32)

    U, S, Vt = mx.linalg.svd(W_f32, stream=mx.cpu)
    mx.eval(S)

    s_max = S[0].item()
    m, d = W.shape

    result = {
        "shape": [int(m), int(d)],
        "rank_bound": min(int(m), int(d)),
        "s_max": float(s_max),
        "s_min": float(S[min(m, d) - 1].item()),
        "effective_ranks": {},
        "null_fracs": {},
    }

    for eps in eps_thresholds:
        threshold = eps * s_max
        r_eff = int((S > threshold).sum().item())
        result["effective_ranks"][str(eps)] = r_eff

        Vt_row = Vt[:r_eff, :]
        mx.eval(Vt_row)

        fracs = []
        for A_i in A_matrices:
            A_f32 = A_i.astype(mx.float32)
            proj = Vt_row @ A_f32.T
            mx.eval(proj)
            residual_sq = (proj * proj).sum().item()
            total_sq = (A_f32 * A_f32).sum().item()
            row_frac = residual_sq / total_sq
            null_frac = 1.0 - row_frac
            fracs.append(null_frac)

        result["null_fracs"][str(eps)] = {
            "mean": sum(fracs) / len(fracs),
            "min": min(fracs),
            "max": max(fracs),
            "per_adapter": fracs,
        }

    del U, Vt, W_f32
    mx.clear_cache()
    gc.collect()

    return result, S


def phase1_effective_rank(model):
    """Phase 1: Measure effective rank and null-space fraction per layer."""
    print("\n=== PHASE 1: Effective Rank & Null-Space Fraction ===")
    lm = model.language_model

    all_results = {}
    for proj_name in ["v_proj", "o_proj"]:
        proj_results = {}
        for layer_idx in LAYER_INDICES:
            layer = lm.model.layers[layer_idx]
            qlinear = getattr(layer.self_attn, proj_name)
            W = dequantize_weight(qlinear)

            d_in = W.shape[1]
            A_matrices = grassmannian_A(d_in, RANK, N_ADAPTERS, seed=42 + layer_idx)

            result, _ = compute_effective_rank_and_null_frac(W, A_matrices, EPS_THRESHOLDS)
            proj_results[str(layer_idx)] = result

            nf_001 = result["null_fracs"]["0.01"]["mean"]
            r_eff_001 = result["effective_ranks"]["0.01"]
            print(f"  Layer {layer_idx:2d} {proj_name}: shape={result['shape']}, "
                  f"r_eff(0.01)={r_eff_001}, null_frac={nf_001:.4f}")

            del W
            mx.clear_cache()
            gc.collect()

        all_results[proj_name] = proj_results

    return all_results


def phase2_projection_summary(phase1_results):
    """Phase 2: Aggregate projection residual statistics."""
    print("\n=== PHASE 2: Projection Residual Summary ===")
    summary = {}

    for proj_name, proj_data in phase1_results.items():
        null_fracs_001 = [v["null_fracs"]["0.01"]["mean"] for v in proj_data.values()]
        row_fracs_001 = [1.0 - nf for nf in null_fracs_001]

        mean_row_frac = sum(row_fracs_001) / len(row_fracs_001)
        mean_null_frac = sum(null_fracs_001) / len(null_fracs_001)

        summary[proj_name] = {
            "mean_null_frac_eps001": mean_null_frac,
            "mean_row_frac_eps001": mean_row_frac,
            "min_null_frac": min(null_fracs_001),
            "max_null_frac": max(null_fracs_001),
            "K2020_residual": mean_row_frac,
            "K2020_threshold": 0.05,
            "K2020_pass": mean_row_frac <= 0.05,
        }

        print(f"  {proj_name}: mean_null_frac={mean_null_frac:.4f}, "
              f"mean_row_frac={mean_row_frac:.4f}, "
              f"K2020={'PASS' if mean_row_frac <= 0.05 else 'FAIL'}")

    return summary


def phase3_composition_tau(model, tokenizer):
    """Phase 3: Measure composition residual tau with standard vs null-projected adapters."""
    print("\n=== PHASE 3: Composition Quality (tau) ===")
    lm = model.language_model

    prompts = [
        "Explain the concept of recursion in programming.",
        "What is the capital of France?",
        "Solve: 2x + 3 = 7",
        "Write a Python function to sort a list.",
        "What are the symptoms of diabetes?",
        "Describe the process of photosynthesis.",
        "What is machine learning?",
        "How does TCP/IP work?",
        "What is the Pythagorean theorem?",
        "Explain object-oriented programming.",
    ][:N_PROMPTS]

    test_layer = 20
    proj_name = "v_proj"

    layer = lm.model.layers[test_layer]
    qlinear = getattr(layer.self_attn, proj_name)
    W = dequantize_weight(qlinear)
    d_in = W.shape[1]

    A_matrices = grassmannian_A(d_in, RANK, 3, seed=42)

    key = mx.random.key(99)
    B_matrices = []
    for i in range(3):
        B_i = mx.random.normal(key=mx.random.split(key, 2)[0],
                               shape=(W.shape[0], RANK)) * 0.01
        mx.eval(B_i)
        B_matrices.append(B_i)
        key = mx.random.split(key, 2)[1]

    W_f32 = W.astype(mx.float32)
    U, S, Vt = mx.linalg.svd(W_f32, stream=mx.cpu)
    mx.eval(S, Vt)
    s_max = S[0].item()
    r_eff = int((S > 0.01 * s_max).sum().item())
    Vt_row = Vt[:r_eff, :]
    Vt_null = Vt[r_eff:, :]
    mx.eval(Vt_row, Vt_null)
    del U, S, W_f32
    mx.clear_cache()

    A_null_projected = []
    for A_i in A_matrices:
        A_f32 = A_i.astype(mx.float32)
        A_null = A_f32 @ Vt_null.T @ Vt_null
        mx.eval(A_null)
        norms = mx.linalg.norm(A_null, axis=1, keepdims=True)
        mx.eval(norms)
        norms = mx.maximum(norms, 1e-8)
        A_null_normed = A_null / norms * mx.linalg.norm(A_f32, axis=1, keepdims=True)
        mx.eval(A_null_normed)
        A_null_projected.append(A_null_normed.astype(A_i.dtype))
        del A_f32, A_null, norms, A_null_normed
    mx.clear_cache()

    del Vt_row, Vt_null, Vt
    mx.clear_cache()
    gc.collect()

    def compute_delta_W(B_list, A_list):
        dW = None
        for B_i, A_i in zip(B_list, A_list):
            term = B_i @ A_i
            mx.eval(term)
            dW = term if dW is None else dW + term
        mx.eval(dW)
        return dW

    dW_standard = compute_delta_W(B_matrices, A_matrices)
    dW_null = compute_delta_W(B_matrices, A_null_projected)

    taus_standard = []
    taus_null = []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        x = mx.array([tokens])
        mx.eval(x)

        h_base_list = []
        h_std_list = []
        h_null_list = []

        original_weight_q = qlinear.weight
        original_scales = qlinear.scales
        original_biases = qlinear.biases

        h_base = W @ mx.ones((d_in,), dtype=W.dtype)
        mx.eval(h_base)

        h_std = (W + dW_standard) @ mx.ones((d_in,), dtype=W.dtype)
        mx.eval(h_std)

        h_null = (W + dW_null) @ mx.ones((d_in,), dtype=W.dtype)
        mx.eval(h_null)

        individual_deltas_std = []
        individual_deltas_null = []
        for i in range(3):
            dW_i_std = B_matrices[i] @ A_matrices[i]
            dW_i_null = B_matrices[i] @ A_null_projected[i]
            mx.eval(dW_i_std, dW_i_null)

            h_i_std = (W + dW_i_std) @ mx.ones((d_in,), dtype=W.dtype)
            h_i_null = (W + dW_i_null) @ mx.ones((d_in,), dtype=W.dtype)
            mx.eval(h_i_std, h_i_null)

            individual_deltas_std.append(h_i_std - h_base)
            individual_deltas_null.append(h_i_null - h_base)

        sum_deltas_std = individual_deltas_std[0] + individual_deltas_std[1] + individual_deltas_std[2]
        sum_deltas_null = individual_deltas_null[0] + individual_deltas_null[1] + individual_deltas_null[2]
        mx.eval(sum_deltas_std, sum_deltas_null)

        h_additive_std = h_base + sum_deltas_std
        h_additive_null = h_base + sum_deltas_null
        mx.eval(h_additive_std, h_additive_null)

        residual_std = h_std - h_additive_std
        residual_null = h_null - h_additive_null
        mx.eval(residual_std, residual_null)

        r_norm_std = mx.linalg.norm(residual_std).item()
        d_norm_std = mx.linalg.norm(sum_deltas_std).item()
        r_norm_null = mx.linalg.norm(residual_null).item()
        d_norm_null = mx.linalg.norm(sum_deltas_null).item()

        tau_std = r_norm_std / max(d_norm_std, 1e-12)
        tau_null = r_norm_null / max(d_norm_null, 1e-12)
        taus_standard.append(tau_std)
        taus_null.append(tau_null)

        del h_base, h_std, h_null, residual_std, residual_null
        mx.clear_cache()

    mean_tau_std = sum(taus_standard) / len(taus_standard)
    mean_tau_null = sum(taus_null) / len(taus_null)
    tau_reduction = (mean_tau_std - mean_tau_null) / max(mean_tau_std, 1e-12)

    print(f"  tau (standard Grassmannian): {mean_tau_std:.6f}")
    print(f"  tau (null-projected):        {mean_tau_null:.6f}")
    print(f"  tau reduction:               {tau_reduction:.4f} ({tau_reduction*100:.1f}%)")

    k2021_pass = tau_reduction > 0.20

    return {
        "mean_tau_standard": mean_tau_std,
        "mean_tau_null_projected": mean_tau_null,
        "tau_reduction_fraction": tau_reduction,
        "taus_standard": taus_standard,
        "taus_null_projected": taus_null,
        "test_layer": test_layer,
        "proj_name": proj_name,
        "n_adapters": 3,
        "K2021_pass": k2021_pass,
        "K2021_threshold": 0.20,
    }


def main():
    print(f"E2: Null-Space Composition Theorem")
    print(f"Model: {MODEL_ID}")
    print(f"Smoke test: {SMOKE_TEST}")
    print(f"Layers: {LAYER_INDICES}")
    start_time = time.time()

    import mlx_lm
    mlx_lm_version = getattr(mlx_lm, "__version__", "unknown")
    print(f"mlx-lm version: {mlx_lm_version}")

    model, tokenizer = load_model()

    phase1_results = phase1_effective_rank(model)

    phase2_summary = phase2_projection_summary(phase1_results)

    mx.clear_cache()
    gc.collect()

    phase3_results = phase3_composition_tau(model, tokenizer)

    k2020_pass_v = phase2_summary.get("v_proj", {}).get("K2020_pass", False)
    k2020_pass_o = phase2_summary.get("o_proj", {}).get("K2020_pass", False)
    k2020_pass = k2020_pass_v and k2020_pass_o
    k2021_pass = phase3_results["K2021_pass"]

    all_pass = k2020_pass and k2021_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"

    if SMOKE_TEST:
        verdict = "PROVISIONAL"
        all_pass = None

    elapsed = time.time() - start_time

    results = {
        "experiment": "exp_e2_null_space_composition_theorem",
        "model": MODEL_ID,
        "mlx_lm_version": mlx_lm_version,
        "smoke_test": SMOKE_TEST,
        "elapsed_seconds": elapsed,
        "phase1_effective_rank": phase1_results,
        "phase2_projection_summary": phase2_summary,
        "phase3_composition_tau": phase3_results,
        "kill_criteria": {
            "K2020": {
                "description": "Mean projection residual (row-space fraction) across layers <= 0.05",
                "v_proj_pass": k2020_pass_v,
                "o_proj_pass": k2020_pass_o,
                "pass": k2020_pass,
            },
            "K2021": {
                "description": "Null-space projection reduces tau by >20%",
                "pass": k2021_pass,
                "tau_standard": phase3_results["mean_tau_standard"],
                "tau_null": phase3_results["mean_tau_null_projected"],
                "reduction": phase3_results["tau_reduction_fraction"],
            },
        },
        "all_pass": all_pass,
        "verdict": verdict,
        "is_smoke": SMOKE_TEST,
    }

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {results_path}")
    print(f"\nVerdict: {verdict}")
    print(f"K2020 (null-space): {'PASS' if k2020_pass else 'FAIL'}")
    print(f"K2021 (tau reduction): {'PASS' if k2021_pass else 'FAIL'}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
