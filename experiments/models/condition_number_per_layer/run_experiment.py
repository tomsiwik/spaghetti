#!/usr/bin/env python3
"""Measure condition number κ(W) per layer for Qwen3-0.6B-4bit.

Kill criteria:
  K942: All layers have finite kappa (no degenerate layers)
  K943: Mean kappa > 200 (KILL: promotion fundamentally unsafe)

Method: Gram matrix eigendecomposition — for W ∈ R^(m×n):
  G = Wᵀ W (if m≥n) or W Wᵀ (if m<n)
  κ(W) = sqrt(λ_max(G) / λ_min(G))

References:
  BitNet b1.58 (Ma et al., arXiv 2402.17764) — ternary weight quantization
  Stewart (1973) — matrix perturbation theory, condition number bounds
  Aghajanyan et al. (arXiv 2012.13255) — intrinsic dimensionality of trained weights

Supports SMOKE_TEST=1 for quick validation (first 3 layers only).
"""

import gc
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

# Suppress overflow warnings from float64 matmul on large matrices
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*matmul.*")

import mlx.core as mx
import mlx.nn as nn

# Memory safety — per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load as mlx_load

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
RESULTS_DIR = Path(__file__).parent
RESULTS_PATH = RESULTS_DIR / "results.json"

# Weight types to measure per attention layer
ATTN_WEIGHTS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_WEIGHTS = ["gate_proj", "up_proj", "down_proj"]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


def dequantize_weight(module) -> np.ndarray:
    """Extract float32 weight from a QuantizedLinear or Linear module."""
    if hasattr(module, "scales") and hasattr(module, "biases"):
        # QuantizedLinear — dequantize packed integers to float
        W_float = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            module.group_size,
            module.bits,
        )
    elif hasattr(module, "weight") and module.weight is not None:
        W_float = module.weight
    else:
        return None

    mx.eval(W_float)
    W_np = np.array(W_float.astype(mx.float32))
    # Check for NaN/Inf from bad dequantization
    if not np.isfinite(W_np).all():
        n_bad = np.sum(~np.isfinite(W_np))
        log(f"    WARNING: {n_bad}/{W_np.size} non-finite values in dequantized weight — replacing with 0")
        W_np = np.nan_to_num(W_np, nan=0.0, posinf=0.0, neginf=0.0)
    del W_float
    mx.clear_cache()
    return W_np


def compute_condition_number(W_np: np.ndarray) -> dict:
    """Compute condition number via Gram matrix eigendecomposition."""
    m, n = W_np.shape

    # Cast to float64 to avoid overflow/underflow in Gram matrix computation
    W64 = W_np.astype(np.float64)

    # Use smaller Gram matrix for efficiency
    if m >= n:
        G = W64.T @ W64  # (n, n)
        gram_dim = n
    else:
        G = W64 @ W64.T  # (m, m)
        gram_dim = m

    del W64

    # Symmetric eigendecomposition (faster than full SVD)
    eigvals = np.linalg.eigvalsh(G)  # sorted ascending

    # Eigenvalues should be non-negative (Gram matrix), clip numerical noise
    eigvals = np.clip(eigvals, 0, None)

    sigma_max = np.sqrt(eigvals[-1])
    sigma_min = np.sqrt(eigvals[0])

    if sigma_max < 1e-12:
        # Essentially zero matrix — degenerate
        return {
            "kappa": float("inf"),
            "sigma_max": 0.0,
            "sigma_min": 0.0,
            "rank_deficient": True,
            "gram_dim": gram_dim,
            "shape": list(W_np.shape),
        }

    if sigma_min < 1e-10:
        # Rank-deficient but non-zero
        kappa = float("inf")
    else:
        kappa = float(sigma_max / sigma_min)

    return {
        "kappa": kappa,
        "sigma_max": float(sigma_max),
        "sigma_min": float(sigma_min),
        "rank_deficient": sigma_min < 1e-10,
        "gram_dim": gram_dim,
        "shape": list(W_np.shape),
    }


def phase_measure_condition_numbers() -> dict:
    """Load model and measure condition numbers for all layers."""
    log(f"Loading {MODEL_ID}...")
    model, _tokenizer = mlx_load(MODEL_ID)
    layers = model.model.layers

    n_layers = len(layers)
    if SMOKE_TEST:
        n_layers = min(3, n_layers)
        log(f"SMOKE_TEST: measuring first {n_layers} layers only")

    log(f"Model loaded. Measuring {n_layers} layers × {len(ATTN_WEIGHTS)+len(MLP_WEIGHTS)} weight types...")

    all_results = []

    for layer_idx in range(n_layers):
        layer = layers[layer_idx]
        layer_results = {"layer_idx": layer_idx, "weights": {}}

        # Attention weights
        attn = layer.self_attn
        for wname in ATTN_WEIGHTS:
            module = getattr(attn, wname, None)
            if module is None:
                log(f"  L{layer_idx} {wname}: NOT FOUND, skipping")
                continue

            W_np = dequantize_weight(module)
            if W_np is None:
                log(f"  L{layer_idx} {wname}: no weight, skipping")
                continue

            stats = compute_condition_number(W_np)
            layer_results["weights"][wname] = stats

            stats["w_absmax"] = float(np.abs(W_np).max())
            stats["w_nnz_frac"] = float(np.count_nonzero(W_np) / W_np.size)
            kappa_str = f"{stats['kappa']:.1f}" if not np.isinf(stats['kappa']) else "∞"
            log(f"  L{layer_idx:02d} {wname:8s} {W_np.shape[0]}×{W_np.shape[1]:4d}  κ={kappa_str}  absmax={stats['w_absmax']:.4f}  nnz={stats['w_nnz_frac']:.3f}")
            del W_np

        # MLP weights
        mlp = layer.mlp
        for wname in MLP_WEIGHTS:
            module = getattr(mlp, wname, None)
            if module is None:
                log(f"  L{layer_idx} mlp.{wname}: NOT FOUND, skipping")
                continue

            W_np = dequantize_weight(module)
            if W_np is None:
                continue

            stats = compute_condition_number(W_np)
            stats["w_absmax"] = float(np.abs(W_np).max())
            stats["w_nnz_frac"] = float(np.count_nonzero(W_np) / W_np.size)
            kappa_str = f"{stats['kappa']:.1f}" if not np.isinf(stats['kappa']) else "∞"
            log(f"  L{layer_idx:02d} mlp.{wname:8s} {W_np.shape[0]}×{W_np.shape[1]:4d}  κ={kappa_str}  absmax={stats['w_absmax']:.4f}  nnz={stats['w_nnz_frac']:.3f}")
            del W_np

        all_results.append(layer_results)

        # Periodic memory cleanup
        if (layer_idx + 1) % 7 == 0:
            log(f"  [Memory] Active: {mx.get_active_memory()/1e9:.2f}GB  Cache: {mx.get_cache_memory()/1e9:.2f}GB")
            mx.clear_cache()

    cleanup(model)
    return {"layers": all_results}


def compute_summary(measurements: dict) -> dict:
    """Compute summary statistics and kill criterion results."""
    layers = measurements["layers"]

    all_kappas = []
    inf_count = 0
    kappas_by_type = {w: [] for w in ATTN_WEIGHTS + MLP_WEIGHTS}

    for layer_data in layers:
        for wname, stats in layer_data["weights"].items():
            kappa = stats["kappa"]
            if np.isinf(kappa):
                inf_count += 1
            else:
                all_kappas.append(kappa)
                if wname in kappas_by_type:
                    kappas_by_type[wname].append(kappa)

    n_total = len(all_kappas) + inf_count
    n_finite = len(all_kappas)

    # K942: All layers finite (no degenerate)
    k942_pass = inf_count == 0
    k942_result = f"{'PASS' if k942_pass else 'FAIL'}: {inf_count}/{n_total} inf kappas"

    # K943: Mean kappa > 200 → KILL
    mean_kappa = float(np.mean(all_kappas)) if all_kappas else float("inf")
    k943_killed = mean_kappa > 200
    k943_result = f"{'KILL' if k943_killed else 'PASS'}: mean κ = {mean_kappa:.1f} (threshold 200)"

    # Per-type statistics
    type_stats = {}
    for wname, kappas in kappas_by_type.items():
        if kappas:
            type_stats[wname] = {
                "mean": float(np.mean(kappas)),
                "median": float(np.median(kappas)),
                "p95": float(np.percentile(kappas, 95)),
                "max": float(np.max(kappas)),
                "min": float(np.min(kappas)),
                "n": len(kappas),
            }

    # Safety zone classification (from MATH.md calibration targets)
    if mean_kappa < 20:
        safety_zone = "SAFE_K5PLUS"
        safety_note = "κ < 20: promotion safe for K > 5 cycles"
    elif mean_kappa < 100:
        safety_zone = "SAFE_K5"
        safety_note = "κ < 100: promotion safe for K ≤ 5 cycles"
    elif mean_kappa < 200:
        safety_zone = "NEEDS_SCALE_REDUCTION"
        safety_note = "κ < 200: promotion requires scale reduction"
    else:
        safety_zone = "UNSAFE"
        safety_note = "κ ≥ 200: KILL — promotion fundamentally unsafe"

    return {
        "n_total": n_total,
        "n_finite": n_finite,
        "n_inf": inf_count,
        "mean_kappa": mean_kappa,
        "median_kappa": float(np.median(all_kappas)) if all_kappas else float("inf"),
        "p95_kappa": float(np.percentile(all_kappas, 95)) if all_kappas else float("inf"),
        "max_kappa": float(np.max(all_kappas)) if all_kappas else float("inf"),
        "min_kappa": float(np.min(all_kappas)) if all_kappas else float("inf"),
        "safety_zone": safety_zone,
        "safety_note": safety_note,
        "type_stats": type_stats,
        "k942_pass": k942_pass,
        "k942_result": k942_result,
        "k943_killed": k943_killed,
        "k943_result": k943_result,
    }


def main():
    log("=" * 60)
    log("Condition Number Measurement — Qwen3-0.6B-4bit")
    log("=" * 60)

    t0 = time.time()

    # Phase 1: Measure condition numbers
    measurements = phase_measure_condition_numbers()

    # Phase 2: Compute summary
    summary = compute_summary(measurements)

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.1f}s")

    # Print kill criterion results
    log("\n" + "=" * 60)
    log("KILL CRITERION RESULTS")
    log("=" * 60)
    log(f"K942: {summary['k942_result']}")
    log(f"K943: {summary['k943_result']}")
    log(f"\nSafety Zone: {summary['safety_zone']}")
    log(f"  {summary['safety_note']}")
    log(f"\nGlobal statistics (n={summary['n_finite']} finite, {summary['n_inf']} inf):")
    log(f"  mean  κ = {summary['mean_kappa']:.1f}")
    log(f"  median κ = {summary['median_kappa']:.1f}")
    log(f"  p95   κ = {summary['p95_kappa']:.1f}")
    log(f"  max   κ = {summary['max_kappa']:.1f}")
    log(f"  min   κ = {summary['min_kappa']:.1f}")
    log("\nPer-weight-type mean κ:")
    for wname, stats in summary["type_stats"].items():
        log(f"  {wname:10s}: mean={stats['mean']:.1f}  median={stats['median']:.1f}  max={stats['max']:.1f}")

    # Write results
    results = {
        "experiment": "exp_condition_number_per_layer",
        "model": MODEL_ID,
        "smoke_test": SMOKE_TEST,
        "elapsed_s": elapsed,
        "summary": summary,
        "measurements": measurements,
        "k942_pass": summary["k942_pass"],
        "k943_killed": summary["k943_killed"],
    }

    def _json_safe(obj):
        """Convert numpy types to Python native for JSON serialization."""
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, allow_nan=True, default=_json_safe))
    log(f"\nResults written to {RESULTS_PATH}")
    log("DONE")


if __name__ == "__main__":
    main()
