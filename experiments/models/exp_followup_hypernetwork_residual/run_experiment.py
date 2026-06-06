#!/usr/bin/env python3
"""Residual hypernetwork: B_pred = mean_B + delta(emb).

Pre-registered KCs (see MATH.md, locked at commit):

  K1  (primary, real-data): mean_t( PPL_residual(t) / PPL_mean_baseline(t) ) <= 0.95
  K2  (secondary, real):    median LOO cosine rho_t > 0.1 across 24 folds
  K3  (infra):              peak memory <= 40 GB
  K_vacate:                 if parent adapter artifacts missing ->
                            synthetic-proxy sub-test, verdict=provisional

Synthetic sub-KCs (activate on K_vacate, see MATH.md sec 6):
  K1s: mean_t( MSE_residual / MSE_baseline ) <= 0.95
  K2s: median rho_t > 0.1

Platform: Apple M5 Pro 48GB, MLX.  mlx-lm 0.31.1.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters" / "adapters"
SKELETON_PATH = ADAPTERS_DIR / "grassmannian_skeleton_n24.npz"

LORA_SCALE = 5.0  # antipattern-003 fix (parent used 20)
SEED = 42

DOMAINS = [
    "medical", "code", "math", "legal", "finance", "science",
    "history", "philosophy", "creative_writing", "cooking",
    "health_fitness", "psychology", "education", "engineering",
    "agriculture", "environmental", "politics", "economics",
    "sociology", "linguistics", "cybersecurity", "marketing",
    "sports", "music",
]

EVAL_DOMAINS = ["medical", "code", "math", "legal", "cooking", "sports"]


# ============================================================================
# Infrastructure probe
# ============================================================================
def probe_infrastructure() -> dict:
    """Detect presence of parent adapter artifacts (K_vacate trigger)."""
    present = {}
    missing = []
    for d in DOMAINS:
        p = ADAPTERS_DIR / d / "adapter.npz"
        present[d] = p.exists()
        if not p.exists():
            missing.append(d)
    skel_ok = SKELETON_PATH.exists()
    return {
        "adapters_dir": str(ADAPTERS_DIR),
        "adapters_present_count": sum(present.values()),
        "adapters_total": len(DOMAINS),
        "missing_domains": missing,
        "skeleton_present": skel_ok,
        "skeleton_path": str(SKELETON_PATH),
        "vacate_K1_K2": (sum(present.values()) < len(DOMAINS)) or (not skel_ok),
    }


# ============================================================================
# Synthetic-proxy sub-test
# ============================================================================
def _generate_synthetic(
    n_domains: int = 24,
    d_embed: int = 64,
    r: int = 16,
    d_out: int = 2560,
    k: int = 8,
    emb_noise: float = 0.1,
    adapter_noise: float = 0.01,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (emb_i, B_i_flat) pairs from a shared k-dim topic manifold."""
    rng = np.random.default_rng(seed)
    dim_b = r * d_out

    # Topic vectors (k-dim shared manifold)
    z = rng.standard_normal((n_domains, k))

    # Orthogonal projectors
    u_e, _ = np.linalg.qr(rng.standard_normal((d_embed, k)))
    u_b, _ = np.linalg.qr(rng.standard_normal((dim_b, k)))

    embeddings = z @ u_e.T + rng.standard_normal((n_domains, d_embed)) * emb_noise
    adapters = z @ u_b.T + rng.standard_normal((n_domains, dim_b)) * adapter_noise
    return embeddings.astype(np.float32), adapters.astype(np.float32)


def _train_ridge_hyper(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 1e-2,
) -> tuple[np.ndarray, float]:
    """Closed-form ridge regression: Y = X @ W + b.

    A linear hypernetwork (0-hidden-layer MLP, universal at linear
    scale). If ridge fails to recover the mapping under a known-linear
    generative model, the residual-hypernetwork mechanism cannot
    generalise at N=24 regardless of nonlinearity.

    Returns (prediction_on_X_test, train_MSE).
    """
    n, d_in = X_train.shape
    # Centre inputs; keep outputs as-is (intercept handled separately)
    x_mean = X_train.mean(0, keepdims=True)
    y_mean = Y_train.mean(0, keepdims=True)
    Xc = X_train - x_mean
    Yc = Y_train - y_mean

    # W = (Xc^T Xc + alpha I)^-1 Xc^T Yc
    gram = Xc.T @ Xc + alpha * np.eye(d_in)
    W = np.linalg.solve(gram, Xc.T @ Yc)
    # Predict on test
    pred = (X_test - x_mean) @ W + y_mean

    # Training MSE
    train_pred = Xc @ W + y_mean
    train_mse = float(np.mean((train_pred - Y_train) ** 2))
    return pred, train_mse


def run_synthetic_proxy(seed: int = SEED) -> dict:
    """LOO residual hypernetwork on synthetic structured data."""
    print("\n=== Synthetic-proxy sub-test (K_vacate branch) ===")
    t0 = time.time()
    X, Y = _generate_synthetic(seed=seed)
    n = X.shape[0]
    dim_b = Y.shape[1]
    print(f"  synth shapes: emb={X.shape} adapter={Y.shape}")

    cosines = []
    mse_ratios = []
    norm_ratios = []
    per_fold = []

    for t_idx in range(n):
        mask = np.ones(n, dtype=bool)
        mask[t_idx] = False
        X_train, X_test = X[mask], X[t_idx:t_idx + 1]
        Y_train, Y_test = Y[mask], Y[t_idx:t_idx + 1]

        mu = Y_train.mean(axis=0, keepdims=True)           # (1, dim_b)
        delta_train = Y_train - mu                          # true residuals
        delta_true = Y_test - mu                            # held-out truth

        # Normalise inputs/targets per-fold for numerical stability
        # (prediction remapped back below)
        x_mean = X_train.mean(0, keepdims=True)
        x_std = X_train.std(0, keepdims=True) + 1e-8
        X_train_n = (X_train - x_mean) / x_std
        X_test_n = (X_test - x_mean) / x_std

        d_t_std = float(delta_train.std()) + 1e-8
        d_train_n = delta_train / d_t_std

        # Linear hypernetwork via closed-form ridge on full-dim targets.
        # (Random projection would push the prediction into a k_out-dim
        # subspace near-orthogonal to the true rank-k signal subspace —
        # destructive. Direct full-dim ridge is tractable: 23 x 40960
        # targets, W ~ 64 x 40960 ~ 10 MB.)
        delta_pred_n, best_loss = _train_ridge_hyper(
            X_train_n, d_train_n, X_test_n, alpha=1e-2,
        )
        delta_pred = delta_pred_n * d_t_std

        # Metrics
        flat_true = delta_true.ravel()
        flat_pred = delta_pred.ravel()
        cos = float(
            np.dot(flat_true, flat_pred)
            / (np.linalg.norm(flat_true) * np.linalg.norm(flat_pred) + 1e-12)
        )
        cosines.append(cos)

        mse_baseline = float(np.mean((mu[0] - Y_test[0]) ** 2))            # eq (4)
        mse_residual = float(np.mean((mu[0] + delta_pred[0] - Y_test[0]) ** 2))  # eq (3)
        mse_ratio = mse_residual / (mse_baseline + 1e-12)
        mse_ratios.append(mse_ratio)

        norm_ratio = float(np.linalg.norm(flat_pred) / (np.linalg.norm(flat_true) + 1e-12))
        norm_ratios.append(norm_ratio)

        per_fold.append({
            "held_out": DOMAINS[t_idx],
            "rho": round(cos, 4),
            "mse_ratio": round(mse_ratio, 4),
            "sigma_hat_over_sigma": round(norm_ratio, 4),
            "train_loss": round(best_loss, 6),
        })
        if t_idx < 5 or t_idx == n - 1:
            print(f"  fold {t_idx:2d} ({DOMAINS[t_idx]:15s}): "
                  f"rho={cos:+.4f}  mse_ratio={mse_ratio:.4f}  "
                  f"sigma_ratio={norm_ratio:.4f}")

    median_rho = float(np.median(cosines))
    mean_mse_ratio = float(np.mean(mse_ratios))
    median_norm_ratio = float(np.median(norm_ratios))

    k1s_pass = mean_mse_ratio <= 0.95
    k2s_pass = median_rho > 0.1
    p4_pass = 0.5 <= median_norm_ratio <= 2.0

    elapsed = time.time() - t0
    print(f"  synth summary: median_rho={median_rho:.4f}  "
          f"mean_mse_ratio={mean_mse_ratio:.4f}  "
          f"median_sigma_ratio={median_norm_ratio:.4f}  "
          f"time={elapsed:.1f}s")

    return {
        "n_domains": n,
        "dim_b": dim_b,
        "median_rho": round(median_rho, 4),
        "mean_mse_ratio": round(mean_mse_ratio, 4),
        "median_sigma_hat_over_sigma": round(median_norm_ratio, 4),
        "per_fold": per_fold,
        "K1s_pass": k1s_pass,
        "K2s_pass": k2s_pass,
        "P4_pass": p4_pass,
        "time_s": round(elapsed, 1),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    print("=" * 70)
    print("exp_followup_hypernetwork_residual")
    print("=" * 70)

    # Phase 0: infrastructure probe (sets K_vacate)
    infra = probe_infrastructure()
    print(f"\n[infra] adapters present: {infra['adapters_present_count']}/{infra['adapters_total']}")
    print(f"[infra] skeleton present:  {infra['skeleton_present']}")
    print(f"[infra] K_vacate triggered: {infra['vacate_K1_K2']}")

    k1_result = None
    k2_result = None
    k3_peak_gb = None
    synth = None

    if infra["vacate_K1_K2"]:
        print("\n[K_vacate] Parent adapter artifacts missing -> running synthetic proxy.")
        synth = run_synthetic_proxy()
        verdict = "PROVISIONAL"
        # Synthetic verdict only drives the residual-mechanism claim, never
        # upgrades the real-data KCs.
        if synth["K1s_pass"] and synth["K2s_pass"] and synth["P4_pass"]:
            mechanism_verdict = "mechanism-plausible"
        else:
            mechanism_verdict = "mechanism-falsified"
    else:
        # Real-data path (not reachable in this run; kept for the future)
        print("\n[real-data] Adapter artifacts present; would run BitNet LOO here.")
        print("           (Skipping real-data path in this iteration — see NOTE.)")
        verdict = "INCONCLUSIVE"
        mechanism_verdict = "not-tested"

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s")

    # Antipattern self-check (mirrors MATH.md sec 5)
    antipattern_check = {
        "lora_scale": LORA_SCALE,                # 5.0, antipattern-003 fixed
        "tautological_routing": False,           # LOO holds out test domain
        "composition_bug": False,                # no composition in this exp
        "thinking_mode_truncation": False,       # no thinking channel
        "smoke_hardcoded_supported": False,      # KCs computed from data
        "shutil_copy_as_adapter": False,
        "hardcoded_pass_true": False,
        "copy_paste_domain_keywords": False,
    }

    results = {
        "experiment": "exp_followup_hypernetwork_residual",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_target": "microsoft/BitNet-b1.58-2B-4T",
        "mlx_lm_version_expected": "0.31.1",
        "lora_scale": LORA_SCALE,
        "seed": SEED,
        "is_smoke": False,
        "infra": infra,
        "K_vacate_active": bool(infra["vacate_K1_K2"]),
        "synthetic_proxy": synth,
        "K1": {"evaluated": not infra["vacate_K1_K2"],
               "pass": k1_result,
               "note": "Vacated — parent adapter artifacts missing" if infra["vacate_K1_K2"] else None},
        "K2": {"evaluated": not infra["vacate_K1_K2"],
               "pass": k2_result,
               "note": "Vacated — parent adapter artifacts missing" if infra["vacate_K1_K2"] else None},
        "K3": {"evaluated": False,
               "peak_memory_gb": k3_peak_gb,
               "note": "Not evaluated — real-data path vacated"},
        "P1_infra_missing": bool(infra["vacate_K1_K2"]),
        "P2_synth_mse_pass": (synth or {}).get("K1s_pass"),
        "P3_synth_rho_pass": (synth or {}).get("K2s_pass"),
        "P4_norm_bounded_pass": (synth or {}).get("P4_pass"),
        "P5_combined_synth_pass": (
            (synth or {}).get("K1s_pass") and (synth or {}).get("K2s_pass")
            if synth else None
        ),
        "mechanism_verdict": mechanism_verdict,
        "verdict": verdict,
        "all_pass": None,
        "antipattern_check": antipattern_check,
        "total_time_s": round(total_time, 1),
        "blocker_followup_tasks": [
            "Regenerate 24 adapters under exp_real_data_25_domain_adapters at LORA_SCALE=5 "
            "to unblock K1/K2 on BitNet-2B-4T."
        ] if infra["vacate_K1_K2"] else [],
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Verdict: {verdict}  (mechanism: {mechanism_verdict})")


if __name__ == "__main__":
    main()
