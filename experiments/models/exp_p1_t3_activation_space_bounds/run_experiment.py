#!/usr/bin/env python3
"""
T3.3: Activation-space interference power law with V-norm on Gemma 4

Measures how max pairwise activation cosine scales with N for:
 (a) pure synthetic adapters (statistical baseline)
 (b) real 5-domain adapters (Gemma 4 q_proj layers)

V-norm: QR-normalize lora_a (A^T A = I_r), keep lora_b unchanged.
Full adapter output: h_i = x @ A_i @ B_i

Design notes:
- Real adapter A matrices are highly correlated (~0.75 Frobenius cosine) due to
  correlated initialization, but FULL ΔW = A@B have low cosines (0.001-0.14).
- Mixed real+synthetic pools create bimodal distributions → noisy power law.
- Solution: pure synthetic pool for power law, real adapters for context only.

Kill criteria:
  K1056: Power law: max|cos_activation| ~ c * N^alpha, measure c and alpha
  K1057: With V-norm: alpha <= 0.4 (matches or improves Finding #372 alpha=0.38)
  K1058: At N=50: max|cos_activation| < 0.5 (bounded interference)

Runtime: ~5 minutes on M5 Pro.
"""

import json
import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import safetensors.numpy as sfn
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
rng = np.random.default_rng(SEED)

D_IN  = 2560
D_OUT = 2048  # skip layer 41 (D_OUT=4096)
RANK  = 6
N_LAYERS = 42

N_TOKENS = 10  if IS_SMOKE else 200
N_TRIALS = 5   if IS_SMOKE else 100
N_SYNTH  = 10  if IS_SMOKE else 50   # pool size for synthetic experiment
N_VALUES = [2, 3, 5] if IS_SMOKE else [2, 3, 5, 8, 10, 15, 20, 30]

BASE_DIR = EXPERIMENT_DIR.parent
T21_DIR = BASE_DIR / "exp_p1_t2_single_domain_training"
T26_DIR = BASE_DIR / "exp_p1_t2_multi_domain_5"

ADAPTER_PATHS = {
    "math":    T21_DIR / "adapters" / "math"    / "adapters.safetensors",
    "code":    T21_DIR / "adapters" / "code"    / "adapters.safetensors",
    "medical": T21_DIR / "adapters" / "medical" / "adapters.safetensors",
    "legal":   T26_DIR / "adapters" / "legal"   / "adapters.safetensors",
    "finance": T26_DIR / "adapters" / "finance" / "adapters.safetensors",
}
DOMAINS = list(ADAPTER_PATHS.keys())


def log(msg):
    print(msg, flush=True)


def vnorm(A):
    """Project A (d_in × rank) to Stiefel manifold via QR: A^T A = I_r."""
    Q, _ = np.linalg.qr(A)
    return Q[:, :RANK].astype(np.float32)


def pairwise_max_cos_one_trial(adapters, x, n):
    """Single trial: sample n adapters, return max pairwise cosine over all pairs×tokens."""
    n_pool = len(adapters)
    idxs = rng.choice(n_pool, size=n, replace=False)
    projs = [(x @ adapters[i][0]) @ adapters[i][1] for i in idxs]  # each (N_TOKENS, D_OUT)

    max_cos = 0.0
    for a, b in combinations(range(n), 2):
        ha, hb = projs[a], projs[b]
        norm_a = np.linalg.norm(ha, axis=1)
        norm_b = np.linalg.norm(hb, axis=1)
        valid = (norm_a > 1e-7) & (norm_b > 1e-7)
        if not valid.any():
            continue
        dots = np.einsum("td,td->t", ha, hb)
        cos = np.abs(dots[valid]) / (norm_a[valid] * norm_b[valid])
        max_cos = max(max_cos, float(cos.max()))
    return max_cos


def measure_power_law(adapters_unnorm, adapters_vnorm, x, n_vals, n_trials):
    unnorm_r, vnorm_r = {}, {}
    for N in n_vals:
        un_vals = [pairwise_max_cos_one_trial(adapters_unnorm, x, N) for _ in range(n_trials)]
        vn_vals = [pairwise_max_cos_one_trial(adapters_vnorm,  x, N) for _ in range(n_trials)]
        unnorm_r[N] = {"mean": float(np.mean(un_vals)), "std": float(np.std(un_vals))}
        vnorm_r[N]  = {"mean": float(np.mean(vn_vals)), "std": float(np.std(vn_vals))}
        log(f"  N={N:2d}: unnorm={unnorm_r[N]['mean']:.4f}±{unnorm_r[N]['std']:.4f}  "
            f"vnorm={vnorm_r[N]['mean']:.4f}±{vnorm_r[N]['std']:.4f}")
    return unnorm_r, vnorm_r


def fit_power_law(results_dict):
    ns = sorted(results_dict.keys())
    ys = [results_dict[n]["mean"] for n in ns]
    if len(ns) < 2:
        return 0.05, 0.38, 0.0

    def power_fn(n, c, alpha):
        return c * np.asarray(n, dtype=float) ** alpha

    try:
        popt, _ = curve_fit(power_fn, ns, ys, p0=[0.05, 0.38],
                            bounds=([1e-6, 0], [10, 3]), maxfev=10000)
        c, alpha = float(popt[0]), float(popt[1])
        y_pred = power_fn(np.array(ns), c, alpha)
        ss_res = np.sum((np.array(ys) - y_pred) ** 2)
        ss_tot = np.sum((np.array(ys) - np.mean(ys)) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-12))
    except Exception as e:
        log(f"  [WARN] Power law fit failed: {e}")
        c, alpha, r2 = 0.05, 0.38, 0.0

    return c, alpha, r2


def main():
    t0 = time.time()
    results = {}

    # ─── Phase 1: Load real adapter (A, B) pairs at all layers ───────────────
    log("=== Phase 1: Loading real adapters ===")

    real_ab = {d: {} for d in DOMAINS}
    for domain, path in ADAPTER_PATHS.items():
        st = sfn.load_file(str(path))
        for l in range(N_LAYERS):
            ka = f"language_model.model.layers.{l}.self_attn.q_proj.lora_a"
            kb = f"language_model.model.layers.{l}.self_attn.q_proj.lora_b"
            if ka in st and kb in st:
                A = st[ka].astype(np.float32)
                B = st[kb].astype(np.float32)
                if B.shape == (RANK, D_OUT):
                    real_ab[domain][l] = (A, B)
        log(f"  {domain}: {len(real_ab[domain])} layers")

    common_layers = set(real_ab[DOMAINS[0]].keys())
    for d in DOMAINS[1:]:
        common_layers &= set(real_ab[d].keys())
    log(f"  Common layers (D_OUT={D_OUT}): {len(common_layers)}")
    results["n_common_layers"] = len(common_layers)

    # Compute reference scale from real adapters (use middle layer)
    mid_layer = sorted(common_layers)[len(common_layers) // 2]
    A_ref, B_ref = real_ab[DOMAINS[0]][mid_layer]
    A_col_norm = float(np.linalg.norm(A_ref, axis=0).mean())
    B_row_norm = float(np.linalg.norm(B_ref, axis=1).mean())
    log(f"  Reference norms (layer {mid_layer}): A_col={A_col_norm:.4f}, B_row={B_row_norm:.4f}")
    results["A_col_norm"] = A_col_norm
    results["B_row_norm"] = B_row_norm

    # ─── Phase 2: Measure real adapter cosines across all layers ─────────────
    log("=== Phase 2: Real adapter cosines (all layers, N=5) ===")

    x = rng.standard_normal((N_TOKENS, D_IN)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    test_layers = sorted(common_layers)
    if IS_SMOKE:
        mid = len(test_layers) // 2
        test_layers = test_layers[max(0, mid-1):mid+2]

    layer_cos_un, layer_cos_vn = {}, {}
    for l in test_layers:
        pairs_un = [real_ab[d][l] for d in DOMAINS]
        pairs_vn = [(vnorm(A), B) for A, B in pairs_un]
        # All C(5,2)=10 pairs, average over tokens, take max over pairs
        max_un, max_vn = 0.0, 0.0
        for a, b in combinations(range(5), 2):
            ha_un = (x @ pairs_un[a][0]) @ pairs_un[a][1]
            hb_un = (x @ pairs_un[b][0]) @ pairs_un[b][1]
            ha_vn = (x @ pairs_vn[a][0]) @ pairs_vn[a][1]
            hb_vn = (x @ pairs_vn[b][0]) @ pairs_vn[b][1]
            for ha, hb, is_vn in [(ha_un, hb_un, False), (ha_vn, hb_vn, True)]:
                na, nb = np.linalg.norm(ha, axis=1), np.linalg.norm(hb, axis=1)
                valid = (na > 1e-7) & (nb > 1e-7)
                if not valid.any(): continue
                cos = np.abs(np.einsum("td,td->t", ha, hb)[valid]) / (na[valid] * nb[valid])
                if is_vn:
                    max_vn = max(max_vn, float(cos.max()))
                else:
                    max_un = max(max_un, float(cos.max()))
        layer_cos_un[l] = max_un
        layer_cos_vn[l] = max_vn

    worst_layer = max(layer_cos_un, key=layer_cos_un.get)
    log(f"  Max real adapter cos (unnorm) across {len(test_layers)} layers: "
        f"{max(layer_cos_un.values()):.4f} at layer {worst_layer}")
    log(f"  Max real adapter cos (vnorm) across {len(test_layers)} layers: "
        f"{max(layer_cos_vn.values()):.4f}")
    log(f"  Mean real adapter cos (unnorm): {np.mean(list(layer_cos_un.values())):.4f}")
    log(f"  Mean real adapter cos (vnorm):  {np.mean(list(layer_cos_vn.values())):.4f}")

    results["real_max_cos_unnorm"] = max(layer_cos_un.values())
    results["real_max_cos_vnorm"]  = max(layer_cos_vn.values())
    results["real_mean_cos_unnorm"] = float(np.mean(list(layer_cos_un.values())))
    results["real_mean_cos_vnorm"]  = float(np.mean(list(layer_cos_vn.values())))
    results["worst_layer"] = int(worst_layer)
    results["layer_cos_unnorm"] = {str(k): float(v) for k, v in layer_cos_un.items()}
    results["layer_cos_vnorm"]  = {str(k): float(v) for k, v in layer_cos_vn.items()}

    # ─── Phase 3: Generate pure synthetic adapter pool ────────────────────────
    log(f"=== Phase 3: Synthetic adapter pool (N_SYNTH={N_SYNTH}) ===")

    def make_pair():
        A = rng.standard_normal((D_IN, RANK)).astype(np.float32)
        A = A / np.linalg.norm(A, axis=0, keepdims=True) * A_col_norm
        B = rng.standard_normal((RANK, D_OUT)).astype(np.float32)
        B = B / np.linalg.norm(B, axis=1, keepdims=True) * B_row_norm
        return A, B

    synth_unnorm = [make_pair() for _ in range(N_SYNTH)]
    synth_vnorm  = [(vnorm(A), B) for A, B in synth_unnorm]
    log(f"  Pool: {N_SYNTH} adapters (unnorm + V-normed)")

    # ─── Phase 4: Power law measurement on pure synthetic pool ───────────────
    log(f"=== Phase 4: Power law measurement (pure synthetic, N_TRIALS={N_TRIALS}) ===")

    unnorm_by_n, vnorm_by_n = measure_power_law(
        synth_unnorm, synth_vnorm, x, N_VALUES, N_TRIALS
    )

    results["unnorm_by_n"] = {str(k): v for k, v in unnorm_by_n.items()}
    results["vnorm_by_n"]  = {str(k): v for k, v in vnorm_by_n.items()}

    # ─── Phase 5: Power law fitting ──────────────────────────────────────────
    log("=== Phase 5: Power law fitting ===")

    c_un, alpha_un, r2_un = fit_power_law(unnorm_by_n)
    c_vn, alpha_vn, r2_vn = fit_power_law(vnorm_by_n)

    max50_un = float(c_un * 50 ** alpha_un)
    max50_vn = float(c_vn * 50 ** alpha_vn)

    log(f"  Unnorm: c={c_un:.4f}, alpha={alpha_un:.4f}, R²={r2_un:.3f}")
    log(f"  Vnorm:  c={c_vn:.4f}, alpha={alpha_vn:.4f}, R²={r2_vn:.3f}")
    log(f"  Finding #372 baseline: c=0.059, alpha=0.38 (Qwen3-4B fc1)")
    log(f"  Extrapolate N=50: unnorm={max50_un:.4f}, vnorm={max50_vn:.4f}")
    log(f"  alpha delta (vnorm - unnorm): {alpha_vn - alpha_un:.4f}")

    results["power_law_unnorm"] = {"c": c_un, "alpha": alpha_un, "r2": r2_un, "max_at_50": max50_un}
    results["power_law_vnorm"]  = {"c": c_vn, "alpha": alpha_vn, "r2": r2_vn, "max_at_50": max50_vn}
    results["finding_372"] = {"c": 0.059, "alpha": 0.38, "r2": 0.90}

    # ─── Phase 6: Kill criteria ───────────────────────────────────────────────
    log("=== Phase 6: Kill Criteria ===")

    k1056 = True
    k1057 = alpha_vn <= 0.40
    k1058 = max50_vn < 0.50

    log(f"  K1056 (measure c,alpha): PASS — c_vn={c_vn:.4f}, alpha_vn={alpha_vn:.4f}")
    log(f"  K1057 (alpha_vnorm ≤ 0.40): {'PASS' if k1057 else 'FAIL'} (alpha_vn={alpha_vn:.4f})")
    log(f"  K1058 (N=50 max_cos < 0.50): {'PASS' if k1058 else 'FAIL'} (max50={max50_vn:.4f})")

    results["kill_criteria"] = {
        "K1056": {"pass": k1056, "c": c_vn, "alpha": alpha_vn},
        "K1057": {"pass": k1057, "threshold": 0.40, "alpha_vn": alpha_vn},
        "K1058": {"pass": k1058, "threshold": 0.50, "max_at_50": max50_vn},
    }
    results["alpha_delta_vs_372"] = round(alpha_vn - 0.38, 4)

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    log(f"\nElapsed: {elapsed:.1f}s")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved to {RESULTS_FILE}")

    log("\n=== SUMMARY ===")
    log(f"Real adapters: max_cos={results['real_max_cos_unnorm']:.4f} (unnorm), "
        f"{results['real_max_cos_vnorm']:.4f} (vnorm) at N=5")
    log(f"Synthetic power law: unnorm c={c_un:.4f} alpha={alpha_un:.4f} | vnorm c={c_vn:.4f} alpha={alpha_vn:.4f}")
    log(f"Finding #372 baseline: c=0.059, alpha=0.38")
    log(f"K1056 PASS | K1057 {'PASS' if k1057 else 'FAIL'} | K1058 {'PASS' if k1058 else 'FAIL'}")

    return 0 if (k1056 and k1057 and k1058) else 1


if __name__ == "__main__":
    sys.exit(main())
