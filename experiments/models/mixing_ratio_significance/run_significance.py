#!/usr/bin/env python3
"""
Statistical Significance of Optimal Data Mixing Ratio

Follows up on exp_synthetic_vs_real_data (supported) where:
  - K2 survived: 20% synthetic mix improved +11.2% over pure real (5 seeds)
  - Adversarial review flagged overlapping CIs (std=0.0177 vs 0.0106)

This experiment:
  1. Runs 20 seeds (up from 5)
  2. Collects PAIRED per-seed differences: quality(ratio=alpha) - quality(ratio=0.0)
  3. Applies Wilcoxon signed-rank test for ratio=0.2 vs 0.0
  4. Sweeps alpha in 0.05 increments to find optimal ratio with bootstrap CI
  5. Tests stability: does optimal ratio shift by >0.15 across seed subsets?

Kill criteria:
  K1: Wilcoxon p > 0.05 for ratio=0.2 vs ratio=0.0 -> mixing benefit is noise
  K2: Optimal ratio changes by >0.15 across 20 seeds -> ratio unstable

Reuses data generation and LoRA training from parent experiment.
Pure numpy/scipy. Runs in <2 minutes on Apple Silicon.
"""

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon, bootstrap

# ── Configuration ────────────────────────────────────────────────
# Match parent experiment exactly
D_MODEL = 64
RANK = 8
N_TRAIN = 1000
N_EVAL = 500
N_GRADIENT_STEPS = 500
LR = 0.01
BATCH_SIZE = 64

# This experiment: 20 seeds, finer sweep
N_SEEDS = 20
SEEDS = list(range(42, 42 + N_SEEDS))  # deterministic, non-overlapping
MIXING_RATIOS = [round(x * 0.05, 2) for x in range(21)]  # 0.00, 0.05, ..., 1.00

# Synthetic data config (from parent)
SYNTH_N_MODES = 5
SYNTH_CONCENTRATION = 0.5
SYNTH_INPUT_NOISE = 0.3
SYNTH_LABEL_NOISE = 0.05
SYNTH_BIAS = 0.25

# Real data config (from parent)
REAL_N_MODES = 20
REAL_CONCENTRATION = 2.0
REAL_INPUT_NOISE = 0.8
REAL_LABEL_NOISE = 0.30


# ── Data Generation (from parent, simplified) ───────────────────

def make_mode_centers(n_modes: int, d: int, rng: np.random.Generator) -> np.ndarray:
    centers = rng.standard_normal((n_modes, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return centers


def generate_inputs(n_modes, concentration, input_noise, bias_mag,
                    n, d, rng, mode_centers, bias_dir=None):
    alpha = np.ones(n_modes) * concentration
    weights = rng.dirichlet(alpha)
    assignments = rng.choice(n_modes, size=n, p=weights)
    X = np.zeros((n, d))
    for i in range(n):
        X[i] = mode_centers[assignments[i]] + rng.standard_normal(d) * input_noise
    if bias_mag > 0 and bias_dir is not None:
        X += bias_mag * bias_dir[np.newaxis, :]
    return X


def generate_labels(X, W_star, noise_std, rng):
    return X @ W_star + rng.standard_normal(X.shape) * noise_std


def train_lora(X, y, d, rank, n_steps, lr, rng):
    n = X.shape[0]
    A = rng.standard_normal((rank, d)) * (1.0 / np.sqrt(d))
    B = np.zeros((d, rank))
    for step in range(n_steps):
        idx = rng.integers(0, n, size=min(BATCH_SIZE, n))
        xb, yb = X[idx], y[idx]
        y_pred = xb @ A.T @ B.T
        error = y_pred - yb
        grad_B = error.T @ xb @ A.T / len(idx)
        decay = 0.5 * (1 + np.cos(np.pi * step / n_steps))
        B -= lr * (0.1 + 0.9 * decay) * grad_B
    return A, B


def quality(A, B, X_eval, W_star):
    y_pred = X_eval @ A.T @ B.T
    y_true = X_eval @ W_star
    return max(0, 1 - np.linalg.norm(y_pred - y_true, 'fro') /
               (np.linalg.norm(y_true, 'fro') + 1e-10))


# ── Per-seed sweep ──────────────────────────────────────────────

def run_seed(seed: int) -> Dict:
    """Run full mixing ratio sweep for one seed. Returns dict of ratio -> quality."""
    rng = np.random.default_rng(seed)

    # Shared ground truth
    U = rng.standard_normal((D_MODEL, RANK))
    V = rng.standard_normal((RANK, D_MODEL))
    W_star = U @ V * 0.1

    bias_dir = rng.standard_normal(D_MODEL)
    bias_dir /= np.linalg.norm(bias_dir)

    synth_modes = make_mode_centers(SYNTH_N_MODES, D_MODEL, rng)
    real_modes = make_mode_centers(REAL_N_MODES, D_MODEL, rng)

    # Uniform eval data
    X_eval = rng.standard_normal((N_EVAL, D_MODEL)) * 0.5

    results = {}
    for ratio in MIXING_RATIOS:
        n_s = int(N_TRAIN * ratio)
        n_r = N_TRAIN - n_s
        parts_X, parts_y = [], []

        if n_s > 0:
            Xs = generate_inputs(SYNTH_N_MODES, SYNTH_CONCENTRATION, SYNTH_INPUT_NOISE,
                                 SYNTH_BIAS, n_s, D_MODEL, rng, synth_modes, bias_dir)
            ys = generate_labels(Xs, W_star, SYNTH_LABEL_NOISE, rng)
            parts_X.append(Xs)
            parts_y.append(ys)
        if n_r > 0:
            Xr = generate_inputs(REAL_N_MODES, REAL_CONCENTRATION, REAL_INPUT_NOISE,
                                 0.0, n_r, D_MODEL, rng, real_modes, None)
            yr = generate_labels(Xr, W_star, REAL_LABEL_NOISE, rng)
            parts_X.append(Xr)
            parts_y.append(yr)

        Xm = np.vstack(parts_X)
        ym = np.vstack(parts_y)
        perm = rng.permutation(len(Xm))
        Xm, ym = Xm[perm], ym[perm]

        A, B = train_lora(Xm, ym, D_MODEL, RANK, N_GRADIENT_STEPS, LR, rng)
        results[ratio] = quality(A, B, X_eval, W_star)

    return results


# ── Analysis ────────────────────────────────────────────────────

def analyze(all_seed_results: List[Dict]) -> Dict:
    """
    all_seed_results: list of {ratio: quality} dicts, one per seed.
    """
    n = len(all_seed_results)
    ratios = MIXING_RATIOS

    # Build matrix: seeds x ratios
    Q = np.zeros((n, len(ratios)))
    for i, sr in enumerate(all_seed_results):
        for j, r in enumerate(ratios):
            Q[i, j] = sr[r]

    # Per-ratio statistics
    ratio_stats = {}
    for j, r in enumerate(ratios):
        vals = Q[:, j]
        ratio_stats[f"{r:.2f}"] = {
            "ratio": r,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci95": float(1.96 * np.std(vals, ddof=1) / np.sqrt(n)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # ── K1: Wilcoxon signed-rank test, ratio=0.2 vs ratio=0.0 ──
    idx_0 = ratios.index(0.0)
    idx_02 = ratios.index(0.2)
    diffs_02 = Q[:, idx_02] - Q[:, idx_0]

    # Also test ratio=0.1 and 0.15 for completeness
    test_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    paired_tests = {}
    for tr in test_ratios:
        idx_tr = ratios.index(tr)
        d = Q[:, idx_tr] - Q[:, idx_0]
        # Wilcoxon signed-rank (two-sided first, then one-sided)
        if np.all(d == 0):
            stat, p_two, p_one = 0.0, 1.0, 1.0
        else:
            stat, p_two = wilcoxon(d, alternative='two-sided')
            _, p_one = wilcoxon(d, alternative='greater')
        paired_tests[f"{tr:.2f}"] = {
            "ratio": tr,
            "mean_diff": float(np.mean(d)),
            "std_diff": float(np.std(d, ddof=1)),
            "median_diff": float(np.median(d)),
            "n_positive": int(np.sum(d > 0)),
            "n_negative": int(np.sum(d < 0)),
            "n_zero": int(np.sum(d == 0)),
            "wilcoxon_stat": float(stat),
            "p_two_sided": float(p_two),
            "p_one_sided_greater": float(p_one),
            "significant_0.05": bool(p_two < 0.05),
            "significant_0.01": bool(p_two < 0.01),
        }

    k1_p = paired_tests["0.20"]["p_two_sided"]
    k1_verdict = "KILLED" if k1_p > 0.05 else "SURVIVES"

    # ── K2: Optimal ratio stability ──
    # Find per-seed optimal ratio
    per_seed_optimal = []
    for i in range(n):
        # Only consider mixed ratios (0 < r < 1)
        mixed_idx = [j for j, r in enumerate(ratios) if 0 < r < 1]
        best_j = max(mixed_idx, key=lambda j: Q[i, j])
        per_seed_optimal.append(ratios[best_j])

    per_seed_optimal = np.array(per_seed_optimal)
    opt_mean = float(np.mean(per_seed_optimal))
    opt_std = float(np.std(per_seed_optimal))
    opt_range = float(np.max(per_seed_optimal) - np.min(per_seed_optimal))
    opt_median = float(np.median(per_seed_optimal))

    # Bootstrap CI for optimal ratio
    # For each bootstrap resample of seeds, find the ratio with best mean quality
    rng_boot = np.random.default_rng(999)
    n_boot = 10000
    boot_optima = []
    for _ in range(n_boot):
        idx = rng_boot.integers(0, n, size=n)
        Q_boot = Q[idx, :]
        mean_boot = Q_boot.mean(axis=0)
        # Only mixed ratios
        mixed_idx = [j for j, r in enumerate(ratios) if 0 < r < 1]
        best_j = max(mixed_idx, key=lambda j: mean_boot[j])
        boot_optima.append(ratios[best_j])

    boot_optima = np.array(boot_optima)
    boot_ci_lo = float(np.percentile(boot_optima, 2.5))
    boot_ci_hi = float(np.percentile(boot_optima, 97.5))
    boot_ci_width = boot_ci_hi - boot_ci_lo

    k2_verdict = "KILLED" if boot_ci_width > 0.15 else "SURVIVES"

    # ── Effect size (Cohen's d for paired samples) ──
    d_02 = diffs_02
    cohens_d = float(np.mean(d_02) / (np.std(d_02, ddof=1) + 1e-10))

    # ── Global best ratio from pooled mean ──
    mean_q = Q.mean(axis=0)
    mixed_idx = [j for j, r in enumerate(ratios) if 0 < r < 1]
    global_best_j = max(mixed_idx, key=lambda j: mean_q[j])
    global_best_ratio = ratios[global_best_j]
    global_best_q = float(mean_q[global_best_j])
    baseline_q = float(mean_q[idx_0])
    global_improvement = (global_best_q - baseline_q) / (baseline_q + 1e-10) * 100

    results = {
        "n_seeds": n,
        "config": {
            "d": D_MODEL, "r": RANK, "n_train": N_TRAIN,
            "n_eval": N_EVAL, "steps": N_GRADIENT_STEPS,
            "lr": LR, "n_ratios": len(ratios),
        },
        "ratio_stats": ratio_stats,
        "paired_tests": paired_tests,
        "optimal_ratio": {
            "global_best": global_best_ratio,
            "global_best_quality": global_best_q,
            "baseline_quality": baseline_q,
            "improvement_pct": round(global_improvement, 2),
            "per_seed_mean": opt_mean,
            "per_seed_std": opt_std,
            "per_seed_median": opt_median,
            "per_seed_range": opt_range,
            "per_seed_values": per_seed_optimal.tolist(),
            "bootstrap_ci_lo": boot_ci_lo,
            "bootstrap_ci_hi": boot_ci_hi,
            "bootstrap_ci_width": round(boot_ci_width, 4),
        },
        "effect_size": {
            "cohens_d_02_vs_00": round(cohens_d, 4),
            "interpretation": (
                "large" if abs(cohens_d) > 0.8 else
                "medium" if abs(cohens_d) > 0.5 else
                "small" if abs(cohens_d) > 0.2 else "negligible"
            ),
        },
        "kill_criteria": {
            "K1": {
                "desc": "Wilcoxon p > 0.05 for ratio=0.2 vs ratio=0.0",
                "p_value": round(k1_p, 6),
                "threshold": 0.05,
                "killed": k1_p > 0.05,
                "verdict": k1_verdict,
                "detail": f"p={k1_p:.4f}, mean_diff={np.mean(diffs_02):.5f}, "
                          f"n_positive={np.sum(diffs_02 > 0)}/{n}",
            },
            "K2": {
                "desc": "Optimal ratio CI width > 0.15 (ratio unstable)",
                "ci_width": round(boot_ci_width, 4),
                "ci": [boot_ci_lo, boot_ci_hi],
                "threshold": 0.15,
                "killed": boot_ci_width > 0.15,
                "verdict": k2_verdict,
                "detail": f"95% CI=[{boot_ci_lo:.2f}, {boot_ci_hi:.2f}], "
                          f"width={boot_ci_width:.4f}",
            },
        },
    }

    return results


def print_report(results):
    print("=" * 74)
    print("MIXING RATIO SIGNIFICANCE: STATISTICAL ANALYSIS")
    print("=" * 74)
    c = results["config"]
    print(f"\nConfig: d={c['d']}, r={c['r']}, N_train={c['n_train']}, "
          f"steps={c['steps']}, seeds={results['n_seeds']}")
    print(f"Ratios tested: {c['n_ratios']} (0.00 to 1.00 in 0.05 steps)")

    # Ratio quality table
    print("\n--- Quality by Mixing Ratio (uniform eval) ---")
    print(f"{'Ratio':<7} {'Mean':<10} {'Std':<10} {'95% CI':<14} {'Range'}")
    for k in sorted(results["ratio_stats"], key=float):
        s = results["ratio_stats"][k]
        print(f"{float(k):<7.2f} {s['mean']:<10.5f} {s['std']:<10.5f} "
              f"+/-{s['ci95']:<10.5f} [{s['min']:.4f}, {s['max']:.4f}]")

    # Paired tests
    print("\n--- Paired Wilcoxon Signed-Rank Tests (ratio vs 0.0) ---")
    print(f"{'Ratio':<7} {'Mean diff':<12} {'Median diff':<12} "
          f"{'W stat':<10} {'p (2-sided)':<12} {'p (1-sided)':<12} {'Sig?'}")
    for k in sorted(results["paired_tests"], key=float):
        t = results["paired_tests"][k]
        sig = "***" if t["p_two_sided"] < 0.001 else (
              "**" if t["p_two_sided"] < 0.01 else (
              "*" if t["p_two_sided"] < 0.05 else "ns"))
        print(f"{float(k):<7.2f} {t['mean_diff']:<12.6f} {t['median_diff']:<12.6f} "
              f"{t['wilcoxon_stat']:<10.1f} {t['p_two_sided']:<12.6f} "
              f"{t['p_one_sided_greater']:<12.6f} {sig}")
        print(f"        (+:{t['n_positive']}, -:{t['n_negative']}, 0:{t['n_zero']})")

    # Optimal ratio
    print("\n--- Optimal Mixing Ratio ---")
    opt = results["optimal_ratio"]
    print(f"Global best ratio: {opt['global_best']:.2f} "
          f"(quality={opt['global_best_quality']:.5f}, "
          f"+{opt['improvement_pct']:.1f}% over pure real)")
    print(f"Per-seed optimal: mean={opt['per_seed_mean']:.3f}, "
          f"std={opt['per_seed_std']:.3f}, "
          f"median={opt['per_seed_median']:.3f}, "
          f"range={opt['per_seed_range']:.3f}")
    print(f"Bootstrap 95% CI for optimal ratio: "
          f"[{opt['bootstrap_ci_lo']:.2f}, {opt['bootstrap_ci_hi']:.2f}] "
          f"(width={opt['bootstrap_ci_width']:.4f})")

    # Effect size
    es = results["effect_size"]
    print(f"\nEffect size (Cohen's d, ratio=0.2 vs 0.0): "
          f"{es['cohens_d_02_vs_00']:.3f} ({es['interpretation']})")

    # Kill criteria
    print("\n--- Kill Criteria ---")
    k1 = results["kill_criteria"]["K1"]
    print(f"K1: {k1['desc']}")
    print(f"    {k1['detail']}")
    print(f"    -> {k1['verdict']}")

    k2 = results["kill_criteria"]["K2"]
    print(f"K2: {k2['desc']}")
    print(f"    {k2['detail']}")
    print(f"    -> {k2['verdict']}")

    # Overall
    print("\n" + "=" * 74)
    if k1["killed"] and k2["killed"]:
        print("OVERALL: KILLED (both criteria failed)")
    elif k1["killed"]:
        print("OVERALL: KILLED (K1 - mixing benefit is not significant)")
    elif k2["killed"]:
        print("OVERALL: KILLED (K2 - optimal ratio is unstable)")
    else:
        print("OVERALL: SURVIVES (mixing benefit is significant and ratio is stable)")
    print("=" * 74)


def main():
    t0 = time.time()
    print("Running mixing ratio significance analysis (20 seeds)...")

    all_results = []
    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({i+1}/{N_SEEDS})...", end=" ", flush=True)
        sr = run_seed(seed)
        all_results.append(sr)
        print(f"done (q@0.0={sr[0.0]:.4f}, q@0.2={sr[0.2]:.4f}, "
              f"diff={sr[0.2]-sr[0.0]:+.4f})")

    results = analyze(all_results)

    # Save raw per-seed data too
    results["raw_per_seed"] = [
        {f"{r:.2f}": sr[r] for r in MIXING_RATIOS}
        for sr in all_results
    ]
    results["seeds"] = SEEDS
    results["runtime_seconds"] = round(time.time() - t0, 1)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    print_report(results)
    print(f"\nTotal runtime: {results['runtime_seconds']:.1f}s")

    return results


if __name__ == "__main__":
    main()
