#!/usr/bin/env python3
"""
Mixing Ratio Statistical Significance

Extends exp_synthetic_vs_real_data with:
  - 20 seeds (up from 5) for statistical power
  - Wilcoxon signed-rank paired test on per-seed (ratio=0.2 - ratio=0.0)
  - Alpha sweep in 0.05 increments to find optimal ratio per seed
  - Bootstrap CI for mean improvement

Kill criteria:
  K1: Wilcoxon p > 0.05 for alpha=0.2 vs 0.0 (benefit is noise)
  K2: optimal alpha changes by >0.15 across 20 seeds (ratio unstable)

CPU only. numpy/scipy. ~2min on Apple Silicon.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# ── Config (matches parent experiment) ──────────────────────────
D_MODEL = 64
RANK = 8
N_TRAIN = 1000
N_EVAL = 500
N_GRADIENT_STEPS = 500
LR = 0.01
BATCH_SIZE = 64
N_SEEDS = 20
ALPHA_SWEEP = [round(a * 0.05, 2) for a in range(21)]  # 0.00 to 1.00 by 0.05
N_BOOTSTRAP = 10000

# Data source params (from parent)
SYNTH_MODES = 5
SYNTH_CONC = 0.5
SYNTH_NOISE = 0.3
SYNTH_LABEL_STD = 0.05
SYNTH_BIAS = 0.25
REAL_MODES = 20
REAL_CONC = 2.0
REAL_NOISE = 0.8
REAL_LABEL_STD = 0.30


# ── Data generation (from parent) ──────────────────────────────

def make_mode_centers(n_modes, d, rng):
    centers = rng.standard_normal((n_modes, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return centers


def generate_inputs(n_modes, concentration, noise_scale, bias_mag,
                    n, d, rng, mode_centers, bias_dir=None):
    alpha = np.ones(n_modes) * concentration
    weights = rng.dirichlet(alpha)
    assignments = rng.choice(n_modes, size=n, p=weights)
    X = np.zeros((n, d))
    for i in range(n):
        X[i] = mode_centers[assignments[i]] + rng.standard_normal(d) * noise_scale
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


# ── Main experiment ─────────────────────────────────────────────

def run_seed(seed):
    """Run mixing ratio sweep for one seed. Returns {alpha: quality_uniform}."""
    rng = np.random.default_rng(seed)

    # Ground truth
    U = rng.standard_normal((D_MODEL, RANK))
    V = rng.standard_normal((RANK, D_MODEL))
    W_star = U @ V * 0.1

    bias_dir = rng.standard_normal(D_MODEL)
    bias_dir /= np.linalg.norm(bias_dir)

    synth_modes = make_mode_centers(SYNTH_MODES, D_MODEL, rng)
    real_modes = make_mode_centers(REAL_MODES, D_MODEL, rng)

    # Eval data (uniform)
    X_eval = rng.standard_normal((N_EVAL, D_MODEL)) * 0.5

    results = {}
    for alpha in ALPHA_SWEEP:
        n_s = int(N_TRAIN * alpha)
        n_r = N_TRAIN - n_s
        parts_X, parts_y = [], []

        if n_s > 0:
            Xs = generate_inputs(SYNTH_MODES, SYNTH_CONC, SYNTH_NOISE, SYNTH_BIAS,
                                 n_s, D_MODEL, rng, synth_modes, bias_dir)
            ys = generate_labels(Xs, W_star, SYNTH_LABEL_STD, rng)
            parts_X.append(Xs)
            parts_y.append(ys)
        if n_r > 0:
            Xr = generate_inputs(REAL_MODES, REAL_CONC, REAL_NOISE, 0.02,
                                 n_r, D_MODEL, rng, real_modes, None)
            yr = generate_labels(Xr, W_star, REAL_LABEL_STD, rng)
            parts_X.append(Xr)
            parts_y.append(yr)

        Xm = np.vstack(parts_X)
        ym = np.vstack(parts_y)
        perm = rng.permutation(len(Xm))
        Xm, ym = Xm[perm], ym[perm]

        A, B = train_lora(Xm, ym, D_MODEL, RANK, N_GRADIENT_STEPS, LR, rng)
        results[alpha] = quality(A, B, X_eval, W_star)

    return results


def bootstrap_ci(diffs, n_boot=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap percentile CI for mean of diffs."""
    rng = np.random.default_rng(42)
    n = len(diffs)
    boot_means = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi), boot_means


def main():
    t0 = time.time()
    print("=" * 70)
    print("  MIXING RATIO STATISTICAL SIGNIFICANCE")
    print(f"  {N_SEEDS} seeds, alpha sweep 0.00-1.00 by 0.05")
    print("=" * 70)

    # Run all seeds
    all_seed_results = {}
    seeds = list(range(42, 42 + N_SEEDS))
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i+1}/{N_SEEDS})...", end=" ", flush=True)
        all_seed_results[seed] = run_seed(seed)
        print("done")

    # ── Analysis 1: Paired differences at alpha=0.2 vs 0.0 ────
    print("\n--- Analysis 1: Paired Wilcoxon at alpha=0.2 vs 0.0 ---")

    diffs_02 = np.array([
        all_seed_results[s][0.2] - all_seed_results[s][0.0]
        for s in seeds
    ])

    mean_diff = float(np.mean(diffs_02))
    std_diff = float(np.std(diffs_02, ddof=1))
    wilcoxon_stat, wilcoxon_p = sp_stats.wilcoxon(diffs_02, alternative='two-sided')
    # Also t-test for comparison
    t_stat, t_p = sp_stats.ttest_1samp(diffs_02, 0)
    # Bootstrap CI
    boot_lo, boot_hi, boot_means = bootstrap_ci(diffs_02)

    n_positive = int(np.sum(diffs_02 > 0))
    n_negative = int(np.sum(diffs_02 < 0))

    print(f"  Mean diff (0.2 - 0.0): {mean_diff:.6f}")
    print(f"  Std diff: {std_diff:.6f}")
    print(f"  Positive/Negative: {n_positive}/{n_negative}")
    print(f"  Wilcoxon signed-rank: W={wilcoxon_stat:.1f}, p={wilcoxon_p:.6f}")
    print(f"  t-test: t={t_stat:.3f}, p={t_p:.6f}")
    print(f"  Bootstrap 95% CI: [{boot_lo:.6f}, {boot_hi:.6f}]")

    k1_pass = wilcoxon_p <= 0.05
    print(f"  K1 VERDICT: {'SURVIVES (significant)' if k1_pass else 'KILLED (not significant)'}")

    # ── Analysis 2: Optimal alpha stability ────────────────────
    print("\n--- Analysis 2: Optimal alpha stability across seeds ---")

    optimal_alphas = []
    for s in seeds:
        best_alpha = max(ALPHA_SWEEP, key=lambda a: all_seed_results[s][a])
        optimal_alphas.append(best_alpha)

    optimal_alphas = np.array(optimal_alphas)
    mean_opt = float(np.mean(optimal_alphas))
    std_opt = float(np.std(optimal_alphas))
    median_opt = float(np.median(optimal_alphas))

    # Count how many seeds pick each alpha
    from collections import Counter
    alpha_counts = Counter(optimal_alphas)
    most_common = alpha_counts.most_common(5)

    print(f"  Mean optimal alpha: {mean_opt:.3f}")
    print(f"  Median optimal alpha: {median_opt:.3f}")
    print(f"  Std optimal alpha: {std_opt:.3f}")
    print(f"  Range: [{float(optimal_alphas.min()):.2f}, {float(optimal_alphas.max()):.2f}]")
    print(f"  Most common: {[(f'{a:.2f}', c) for a, c in most_common]}")

    k2_pass = std_opt <= 0.15
    print(f"  K2 VERDICT: {'SURVIVES (stable)' if k2_pass else 'KILLED (unstable)'}")

    # ── Analysis 3: Full alpha sweep with CIs ──────────────────
    print("\n--- Analysis 3: Full alpha sweep with confidence intervals ---")
    print(f"  {'Alpha':<8} {'Mean Q':<12} {'Std Q':<12} {'CI95':<20} {'vs 0.0':<12}")

    q_at_0 = np.array([all_seed_results[s][0.0] for s in seeds])
    sweep_summary = {}

    for alpha in ALPHA_SWEEP:
        q_vals = np.array([all_seed_results[s][alpha] for s in seeds])
        mean_q = float(np.mean(q_vals))
        std_q = float(np.std(q_vals, ddof=1))
        ci = 1.96 * std_q / np.sqrt(N_SEEDS)
        diff_vs_0 = float(np.mean(q_vals - q_at_0))

        sweep_summary[f"{alpha:.2f}"] = {
            "mean": mean_q, "std": std_q,
            "ci95_lo": mean_q - ci, "ci95_hi": mean_q + ci,
            "diff_vs_0": diff_vs_0
        }
        print(f"  {alpha:<8.2f} {mean_q:<12.6f} {std_q:<12.6f} "
              f"[{mean_q-ci:.6f}, {mean_q+ci:.6f}] {diff_vs_0:+.6f}")

    # ── Analysis 4: Multi-comparison correction ────────────────
    print("\n--- Analysis 4: All pairwise tests vs alpha=0.0 (Holm-Bonferroni) ---")

    pairwise_tests = []
    for alpha in ALPHA_SWEEP:
        if alpha == 0.0:
            continue
        diffs = np.array([all_seed_results[s][alpha] - all_seed_results[s][0.0] for s in seeds])
        if np.all(diffs == 0):
            pairwise_tests.append((alpha, 1.0, float(np.mean(diffs))))
        else:
            _, p = sp_stats.wilcoxon(diffs, alternative='two-sided')
            pairwise_tests.append((alpha, p, float(np.mean(diffs))))

    # Holm-Bonferroni correction
    pairwise_tests.sort(key=lambda x: x[1])
    m = len(pairwise_tests)
    corrected = []
    for i, (alpha, p, diff) in enumerate(pairwise_tests):
        p_adj = min(1.0, p * (m - i))
        sig = p_adj <= 0.05
        corrected.append((alpha, p, p_adj, diff, sig))

    corrected.sort(key=lambda x: x[0])
    print(f"  {'Alpha':<8} {'Raw p':<12} {'Adj p':<12} {'MeanDiff':<12} {'Sig?'}")
    for alpha, p, p_adj, diff, sig in corrected:
        print(f"  {alpha:<8.2f} {p:<12.6f} {p_adj:<12.6f} {diff:+<12.6f} {'YES' if sig else 'no'}")

    sig_alphas = [a for a, _, p_adj, d, s in corrected if s and d > 0]
    if sig_alphas:
        print(f"\n  Significantly better than pure real (after correction): {[f'{a:.2f}' for a in sig_alphas]}")
    else:
        print(f"\n  No alpha significantly better than pure real after correction.")

    # ── Analysis 5: Effect size ────────────────────────────────
    print("\n--- Analysis 5: Effect size (Cohen's d) ---")
    cohens_d = mean_diff / (std_diff + 1e-10)
    effect_label = "large" if abs(cohens_d) >= 0.8 else "medium" if abs(cohens_d) >= 0.5 else "small"
    print(f"  Cohen's d for alpha=0.2 vs 0.0: {cohens_d:.3f} ({effect_label})")

    # ── Overall verdict ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OVERALL VERDICT")
    print("=" * 70)
    print(f"  K1 (Wilcoxon p <= 0.05): {'PASS' if k1_pass else 'FAIL'} (p={wilcoxon_p:.6f})")
    print(f"  K2 (std(alpha*) <= 0.15): {'PASS' if k2_pass else 'FAIL'} (std={std_opt:.3f})")

    if k1_pass and k2_pass:
        verdict = "proven"
        print(f"  STATUS: PROVEN — mixing benefit is real and optimal ratio is stable")
    elif k1_pass:
        verdict = "supported"
        print(f"  STATUS: SUPPORTED — benefit is real but ratio is unstable")
    elif k2_pass:
        verdict = "killed"
        print(f"  STATUS: KILLED — K1 failed, benefit is not significant")
    else:
        verdict = "killed"
        print(f"  STATUS: KILLED — both K1 and K2 failed")

    # ── Save results ───────────────────────────────────────────
    elapsed = time.time() - t0

    results = {
        "experiment": "mixing_ratio_significance",
        "n_seeds": N_SEEDS,
        "config": {"d": D_MODEL, "r": RANK, "n_train": N_TRAIN,
                   "n_eval": N_EVAL, "steps": N_GRADIENT_STEPS},
        "k1": {
            "test": "Wilcoxon signed-rank (alpha=0.2 vs 0.0)",
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "wilcoxon_W": float(wilcoxon_stat),
            "wilcoxon_p": float(wilcoxon_p),
            "t_stat": float(t_stat),
            "t_p": float(t_p),
            "bootstrap_ci_95": [boot_lo, boot_hi],
            "cohens_d": cohens_d,
            "verdict": "PASS" if k1_pass else "FAIL",
        },
        "k2": {
            "test": "Optimal alpha stability (std across 20 seeds)",
            "mean_optimal_alpha": mean_opt,
            "median_optimal_alpha": median_opt,
            "std_optimal_alpha": std_opt,
            "range": [float(optimal_alphas.min()), float(optimal_alphas.max())],
            "distribution": {f"{a:.2f}": int(c) for a, c in alpha_counts.most_common()},
            "verdict": "PASS" if k2_pass else "FAIL",
        },
        "sweep_summary": sweep_summary,
        "significant_alphas": [f"{a:.2f}" for a in sig_alphas],
        "per_seed_diffs_02": diffs_02.tolist(),
        "per_seed_optimal_alpha": optimal_alphas.tolist(),
        "verdict": verdict,
        "elapsed_s": elapsed,
    }

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    main()
