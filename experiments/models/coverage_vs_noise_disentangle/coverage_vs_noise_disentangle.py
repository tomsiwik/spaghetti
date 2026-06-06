#!/usr/bin/env python3
"""
Coverage vs Noise Disentangle: 2x2 Factorial Ablation

The parent experiment (synthetic_vs_real_data) found a 58% quality gap between
synthetic-only and real-only LoRA training. The adversarial review identified
that coverage (5 vs 20 modes) and noise (sigma=0.05 vs 0.30) are confounded.

This experiment performs a full 2x2 factorial ablation:

  Factor A: COVERAGE  -- number of input modes (5 = low, 20 = high)
  Factor B: NOISE     -- label noise std (0.05 = low, 0.30 = high)

  Condition 1: low-cov / low-noise   (original "synthetic")
  Condition 2: low-cov / high-noise  (NEW: isolates coverage penalty)
  Condition 3: high-cov / low-noise  (NEW: isolates noise penalty)
  Condition 4: high-cov / high-noise (original "real")

The quality gap decomposes as:

  total_gap = Q(high-cov/high-noise) - Q(low-cov/low-noise)
            = coverage_effect + noise_effect + interaction

where:
  coverage_effect = [Q(high-cov/*) - Q(low-cov/*)] averaged over noise levels
  noise_effect    = [Q(*/low-noise) - Q(*/high-noise)] averaged over coverage
  interaction     = total - coverage - noise

Kill criteria:
  K1: coverage alone explains <50% of the 58% gap (noise is the driver)
  K2: noise alone explains >80% of the gap (coverage is irrelevant)

Same frozen-A linear regression setup as parent experiment.
Pure numpy/scipy. Runs in <60 seconds on Apple Silicon.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats as sp_stats

# ── Configuration ────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 4567, 5678, 9999]  # 10 seeds for better stats
D_MODEL = 64
RANK = 8
N_TRAIN = 1000
N_EVAL = 500
N_EXPERTS = 4
N_GRADIENT_STEPS = 500
LR = 0.01
BATCH_SIZE = 64

# Factor levels
COVERAGE_LEVELS = {"low": 5, "high": 20}
NOISE_LEVELS = {"low": 0.05, "high": 0.30}

# Shared parameters (held constant across conditions)
MODE_CONCENTRATION_LOW_COV = 0.5    # Dirichlet alpha for 5 modes
MODE_CONCENTRATION_HIGH_COV = 2.0   # Dirichlet alpha for 20 modes
INPUT_NOISE_SCALE_LOW_COV = 0.3     # tight clusters
INPUT_NOISE_SCALE_HIGH_COV = 0.8    # wide spread
SYSTEMATIC_BIAS = 0.0               # REMOVED: bias confounds with coverage


# ── Data Source Models ───────────────────────────────────────────

@dataclass
class ConditionConfig:
    """A single cell in the 2x2 factorial design."""
    name: str
    n_modes: int
    mode_concentration: float
    input_noise_scale: float
    label_noise_std: float


def make_conditions() -> Dict[str, ConditionConfig]:
    """Create all 4 conditions of the 2x2 design."""
    conditions = {}
    for cov_name, n_modes in COVERAGE_LEVELS.items():
        for noise_name, sigma in NOISE_LEVELS.items():
            key = f"{cov_name}_cov_{noise_name}_noise"
            # Input distribution params scale with coverage
            if n_modes == 5:
                alpha = MODE_CONCENTRATION_LOW_COV
                input_scale = INPUT_NOISE_SCALE_LOW_COV
            else:
                alpha = MODE_CONCENTRATION_HIGH_COV
                input_scale = INPUT_NOISE_SCALE_HIGH_COV
            conditions[key] = ConditionConfig(
                name=key,
                n_modes=n_modes,
                mode_concentration=alpha,
                input_noise_scale=input_scale,
                label_noise_std=sigma,
            )
    return conditions


# ── Data Generation (reused from parent) ─────────────────────────

def make_mode_centers(n_modes: int, d: int, rng: np.random.Generator) -> np.ndarray:
    centers = rng.standard_normal((n_modes, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return centers


def generate_inputs(cfg: ConditionConfig, n: int, d: int,
                    rng: np.random.Generator,
                    mode_centers: np.ndarray) -> np.ndarray:
    alpha = np.ones(cfg.n_modes) * cfg.mode_concentration
    weights = rng.dirichlet(alpha)
    assignments = rng.choice(cfg.n_modes, size=n, p=weights)
    X = np.zeros((n, d))
    for i in range(n):
        X[i] = mode_centers[assignments[i]] + \
               rng.standard_normal(d) * cfg.input_noise_scale
    return X


def generate_labels(X: np.ndarray, W_star: np.ndarray,
                    noise_std: float, rng: np.random.Generator) -> np.ndarray:
    return X @ W_star + rng.standard_normal(X.shape) * noise_std


# ── LoRA Training (reused from parent) ───────────────────────────

def train_lora(X: np.ndarray, y: np.ndarray, d: int, rank: int,
               n_steps: int, lr: float,
               rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Train B (learned) with A frozen. Returns (A, B)."""
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


# ── Metrics (reused from parent) ────────────────────────────────

def quality(A, B, X_eval, W_star):
    y_pred = X_eval @ A.T @ B.T
    y_true = X_eval @ W_star
    return max(0, 1 - np.linalg.norm(y_pred - y_true, 'fro') /
               (np.linalg.norm(y_true, 'fro') + 1e-10))


def effective_rank(X):
    S = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = S / (S.sum() + 1e-10)
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


# ── ANOVA-style Decomposition ───────────────────────────────────

def decompose_effects(cell_means: Dict[str, float]) -> Dict:
    """
    Two-way ANOVA decomposition of the 2x2 factorial.

    cell_means keys: "low_cov_low_noise", "low_cov_high_noise",
                     "high_cov_low_noise", "high_cov_high_noise"

    Returns main effects, interaction, and variance explained.
    """
    q_ll = cell_means["low_cov_low_noise"]    # low coverage, low noise
    q_lh = cell_means["low_cov_high_noise"]   # low coverage, high noise
    q_hl = cell_means["high_cov_low_noise"]   # high coverage, low noise
    q_hh = cell_means["high_cov_high_noise"]  # high coverage, high noise

    grand_mean = (q_ll + q_lh + q_hl + q_hh) / 4.0

    # Main effect of COVERAGE: high_cov mean - low_cov mean
    cov_high_mean = (q_hl + q_hh) / 2.0
    cov_low_mean = (q_ll + q_lh) / 2.0
    coverage_effect = cov_high_mean - cov_low_mean

    # Main effect of NOISE: low_noise mean - high_noise mean (positive = low noise better)
    noise_low_mean = (q_ll + q_hl) / 2.0
    noise_high_mean = (q_lh + q_hh) / 2.0
    noise_effect = noise_low_mean - noise_high_mean

    # Interaction: deviation from additivity
    # If effects were purely additive: q_hh = grand_mean + coverage/2 - noise/2
    # Interaction = q_hh - (grand_mean + coverage_effect/2 - noise_effect/2)
    # More standard: interaction = (q_ll + q_hh)/2 - (q_lh + q_hl)/2
    interaction = (q_ll + q_hh) / 2.0 - (q_lh + q_hl) / 2.0

    # The "total gap" from parent experiment: Q(real) - Q(synthetic)
    # = Q(high_cov/high_noise) - Q(low_cov/low_noise) = q_hh - q_ll
    total_gap = q_hh - q_ll

    # Decompose total_gap into:
    # total_gap = coverage_contribution + noise_contribution + interaction_contribution
    # Going from (low_cov, low_noise) to (high_cov, high_noise):
    #   coverage_contribution = coverage_effect (changing coverage, averaged over noise)
    #   noise_contribution = -noise_effect (changing from low to high noise, averaged over coverage)
    #   interaction adjusts for non-additivity
    coverage_contribution = coverage_effect
    noise_contribution = -noise_effect  # negative because going low->high noise hurts

    # Verify: total_gap = coverage + noise + interaction
    # q_hh - q_ll = [(q_hl+q_hh)/2 - (q_ll+q_lh)/2] + [(q_lh+q_hh)/2 - (q_ll+q_hl)/2]
    #               + [(q_ll+q_hh)/2 - (q_lh+q_hl)/2]
    # Simplify: = (q_hl+q_hh-q_ll-q_lh)/2 + (q_lh+q_hh-q_ll-q_hl)/2 + (q_ll+q_hh-q_lh-q_hl)/2
    #           = (q_hl+q_hh-q_ll-q_lh + q_lh+q_hh-q_ll-q_hl + q_ll+q_hh-q_lh-q_hl) / 2
    #           = (3*q_hh - q_ll - q_lh - q_hl) / 2
    # This doesn't simplify to q_hh - q_ll. The decomposition is:
    #   q_hh - q_ll = coverage_effect + (-noise_effect) + interaction - interaction
    #   ... Actually, for a proper path decomposition:
    #
    # Path from (low,low) to (high,high):
    #   Via (high,low): [q_hl - q_ll] + [q_hh - q_hl]
    #                 = pure_coverage_at_low_noise + pure_noise_at_high_coverage
    #   Via (low,high): [q_lh - q_ll] + [q_hh - q_lh]
    #                 = pure_noise_at_low_coverage + pure_coverage_at_high_noise

    # Simple effects (conditional on the other factor level)
    coverage_at_low_noise = q_hl - q_ll    # coverage effect when noise is low
    coverage_at_high_noise = q_hh - q_lh   # coverage effect when noise is high
    noise_at_low_cov = q_ll - q_lh         # noise effect (low-high) when coverage is low
    noise_at_high_cov = q_hl - q_hh        # noise effect (low-high) when coverage is high

    # Variance decomposition (SS / SS_total)
    ss_total = sum((q - grand_mean)**2 for q in [q_ll, q_lh, q_hl, q_hh])
    ss_coverage = 2 * (cov_high_mean - grand_mean)**2 + 2 * (cov_low_mean - grand_mean)**2
    ss_noise = 2 * (noise_low_mean - grand_mean)**2 + 2 * (noise_high_mean - grand_mean)**2
    ss_interaction = (q_ll - cov_low_mean - noise_low_mean + grand_mean)**2 + \
                     (q_lh - cov_low_mean - noise_high_mean + grand_mean)**2 + \
                     (q_hl - cov_high_mean - noise_low_mean + grand_mean)**2 + \
                     (q_hh - cov_high_mean - noise_high_mean + grand_mean)**2

    # Percent of total variance explained
    if ss_total > 1e-15:
        pct_coverage = ss_coverage / ss_total * 100
        pct_noise = ss_noise / ss_total * 100
        pct_interaction = ss_interaction / ss_total * 100
    else:
        pct_coverage = pct_noise = pct_interaction = 0.0

    return {
        "cell_means": {
            "low_cov_low_noise": float(q_ll),
            "low_cov_high_noise": float(q_lh),
            "high_cov_low_noise": float(q_hl),
            "high_cov_high_noise": float(q_hh),
        },
        "grand_mean": float(grand_mean),
        "main_effects": {
            "coverage": float(coverage_effect),
            "noise": float(noise_effect),
        },
        "interaction": float(interaction),
        "simple_effects": {
            "coverage_at_low_noise": float(coverage_at_low_noise),
            "coverage_at_high_noise": float(coverage_at_high_noise),
            "noise_at_low_cov": float(noise_at_low_cov),
            "noise_at_high_cov": float(noise_at_high_cov),
        },
        "variance_explained_pct": {
            "coverage": float(pct_coverage),
            "noise": float(pct_noise),
            "interaction": float(pct_interaction),
        },
        "total_gap": float(total_gap),
    }


# ── Main Experiment ──────────────────────────────────────────────

def run_seed(seed: int) -> Dict:
    rng = np.random.default_rng(seed)

    # Ground truth: rank-r task
    U = rng.standard_normal((D_MODEL, RANK))
    V = rng.standard_normal((RANK, D_MODEL))
    W_star = U @ V * 0.1

    # Uniform evaluation data (the unbiased test)
    X_eval = rng.standard_normal((N_EVAL, D_MODEL)) * 0.5

    conditions = make_conditions()

    # Pre-generate mode centers (shared across experts within a condition)
    # Use separate rng draws for 5 and 20 modes
    mode_centers = {
        5: make_mode_centers(5, D_MODEL, rng),
        20: make_mode_centers(20, D_MODEL, rng),
    }

    result = {"seed": seed, "conditions": {}}

    for cond_key, cfg in conditions.items():
        qs = []
        divs = []
        for _ in range(N_EXPERTS):
            X = generate_inputs(cfg, N_TRAIN, D_MODEL, rng,
                                mode_centers[cfg.n_modes])
            y = generate_labels(X, W_star, cfg.label_noise_std, rng)
            divs.append(effective_rank(X))
            A, B = train_lora(X, y, D_MODEL, RANK, N_GRADIENT_STEPS, LR, rng)
            qs.append(quality(A, B, X_eval, W_star))

        result["conditions"][cond_key] = {
            "quality_mean": float(np.mean(qs)),
            "quality_std": float(np.std(qs)),
            "quality_all": [float(q) for q in qs],
            "effective_rank_mean": float(np.mean(divs)),
        }

    return result


def run_statistical_tests(all_results: List[Dict]) -> Dict:
    """Run paired statistical tests on per-seed means."""
    n = len(all_results)
    conditions = list(all_results[0]["conditions"].keys())

    # Collect per-seed means for each condition
    per_seed = {c: [] for c in conditions}
    for r in all_results:
        for c in conditions:
            per_seed[c].append(r["conditions"][c]["quality_mean"])

    tests = {}

    # Paired t-tests for simple effects
    pairs = [
        ("coverage_at_low_noise", "high_cov_low_noise", "low_cov_low_noise"),
        ("coverage_at_high_noise", "high_cov_high_noise", "low_cov_high_noise"),
        ("noise_at_low_cov", "low_cov_low_noise", "low_cov_high_noise"),
        ("noise_at_high_cov", "high_cov_low_noise", "high_cov_high_noise"),
    ]

    for name, a_key, b_key in pairs:
        a = np.array(per_seed[a_key])
        b = np.array(per_seed[b_key])
        diff = a - b
        t, p = sp_stats.ttest_rel(a, b)
        tests[name] = {
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "t_stat": float(t),
            "p_value": float(p),
            "significant_005": bool(p < 0.05),
            "n": n,
        }

    # Two-way ANOVA F-tests (using per-seed decomposition)
    cov_effects = []
    noise_effects = []
    interactions = []
    for r in all_results:
        cm = {c: r["conditions"][c]["quality_mean"] for c in conditions}
        d = decompose_effects(cm)
        cov_effects.append(d["main_effects"]["coverage"])
        noise_effects.append(d["main_effects"]["noise"])
        interactions.append(d["interaction"])

    # One-sample t-test: is each effect significantly different from zero?
    for name, vals in [("coverage_main", cov_effects),
                       ("noise_main", noise_effects),
                       ("interaction", interactions)]:
        arr = np.array(vals)
        t, p = sp_stats.ttest_1samp(arr, 0)
        tests[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "t_stat": float(t),
            "p_value": float(p),
            "significant_005": bool(p < 0.05),
            "n": n,
        }

    return tests


def aggregate(all_results: List[Dict]) -> Dict:
    n = len(all_results)
    conditions = list(all_results[0]["conditions"].keys())

    agg = {
        "n_seeds": n,
        "config": {
            "d": D_MODEL, "r": RANK, "n_train": N_TRAIN,
            "n_eval": N_EVAL, "n_experts": N_EXPERTS,
            "steps": N_GRADIENT_STEPS, "lr": LR,
            "coverage_levels": COVERAGE_LEVELS,
            "noise_levels": NOISE_LEVELS,
        },
        "conditions": {},
    }

    # Per-condition aggregation
    for c in conditions:
        qs = [r["conditions"][c]["quality_mean"] for r in all_results]
        ers = [r["conditions"][c]["effective_rank_mean"] for r in all_results]
        agg["conditions"][c] = {
            "quality": {
                "mean": float(np.mean(qs)),
                "std": float(np.std(qs)),
                "ci95": float(1.96 * np.std(qs) / np.sqrt(n)),
                "per_seed": [float(q) for q in qs],
            },
            "effective_rank": {
                "mean": float(np.mean(ers)),
                "std": float(np.std(ers)),
            },
        }

    # ANOVA decomposition on the aggregated means
    cell_means = {c: agg["conditions"][c]["quality"]["mean"] for c in conditions}
    agg["anova"] = decompose_effects(cell_means)

    # Also compute per-seed ANOVA to get confidence intervals on effects
    per_seed_coverage = []
    per_seed_noise = []
    per_seed_interaction = []
    per_seed_pct_cov = []
    per_seed_pct_noise = []
    per_seed_pct_int = []

    for r in all_results:
        cm = {c: r["conditions"][c]["quality_mean"] for c in conditions}
        d = decompose_effects(cm)
        per_seed_coverage.append(d["main_effects"]["coverage"])
        per_seed_noise.append(d["main_effects"]["noise"])
        per_seed_interaction.append(d["interaction"])
        per_seed_pct_cov.append(d["variance_explained_pct"]["coverage"])
        per_seed_pct_noise.append(d["variance_explained_pct"]["noise"])
        per_seed_pct_int.append(d["variance_explained_pct"]["interaction"])

    agg["effects_distribution"] = {
        "coverage": {
            "mean": float(np.mean(per_seed_coverage)),
            "std": float(np.std(per_seed_coverage)),
            "ci95": float(1.96 * np.std(per_seed_coverage) / np.sqrt(n)),
        },
        "noise": {
            "mean": float(np.mean(per_seed_noise)),
            "std": float(np.std(per_seed_noise)),
            "ci95": float(1.96 * np.std(per_seed_noise) / np.sqrt(n)),
        },
        "interaction": {
            "mean": float(np.mean(per_seed_interaction)),
            "std": float(np.std(per_seed_interaction)),
            "ci95": float(1.96 * np.std(per_seed_interaction) / np.sqrt(n)),
        },
        "variance_explained_pct": {
            "coverage": {
                "mean": float(np.mean(per_seed_pct_cov)),
                "std": float(np.std(per_seed_pct_cov)),
            },
            "noise": {
                "mean": float(np.mean(per_seed_pct_noise)),
                "std": float(np.std(per_seed_pct_noise)),
            },
            "interaction": {
                "mean": float(np.mean(per_seed_pct_int)),
                "std": float(np.std(per_seed_pct_int)),
            },
        },
    }

    # Statistical tests
    agg["statistical_tests"] = run_statistical_tests(all_results)

    # Kill criteria evaluation
    pct_cov = agg["effects_distribution"]["variance_explained_pct"]["coverage"]["mean"]
    pct_noise = agg["effects_distribution"]["variance_explained_pct"]["noise"]["mean"]

    agg["kill_criteria"] = {
        "K1": {
            "desc": "coverage alone explains <50% of the quality gap",
            "coverage_pct": round(pct_cov, 1),
            "threshold": 50,
            "killed": pct_cov < 50,
            "verdict": "KILLED" if pct_cov < 50 else "SURVIVES",
        },
        "K2": {
            "desc": "noise alone explains >80% of the gap",
            "noise_pct": round(pct_noise, 1),
            "threshold": 80,
            "killed": pct_noise > 80,
            "verdict": "KILLED" if pct_noise > 80 else "SURVIVES",
        },
    }

    return agg


def print_report(agg):
    print("=" * 78)
    print("COVERAGE vs NOISE DISENTANGLE: 2x2 FACTORIAL ABLATION")
    print("=" * 78)
    c = agg["config"]
    print(f"\nConfig: d={c['d']}, r={c['r']}, N_train={c['n_train']}, "
          f"steps={c['steps']}, seeds={agg['n_seeds']}")

    # 2x2 quality table
    print("\n--- Quality (uniform eval, 1 - relative Frobenius error) ---")
    print(f"{'':20s} {'Low Noise (0.05)':>20s} {'High Noise (0.30)':>20s}")
    print(f"{'':20s} {'':>20s} {'':>20s}")

    for cov_name in ["low", "high"]:
        label = f"{cov_name.capitalize()} Coverage ({COVERAGE_LEVELS[cov_name]})"
        vals = []
        for noise_name in ["low", "high"]:
            key = f"{cov_name}_cov_{noise_name}_noise"
            q = agg["conditions"][key]["quality"]
            vals.append(f"{q['mean']:.4f} +/- {q['ci95']:.4f}")
        print(f"{label:<20s} {vals[0]:>20s} {vals[1]:>20s}")

    # Effective rank table
    print("\n--- Effective Rank ---")
    for cov_name in ["low", "high"]:
        for noise_name in ["low", "high"]:
            key = f"{cov_name}_cov_{noise_name}_noise"
            er = agg["conditions"][key]["effective_rank"]
            print(f"  {key}: {er['mean']:.1f}")

    # ANOVA decomposition
    anova = agg["anova"]
    print("\n--- ANOVA Decomposition (on aggregate means) ---")
    print(f"  Grand mean:       {anova['grand_mean']:.5f}")
    print(f"  Coverage effect:  {anova['main_effects']['coverage']:+.5f} "
          f"(high - low coverage)")
    print(f"  Noise effect:     {anova['main_effects']['noise']:+.5f} "
          f"(low - high noise, positive = low noise better)")
    print(f"  Interaction:      {anova['interaction']:+.5f}")

    # Variance explained
    ve = agg["effects_distribution"]["variance_explained_pct"]
    print(f"\n--- Variance Explained (mean across seeds) ---")
    print(f"  Coverage:    {ve['coverage']['mean']:5.1f}% +/- {ve['coverage']['std']:.1f}%")
    print(f"  Noise:       {ve['noise']['mean']:5.1f}% +/- {ve['noise']['std']:.1f}%")
    print(f"  Interaction: {ve['interaction']['mean']:5.1f}% +/- {ve['interaction']['std']:.1f}%")

    # Simple effects
    se = anova["simple_effects"]
    print(f"\n--- Simple Effects ---")
    print(f"  Coverage at low noise:  {se['coverage_at_low_noise']:+.5f}")
    print(f"  Coverage at high noise: {se['coverage_at_high_noise']:+.5f}")
    print(f"  Noise at low coverage:  {se['noise_at_low_cov']:+.5f}")
    print(f"  Noise at high coverage: {se['noise_at_high_cov']:+.5f}")

    # Statistical tests
    print(f"\n--- Statistical Tests (paired t-tests, n={agg['n_seeds']}) ---")
    for name, t in agg["statistical_tests"].items():
        sig = "*" if t.get("significant_005", False) else ""
        # Simple effects use "mean_diff", main effects use "mean"
        mean_val = t.get("mean", t.get("mean_diff", 0.0))
        print(f"  {name:<25s} mean={mean_val:+.5f}  t={t['t_stat']:+.3f}  "
              f"p={t['p_value']:.4f} {sig}")

    # Kill criteria
    print("\n--- Kill Criteria ---")
    k1 = agg["kill_criteria"]["K1"]
    print(f"  K1: coverage explains {k1['coverage_pct']:.1f}% "
          f"(threshold <50%) -> {k1['verdict']}")
    k2 = agg["kill_criteria"]["K2"]
    print(f"  K2: noise explains {k2['noise_pct']:.1f}% "
          f"(threshold >80%) -> {k2['verdict']}")

    # Key findings
    print("\n" + "=" * 78)
    print("KEY FINDINGS")
    print("=" * 78)

    cov_pct = ve["coverage"]["mean"]
    noise_pct = ve["noise"]["mean"]
    int_pct = ve["interaction"]["mean"]

    if cov_pct > noise_pct:
        print(f"1. COVERAGE DOMINATES: {cov_pct:.1f}% of variance vs "
              f"noise {noise_pct:.1f}%")
    elif noise_pct > cov_pct:
        print(f"1. NOISE DOMINATES: {noise_pct:.1f}% of variance vs "
              f"coverage {cov_pct:.1f}%")
    else:
        print(f"1. BALANCED: coverage {cov_pct:.1f}% vs noise {noise_pct:.1f}%")

    print(f"2. INTERACTION: {int_pct:.1f}% of variance "
          f"({'significant' if int_pct > 10 else 'negligible'})")

    # Total gap reproduction
    q_synth = agg["conditions"]["low_cov_low_noise"]["quality"]["mean"]
    q_real = agg["conditions"]["high_cov_high_noise"]["quality"]["mean"]
    gap = (q_real - q_synth) / (q_real + 1e-10) * 100
    print(f"3. TOTAL GAP REPRODUCED: {gap:.1f}% "
          f"(parent: 58.1%, {q_synth:.4f} vs {q_real:.4f})")

    # The counterfactual: what would happen with coverage only?
    q_best = agg["conditions"]["high_cov_low_noise"]["quality"]["mean"]
    print(f"4. BEST CONDITION: high-cov/low-noise = {q_best:.4f} "
          f"(coverage + clean labels)")

    # Practical recommendation
    cov_test = agg["statistical_tests"].get("coverage_main", {})
    noise_test = agg["statistical_tests"].get("noise_main", {})
    print(f"5. STATISTICAL SIGNIFICANCE: coverage p={cov_test.get('p_value', 'N/A'):.4f}, "
          f"noise p={noise_test.get('p_value', 'N/A'):.4f}")


def main():
    print("Running coverage vs noise disentangle 2x2 factorial...")
    all_results = []
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        all_results.append(run_seed(seed))
        print("done")

    agg = aggregate(all_results)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved to {out}")

    print_report(agg)
    return agg


if __name__ == "__main__":
    main()
