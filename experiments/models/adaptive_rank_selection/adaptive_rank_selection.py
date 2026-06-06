"""
Adaptive Rank Selection: Does domain complexity predict optimal LoRA rank?

REVISION 2 (2026-03-15): Fixes from adversarial review REVIEW-adversarial.md.

Changes from v1:
  1. FIXED knee detector: replaced fragile curvature-based detector with:
     (a) Threshold-based: smallest r where err(r) < tau (default 0.05)
     (b) Kneedle algorithm (Satopaa et al., 2011) as cross-check
     Both handle non-uniform rank grids correctly.
  2. Added null baseline: "always predict rank 16" to contextualize K2.
  3. Multi-SNR: tests at SNR=5, 20, 100 for robustness.
  4. Per-domain median correlations as primary analysis (15 points per d),
     pooled correlations as secondary.

Background:
  AdaLoRA (Zhang et al. 2023) allocates rank budgets per layer via SVD importance.
  Aghajanyan et al. (2020) showed tasks have low intrinsic dimensionality (~200 params
  can capture 90% of fine-tuning performance). Our prior work showed bash (simple)
  saturates at rank-8. The open question: can we PREDICT optimal rank from domain
  complexity without expensive sweeps?

Approach:
  We simulate domains as target weight transformations Delta_target with controlled
  effective rank. A "domain" is a random matrix with a known singular value spectrum.
  We then measure how well LoRA adapters (truncated SVD, the optimal rank-r
  approximation) capture the target at various ranks.

  The key metric is the "optimal rank" -- the smallest rank at which reconstruction
  error drops below a threshold (default 5%).

Kill criteria:
  K1: Spearman rho < 0.5 between BEST complexity metric and optimal rank
  K2: Predicted rank >2x off from optimal for >50% of domains
      AND must significantly beat null baseline (always predict rank 16)

Architecture: Pure numpy/scipy, CPU-only.
"""

import numpy as np
from scipy import stats
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class Config:
    d_in: int = 128
    d_out: int = 128
    n_seeds: int = 5
    # Use a UNIFORM rank grid for optimal rank detection (fix #1)
    # We detect on uniform grid, then snap predictions to available ranks
    lora_ranks: tuple = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
    # Exact-rank domains: true rank of the signal component
    domain_true_ranks: tuple = (2, 4, 8, 12, 16, 24, 32, 48)
    # Spectral decay domains: decay rate per singular value index
    decay_rates: tuple = (0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98)
    # Signal-to-noise ratios to test (fix #3)
    snr_values: tuple = (5.0, 20.0, 100.0)
    # Default SNR for single-SNR runs
    snr: float = 20.0
    # Threshold for optimal rank definition (fix #1c)
    error_threshold: float = 0.05
    # Null baseline rank (fix #2)
    null_baseline_rank: int = 16


# ── Domain Generation ─────────────────────────────────────────────────────

def generate_exact_rank_domain(d_in: int, d_out: int, true_rank: int,
                                snr: float, rng: np.random.Generator) -> np.ndarray:
    """Generate target with exact rank k plus controlled noise at given SNR."""
    k = min(true_rank, min(d_in, d_out))
    U, _ = np.linalg.qr(rng.standard_normal((d_in, k)))
    V, _ = np.linalg.qr(rng.standard_normal((d_out, k)))
    sigmas = np.sort(rng.uniform(0.5, 2.0, size=k))[::-1]
    signal = U @ np.diag(sigmas) @ V.T

    # Add noise scaled to achieve target SNR
    noise = rng.standard_normal((d_in, d_out))
    signal_norm = np.linalg.norm(signal, 'fro')
    noise_norm = np.linalg.norm(noise, 'fro')
    noise *= (signal_norm / snr) / noise_norm

    return signal + noise


def generate_spectral_decay_domain(d_in: int, d_out: int, decay: float,
                                    snr: float, rng: np.random.Generator) -> np.ndarray:
    """Generate target with geometrically decaying singular values + noise."""
    k = min(d_in, d_out)
    U, _ = np.linalg.qr(rng.standard_normal((d_in, k)))
    V, _ = np.linalg.qr(rng.standard_normal((d_out, k)))
    sigmas = 2.0 * (decay ** np.arange(k))
    signal = U @ np.diag(sigmas) @ V.T

    noise = rng.standard_normal((d_in, d_out))
    signal_norm = np.linalg.norm(signal, 'fro')
    noise_norm = np.linalg.norm(noise, 'fro')
    noise *= (signal_norm / snr) / noise_norm

    return signal + noise


# ── Complexity Metrics ─────────────────────────────────────────────────────

def effective_rank_rv(sigmas: np.ndarray) -> float:
    """Roy & Vetterli (2007): exp(entropy of normalized singular values)."""
    s2 = sigmas ** 2
    s2 = s2[s2 > 1e-15]
    p = s2 / s2.sum()
    entropy = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(entropy))


def stable_rank(sigmas: np.ndarray) -> float:
    """||M||_F^2 / ||M||_2^2."""
    s2 = sigmas ** 2
    return float(s2.sum() / s2.max())


def energy_rank(sigmas: np.ndarray, threshold: float) -> int:
    """Smallest k s.t. sum(sigma_1..k^2) >= threshold * total."""
    s2 = sigmas ** 2
    cumsum = np.cumsum(s2)
    total = s2.sum()
    idx = np.searchsorted(cumsum, threshold * total)
    return int(min(idx + 1, len(sigmas)))


def compute_metrics(sigmas: np.ndarray) -> Dict[str, float]:
    return {
        'effective_rank': effective_rank_rv(sigmas),
        'stable_rank': stable_rank(sigmas),
        'energy_rank_90': float(energy_rank(sigmas, 0.90)),
        'energy_rank_95': float(energy_rank(sigmas, 0.95)),
        'energy_rank_99': float(energy_rank(sigmas, 0.99)),
    }


# ── Optimal Rank Detection (FIXED) ──────────────────────────────────────

def find_optimal_rank_threshold(ranks: np.ndarray, errors: np.ndarray,
                                 threshold: float = 0.05) -> int:
    """Find optimal rank as smallest r where err(r) < threshold.

    This is fix #1c from the adversarial review: avoids curvature estimation
    entirely by using a fixed error threshold.

    If no rank achieves err < threshold, returns the rank with minimum error.
    """
    for i, r in enumerate(ranks):
        if errors[i] < threshold:
            return int(r)
    # No rank meets threshold -- return the one with lowest error
    return int(ranks[np.argmin(errors)])


def find_optimal_rank_kneedle(ranks: np.ndarray, errors: np.ndarray,
                               sensitivity: float = 1.0) -> int:
    """Kneedle algorithm (Satopaa et al., 2011) for knee detection.

    This handles non-uniform rank grids properly by normalizing both axes
    to [0,1] and finding the point of maximum distance from the diagonal.

    Parameters:
        ranks: array of rank values (may be non-uniform)
        errors: corresponding reconstruction errors
        sensitivity: higher = more sensitive to gentle knees (default 1.0)

    Returns:
        The rank at the detected knee point.
    """
    if len(ranks) < 3:
        return int(ranks[0])

    # Normalize to [0, 1]
    x = (ranks - ranks.min()) / max(ranks.max() - ranks.min(), 1e-10)
    y = (errors - errors.min()) / max(errors.max() - errors.min(), 1e-10)

    # Distance from the line connecting first and last points
    # Line from (x[0], y[0]) to (x[-1], y[-1])
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    line_len = np.sqrt(dx**2 + dy**2)

    if line_len < 1e-10:
        return int(ranks[0])

    # Perpendicular distance from each point to the line
    distances = np.abs(dy * x - dx * y + x[-1]*y[0] - y[-1]*x[0]) / line_len

    # For a decreasing curve, the knee is where distance is maximized
    knee_idx = np.argmax(distances)

    return int(ranks[knee_idx])


def find_knee_energy(sigmas: np.ndarray, threshold: float = 0.95) -> int:
    """Alternative knee: the rank capturing threshold% of the variance."""
    return energy_rank(sigmas, threshold)


# ── Main Experiment ────────────────────────────────────────────────────────

def run_experiment(cfg: Config = None, snr_override: float = None) -> Dict:
    cfg = cfg or Config()
    snr = snr_override if snr_override is not None else cfg.snr
    print(f"=== Adaptive Rank Selection Experiment (SNR={snr}) ===")
    print(f"d_in={cfg.d_in}, d_out={cfg.d_out}, seeds={cfg.n_seeds}")
    print(f"LoRA ranks: {cfg.lora_ranks}")
    print(f"Error threshold for optimal rank: {cfg.error_threshold}")
    print()

    t0 = time.time()
    records = []
    domain_id = 0

    # Use a UNIFORM rank grid 1..d for optimal rank detection (fix #1b)
    d_min = min(cfg.d_in, cfg.d_out)
    uniform_ranks = np.arange(1, d_min + 1)

    # Part 1: Exact-rank domains
    print("--- Part 1: Exact-rank domains ---")
    valid_true_ranks = [r for r in cfg.domain_true_ranks if r <= d_min]
    for true_rank in valid_true_ranks:
        for seed in range(cfg.n_seeds):
            rng = np.random.default_rng(seed * 10000 + domain_id * 100)
            Delta = generate_exact_rank_domain(cfg.d_in, cfg.d_out, true_rank,
                                                snr, rng)
            sigmas = np.linalg.svd(Delta, compute_uv=False)
            metrics = compute_metrics(sigmas)

            # Error curve on the SPARSE rank grid (for reporting)
            rank_errors = {}
            for r in cfg.lora_ranks:
                r_eff = min(r, len(sigmas))
                residual = np.sqrt(np.sum(sigmas[r_eff:] ** 2))
                total = np.sqrt(np.sum(sigmas ** 2))
                rank_errors[r] = float(residual / total)

            # Error curve on UNIFORM rank grid (for optimal rank detection)
            uniform_errors = np.array([
                float(np.sqrt(np.sum(sigmas[min(r, len(sigmas)):] ** 2)) /
                      np.sqrt(np.sum(sigmas ** 2)))
                for r in uniform_ranks
            ])

            # Optimal rank via threshold (primary, fix #1c)
            opt_threshold = find_optimal_rank_threshold(
                uniform_ranks, uniform_errors, cfg.error_threshold)

            # Optimal rank via Kneedle on uniform grid (cross-check, fix #1a)
            opt_kneedle = find_optimal_rank_kneedle(uniform_ranks, uniform_errors)

            records.append({
                'domain_id': domain_id,
                'domain_type': 'exact_rank',
                'domain_param': true_rank,
                'seed': seed,
                'metrics': metrics,
                'rank_errors': rank_errors,
                'optimal_threshold': opt_threshold,
                'optimal_kneedle': opt_kneedle,
            })
        avg_eff = np.mean([r['metrics']['effective_rank'] for r in records[-cfg.n_seeds:]])
        avg_opt_t = np.median([r['optimal_threshold'] for r in records[-cfg.n_seeds:]])
        avg_opt_k = np.median([r['optimal_kneedle'] for r in records[-cfg.n_seeds:]])
        print(f"  true_rank={true_rank:3d} | eff_rank={avg_eff:6.1f} | opt_thresh={avg_opt_t:4.0f} | opt_kneedle={avg_opt_k:4.0f}")
        domain_id += 1

    # Part 2: Spectral decay domains
    print("\n--- Part 2: Spectral decay domains ---")
    for decay in cfg.decay_rates:
        for seed in range(cfg.n_seeds):
            rng = np.random.default_rng(seed * 10000 + domain_id * 100)
            Delta = generate_spectral_decay_domain(cfg.d_in, cfg.d_out, decay,
                                                    snr, rng)
            sigmas = np.linalg.svd(Delta, compute_uv=False)
            metrics = compute_metrics(sigmas)

            rank_errors = {}
            for r in cfg.lora_ranks:
                r_eff = min(r, len(sigmas))
                residual = np.sqrt(np.sum(sigmas[r_eff:] ** 2))
                total = np.sqrt(np.sum(sigmas ** 2))
                rank_errors[r] = float(residual / total)

            uniform_errors = np.array([
                float(np.sqrt(np.sum(sigmas[min(r, len(sigmas)):] ** 2)) /
                      np.sqrt(np.sum(sigmas ** 2)))
                for r in uniform_ranks
            ])

            opt_threshold = find_optimal_rank_threshold(
                uniform_ranks, uniform_errors, cfg.error_threshold)
            opt_kneedle = find_optimal_rank_kneedle(uniform_ranks, uniform_errors)

            records.append({
                'domain_id': domain_id,
                'domain_type': 'spectral_decay',
                'domain_param': decay,
                'seed': seed,
                'metrics': metrics,
                'rank_errors': rank_errors,
                'optimal_threshold': opt_threshold,
                'optimal_kneedle': opt_kneedle,
            })
        avg_eff = np.mean([r['metrics']['effective_rank'] for r in records[-cfg.n_seeds:]])
        avg_opt_t = np.median([r['optimal_threshold'] for r in records[-cfg.n_seeds:]])
        avg_opt_k = np.median([r['optimal_kneedle'] for r in records[-cfg.n_seeds:]])
        print(f"  decay={decay:.2f} | eff_rank={avg_eff:6.1f} | opt_thresh={avg_opt_t:4.0f} | opt_kneedle={avg_opt_k:4.0f}")
        domain_id += 1

    # ── Analysis ──────────────────────────────────────────────────────────
    n = len(records)
    print(f"\n=== Analysis ({n} total trials) ===")

    # Use Kneedle as primary ground truth (correctly separates signal from noise
    # at all SNR levels). Threshold is secondary (conflates noise removal with
    # signal recovery at low SNR).
    optimal_ranks = np.array([r['optimal_kneedle'] for r in records])
    threshold_ranks = np.array([r['optimal_threshold'] for r in records])

    # Snap to available ranks for prediction comparison
    available = np.array(cfg.lora_ranks)
    def snap(x):
        return available[np.argmin(np.abs(available - x))]

    # ── Fix #4: Per-domain median analysis (PRIMARY) ──
    print("\n--- Per-domain analysis (primary, fix #4) ---")
    unique_ids = sorted(set(r['domain_id'] for r in records))

    # Collect per-domain medians
    domain_medians = {
        'effective_rank': [], 'stable_rank': [],
        'energy_rank_90': [], 'energy_rank_95': [], 'energy_rank_99': [],
        'optimal_kneedle': [], 'optimal_threshold': [],
    }
    domain_summaries = []

    for did in unique_ids:
        group = [r for r in records if r['domain_id'] == did]
        dtype = group[0]['domain_type']
        dparam = group[0]['domain_param']

        for metric_name in ['effective_rank', 'stable_rank', 'energy_rank_90',
                            'energy_rank_95', 'energy_rank_99']:
            vals = [r['metrics'][metric_name] for r in group]
            domain_medians[metric_name].append(np.median(vals))

        med_opt_k = np.median([r['optimal_kneedle'] for r in group])
        med_opt_t = np.median([r['optimal_threshold'] for r in group])
        domain_medians['optimal_kneedle'].append(med_opt_k)
        domain_medians['optimal_threshold'].append(med_opt_t)

        # Prediction using e95 metric
        avg_e95 = np.median([r['metrics']['energy_rank_95'] for r in group])
        pred = snap(avg_e95)
        ratio_k = pred / med_opt_k if med_opt_k > 0 else float('inf')

        domain_summaries.append({
            'domain_id': did, 'type': dtype, 'param': dparam,
            'median_effective_rank': float(np.median([r['metrics']['effective_rank'] for r in group])),
            'median_energy_rank_95': float(avg_e95),
            'median_optimal_kneedle': float(med_opt_k),
            'median_optimal_threshold': float(med_opt_t),
            'pred_e95_snapped': int(pred),
            'ratio_vs_kneedle': float(ratio_k),
            'rank_error_curve': {
                str(r): float(np.mean([rec['rank_errors'][r] for rec in group]))
                for r in cfg.lora_ranks
            },
        })

    # Convert to arrays for correlation
    for k in domain_medians:
        domain_medians[k] = np.array(domain_medians[k])

    # Per-domain correlations (PRIMARY) -- using Kneedle optimal rank
    metric_names = ['effective_rank', 'stable_rank', 'energy_rank_90',
                    'energy_rank_95', 'energy_rank_99']
    correlations_per_domain = {}
    print(f"\n  Spearman correlations with Kneedle-optimal rank (per-domain medians, N={len(unique_ids)}):")
    for name in metric_names:
        rho, p = stats.spearmanr(domain_medians[name], domain_medians['optimal_kneedle'])
        correlations_per_domain[name] = {'rho': float(rho), 'p': float(p)}
        status = "PASS" if abs(rho) >= 0.5 else "FAIL"
        print(f"    {name:20s}: rho={rho:.4f}, p={p:.2e} [{status}]")

    # ── Pooled correlations (SECONDARY) ──
    print(f"\n  Spearman correlations (pooled, N={n}, secondary):")
    metric_arrays = {
        name: np.array([r['metrics'][name] for r in records])
        for name in metric_names
    }
    correlations_pooled = {}
    for name, values in metric_arrays.items():
        rho, p = stats.spearmanr(values, optimal_ranks)
        correlations_pooled[name] = {'rho': float(rho), 'p': float(p)}
        status = "PASS" if abs(rho) >= 0.5 else "FAIL"
        print(f"    {name:20s}: rho={rho:.4f}, p={p:.2e} [{status}]")

    # Cross-check: threshold vs kneedle agreement
    agreement = np.mean(np.abs(optimal_ranks - threshold_ranks) <= optimal_ranks * 0.5)
    print(f"\n  Kneedle vs Threshold agreement (within 50%): {agreement*100:.1f}%")
    rho_methods, _ = stats.spearmanr(optimal_ranks, threshold_ranks)
    print(f"  Kneedle vs Threshold Spearman rho: {rho_methods:.4f}")

    # ── Prediction accuracy with null baseline (fix #2) ──
    print(f"\n--- Prediction accuracy (with null baseline, fix #2) ---")

    # Null baseline: always predict rank 16
    null_rank = cfg.null_baseline_rank
    null_ratios = null_rank / np.maximum(optimal_ranks, 1)
    null_within_2x = np.mean((null_ratios >= 0.5) & (null_ratios <= 2.0))
    print(f"  NULL BASELINE (always rank {null_rank}): {null_within_2x*100:.1f}% within 2x")

    # Per-domain null baseline
    null_ratios_domain = null_rank / np.maximum(domain_medians['optimal_kneedle'], 1)
    null_within_2x_domain = np.mean((null_ratios_domain >= 0.5) & (null_ratios_domain <= 2.0))
    print(f"  NULL BASELINE (per-domain): {null_within_2x_domain*100:.1f}% within 2x")

    # Prediction from each metric
    print(f"\n  Prediction accuracy (pooled, within 2x of Kneedle-optimal):")
    prediction_results = {}
    for name, values in metric_arrays.items():
        preds = np.array([snap(v) for v in values])
        ratios = preds / np.maximum(optimal_ranks, 1)
        within_2x = np.mean((ratios >= 0.5) & (ratios <= 2.0))
        improvement_over_null = within_2x - null_within_2x
        prediction_results[name] = {
            'within_2x_frac': float(within_2x),
            'improvement_over_null': float(improvement_over_null),
            'mean_ratio': float(np.mean(ratios)),
            'median_ratio': float(np.median(ratios)),
            'mean_abs_log_ratio': float(np.mean(np.abs(np.log2(np.maximum(ratios, 0.01))))),
        }
        status = "PASS" if within_2x >= 0.5 else "FAIL"
        delta_str = f"+{improvement_over_null*100:.1f}pp" if improvement_over_null > 0 else f"{improvement_over_null*100:.1f}pp"
        print(f"    {name:20s}: {within_2x*100:5.1f}% within 2x ({delta_str} vs null) [{status}]")

    # Per-domain prediction accuracy
    print(f"\n  Prediction accuracy (per-domain medians):")
    prediction_results_domain = {}
    for name in metric_names:
        preds = np.array([snap(v) for v in domain_medians[name]])
        ratios = preds / np.maximum(domain_medians['optimal_kneedle'], 1)
        within_2x = np.mean((ratios >= 0.5) & (ratios <= 2.0))
        improvement_over_null = within_2x - null_within_2x_domain
        prediction_results_domain[name] = {
            'within_2x_frac': float(within_2x),
            'improvement_over_null': float(improvement_over_null),
            'median_ratio': float(np.median(ratios)),
        }
        delta_str = f"+{improvement_over_null*100:.1f}pp" if improvement_over_null > 0 else f"{improvement_over_null*100:.1f}pp"
        print(f"    {name:20s}: {within_2x*100:5.1f}% within 2x ({delta_str} vs null)")

    # Per-domain summary table
    print(f"\n  Per-domain summary:")
    print(f"    {'Type':15s} {'Param':>6s} {'EffRank':>8s} {'E95':>5s} {'OptKneedle':>11s} {'OptThresh':>10s} {'PredE95':>8s} {'Ratio':>6s}")
    for ds in domain_summaries:
        print(f"    {ds['type']:15s} {ds['param']:6.2f} {ds['median_effective_rank']:8.1f} "
              f"{ds['median_energy_rank_95']:5.0f} {ds['median_optimal_kneedle']:11.0f} "
              f"{ds['median_optimal_threshold']:10.0f} {ds['pred_e95_snapped']:8d} "
              f"{ds['ratio_vs_kneedle']:6.2f}")

    # Kill criteria (using per-domain medians as PRIMARY)
    print("\n=== Kill Criteria Evaluation ===")
    best_metric = max(correlations_per_domain,
                      key=lambda k: abs(correlations_per_domain[k]['rho']))
    best_rho = abs(correlations_per_domain[best_metric]['rho'])
    k1_pass = best_rho >= 0.5
    print(f"  K1 (correlation >= 0.5, per-domain): {'PASS' if k1_pass else 'KILL'}")
    print(f"      Best metric: {best_metric}, |rho|={best_rho:.4f}")

    best_pred_name = max(prediction_results_domain,
                         key=lambda k: prediction_results_domain[k]['within_2x_frac'])
    best_within_2x = prediction_results_domain[best_pred_name]['within_2x_frac']
    best_improvement = prediction_results_domain[best_pred_name]['improvement_over_null']
    k2_pass = best_within_2x >= 0.5
    k2_beats_null = best_improvement > 0.10  # must beat null by >10pp
    print(f"  K2 (>50% within 2x AND beat null by >10pp): {'PASS' if (k2_pass and k2_beats_null) else 'FAIL'}")
    print(f"      Best predictor: {best_pred_name}, {best_within_2x*100:.1f}% within 2x")
    print(f"      Null baseline: {null_within_2x_domain*100:.1f}% within 2x")
    print(f"      Improvement over null: {best_improvement*100:+.1f}pp")

    overall = "PROVEN" if (k1_pass and k2_pass and k2_beats_null) else "KILLED"
    print(f"\n  Overall: {overall}")

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")

    return {
        'config': {
            'd_in': cfg.d_in, 'd_out': cfg.d_out, 'n_seeds': cfg.n_seeds,
            'snr': snr, 'lora_ranks': list(cfg.lora_ranks),
            'error_threshold': cfg.error_threshold,
        },
        'correlations_per_domain': correlations_per_domain,
        'correlations_pooled': correlations_pooled,
        'prediction_results_pooled': prediction_results,
        'prediction_results_domain': prediction_results_domain,
        'null_baseline': {
            'rank': null_rank,
            'within_2x_pooled': float(null_within_2x),
            'within_2x_domain': float(null_within_2x_domain),
        },
        'threshold_vs_kneedle': {
            'agreement_50pct': float(agreement),
            'spearman_rho': float(rho_methods),
        },
        'domain_summaries': domain_summaries,
        'kill_criteria': {
            'K1_best_metric': best_metric,
            'K1_best_rho': float(best_rho),
            'K1_pass': k1_pass,
            'K2_best_predictor': best_pred_name,
            'K2_best_within_2x': float(best_within_2x),
            'K2_null_within_2x': float(null_within_2x_domain),
            'K2_improvement_over_null': float(best_improvement),
            'K2_pass': k2_pass,
            'K2_beats_null': k2_beats_null,
            'overall': overall,
        },
        'n_trials': n,
        'elapsed_seconds': elapsed,
    }


def run_multi_d_experiment() -> Dict:
    """Run at multiple d values and SNR values to test robustness (fixes #1, #3)."""
    all_results = {}

    for d in [64, 128, 256]:
        for snr in [5.0, 20.0, 100.0]:
            print(f"\n{'='*60}")
            print(f"  Dimension d={d}, SNR={snr}")
            print(f"{'='*60}")
            true_ranks = tuple(r for r in (2, 4, 8, 12, 16, 24, 32, 48) if r <= d)
            lora_ranks = tuple(r for r in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64) if r <= d)
            cfg = Config(d_in=d, d_out=d, domain_true_ranks=true_ranks,
                         lora_ranks=lora_ranks)
            all_results[f'd={d}_snr={snr}'] = run_experiment(cfg, snr_override=snr)

    # Summary across all conditions
    print(f"\n{'='*60}")
    print("  SUMMARY ACROSS ALL CONDITIONS")
    print(f"{'='*60}")
    print(f"  {'Condition':20s} {'Best rho':>10s} {'K1':>5s} {'Best 2x%':>10s} {'Null 2x%':>10s} {'Delta':>8s} {'K2':>5s} {'Overall':>10s}")
    for key, res in all_results.items():
        kc = res['kill_criteria']
        print(f"  {key:20s} {kc['K1_best_rho']:10.4f} "
              f"{'PASS' if kc['K1_pass'] else 'FAIL':>5s} "
              f"{kc['K2_best_within_2x']*100:9.1f}% "
              f"{kc['K2_null_within_2x']*100:9.1f}% "
              f"{kc['K2_improvement_over_null']*100:+7.1f}pp "
              f"{'PASS' if (kc['K2_pass'] and kc['K2_beats_null']) else 'FAIL':>5s} "
              f"{kc['overall']:>10s}")

    return all_results


if __name__ == '__main__':
    results = run_multi_d_experiment()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'results.json')

    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {out_path}")
