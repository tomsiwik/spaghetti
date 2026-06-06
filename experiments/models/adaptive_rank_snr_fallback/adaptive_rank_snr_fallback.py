"""
SNR-Aware Rank Predictor with Fallback Heuristic

Parent experiment (exp_adaptive_rank_selection) proved energy_rank_99 is the best
rank predictor overall (rho=0.94-0.99 across 9 conditions). However, the adversarial
review found that at SNR=5 d=64, r_99 is WORSE than the null baseline (-13.3pp).
Noise inflates the 99% energy threshold, causing systematic overprediction.

This experiment tests a compound heuristic:
  - Default: use r_99 (the unconditional best)
  - Fallback trigger: if r_99 / r_95 > threshold (default 2.0), the spectral tail
    is noise-dominated, so fall back to r_95 (or effective_rank)
  - The ratio r_99/r_95 is a diagnostic for noise contamination: in a clean signal,
    the 95%-to-99% energy gap is small; under noise, the gap explodes

Kill criteria:
  K1: fallback heuristic (r_99 if r_99 <= 2*r_95 else r_95) does not improve
      over r_99 alone at SNR<=10
  K2: no condition where the compound predictor is worse than null baseline
      (after applying fallback)

Architecture: Pure numpy/scipy, CPU-only. Reuses parent's domain generation
and metric functions.
"""

import numpy as np
from scipy import stats
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Import parent experiment's functions
from experiments.models.adaptive_rank_selection.adaptive_rank_selection import (
    Config as ParentConfig,
    generate_exact_rank_domain,
    generate_spectral_decay_domain,
    compute_metrics,
    find_optimal_rank_kneedle,
    find_optimal_rank_threshold,
    energy_rank,
    effective_rank_rv,
)


@dataclass
class FallbackConfig:
    """Configuration for SNR-aware fallback experiment."""
    dimensions: tuple = (64, 128, 256)
    snr_values: tuple = (5.0, 10.0, 20.0, 100.0)  # Added SNR=10 for K1 boundary
    n_seeds: int = 5
    domain_true_ranks_base: tuple = (2, 4, 8, 12, 16, 24, 32, 48)
    decay_rates: tuple = (0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98)
    lora_ranks_base: tuple = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
    error_threshold: float = 0.05
    null_baseline_rank: int = 16
    # Fallback parameters to sweep
    ratio_thresholds: tuple = (1.5, 2.0, 2.5, 3.0)


# ── Compound Heuristic Functions ──────────────────────────────────────────

def compound_r95_predictor(metrics: Dict, ratio_threshold: float = 2.0) -> float:
    """Compound heuristic: r_99 with r_95 fallback.

    If r_99 / r_95 > ratio_threshold, the spectral tail between 95% and 99%
    energy is noise-dominated. Fall back to r_95.

    Returns the predicted rank (continuous, needs snapping to grid).
    """
    r_99 = metrics['energy_rank_99']
    r_95 = metrics['energy_rank_95']
    if r_95 > 0 and r_99 / r_95 > ratio_threshold:
        return r_95
    return r_99


def compound_effrank_predictor(metrics: Dict, ratio_threshold: float = 2.0) -> float:
    """Compound heuristic: r_99 with effective_rank fallback.

    If r_99 / effective_rank > ratio_threshold, fall back to effective_rank.
    """
    r_99 = metrics['energy_rank_99']
    eff = metrics['effective_rank']
    if eff > 0 and r_99 / eff > ratio_threshold:
        return eff
    return r_99


# ── Evaluation Logic ──────────────────────────────────────────────────────

def evaluate_predictors(records: List[Dict], available_ranks: np.ndarray,
                        null_rank: int, ratio_thresholds: tuple) -> Dict:
    """Evaluate all predictors on a set of records.

    Returns per-predictor accuracy metrics.
    """
    def snap(x):
        return int(available_ranks[np.argmin(np.abs(available_ranks - x))])

    # Collect per-domain medians
    domain_ids = sorted(set(r['domain_id'] for r in records))
    n_domains = len(domain_ids)

    domain_data = []
    for did in domain_ids:
        group = [r for r in records if r['domain_id'] == did]
        med_optimal = np.median([r['optimal_kneedle'] for r in group])
        med_metrics = {}
        for m in ['effective_rank', 'stable_rank', 'energy_rank_90',
                   'energy_rank_95', 'energy_rank_99']:
            med_metrics[m] = float(np.median([r['metrics'][m] for r in group]))
        domain_data.append({
            'domain_id': did,
            'type': group[0]['domain_type'],
            'param': group[0]['domain_param'],
            'optimal': med_optimal,
            'metrics': med_metrics,
        })

    optimal_arr = np.array([d['optimal'] for d in domain_data])

    # Define all predictors
    predictors = {}

    # Baseline predictors
    predictors['null_16'] = np.full(n_domains, null_rank, dtype=float)
    predictors['r_99'] = np.array([snap(d['metrics']['energy_rank_99']) for d in domain_data], dtype=float)
    predictors['r_95'] = np.array([snap(d['metrics']['energy_rank_95']) for d in domain_data], dtype=float)
    predictors['effective_rank'] = np.array([snap(d['metrics']['effective_rank']) for d in domain_data], dtype=float)

    # Compound predictors at various thresholds
    for thresh in ratio_thresholds:
        key_95 = f'compound_r95_t{thresh}'
        key_eff = f'compound_eff_t{thresh}'
        preds_95 = []
        preds_eff = []
        for d in domain_data:
            preds_95.append(snap(compound_r95_predictor(d['metrics'], thresh)))
            preds_eff.append(snap(compound_effrank_predictor(d['metrics'], thresh)))
        predictors[key_95] = np.array(preds_95, dtype=float)
        predictors[key_eff] = np.array(preds_eff, dtype=float)

    # Evaluate each predictor
    results = {}
    for name, preds in predictors.items():
        ratios = preds / np.maximum(optimal_arr, 1)
        within_2x = np.mean((ratios >= 0.5) & (ratios <= 2.0))

        # Spearman correlation (skip for null)
        if name == 'null_16':
            rho, p_val = 0.0, 1.0
        else:
            rho, p_val = stats.spearmanr(preds, optimal_arr)

        # Mean absolute error in rank
        mae = float(np.mean(np.abs(preds - optimal_arr)))

        # Mean absolute log2 ratio (0 = perfect, 1 = off by 2x)
        log2_ratio = float(np.mean(np.abs(np.log2(np.maximum(ratios, 0.01)))))

        results[name] = {
            'within_2x': float(within_2x),
            'spearman_rho': float(rho),
            'p_value': float(p_val),
            'mae': mae,
            'log2_ratio': log2_ratio,
            'predictions': preds.tolist(),
        }

    return results, domain_data


def run_single_condition(d: int, snr: float, cfg: FallbackConfig) -> Dict:
    """Run experiment for a single (d, SNR) condition."""
    true_ranks = tuple(r for r in cfg.domain_true_ranks_base if r <= d)
    lora_ranks = tuple(r for r in cfg.lora_ranks_base if r <= d)
    available = np.array(lora_ranks)

    uniform_ranks = np.arange(1, d + 1)
    records = []
    domain_id = 0

    # Part 1: Exact-rank domains
    for true_rank in true_ranks:
        for seed in range(cfg.n_seeds):
            rng = np.random.default_rng(seed * 10000 + domain_id * 100)
            Delta = generate_exact_rank_domain(d, d, true_rank, snr, rng)
            sigmas = np.linalg.svd(Delta, compute_uv=False)
            metrics = compute_metrics(sigmas)

            uniform_errors = np.array([
                float(np.sqrt(np.sum(sigmas[min(r, len(sigmas)):] ** 2)) /
                      np.sqrt(np.sum(sigmas ** 2)))
                for r in uniform_ranks
            ])

            opt_kneedle = find_optimal_rank_kneedle(uniform_ranks, uniform_errors)

            records.append({
                'domain_id': domain_id,
                'domain_type': 'exact_rank',
                'domain_param': true_rank,
                'seed': seed,
                'metrics': metrics,
                'optimal_kneedle': opt_kneedle,
            })
        domain_id += 1

    # Part 2: Spectral decay domains
    for decay in cfg.decay_rates:
        for seed in range(cfg.n_seeds):
            rng = np.random.default_rng(seed * 10000 + domain_id * 100)
            Delta = generate_spectral_decay_domain(d, d, decay, snr, rng)
            sigmas = np.linalg.svd(Delta, compute_uv=False)
            metrics = compute_metrics(sigmas)

            uniform_errors = np.array([
                float(np.sqrt(np.sum(sigmas[min(r, len(sigmas)):] ** 2)) /
                      np.sqrt(np.sum(sigmas ** 2)))
                for r in uniform_ranks
            ])

            opt_kneedle = find_optimal_rank_kneedle(uniform_ranks, uniform_errors)

            records.append({
                'domain_id': domain_id,
                'domain_type': 'spectral_decay',
                'domain_param': decay,
                'seed': seed,
                'metrics': metrics,
                'optimal_kneedle': opt_kneedle,
            })
        domain_id += 1

    # Evaluate all predictors
    results, domain_data = evaluate_predictors(
        records, available, cfg.null_baseline_rank, cfg.ratio_thresholds)

    return {
        'd': d,
        'snr': snr,
        'n_domains': len(set(r['domain_id'] for r in records)),
        'n_records': len(records),
        'predictors': results,
        'domain_data': [
            {k: v for k, v in dd.items() if k != 'metrics'}
            for dd in domain_data
        ],
    }


def run_full_experiment(cfg: FallbackConfig = None) -> Dict:
    """Run across all conditions and evaluate kill criteria."""
    cfg = cfg or FallbackConfig()
    t0 = time.time()

    print("=" * 70)
    print("  SNR-Aware Rank Predictor with Fallback Heuristic")
    print("=" * 70)
    print(f"  Dimensions: {cfg.dimensions}")
    print(f"  SNR values: {cfg.snr_values}")
    print(f"  Ratio thresholds: {cfg.ratio_thresholds}")
    print(f"  Seeds: {cfg.n_seeds}")
    print()

    all_conditions = {}
    for d in cfg.dimensions:
        for snr in cfg.snr_values:
            key = f'd={d}_snr={snr}'
            print(f"--- {key} ---")
            result = run_single_condition(d, snr, cfg)
            all_conditions[key] = result

            # Print summary for this condition
            null_acc = result['predictors']['null_16']['within_2x']
            r99_acc = result['predictors']['r_99']['within_2x']
            r95_acc = result['predictors']['r_95']['within_2x']
            best_compound = None
            best_compound_acc = -1
            for name, pred in result['predictors'].items():
                if name.startswith('compound_r95'):
                    if pred['within_2x'] > best_compound_acc:
                        best_compound_acc = pred['within_2x']
                        best_compound = name

            print(f"  null={null_acc*100:.1f}%  r_99={r99_acc*100:.1f}%  "
                  f"r_95={r95_acc*100:.1f}%  best_compound={best_compound_acc*100:.1f}% ({best_compound})")
            print()

    # ── Aggregate Analysis ──────────────────────────────────────────────

    print("=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    # Collect all predictor names
    sample = next(iter(all_conditions.values()))
    all_predictor_names = list(sample['predictors'].keys())

    # Summary table
    print(f"\n{'Condition':20s}", end="")
    for name in ['null_16', 'r_99', 'r_95', 'effective_rank',
                 'compound_r95_t2.0', 'compound_eff_t2.0']:
        short = name[:12]
        print(f" {short:>12s}", end="")
    print()
    print("-" * 92)

    for key, res in all_conditions.items():
        print(f"{key:20s}", end="")
        for name in ['null_16', 'r_99', 'r_95', 'effective_rank',
                     'compound_r95_t2.0', 'compound_eff_t2.0']:
            acc = res['predictors'][name]['within_2x'] * 100
            print(f" {acc:11.1f}%", end="")
        print()

    # ── Improvement over r_99 at low SNR ──────────────────────────────

    print(f"\n{'='*70}")
    print("  IMPROVEMENT OVER r_99 AT LOW SNR (pp)")
    print(f"{'='*70}")

    low_snr_conditions = {k: v for k, v in all_conditions.items()
                          if v['snr'] <= 10}

    print(f"\n{'Condition':20s} {'r_99':>8s} {'comp_r95':>10s} {'delta':>8s} {'comp_eff':>10s} {'delta':>8s}")
    print("-" * 66)

    for key, res in sorted(low_snr_conditions.items()):
        r99 = res['predictors']['r_99']['within_2x'] * 100
        comp95 = res['predictors']['compound_r95_t2.0']['within_2x'] * 100
        compeff = res['predictors']['compound_eff_t2.0']['within_2x'] * 100
        print(f"{key:20s} {r99:7.1f}% {comp95:9.1f}% {comp95-r99:+7.1f}pp "
              f"{compeff:9.1f}% {compeff-r99:+7.1f}pp")

    # ── Ratio threshold sweep ─────────────────────────────────────────

    print(f"\n{'='*70}")
    print("  RATIO THRESHOLD SWEEP (compound_r95 at low SNR)")
    print(f"{'='*70}")

    for thresh in cfg.ratio_thresholds:
        name = f'compound_r95_t{thresh}'
        print(f"\n  Threshold = {thresh}:")
        for key, res in sorted(low_snr_conditions.items()):
            acc = res['predictors'][name]['within_2x'] * 100
            r99 = res['predictors']['r_99']['within_2x'] * 100
            null = res['predictors']['null_16']['within_2x'] * 100
            print(f"    {key:20s}: {acc:.1f}% (r99={r99:.1f}%, null={null:.1f}%, "
                  f"delta_r99={acc-r99:+.1f}pp, delta_null={acc-null:+.1f}pp)")

    # ── Fallback trigger analysis ─────────────────────────────────────

    print(f"\n{'='*70}")
    print("  FALLBACK TRIGGER ANALYSIS (r_99/r_95 ratio by condition)")
    print(f"{'='*70}")

    for key, res in sorted(all_conditions.items()):
        domain_data_full = []
        # Reconstruct metrics from records
        cond_d = res['d']
        cond_snr = res['snr']
        true_ranks = tuple(r for r in cfg.domain_true_ranks_base if r <= cond_d)
        domain_id = 0
        ratios_99_95 = []

        for true_rank in true_ranks:
            for seed in range(cfg.n_seeds):
                rng = np.random.default_rng(seed * 10000 + domain_id * 100)
                Delta = generate_exact_rank_domain(cond_d, cond_d, true_rank, cond_snr, rng)
                sigmas = np.linalg.svd(Delta, compute_uv=False)
                metrics = compute_metrics(sigmas)
                ratios_99_95.append(metrics['energy_rank_99'] / max(metrics['energy_rank_95'], 1))
            domain_id += 1

        for decay in cfg.decay_rates:
            for seed in range(cfg.n_seeds):
                rng = np.random.default_rng(seed * 10000 + domain_id * 100)
                Delta = generate_spectral_decay_domain(cond_d, cond_d, decay, cond_snr, rng)
                sigmas = np.linalg.svd(Delta, compute_uv=False)
                metrics = compute_metrics(sigmas)
                ratios_99_95.append(metrics['energy_rank_99'] / max(metrics['energy_rank_95'], 1))
            domain_id += 1

        arr = np.array(ratios_99_95)
        pct_triggered = np.mean(arr > 2.0) * 100
        print(f"  {key:20s}: mean={arr.mean():.2f}, max={arr.max():.2f}, "
              f">2.0: {pct_triggered:.0f}%, >1.5: {np.mean(arr > 1.5)*100:.0f}%")

    # ── Kill Criteria ─────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("  KILL CRITERIA EVALUATION")
    print(f"{'='*70}")

    # K1: compound_r95_t2.0 must improve over r_99 at SNR<=10
    k1_improvements = []
    for key, res in sorted(low_snr_conditions.items()):
        r99_acc = res['predictors']['r_99']['within_2x']
        comp_acc = res['predictors']['compound_r95_t2.0']['within_2x']
        improvement = comp_acc - r99_acc
        k1_improvements.append(improvement)
        print(f"  K1 [{key}]: compound_r95 - r_99 = {improvement*100:+.1f}pp")

    k1_any_improvement = bool(any(imp > 0 for imp in k1_improvements))
    k1_mean_improvement = float(np.mean(k1_improvements))
    k1_pass = bool(k1_any_improvement and k1_mean_improvement > 0)
    print(f"\n  K1 verdict: {'PASS' if k1_pass else 'KILL'}")
    print(f"    Any improvement at SNR<=10: {k1_any_improvement}")
    print(f"    Mean improvement: {k1_mean_improvement*100:+.2f}pp")

    # K2: no condition where compound predictor is worse than null
    k2_failures = []
    for key, res in all_conditions.items():
        null_acc = res['predictors']['null_16']['within_2x']
        comp_acc = res['predictors']['compound_r95_t2.0']['within_2x']
        delta = comp_acc - null_acc
        if delta < 0:
            k2_failures.append((key, delta))
            print(f"  K2 FAILURE [{key}]: compound_r95 - null = {delta*100:+.1f}pp")

    k2_pass = bool(len(k2_failures) == 0)
    if k2_pass:
        print("  K2: No condition where compound predictor is worse than null")
    print(f"\n  K2 verdict: {'PASS' if k2_pass else 'KILL'}")

    # Also check with other thresholds for K2
    print(f"\n  K2 sensitivity to ratio threshold:")
    for thresh in cfg.ratio_thresholds:
        name = f'compound_r95_t{thresh}'
        n_failures = 0
        for key, res in all_conditions.items():
            null_acc = res['predictors']['null_16']['within_2x']
            comp_acc = res['predictors'][name]['within_2x']
            if comp_acc < null_acc:
                n_failures += 1
        print(f"    threshold={thresh}: {n_failures} conditions worse than null")

    # Also check compound_eff
    print(f"\n  K2 for compound_eff_t2.0:")
    k2_eff_failures = []
    for key, res in all_conditions.items():
        null_acc = res['predictors']['null_16']['within_2x']
        comp_acc = res['predictors']['compound_eff_t2.0']['within_2x']
        delta = comp_acc - null_acc
        if delta < 0:
            k2_eff_failures.append((key, delta))
            print(f"    FAILURE [{key}]: compound_eff - null = {delta*100:+.1f}pp")
    if not k2_eff_failures:
        print("    No failures")

    # ── Best overall predictor ────────────────────────────────────────

    print(f"\n{'='*70}")
    print("  BEST PREDICTOR ACROSS ALL CONDITIONS")
    print(f"{'='*70}")

    # For each predictor, compute mean within_2x across all conditions
    predictor_means = {}
    for name in all_predictor_names:
        accs = [res['predictors'][name]['within_2x'] for res in all_conditions.values()]
        predictor_means[name] = np.mean(accs)

    for name, mean_acc in sorted(predictor_means.items(), key=lambda x: -x[1]):
        print(f"  {name:25s}: mean within_2x = {mean_acc*100:.1f}%")

    overall = "PROVEN" if (k1_pass and k2_pass) else "KILLED"
    print(f"\n  OVERALL VERDICT: {overall}")

    elapsed = time.time() - t0
    print(f"  Runtime: {elapsed:.1f}s")

    return {
        'conditions': {k: {
            'd': v['d'], 'snr': v['snr'], 'n_domains': v['n_domains'],
            'predictors': {pn: {pk: pv for pk, pv in pd.items() if pk != 'predictions'}
                           for pn, pd in v['predictors'].items()},
        } for k, v in all_conditions.items()},
        'predictor_means': {k: float(v) for k, v in predictor_means.items()},
        'kill_criteria': {
            'K1_improvements': [float(x) for x in k1_improvements],
            'K1_mean_improvement': float(k1_mean_improvement),
            'K1_any_improvement': k1_any_improvement,
            'K1_pass': k1_pass,
            'K2_failures': [(k, float(d)) for k, d in k2_failures],
            'K2_pass': k2_pass,
            'overall': overall,
        },
        'config': {
            'dimensions': list(cfg.dimensions),
            'snr_values': list(cfg.snr_values),
            'ratio_thresholds': list(cfg.ratio_thresholds),
            'n_seeds': cfg.n_seeds,
        },
        'elapsed_seconds': elapsed,
    }


if __name__ == '__main__':
    results = run_full_experiment()

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
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {out_path}")
