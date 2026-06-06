"""
SNR Transition Zone Validation

The parent experiment (exp_adaptive_rank_snr_fallback) tested SNR={5, 10, 20, 100}
and found that T=2.0 and T=2.5 give identical compound heuristic results. But the
transition zone SNR={6, 7, 8, 9} was never tested, so we don't know:

  1. At what SNR does r_99/r_95 cross the T=2.0 threshold?
  2. Is the transition sharp or gradual?
  3. Does T=2.0 vs T=2.5 ever disagree in the transition zone?
  4. Does compound heuristic accuracy degrade in the transition zone?

Kill criteria:
  K1: compound accuracy drops >10pp below linear interpolation at SNR=7
  K2: T=2.0 and T=2.5 disagree on >20% of domains in the transition zone

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

# Import parent experiment's functions via importlib to avoid experiments/models/__init__.py
# which tries to import all models (including MLX-dependent ones)
import importlib.util
import sys

def _import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
# Add base dir to path so that the fallback module can import from micro.models
sys.path.insert(0, os.path.abspath(_base_dir))
# Pre-load the adaptive_rank_selection module under its full dotted name
# so the fallback module's import statement resolves
_ars_mod = _import_from_file(
    'micro.models.adaptive_rank_selection.adaptive_rank_selection',
    os.path.join(_base_dir, 'micro', 'models', 'adaptive_rank_selection', 'adaptive_rank_selection.py'))
_fb_mod = _import_from_file(
    'adaptive_rank_snr_fallback',
    os.path.join(_base_dir, 'micro', 'models', 'adaptive_rank_snr_fallback', 'adaptive_rank_snr_fallback.py'))

generate_exact_rank_domain = _ars_mod.generate_exact_rank_domain
generate_spectral_decay_domain = _ars_mod.generate_spectral_decay_domain
compute_metrics = _ars_mod.compute_metrics
find_optimal_rank_kneedle = _ars_mod.find_optimal_rank_kneedle
energy_rank = _ars_mod.energy_rank

compound_r95_predictor = _fb_mod.compound_r95_predictor
evaluate_predictors = _fb_mod.evaluate_predictors


@dataclass
class TransitionConfig:
    """Configuration for SNR transition zone experiment."""
    dimensions: tuple = (64, 128, 256)
    # Transition zone + anchors
    snr_values: tuple = (5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    n_seeds: int = 5
    domain_true_ranks_base: tuple = (2, 4, 8, 12, 16, 24, 32, 48)
    decay_rates: tuple = (0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98)
    lora_ranks_base: tuple = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
    error_threshold: float = 0.05
    null_baseline_rank: int = 16
    ratio_thresholds: tuple = (1.5, 2.0, 2.5, 3.0)


def compute_ratio_distribution(d: int, snr: float, cfg: TransitionConfig) -> Dict:
    """Compute r_99/r_95 ratio distribution for a single (d, SNR) condition.

    Returns per-trial ratios and per-domain median ratios.
    """
    true_ranks = tuple(r for r in cfg.domain_true_ranks_base if r <= d)

    trial_ratios = []
    domain_median_ratios = []
    domain_id = 0

    # Part 1: Exact-rank domains
    for true_rank in true_ranks:
        seed_ratios = []
        for seed in range(cfg.n_seeds):
            rng = np.random.default_rng(seed * 10000 + domain_id * 100)
            Delta = generate_exact_rank_domain(d, d, true_rank, snr, rng)
            sigmas = np.linalg.svd(Delta, compute_uv=False)
            metrics = compute_metrics(sigmas)
            r = metrics['energy_rank_99'] / max(metrics['energy_rank_95'], 1)
            trial_ratios.append(r)
            seed_ratios.append(r)
        domain_median_ratios.append(np.median(seed_ratios))
        domain_id += 1

    # Part 2: Spectral decay domains
    for decay in cfg.decay_rates:
        seed_ratios = []
        for seed in range(cfg.n_seeds):
            rng = np.random.default_rng(seed * 10000 + domain_id * 100)
            Delta = generate_spectral_decay_domain(d, d, decay, snr, rng)
            sigmas = np.linalg.svd(Delta, compute_uv=False)
            metrics = compute_metrics(sigmas)
            r = metrics['energy_rank_99'] / max(metrics['energy_rank_95'], 1)
            trial_ratios.append(r)
            seed_ratios.append(r)
        domain_median_ratios.append(np.median(seed_ratios))
        domain_id += 1

    arr = np.array(trial_ratios)
    dom_arr = np.array(domain_median_ratios)

    return {
        'trial_ratios': arr,
        'domain_median_ratios': dom_arr,
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'p10': float(np.percentile(arr, 10)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
        'frac_above_1.5': float(np.mean(arr > 1.5)),
        'frac_above_2.0': float(np.mean(arr > 2.0)),
        'frac_above_2.5': float(np.mean(arr > 2.5)),
        'frac_above_3.0': float(np.mean(arr > 3.0)),
        'domain_frac_above_1.5': float(np.mean(dom_arr > 1.5)),
        'domain_frac_above_2.0': float(np.mean(dom_arr > 2.0)),
        'domain_frac_above_2.5': float(np.mean(dom_arr > 2.5)),
        'domain_frac_above_3.0': float(np.mean(dom_arr > 3.0)),
        'n_trials': len(trial_ratios),
        'n_domains': len(domain_median_ratios),
    }


def run_single_condition(d: int, snr: float, cfg: TransitionConfig) -> Dict:
    """Run full predictor evaluation for a single (d, SNR) condition.

    Reuses the parent's run_single_condition logic but adds transition-specific
    analysis.
    """
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

    # Evaluate predictors
    results, domain_data = evaluate_predictors(
        records, available, cfg.null_baseline_rank, cfg.ratio_thresholds)

    # T=2.0 vs T=2.5 disagreement analysis
    def snap(x):
        return int(available[np.argmin(np.abs(available - x))])

    domain_ids = sorted(set(r['domain_id'] for r in records))
    n_disagree = 0
    disagree_details = []
    for did in domain_ids:
        group = [r for r in records if r['domain_id'] == did]
        med_metrics = {}
        for m in ['effective_rank', 'stable_rank', 'energy_rank_90',
                   'energy_rank_95', 'energy_rank_99']:
            med_metrics[m] = float(np.median([r['metrics'][m] for r in group]))

        pred_t20 = snap(compound_r95_predictor(med_metrics, 2.0))
        pred_t25 = snap(compound_r95_predictor(med_metrics, 2.5))

        if pred_t20 != pred_t25:
            n_disagree += 1
            ratio = med_metrics['energy_rank_99'] / max(med_metrics['energy_rank_95'], 1)
            disagree_details.append({
                'domain_id': did,
                'type': group[0]['domain_type'],
                'param': group[0]['domain_param'],
                'r99_r95_ratio': float(ratio),
                'pred_t20': pred_t20,
                'pred_t25': pred_t25,
            })

    n_domains = len(domain_ids)
    disagree_frac = n_disagree / max(n_domains, 1)

    return {
        'd': d,
        'snr': snr,
        'n_domains': n_domains,
        'n_records': len(records),
        'predictors': results,
        'threshold_disagreement': {
            'n_disagree': n_disagree,
            'n_domains': n_domains,
            'disagree_frac': float(disagree_frac),
            'details': disagree_details,
        },
    }


def run_full_experiment(cfg: TransitionConfig = None) -> Dict:
    """Run across all conditions and analyze the transition zone."""
    cfg = cfg or TransitionConfig()
    t0 = time.time()

    print("=" * 70)
    print("  SNR Transition Zone Validation")
    print("=" * 70)
    print(f"  Dimensions: {cfg.dimensions}")
    print(f"  SNR values: {cfg.snr_values}")
    print(f"  Ratio thresholds: {cfg.ratio_thresholds}")
    print(f"  Seeds: {cfg.n_seeds}")
    print()

    # ── Phase 1: Ratio Distribution Analysis ──────────────────────────
    print("=" * 70)
    print("  PHASE 1: r_99/r_95 RATIO DISTRIBUTION ACROSS TRANSITION ZONE")
    print("=" * 70)

    ratio_data = {}
    for d in cfg.dimensions:
        for snr in cfg.snr_values:
            key = f'd={d}_snr={snr}'
            ratio_data[key] = compute_ratio_distribution(d, snr, cfg)

    # Print ratio summary table
    print(f"\n{'Condition':20s} {'Mean R':>8s} {'Median':>8s} {'P10':>8s} "
          f"{'P90':>8s} {'>1.5':>6s} {'>2.0':>6s} {'>2.5':>6s} {'>3.0':>6s}")
    print("-" * 100)

    for d in cfg.dimensions:
        for snr in cfg.snr_values:
            key = f'd={d}_snr={snr}'
            rd = ratio_data[key]
            print(f"{key:20s} {rd['mean']:8.2f} {rd['median']:8.2f} "
                  f"{rd['p10']:8.2f} {rd['p90']:8.2f} "
                  f"{rd['frac_above_1.5']*100:5.1f}% "
                  f"{rd['frac_above_2.0']*100:5.1f}% "
                  f"{rd['frac_above_2.5']*100:5.1f}% "
                  f"{rd['frac_above_3.0']*100:5.1f}%")
        print()

    # Per-domain median ratio summary
    print(f"\n{'Condition':20s} {'Dom Mean':>9s} {'Dom Med':>8s} "
          f"{'Dom>1.5':>8s} {'Dom>2.0':>8s} {'Dom>2.5':>8s}")
    print("-" * 75)

    for d in cfg.dimensions:
        for snr in cfg.snr_values:
            key = f'd={d}_snr={snr}'
            rd = ratio_data[key]
            dom_mean = float(np.mean(rd['domain_median_ratios']))
            dom_med = float(np.median(rd['domain_median_ratios']))
            print(f"{key:20s} {dom_mean:9.2f} {dom_med:8.2f} "
                  f"{rd['domain_frac_above_1.5']*100:7.1f}% "
                  f"{rd['domain_frac_above_2.0']*100:7.1f}% "
                  f"{rd['domain_frac_above_2.5']*100:7.1f}%")
        print()

    # ── Phase 2: Transition Crossing Analysis ────────────────────────
    print("=" * 70)
    print("  PHASE 2: THRESHOLD CROSSING ANALYSIS")
    print("=" * 70)

    # For each dimension, find the SNR at which >50% of domains cross T=2.0
    for d in cfg.dimensions:
        print(f"\n  d={d}:")
        snr_list = list(cfg.snr_values)
        frac_above_20 = []
        frac_above_25 = []
        for snr in snr_list:
            key = f'd={d}_snr={snr}'
            frac_above_20.append(ratio_data[key]['domain_frac_above_2.0'])
            frac_above_25.append(ratio_data[key]['domain_frac_above_2.5'])

        print(f"    SNR:       {' '.join(f'{s:6.0f}' for s in snr_list)}")
        print(f"    f(R>2.0):  {' '.join(f'{f*100:5.1f}%' for f in frac_above_20)}")
        print(f"    f(R>2.5):  {' '.join(f'{f*100:5.1f}%' for f in frac_above_25)}")

        # Interpolate crossing points
        for label, fracs in [('T=2.0', frac_above_20), ('T=2.5', frac_above_25)]:
            crossing_snr = None
            for i in range(len(snr_list) - 1):
                if fracs[i] > 0.5 and fracs[i+1] <= 0.5:
                    # Linear interpolation
                    f1, f2 = fracs[i], fracs[i+1]
                    s1, s2 = snr_list[i], snr_list[i+1]
                    crossing_snr = s1 + (0.5 - f1) / (f2 - f1) * (s2 - s1)
                    break
            if crossing_snr is not None:
                print(f"    50% crossing for {label}: SNR ~ {crossing_snr:.1f}")
            else:
                # Check if always above or always below
                if all(f > 0.5 for f in fracs):
                    print(f"    50% crossing for {label}: always > 50% (all SNR tested)")
                elif all(f <= 0.5 for f in fracs):
                    print(f"    50% crossing for {label}: never > 50% (or only at SNR=5)")
                else:
                    print(f"    50% crossing for {label}: non-monotone pattern")

    # ── Phase 3: Compound Heuristic Accuracy ────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 3: COMPOUND HEURISTIC ACCURACY IN TRANSITION ZONE")
    print(f"{'='*70}")

    all_conditions = {}
    for d in cfg.dimensions:
        for snr in cfg.snr_values:
            key = f'd={d}_snr={snr}'
            print(f"\n--- {key} ---")
            result = run_single_condition(d, snr, cfg)
            all_conditions[key] = result

            null_acc = result['predictors']['null_16']['within_2x']
            r99_acc = result['predictors']['r_99']['within_2x']
            r95_acc = result['predictors']['r_95']['within_2x']
            comp20_acc = result['predictors']['compound_r95_t2.0']['within_2x']
            comp25_acc = result['predictors']['compound_r95_t2.5']['within_2x']
            print(f"  null={null_acc*100:.1f}%  r99={r99_acc*100:.1f}%  "
                  f"r95={r95_acc*100:.1f}%  comp_t2.0={comp20_acc*100:.1f}%  "
                  f"comp_t2.5={comp25_acc*100:.1f}%")

    # Summary accuracy table
    print(f"\n{'='*70}")
    print("  ACCURACY SUMMARY TABLE")
    print(f"{'='*70}")

    print(f"\n{'Condition':20s} {'null':>6s} {'r_99':>6s} {'r_95':>6s} "
          f"{'c_t2.0':>7s} {'c_t2.5':>7s} {'agree':>6s}")
    print("-" * 64)

    for d in cfg.dimensions:
        for snr in cfg.snr_values:
            key = f'd={d}_snr={snr}'
            res = all_conditions[key]
            null = res['predictors']['null_16']['within_2x'] * 100
            r99 = res['predictors']['r_99']['within_2x'] * 100
            r95 = res['predictors']['r_95']['within_2x'] * 100
            c20 = res['predictors']['compound_r95_t2.0']['within_2x'] * 100
            c25 = res['predictors']['compound_r95_t2.5']['within_2x'] * 100
            agree = "YES" if abs(c20 - c25) < 0.1 else f"{abs(c20-c25):.1f}pp"
            print(f"{key:20s} {null:5.1f}% {r99:5.1f}% {r95:5.1f}% "
                  f"{c20:6.1f}% {c25:6.1f}% {agree:>6s}")
        print()

    # ── Phase 4: Threshold Disagreement Analysis ────────────────────
    print(f"{'='*70}")
    print("  PHASE 4: T=2.0 vs T=2.5 DISAGREEMENT ANALYSIS")
    print(f"{'='*70}")

    for d in cfg.dimensions:
        print(f"\n  d={d}:")
        for snr in cfg.snr_values:
            key = f'd={d}_snr={snr}'
            td = all_conditions[key]['threshold_disagreement']
            n_dis = td['n_disagree']
            n_dom = td['n_domains']
            frac = td['disagree_frac'] * 100
            print(f"    SNR={snr:4.0f}: {n_dis}/{n_dom} domains disagree ({frac:.1f}%)")
            if td['details']:
                for det in td['details']:
                    print(f"      domain {det['domain_id']} ({det['type']}, "
                          f"param={det['param']}): R={det['r99_r95_ratio']:.2f}, "
                          f"t2.0->{det['pred_t20']}, t2.5->{det['pred_t25']}")

    # ── Phase 5: Transition Sharpness ──────────────────────────────
    print(f"\n{'='*70}")
    print("  PHASE 5: TRANSITION SHARPNESS")
    print(f"{'='*70}")

    for d in cfg.dimensions:
        print(f"\n  d={d}: compound_r95_t2.0 accuracy vs SNR")
        snr_list = list(cfg.snr_values)
        accs = []
        for snr in snr_list:
            key = f'd={d}_snr={snr}'
            acc = all_conditions[key]['predictors']['compound_r95_t2.0']['within_2x'] * 100
            accs.append(acc)
            print(f"    SNR={snr:4.0f}: {acc:5.1f}%")

        # Linear interpolation at SNR=7
        acc_5 = accs[0]   # SNR=5
        acc_10 = accs[5]  # SNR=10
        expected_7 = acc_5 + (acc_10 - acc_5) * (7 - 5) / (10 - 5)
        actual_7 = accs[2]  # SNR=7
        delta = actual_7 - expected_7
        print(f"    Interpolated at SNR=7: {expected_7:.1f}%, actual: {actual_7:.1f}%, "
              f"delta: {delta:+.1f}pp")

        # Transition width: SNR range where accuracy changes most
        max_jump = 0
        max_jump_snr = None
        for i in range(len(accs) - 1):
            jump = abs(accs[i+1] - accs[i])
            if jump > max_jump:
                max_jump = jump
                max_jump_snr = (snr_list[i], snr_list[i+1])
        if max_jump_snr:
            print(f"    Largest accuracy jump: {max_jump:.1f}pp between "
                  f"SNR={max_jump_snr[0]:.0f} and SNR={max_jump_snr[1]:.0f}")

    # ── Kill Criteria Evaluation ──────────────────────────────────
    print(f"\n{'='*70}")
    print("  KILL CRITERIA EVALUATION")
    print(f"{'='*70}")

    # K1: accuracy drop at SNR=7 vs interpolation
    k1_results = []
    print("\n  K1: Accuracy at SNR=7 vs linear interpolation (threshold: >10pp drop)")
    for d in cfg.dimensions:
        acc_5 = all_conditions[f'd={d}_snr=5.0']['predictors']['compound_r95_t2.0']['within_2x'] * 100
        acc_10 = all_conditions[f'd={d}_snr=10.0']['predictors']['compound_r95_t2.0']['within_2x'] * 100
        acc_7 = all_conditions[f'd={d}_snr=7.0']['predictors']['compound_r95_t2.0']['within_2x'] * 100
        expected = acc_5 + (acc_10 - acc_5) * (7 - 5) / (10 - 5)
        delta = acc_7 - expected
        k1_pass = delta >= -10.0
        k1_results.append({
            'd': d, 'acc_5': acc_5, 'acc_7': acc_7, 'acc_10': acc_10,
            'expected_7': expected, 'delta': delta, 'pass': k1_pass,
        })
        print(f"    d={d}: acc_5={acc_5:.1f}%, acc_7={acc_7:.1f}%, acc_10={acc_10:.1f}%, "
              f"expected={expected:.1f}%, delta={delta:+.1f}pp -> {'PASS' if k1_pass else 'KILL'}")

    k1_overall = all(r['pass'] for r in k1_results)
    print(f"\n  K1 overall: {'PASS' if k1_overall else 'KILL'}")

    # K2: T=2.0 vs T=2.5 disagreement in transition zone
    print(f"\n  K2: T=2.0 vs T=2.5 disagreement in transition zone (threshold: >20% of domains)")
    k2_results = []
    transition_snrs = [6.0, 7.0, 8.0, 9.0]
    for d in cfg.dimensions:
        for snr in transition_snrs:
            key = f'd={d}_snr={snr}'
            td = all_conditions[key]['threshold_disagreement']
            frac = td['disagree_frac']
            k2_pass = frac <= 0.20
            k2_results.append({
                'd': d, 'snr': snr, 'disagree_frac': frac,
                'n_disagree': td['n_disagree'], 'n_domains': td['n_domains'],
                'pass': k2_pass,
            })
            print(f"    d={d} SNR={snr:.0f}: {td['n_disagree']}/{td['n_domains']} "
                  f"disagree ({frac*100:.1f}%) -> {'PASS' if k2_pass else 'KILL'}")

    k2_overall = all(r['pass'] for r in k2_results)
    print(f"\n  K2 overall: {'PASS' if k2_overall else 'KILL'}")

    overall = "PROVEN" if (k1_overall and k2_overall) else "KILLED"
    print(f"\n  OVERALL VERDICT: {overall}")

    elapsed = time.time() - t0
    print(f"  Runtime: {elapsed:.1f}s")

    # ── Build output ──────────────────────────────────────────────
    # Strip numpy arrays for JSON serialization
    ratio_data_json = {}
    for key, rd in ratio_data.items():
        ratio_data_json[key] = {k: v for k, v in rd.items()
                                if k not in ('trial_ratios', 'domain_median_ratios')}

    conditions_json = {}
    for key, res in all_conditions.items():
        conditions_json[key] = {
            'd': res['d'],
            'snr': res['snr'],
            'n_domains': res['n_domains'],
            'predictors': {
                pn: {pk: pv for pk, pv in pd.items() if pk != 'predictions'}
                for pn, pd in res['predictors'].items()
            },
            'threshold_disagreement': res['threshold_disagreement'],
        }

    return {
        'ratio_distributions': ratio_data_json,
        'conditions': conditions_json,
        'kill_criteria': {
            'K1_results': k1_results,
            'K1_overall': k1_overall,
            'K2_results': k2_results,
            'K2_overall': k2_overall,
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
