#!/usr/bin/env python3
"""
PPL-Probe Temperature Sensitivity Experiment

Tests whether PPL-probe weighting quality is robust across softmax temperatures.
Pure numpy experiment with synthetic LoRA matrices.

Kill criteria:
  K1: tau sweep {0.1, 0.5, 1.0, 2.0, 5.0} shows >5pp variance in mean gap improvement
  K2: optimal tau differs from 1.0 by >2x (tau* < 0.5 or tau* > 2.0)
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy import stats


def make_expert_deltas(K, d, r, rng):
    """Generate K random LoRA expert deltas: Delta_i = B_i @ A_i."""
    deltas = []
    for _ in range(K):
        A = rng.standard_normal((r, d)) * 0.1
        B = rng.standard_normal((d, r)) * 0.1
        deltas.append(B @ A)
    return deltas


def make_target(deltas, d, rng):
    """Create a target that is a noisy combination of experts.

    Target = random_weights @ deltas + noise.
    This ensures the oracle can achieve good quality and the problem
    is not trivially degenerate.
    """
    K = len(deltas)
    # Random true weights (not uniform, to make discrimination useful)
    true_w = rng.dirichlet(np.ones(K) * 0.5)  # Concentrated Dirichlet
    target = sum(w * delta for w, delta in zip(true_w, deltas))
    # Add small noise so oracle is not perfect
    noise_scale = 0.01 * np.linalg.norm(target, 'fro')
    target += rng.standard_normal((d, d)) * noise_scale
    return target, true_w


def compute_synthetic_losses(deltas, target):
    """Compute synthetic 'loss' for each expert: Frobenius distance to target.

    Lower distance = lower loss = better expert for this target.
    """
    losses = []
    for delta in deltas:
        dist = np.linalg.norm(target - delta, 'fro')
        losses.append(float(dist))
    return losses


def softmax_weights(losses, tau):
    """Compute PPL-probe weights from losses via softmax(-loss/tau)."""
    scores = -np.array(losses)
    scaled = scores / max(tau, 1e-10)
    scaled -= scaled.max()  # Numerical stability
    exp_s = np.exp(scaled)
    weights = exp_s / (exp_s.sum() + 1e-10)
    return weights


def compose(deltas, weights):
    """Weight-space composition: sum_i w_i * Delta_i."""
    return sum(w * delta for w, delta in zip(weights, deltas))


def oracle_weights(deltas, target):
    """Compute oracle-optimal weights via least squares.

    min_w ||target - sum_i w_i delta_i||_F^2
    """
    K = len(deltas)
    d = deltas[0].shape[0]
    # Vectorize
    D = np.column_stack([delta.ravel() for delta in deltas])  # (d^2, K)
    t = target.ravel()  # (d^2,)
    # Least squares (may give negative weights, that's fine for oracle)
    w_star, _, _, _ = np.linalg.lstsq(D, t, rcond=None)
    return w_star


def frob_distance(A, B):
    """Frobenius distance between two matrices."""
    return float(np.linalg.norm(A - B, 'fro'))


def run_single_trial(K, d, r, tau_values, rng):
    """Run one trial: generate experts, target, measure quality at each tau."""
    deltas = make_expert_deltas(K, d, r, rng)
    target, true_weights = make_target(deltas, d, rng)

    # Baseline: equal-weight composition
    equal_w = np.ones(K) / K
    composed_equal = compose(deltas, equal_w)
    dist_equal = frob_distance(composed_equal, target)

    # Oracle: least-squares optimal
    w_oracle = oracle_weights(deltas, target)
    composed_oracle = compose(deltas, w_oracle)
    dist_oracle = frob_distance(composed_oracle, target)

    # Synthetic losses (Frobenius distance of each expert to target)
    losses = compute_synthetic_losses(deltas, target)

    # PPL-probe at each temperature
    results = {}
    for tau in tau_values:
        w_probe = softmax_weights(losses, tau)
        composed_probe = compose(deltas, w_probe)
        dist_probe = frob_distance(composed_probe, target)

        # Gap improvement: how much better than equal-weight (percentage points)
        if dist_equal > 1e-10:
            gap_improvement = (dist_equal - dist_probe) / dist_equal * 100
        else:
            gap_improvement = 0.0

        # Oracle gap: how close to oracle
        if dist_equal > 1e-10:
            oracle_gap = (dist_equal - dist_oracle) / dist_equal * 100
        else:
            oracle_gap = 0.0

        # Weight entropy
        entropy = -np.sum(w_probe * np.log(w_probe + 1e-10))
        max_entropy = np.log(K)

        results[tau] = {
            'dist_probe': dist_probe,
            'dist_equal': dist_equal,
            'dist_oracle': dist_oracle,
            'gap_improvement_pp': gap_improvement,
            'oracle_gap_pp': oracle_gap,
            'weights': w_probe.tolist(),
            'entropy_ratio': float(entropy / max_entropy) if max_entropy > 0 else 1.0,
        }

    return results


def run_experiment():
    """Main experiment: sweep tau across K values with multiple seeds."""
    # Parameters
    d = 32
    r = 4
    N = 5  # Expert pool (we test K <= N)
    K_values = [2, 3, 5]
    tau_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    n_seeds = 50

    print("=" * 72)
    print("  PPL-PROBE TEMPERATURE SENSITIVITY EXPERIMENT")
    print("=" * 72)
    print(f"  d={d}, r={r}, N={N}")
    print(f"  K values: {K_values}")
    print(f"  tau values: {tau_values}")
    print(f"  Seeds: {n_seeds}")
    print("=" * 72)

    t0 = time.time()

    all_results = {}

    for K in K_values:
        print(f"\n--- K={K} ---")
        tau_gaps = {tau: [] for tau in tau_values}

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed * 1000 + K)
            trial = run_single_trial(K, d, r, tau_values, rng)

            for tau in tau_values:
                tau_gaps[tau].append(trial[tau]['gap_improvement_pp'])

        # Compute statistics per tau
        tau_stats = {}
        for tau in tau_values:
            gaps = np.array(tau_gaps[tau])
            tau_stats[str(tau)] = {
                'mean_gap_pp': float(np.mean(gaps)),
                'std_gap_pp': float(np.std(gaps)),
                'median_gap_pp': float(np.median(gaps)),
                'min_gap_pp': float(np.min(gaps)),
                'max_gap_pp': float(np.max(gaps)),
                'n_positive': int(np.sum(gaps > 0)),
                'n_negative': int(np.sum(gaps < 0)),
            }

        # Variance across tau means
        mean_gaps = [tau_stats[str(tau)]['mean_gap_pp'] for tau in tau_values]
        var_across_tau = float(np.var(mean_gaps))
        std_across_tau = float(np.std(mean_gaps))
        best_tau = tau_values[int(np.argmax(mean_gaps))]
        worst_tau = tau_values[int(np.argmin(mean_gaps))]

        # Print summary
        print(f"  tau   | mean gap (pp) | std gap (pp) | +/- count")
        print(f"  ------|---------------|--------------|----------")
        for tau in tau_values:
            s = tau_stats[str(tau)]
            print(f"  {tau:5.1f} | {s['mean_gap_pp']:+12.2f} | {s['std_gap_pp']:11.2f} | {s['n_positive']:2d}+ / {s['n_negative']:2d}-")

        print(f"\n  Variance across tau means: {var_across_tau:.4f} (std: {std_across_tau:.2f} pp)")
        print(f"  Best tau: {best_tau}, Worst tau: {worst_tau}")
        print(f"  Range: {max(mean_gaps):.2f} - {min(mean_gaps):.2f} = {max(mean_gaps)-min(mean_gaps):.2f} pp")

        # ANOVA across tau groups
        tau_groups = [tau_gaps[tau] for tau in tau_values]
        f_stat, p_value = stats.f_oneway(*tau_groups)
        print(f"  ANOVA: F={f_stat:.3f}, p={p_value:.4f}")

        # Kruskal-Wallis (non-parametric)
        h_stat, kw_p = stats.kruskal(*tau_groups)
        print(f"  Kruskal-Wallis: H={h_stat:.3f}, p={kw_p:.4f}")

        all_results[f'K={K}'] = {
            'tau_stats': tau_stats,
            'var_across_tau_pp2': var_across_tau,
            'std_across_tau_pp': std_across_tau,
            'best_tau': float(best_tau),
            'worst_tau': float(worst_tau),
            'range_pp': float(max(mean_gaps) - min(mean_gaps)),
            'anova_F': float(f_stat),
            'anova_p': float(p_value),
            'kruskal_H': float(h_stat),
            'kruskal_p': float(kw_p),
        }

    # Kill criteria assessment
    print("\n" + "=" * 72)
    print("  KILL CRITERIA ASSESSMENT")
    print("=" * 72)

    # K1: >5pp variance (i.e. std > 5) across tau in mean gap
    max_std = max(r['std_across_tau_pp'] for r in all_results.values())
    max_range = max(r['range_pp'] for r in all_results.values())
    k1_pass = max_std <= 5.0
    print(f"\n  K1: max std across tau = {max_std:.2f} pp (threshold: 5.0 pp)")
    print(f"      max range across tau = {max_range:.2f} pp")
    print(f"      Result: {'PASS (robust)' if k1_pass else 'FAIL (sensitive to tau)'}")

    # K2: optimal tau differs from 1.0 by >2x
    optimal_taus = [r['best_tau'] for r in all_results.values()]
    k2_violations = [tau for tau in optimal_taus if tau < 0.5 or tau > 2.0]
    k2_pass = len(k2_violations) == 0
    print(f"\n  K2: optimal taus = {optimal_taus}")
    print(f"      Violations (tau* < 0.5 or tau* > 2.0): {k2_violations}")
    print(f"      Result: {'PASS (tau=1.0 is fine)' if k2_pass else 'FAIL (tau=1.0 not optimal)'}")

    # Overall
    killed = not k1_pass or not k2_pass
    status = "KILLED" if killed else "PROVEN (tau-robust)"
    print(f"\n  OVERALL: {status}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Additional analysis: correlation between score spread and tau sensitivity
    print("\n" + "=" * 72)
    print("  ADDITIONAL ANALYSIS")
    print("=" * 72)

    # Test with varying score spreads
    print("\n  Score spread vs tau sensitivity:")
    spread_results = []
    for spread_scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        gaps_by_tau = {tau: [] for tau in tau_values}
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed * 7777 + int(spread_scale * 100))
            K = 3
            deltas = make_expert_deltas(K, d, r, rng)
            target, _ = make_target(deltas, d, rng)

            # Scale the losses to control spread
            raw_losses = compute_synthetic_losses(deltas, target)
            mean_loss = np.mean(raw_losses)
            scaled_losses = [mean_loss + (l - mean_loss) * spread_scale for l in raw_losses]

            equal_w = np.ones(K) / K
            dist_equal = frob_distance(compose(deltas, equal_w), target)

            for tau in tau_values:
                w = softmax_weights(scaled_losses, tau)
                dist_probe = frob_distance(compose(deltas, w), target)
                if dist_equal > 1e-10:
                    gap = (dist_equal - dist_probe) / dist_equal * 100
                else:
                    gap = 0.0
                gaps_by_tau[tau].append(gap)

        means = [np.mean(gaps_by_tau[tau]) for tau in tau_values]
        spread_std = np.std(means)
        spread_results.append({
            'spread_scale': spread_scale,
            'std_across_tau': float(spread_std),
            'range_across_tau': float(max(means) - min(means)),
        })
        print(f"  spread={spread_scale:.1f}: tau-std={spread_std:.2f}pp, range={max(means)-min(means):.2f}pp")

    all_results['spread_sensitivity'] = spread_results
    all_results['kill_criteria'] = {
        'K1_max_std_pp': float(max_std),
        'K1_max_range_pp': float(max_range),
        'K1_threshold_pp': 5.0,
        'K1_pass': k1_pass,
        'K2_optimal_taus': optimal_taus,
        'K2_violations': k2_violations,
        'K2_pass': k2_pass,
        'killed': killed,
        'status': status,
    }
    all_results['config'] = {
        'd': d, 'r': r, 'N': N,
        'K_values': K_values,
        'tau_values': tau_values,
        'n_seeds': n_seeds,
    }
    all_results['elapsed_seconds'] = elapsed

    # Save results
    results_path = Path(__file__).parent / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return all_results


if __name__ == '__main__':
    run_experiment()
