#!/usr/bin/env python3
"""
Decorrelation Filter Scaling: Does the Grassmannian decorrelation filter
get stronger with dimension d?

BACKGROUND:
At d=64, the b_matrix_training_correlation experiment found:
- B-matrix cosine (trained): 0.0298 (2.52x random baseline)
- Full delta cosine (A@B):   0.0017 (0.14x random baseline)
The frozen-A Grassmannian skeleton acts as a "decorrelation filter": even though
B-matrices develop training-induced correlation, the near-orthogonal A-matrices
project them into different subspaces, making full deltas LESS correlated than
random. The ratio trained_delta/random_delta = 0.14 at d=64.

QUESTION:
Does this filter ratio DECREASE with d (filter gets stronger at scale)?
Or does it stay constant (meaning the filter is just the dimensionality effect)?

The minimum_viable_base experiment showed cos ~ 1/sqrt(D_flat) for both LoRA
and random vectors, with ratio ~1.0. But that used synthetic (untrained) adapters.
Here we test with TRAINED adapters where B-matrices develop real correlation.

DESIGN:
- Sweep d = {64, 128, 256, 512}
- N = 6 experts per d (same domains across all d)
- 3 seeds per d
- For each (d, seed):
  1. Build AP skeleton, train 6 experts with frozen AP-A (trained condition)
  2. Train 6 experts with random-orthonormal A (control condition)
  3. Measure:
     a. B-matrix pairwise |cos| (trained vs random baseline)
     b. Full delta (A@B) pairwise |cos| (trained vs random baseline)
     c. Decorrelation filter ratio = trained_delta_cos / random_delta_cos

KILL CRITERIA:
- K1: filter ratio DECREASES with d (monotonically or by power law fit)
- K2: at d>=256, filter ratio < 0.5

CPU only. numpy/scipy. Runtime target: < 10 min.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# Reuse parent infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent / 'grassmannian_expert_init'))
from grassmannian_expert_init import (
    DTYPE, VOCAB_SIZE, CONTEXT_LEN, LORA_RANK, LORA_ALPHA,
    welch_bound,
    alternating_projection, random_grassmannian_points,
    MicroMLP, generate_domain_data,
    init_lora_from_frame, init_lora_random_orthonormal,
    lora_delta_vec, train_lora, cosine_sim,
)

# ============================================================================
# Configuration
# ============================================================================

D_VALUES = [64, 128, 256, 512]
N_EXPERTS = 6
SEEDS = [42, 137, 777]

# Domain IDs -- same domains across all d values for fair comparison
DOMAIN_IDS = [0, 1, 2, 50, 100, 200]

# Architecture config per d: (n_layers, d_ff_mult, steps, lr, n_seq, batch_size)
# Scaled steps with d to ensure convergence at larger dimensions
D_CONFIG = {
    64:  (2, 4, 300, 0.01,  200, 32),
    128: (2, 4, 300, 0.008, 200, 32),
    256: (2, 2, 250, 0.005, 150, 32),
    512: (2, 2, 200, 0.003, 100, 16),
}


def extract_b_vector(B1_list, B2_list):
    """Concatenate all B-matrix parameters into a flat vector."""
    parts = []
    for l in range(len(B1_list)):
        parts.append(B1_list[l].ravel())
        parts.append(B2_list[l].ravel())
    return np.concatenate(parts)


def pairwise_abs_cosines(vectors):
    """Compute all pairwise |cos| for a list of vectors. Returns list of values."""
    N = len(vectors)
    cosines = []
    for i in range(N):
        for j in range(i + 1, N):
            cosines.append(abs(cosine_sim(vectors[i], vectors[j])))
    return cosines


def run_experiment():
    t_start = time.time()

    print("=" * 78)
    print("  EXPERIMENT: Decorrelation Filter Scaling with Dimension")
    print("  K1: filter ratio (trained/random delta cos) decreases with d")
    print("  K2: at d>=256, filter ratio < 0.5")
    print("=" * 78)

    all_results = {}

    for d in D_VALUES:
        nl, d_ff_mult, steps, lr, n_seq, bs = D_CONFIG[d]
        d_ff = d_ff_mult * d
        r = LORA_RANK

        print(f"\n{'='*78}")
        print(f"  d={d}, d_ff={d_ff}, layers={nl}, steps={steps}, lr={lr}")
        print(f"{'='*78}")

        d_results = []

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")

            # Build shared base model for this (d, seed)
            rng_model = np.random.RandomState(seed + d)
            model = MicroMLP(d, nl, d_ff_mult, rng_model)

            # ============================================================
            # Condition 1: AP skeleton + frozen-A training
            # ============================================================
            rng_ap = np.random.RandomState(seed)
            frames, _ = alternating_projection(
                N_EXPERTS, r, d, n_iter=500, mu_factor=1.2, rng=rng_ap
            )

            ap_deltas = []
            ap_b_vecs = []
            ap_losses = []

            for idx, domain_id in enumerate(DOMAIN_IDS):
                x, y = generate_domain_data(domain_id, n_seq)
                A1, B1, A2, B2 = init_lora_from_frame(frames[idx], d, d_ff, nl)
                A1, B1, A2, B2, loss = train_lora(
                    model, x, y, A1, B1, A2, B2, steps, lr, bs
                )
                ap_deltas.append(lora_delta_vec(A1, B1, A2, B2))
                ap_b_vecs.append(extract_b_vector(B1, B2))
                ap_losses.append(float(loss))

            # ============================================================
            # Condition 2: Random-orthonormal A + frozen-A training
            # ============================================================
            ro_deltas = []
            ro_b_vecs = []
            ro_losses = []

            for idx, domain_id in enumerate(DOMAIN_IDS):
                x, y = generate_domain_data(domain_id, n_seq)
                rng_lora = np.random.RandomState(seed + d + idx * 31 + 5000)
                A1, B1, A2, B2 = init_lora_random_orthonormal(d, d_ff, nl, rng_lora)
                A1, B1, A2, B2, loss = train_lora(
                    model, x, y, A1, B1, A2, B2, steps, lr, bs
                )
                ro_deltas.append(lora_delta_vec(A1, B1, A2, B2))
                ro_b_vecs.append(extract_b_vector(B1, B2))
                ro_losses.append(float(loss))

            # ============================================================
            # Random baseline: untrained random vectors for reference
            # ============================================================
            D_b = ap_b_vecs[0].shape[0]
            D_delta = ap_deltas[0].shape[0]
            rng_rand = np.random.RandomState(seed + d + 9999)
            rand_b_vecs = [rng_rand.randn(D_b).astype(DTYPE) for _ in range(N_EXPERTS)]
            rand_delta_vecs = [rng_rand.randn(D_delta).astype(DTYPE) for _ in range(N_EXPERTS)]

            # ============================================================
            # Measure pairwise cosines
            # ============================================================
            ap_b_cos = pairwise_abs_cosines(ap_b_vecs)
            ro_b_cos = pairwise_abs_cosines(ro_b_vecs)
            rand_b_cos = pairwise_abs_cosines(rand_b_vecs)

            ap_delta_cos = pairwise_abs_cosines(ap_deltas)
            ro_delta_cos = pairwise_abs_cosines(ro_deltas)
            rand_delta_cos = pairwise_abs_cosines(rand_delta_vecs)

            # Key metrics
            mean_ap_b = float(np.mean(ap_b_cos))
            mean_ro_b = float(np.mean(ro_b_cos))
            mean_rand_b = float(np.mean(rand_b_cos))

            mean_ap_delta = float(np.mean(ap_delta_cos))
            mean_ro_delta = float(np.mean(ro_delta_cos))
            mean_rand_delta = float(np.mean(rand_delta_cos))

            # Decorrelation filter ratio (the key metric)
            # = trained_delta_cos / random_delta_cos
            # < 1 means filter is active (trained deltas MORE orthogonal than random)
            filter_ratio_ap = mean_ap_delta / max(mean_rand_delta, 1e-12)
            filter_ratio_ro = mean_ro_delta / max(mean_rand_delta, 1e-12)

            # B-matrix correlation ratio (trained / random baseline)
            b_corr_ratio_ap = mean_ap_b / max(mean_rand_b, 1e-12)
            b_corr_ratio_ro = mean_ro_b / max(mean_rand_b, 1e-12)

            print(f"    B-matrix |cos|:  AP={mean_ap_b:.6f}  RO={mean_ro_b:.6f}  "
                  f"Rand={mean_rand_b:.6f}")
            print(f"    B-matrix ratio:  AP/rand={b_corr_ratio_ap:.2f}x  "
                  f"RO/rand={b_corr_ratio_ro:.2f}x")
            print(f"    Delta |cos|:     AP={mean_ap_delta:.6f}  RO={mean_ro_delta:.6f}  "
                  f"Rand={mean_rand_delta:.6f}")
            print(f"    FILTER RATIO:    AP={filter_ratio_ap:.4f}  RO={filter_ratio_ro:.4f}")
            print(f"    Losses:          AP={np.mean(ap_losses):.4f}  RO={np.mean(ro_losses):.4f}")

            d_results.append({
                'seed': seed,
                'd': d,
                'D_b': D_b,
                'D_delta': D_delta,
                'b_cos': {
                    'ap_mean': mean_ap_b,
                    'ro_mean': mean_ro_b,
                    'rand_mean': mean_rand_b,
                    'ap_values': ap_b_cos,
                    'ro_values': ro_b_cos,
                    'rand_values': rand_b_cos,
                },
                'delta_cos': {
                    'ap_mean': mean_ap_delta,
                    'ro_mean': mean_ro_delta,
                    'rand_mean': mean_rand_delta,
                    'ap_values': ap_delta_cos,
                    'ro_values': ro_delta_cos,
                    'rand_values': rand_delta_cos,
                },
                'filter_ratio_ap': filter_ratio_ap,
                'filter_ratio_ro': filter_ratio_ro,
                'b_corr_ratio_ap': b_corr_ratio_ap,
                'b_corr_ratio_ro': b_corr_ratio_ro,
                'losses': {
                    'ap_mean': float(np.mean(ap_losses)),
                    'ro_mean': float(np.mean(ro_losses)),
                },
            })

        all_results[d] = d_results

    # ================================================================
    # AGGREGATE ANALYSIS
    # ================================================================
    print(f"\n{'='*78}")
    print(f"  AGGREGATE ANALYSIS ({len(SEEDS)} seeds per d)")
    print(f"{'='*78}")

    aggregate = {}
    for d in D_VALUES:
        results_d = all_results[d]
        agg = {
            'd': d,
            'D_delta': results_d[0]['D_delta'],
            'D_b': results_d[0]['D_b'],
            'b_cos_ap': float(np.mean([r['b_cos']['ap_mean'] for r in results_d])),
            'b_cos_ap_std': float(np.std([r['b_cos']['ap_mean'] for r in results_d])),
            'b_cos_ro': float(np.mean([r['b_cos']['ro_mean'] for r in results_d])),
            'b_cos_rand': float(np.mean([r['b_cos']['rand_mean'] for r in results_d])),
            'delta_cos_ap': float(np.mean([r['delta_cos']['ap_mean'] for r in results_d])),
            'delta_cos_ap_std': float(np.std([r['delta_cos']['ap_mean'] for r in results_d])),
            'delta_cos_ro': float(np.mean([r['delta_cos']['ro_mean'] for r in results_d])),
            'delta_cos_ro_std': float(np.std([r['delta_cos']['ro_mean'] for r in results_d])),
            'delta_cos_rand': float(np.mean([r['delta_cos']['rand_mean'] for r in results_d])),
            'filter_ratio_ap': float(np.mean([r['filter_ratio_ap'] for r in results_d])),
            'filter_ratio_ap_std': float(np.std([r['filter_ratio_ap'] for r in results_d])),
            'filter_ratio_ro': float(np.mean([r['filter_ratio_ro'] for r in results_d])),
            'filter_ratio_ro_std': float(np.std([r['filter_ratio_ro'] for r in results_d])),
            'b_corr_ratio_ap': float(np.mean([r['b_corr_ratio_ap'] for r in results_d])),
            'b_corr_ratio_ro': float(np.mean([r['b_corr_ratio_ro'] for r in results_d])),
            'filter_ratio_ap_per_seed': [r['filter_ratio_ap'] for r in results_d],
            'filter_ratio_ro_per_seed': [r['filter_ratio_ro'] for r in results_d],
        }
        aggregate[d] = agg

    # Print summary table
    print(f"\n  {'d':>4} | {'D_delta':>10} | {'B-cos AP':>10} | {'B-cos Rand':>10} | "
          f"{'B ratio':>8} | {'Delta AP':>10} | {'Delta Rand':>10} | {'Filter':>8} | {'Filter std':>10}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-"
          f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")
    for d in D_VALUES:
        a = aggregate[d]
        print(f"  {d:4d} | {a['D_delta']:10d} | {a['b_cos_ap']:10.6f} | "
              f"{a['b_cos_rand']:10.6f} | {a['b_corr_ratio_ap']:8.2f}x | "
              f"{a['delta_cos_ap']:10.6f} | {a['delta_cos_rand']:10.6f} | "
              f"{a['filter_ratio_ap']:8.4f} | {a['filter_ratio_ap_std']:10.4f}")

    # ================================================================
    # SCALING ANALYSIS: fit power law to filter ratio vs d
    # ================================================================
    print(f"\n{'='*78}")
    print(f"  SCALING ANALYSIS")
    print(f"{'='*78}")

    d_arr = np.array(D_VALUES, dtype=float)
    filter_arr = np.array([aggregate[d]['filter_ratio_ap'] for d in D_VALUES])
    filter_ro_arr = np.array([aggregate[d]['filter_ratio_ro'] for d in D_VALUES])

    # Also fit individual components
    delta_ap_arr = np.array([aggregate[d]['delta_cos_ap'] for d in D_VALUES])
    delta_rand_arr = np.array([aggregate[d]['delta_cos_rand'] for d in D_VALUES])
    b_ap_arr = np.array([aggregate[d]['b_cos_ap'] for d in D_VALUES])
    b_rand_arr = np.array([aggregate[d]['b_cos_rand'] for d in D_VALUES])

    def fit_power_law(x, y, name):
        """Fit y = a * x^beta in log-log space. Return beta, a, R^2."""
        log_x = np.log(x)
        log_y = np.log(np.maximum(y, 1e-15))
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(log_x, log_y)
        a = np.exp(intercept)
        r2 = r_value ** 2
        print(f"    {name}: y = {a:.4f} * d^({slope:.3f}), R^2={r2:.4f}, p={p_value:.4f}")
        return slope, a, r2

    print(f"\n  Power law fits (y = a * d^beta):")
    beta_filter, a_filter, r2_filter = fit_power_law(d_arr, filter_arr, "Filter ratio (AP)")
    beta_filter_ro, _, r2_filter_ro = fit_power_law(d_arr, filter_ro_arr, "Filter ratio (RO)")
    beta_delta_ap, _, r2_delta_ap = fit_power_law(d_arr, delta_ap_arr, "Delta cos (AP)")
    beta_delta_rand, _, r2_delta_rand = fit_power_law(d_arr, delta_rand_arr, "Delta cos (random)")
    beta_b_ap, _, r2_b_ap = fit_power_law(d_arr, b_ap_arr, "B-cos (AP)")
    beta_b_rand, _, r2_b_rand = fit_power_law(d_arr, b_rand_arr, "B-cos (random)")

    # Check monotonicity
    is_monotone_decreasing = all(
        filter_arr[i] >= filter_arr[i + 1] for i in range(len(filter_arr) - 1)
    )
    print(f"\n  Filter ratio monotonically decreasing? {is_monotone_decreasing}")
    print(f"  Filter ratio values: {[f'{x:.4f}' for x in filter_arr]}")

    # Check if delta scaling is FASTER for AP than random
    print(f"\n  Delta cos scaling comparison:")
    print(f"    AP delta:     d^({beta_delta_ap:.3f})")
    print(f"    Random delta: d^({beta_delta_rand:.3f})")
    print(f"    Difference: {beta_delta_ap - beta_delta_rand:.3f} "
          f"({'AP decays faster' if beta_delta_ap < beta_delta_rand else 'Random decays faster'})")

    # ================================================================
    # KILL CRITERIA ASSESSMENT
    # ================================================================
    print(f"\n{'='*78}")
    print(f"  KILL CRITERIA ASSESSMENT")
    print(f"{'='*78}")

    # K1: filter ratio DECREASES with d
    print(f"\n  K1: Filter ratio (trained/random delta cos) decreases with d?")
    print(f"    Power law exponent (beta): {beta_filter:.3f}")
    print(f"    R^2: {r2_filter:.4f}")
    print(f"    Monotonically decreasing: {is_monotone_decreasing}")

    # Also check using the slope: beta < 0 means ratio decreases with d
    k1_pass = beta_filter < -0.05  # small tolerance for noise
    if k1_pass:
        print(f"    K1 PASS: Filter ratio decreases as d^({beta_filter:.3f})")
    else:
        if abs(beta_filter) < 0.05:
            print(f"    K1 FAIL: Filter ratio is approximately CONSTANT "
                  f"(beta={beta_filter:.3f}, |beta|<0.05)")
            print(f"    This means the decorrelation filter is a FIXED ratio, "
                  f"not dimension-dependent.")
        else:
            print(f"    K1 FAIL: Filter ratio INCREASES with d "
                  f"(beta={beta_filter:.3f} > 0)")

    # K2: at d>=256, filter ratio < 0.5
    print(f"\n  K2: At d>=256, filter ratio < 0.5?")
    k2_pass = True
    for d in D_VALUES:
        if d >= 256:
            fr = aggregate[d]['filter_ratio_ap']
            fr_std = aggregate[d]['filter_ratio_ap_std']
            status = "PASS" if fr < 0.5 else "FAIL"
            if fr >= 0.5:
                k2_pass = False
            print(f"    d={d}: filter ratio = {fr:.4f} +/- {fr_std:.4f} -> {status}")

    if k2_pass:
        print(f"    K2 PASS: Filter ratio < 0.5 at all d >= 256")
    else:
        print(f"    K2 FAIL: Filter ratio >= 0.5 at some d >= 256")

    # ================================================================
    # INTERPRETATION
    # ================================================================
    print(f"\n{'='*78}")
    print(f"  INTERPRETATION")
    print(f"{'='*78}")

    if k1_pass and k2_pass:
        print(f"\n  VERDICT: PROVEN")
        print(f"  The decorrelation filter gets STRONGER with dimension.")
        print(f"  Trained delta cosines decay faster than random ({beta_delta_ap:.3f} vs "
              f"{beta_delta_rand:.3f}).")
        print(f"  This means the Grassmannian skeleton provides INCREASING protection")
        print(f"  against B-matrix correlation at larger model dimensions.")
    elif k1_pass and not k2_pass:
        print(f"\n  VERDICT: SUPPORTED")
        print(f"  Filter ratio decreases with d but has not reached <0.5 at d>=256.")
    elif not k1_pass and k2_pass:
        print(f"\n  VERDICT: SUPPORTED (partial)")
        print(f"  Filter ratio < 0.5 at d>=256 but scaling trend is weak.")
    else:
        print(f"\n  VERDICT: KILLED")
        print(f"  The decorrelation filter does NOT scale with dimension.")
        if abs(beta_filter) < 0.1:
            print(f"  The filter is approximately constant across dimensions,")
            print(f"  meaning it provides a FIXED decorrelation factor independent of d.")
            print(f"  This is still useful (filter exists!) but does not compound with")
            print(f"  the dimensional scaling effect.")
        else:
            print(f"  The filter actually WEAKENS at larger d.")

    # ================================================================
    # Comparison with parent experiments
    # ================================================================
    print(f"\n  --- Comparison with parent experiments ---")
    print(f"  b_matrix_training_correlation (d=64):")
    print(f"    Reported delta cos ratio: 0.14x (trained/random)")
    if 64 in aggregate:
        print(f"    This experiment:          {aggregate[64]['filter_ratio_ap']:.4f}")

    print(f"\n  structural_orthogonality_characterization:")
    print(f"    Gradient cos exponent: -0.722")
    print(f"    Random cos exponent:   -0.936")
    print(f"    This experiment AP delta exponent: {beta_delta_ap:.3f}")
    print(f"    This experiment random delta exponent: {beta_delta_rand:.3f}")

    print(f"\n  minimum_viable_base:")
    print(f"    LoRA/random ratio: 0.93-1.13 (for UNTRAINED synthetic adapters)")
    ratios_str = [f"{aggregate[d]['filter_ratio_ap']:.4f}" for d in D_VALUES]
    print(f"    This experiment AP filter ratio: {ratios_str}")

    print(f"{'='*78}")

    # ================================================================
    # Save results
    # ================================================================
    elapsed = time.time() - t_start

    output = {
        'config': {
            'd_values': D_VALUES,
            'n_experts': N_EXPERTS,
            'seeds': SEEDS,
            'rank': LORA_RANK,
            'domain_ids': DOMAIN_IDS,
            'd_config': {str(d): list(v) for d, v in D_CONFIG.items()},
        },
        'per_d': {
            str(d): [
                {k: v for k, v in r.items()
                 if k not in ('b_cos', 'delta_cos')}  # exclude raw value lists
                for r in all_results[d]
            ]
            for d in D_VALUES
        },
        'per_d_detailed': {
            str(d): [
                {
                    'seed': r['seed'],
                    'b_cos_ap_mean': r['b_cos']['ap_mean'],
                    'b_cos_ro_mean': r['b_cos']['ro_mean'],
                    'b_cos_rand_mean': r['b_cos']['rand_mean'],
                    'delta_cos_ap_mean': r['delta_cos']['ap_mean'],
                    'delta_cos_ro_mean': r['delta_cos']['ro_mean'],
                    'delta_cos_rand_mean': r['delta_cos']['rand_mean'],
                    'filter_ratio_ap': r['filter_ratio_ap'],
                    'filter_ratio_ro': r['filter_ratio_ro'],
                }
                for r in all_results[d]
            ]
            for d in D_VALUES
        },
        'aggregate': {str(d): a for d, a in aggregate.items()},
        'scaling': {
            'filter_ratio_ap': {
                'beta': float(beta_filter),
                'a': float(a_filter),
                'r2': float(r2_filter),
            },
            'filter_ratio_ro': {
                'beta': float(beta_filter_ro),
                'r2': float(r2_filter_ro),
            },
            'delta_cos_ap': {
                'beta': float(beta_delta_ap),
                'r2': float(r2_delta_ap),
            },
            'delta_cos_random': {
                'beta': float(beta_delta_rand),
                'r2': float(r2_delta_rand),
            },
            'b_cos_ap': {
                'beta': float(beta_b_ap),
                'r2': float(r2_b_ap),
            },
            'b_cos_random': {
                'beta': float(beta_b_rand),
                'r2': float(r2_b_rand),
            },
            'monotone_decreasing': is_monotone_decreasing,
        },
        'kill_criteria': {
            'k1_ratio_decreases_with_d': k1_pass,
            'k1_beta': float(beta_filter),
            'k2_ratio_below_half_at_d_ge_256': k2_pass,
            'k2_values': {
                str(d): float(aggregate[d]['filter_ratio_ap'])
                for d in D_VALUES if d >= 256
            },
        },
        'elapsed_seconds': elapsed,
    }

    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    print(f"  Total time: {elapsed:.1f}s")

    return output


if __name__ == "__main__":
    run_experiment()
