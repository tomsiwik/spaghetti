#!/usr/bin/env python3
"""
PPL-Probe Weighting at K=3+ Expert Composition

Extends cross_domain_dilution_vs_k to test whether PPL-probe weighting
scales from K=2 to K=3 and K=5 expert compositions.

Key question: As K grows, equal-weight dilution worsens (waste = (K-K_rel)/K),
so the opportunity for smart weighting grows. But the probe must discriminate
among more experts. Does it still work?

Strategies:
  1. equal_weight   -- W_base + (1/K) * sum(Delta_i)
  2. ppl_probe_weighted -- softmax(-L_probe) weighted composition
  3. loss_weighted  -- softmax(-L_full) weighted (oracle)
  4. top1_oracle    -- best single expert by full loss

Kill criteria:
  K1: PPL-probe vs oracle correlation < r=0.8 at K=3
  K2: K=3 probe-weighted is worse than K=2 probe-weighted
"""

import json
import time
import sys
from pathlib import Path
from itertools import combinations

import numpy as onp

# Import from grandparent experiment (cross_domain_composition)
sys.path.insert(0, str(Path(__file__).parent.parent))
from cross_domain_composition.cross_domain_composition import (
    DOMAIN_GENERATORS,
    CROSS_DOMAIN_GENERATORS,
    CharTokenizer,
    init_model,
    forward,
    compute_loss,
    train_model,
    eval_loss,
    compute_delta,
    svd_truncate_delta,
    apply_delta,
    merge_deltas,
    _prepare_batch,
)


def softmax_weights(scores, temperature=1.0):
    """Convert relevance scores to weights via softmax with temperature."""
    names = list(scores.keys())
    vals = onp.array([scores[n] for n in names])
    vals = vals / max(temperature, 1e-10)
    vals = vals - vals.max()
    exp_vals = onp.exp(vals)
    weights = exp_vals / (exp_vals.sum() + 1e-10)
    return {n: float(w) for n, w in zip(names, weights)}


def weighted_merge(base_params, expert_deltas, weights, involved_domains):
    """Merge experts with given weights."""
    total_w = sum(weights.get(d, 0.0) for d in involved_domains)
    if total_w < 1e-10:
        total_w = 1.0
    merged = {}
    for key in expert_deltas[involved_domains[0]]:
        s = sum(
            weights.get(d, 0.0) / total_w * expert_deltas[d][key]
            for d in involved_domains
        )
        merged[key] = s
    return apply_delta(base_params, merged)


def ppl_probe_scores(base_params, expert_deltas, test_enc, pad_id,
                     involved_domains, n_probe=10):
    """PPL on a small probe subset -- cheap proxy for oracle."""
    probe = test_enc[:n_probe]
    scores = {}
    for dom in involved_domains:
        single_p = apply_delta(base_params, expert_deltas[dom])
        loss = eval_loss(single_p, probe, pad_id)
        scores[dom] = -loss
    return scores


def full_loss_scores(base_params, expert_deltas, test_enc, pad_id,
                     involved_domains):
    """Full-dataset loss scores -- oracle."""
    scores = {}
    for dom in involved_domains:
        single_p = apply_delta(base_params, expert_deltas[dom])
        loss = eval_loss(single_p, test_enc, pad_id)
        scores[dom] = -loss
    return scores


def get_k_tuples(domains, K):
    """Get all C(N,K) tuples of K domains from the pool."""
    return list(combinations(domains, K))


def get_cross_domain_test_for_tuple(k_tuple, cross_domain_generators):
    """For a K-tuple, find cross-domain test types whose involved domains
    are a subset of the K-tuple.

    Returns dict of cross_type_name -> involved_domains for matching types.
    """
    matching = {}
    for cross_name, (gen_fn, involved) in cross_domain_generators.items():
        if all(d in k_tuple for d in involved):
            matching[cross_name] = involved
    return matching


def run_experiment(n_domains=5, rank_per_expert=4, epochs_per_expert=15,
                   d_model=32, n_heads=2, n_layers=2, n_train=200,
                   n_cross_test=50, n_seeds=5,
                   lr=0.001, batch_size=16, n_probe=10):
    """Run PPL-probe scaling experiment across K=2,3,5."""

    domains = list(DOMAIN_GENERATORS.keys())[:n_domains]
    tok = CharTokenizer()
    results_dir = Path(__file__).parent

    K_values = [2, 3, 5]
    strategies = ['equal_weight', 'ppl_probe_weighted', 'loss_weighted', 'top1_oracle']

    print("=" * 80)
    print("  PPL-PROBE K=3+ SCALING EXPERIMENT")
    print("=" * 80)
    print(f"  Architecture: d={d_model}, H={n_heads}, L={n_layers}, V={tok.vocab_size}")
    print(f"  Domains: {domains}")
    print(f"  K values: {K_values}")
    print(f"  Strategies: {strategies}")
    print(f"  Probe buffer: n={n_probe}")
    print(f"  Rank per expert: {rank_per_expert}")
    print(f"  Seeds: {n_seeds}")
    print("=" * 80)

    # Precompute K-tuples
    k_tuples = {K: get_k_tuples(domains, K) for K in K_values}
    for K in K_values:
        print(f"  K={K}: {len(k_tuples[K])} tuples")

    # Storage: per K, per strategy, list of (gap, tuple_name, cross_type, seed)
    all_results = {K: {s: [] for s in strategies} for K in K_values}
    # For correlation: per K, lists of (probe_weight_vector, oracle_weight_vector)
    correlation_data = {K: {'probe_w': [], 'oracle_w': []} for K in K_values}
    all_seed_data = []

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 100
        rng = onp.random.RandomState(seed)
        t0 = time.time()

        print(f"\n{'='*80}")
        print(f"  SEED {seed} ({seed_idx+1}/{n_seeds})")
        print(f"{'='*80}")

        # -- Generate data --
        domain_train_enc = {}
        domain_test_enc = {}
        for dom_name in domains:
            gen = DOMAIN_GENERATORS[dom_name]
            train_data = gen(n_train, onp.random.RandomState(seed + hash(dom_name) % 10000))
            test_data = gen(50, onp.random.RandomState(seed + hash(dom_name) % 10000 + 1))
            domain_train_enc[dom_name] = [tok.encode(s) for s in train_data]
            domain_test_enc[dom_name] = [tok.encode(s) for s in test_data]

        combined_train = []
        for dom_name in domains:
            combined_train.extend(domain_train_enc[dom_name])
        rng.shuffle(combined_train)

        # Generate cross-domain test data
        cross_test_enc = {}
        cross_test_involved = {}
        for cross_name, (gen_fn, involved) in CROSS_DOMAIN_GENERATORS.items():
            test_data = gen_fn(n_cross_test, onp.random.RandomState(
                seed + hash(cross_name) % 10000 + 2))
            cross_test_enc[cross_name] = [tok.encode(s) for s in test_data]
            cross_test_involved[cross_name] = involved

        # -- Train base model --
        print(f"  Training base model...")
        base_params = init_model(tok.vocab_size, d=d_model, H=n_heads, L=n_layers,
                                  max_T=32, seed=seed)
        base_trained = {}
        for k, v in base_params.items():
            if k == '_config':
                base_trained[k] = v
            else:
                base_trained[k] = v.copy()
        base_trained = train_model(base_trained, combined_train, tok.pad_id,
                                    epochs=epochs_per_expert, lr=lr,
                                    batch_size=batch_size, verbose=False)

        # -- Train domain experts --
        print(f"  Training {n_domains} domain experts...")
        expert_deltas = {}
        for dom_name in domains:
            expert_params = {}
            for k, v in base_params.items():
                if k == '_config':
                    expert_params[k] = v
                else:
                    expert_params[k] = v.copy()
            expert_params = train_model(
                expert_params, domain_train_enc[dom_name], tok.pad_id,
                epochs=epochs_per_expert, lr=lr, batch_size=batch_size,
                verbose=False
            )
            delta = compute_delta(base_params, expert_params)
            delta_trunc, _ = svd_truncate_delta(delta, rank_per_expert)
            expert_deltas[dom_name] = delta_trunc

        # -- Evaluate for each K value --
        seed_k_results = {}

        for K in K_values:
            print(f"\n  --- K={K} ---")
            tuples = k_tuples[K]
            k_results = []

            for k_tuple in tuples:
                tuple_name = "+".join(k_tuple)

                # Find cross-domain test types that fit this tuple
                matching_types = get_cross_domain_test_for_tuple(
                    k_tuple, CROSS_DOMAIN_GENERATORS)

                if not matching_types:
                    # K=5 has no cross-domain types that span all 5 domains,
                    # so we use ALL cross-domain types (each involves 2 domains,
                    # all of which are in the K=5 tuple)
                    if K == 5:
                        matching_types = {cn: ci for cn, (_, ci)
                                         in CROSS_DOMAIN_GENERATORS.items()}
                    else:
                        continue

                for cross_name, involved_pair in matching_types.items():
                    test_enc = cross_test_enc[cross_name]
                    base_loss = eval_loss(base_trained, test_enc, tok.pad_id)

                    k_domains = list(k_tuple)

                    # Strategy 1: equal_weight
                    eq_delta = {}
                    for key in expert_deltas[k_domains[0]]:
                        eq_delta[key] = sum(expert_deltas[d][key]
                                           for d in k_domains) / K
                    eq_params = apply_delta(base_params, eq_delta)
                    eq_loss = eval_loss(eq_params, test_enc, tok.pad_id)

                    # Strategy 2: ppl_probe_weighted
                    probe_sc = ppl_probe_scores(base_params, expert_deltas,
                                                test_enc, tok.pad_id,
                                                k_domains, n_probe=n_probe)
                    probe_w = softmax_weights(probe_sc, temperature=1.0)
                    probe_params = weighted_merge(base_params, expert_deltas,
                                                  probe_w, k_domains)
                    probe_loss = eval_loss(probe_params, test_enc, tok.pad_id)

                    # Strategy 3: loss_weighted (oracle)
                    oracle_sc = full_loss_scores(base_params, expert_deltas,
                                                 test_enc, tok.pad_id,
                                                 k_domains)
                    oracle_w = softmax_weights(oracle_sc, temperature=1.0)
                    oracle_params = weighted_merge(base_params, expert_deltas,
                                                    oracle_w, k_domains)
                    oracle_loss = eval_loss(oracle_params, test_enc, tok.pad_id)

                    # Strategy 4: top1_oracle
                    best_dom = min(k_domains, key=lambda d: -oracle_sc[d])
                    # oracle_sc is -loss, so min of -(-loss) = min of loss
                    best_dom = max(k_domains, key=lambda d: oracle_sc[d])
                    top1_params = apply_delta(base_params, expert_deltas[best_dom])
                    top1_loss = eval_loss(top1_params, test_enc, tok.pad_id)

                    # Compute gaps vs base
                    strategy_losses = {
                        'equal_weight': eq_loss,
                        'ppl_probe_weighted': probe_loss,
                        'loss_weighted': oracle_loss,
                        'top1_oracle': top1_loss,
                    }
                    strategy_gaps = {}
                    for s_name, s_loss in strategy_losses.items():
                        if base_loss > 0:
                            gap = (s_loss - base_loss) / base_loss * 100
                        else:
                            gap = 0.0
                        strategy_gaps[s_name] = gap
                        all_results[K][s_name].append(gap)

                    # Correlation data: store weight vectors for probe vs oracle
                    probe_vec = [probe_w.get(d, 0.0) for d in k_domains]
                    oracle_vec = [oracle_w.get(d, 0.0) for d in k_domains]
                    correlation_data[K]['probe_w'].append(probe_vec)
                    correlation_data[K]['oracle_w'].append(oracle_vec)

                    k_results.append({
                        'tuple': tuple_name,
                        'cross_type': cross_name,
                        'base_loss': float(base_loss),
                        'strategy_losses': {k: float(v) for k, v in strategy_losses.items()},
                        'strategy_gaps': strategy_gaps,
                        'probe_weights': probe_w,
                        'oracle_weights': oracle_w,
                    })

                    print(f"    {tuple_name:30s} [{cross_name:18s}] "
                          f"eq={eq_loss:.3f}({strategy_gaps['equal_weight']:+.1f}%) "
                          f"probe={probe_loss:.3f}({strategy_gaps['ppl_probe_weighted']:+.1f}%) "
                          f"oracle={oracle_loss:.3f}({strategy_gaps['loss_weighted']:+.1f}%) "
                          f"top1={top1_loss:.3f}({strategy_gaps['top1_oracle']:+.1f}%)")

            seed_k_results[K] = k_results

        all_seed_data.append({
            'seed': seed,
            'k_results': seed_k_results,
            'elapsed': time.time() - t0,
        })
        print(f"\n  Seed {seed} time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Aggregate results
    # ==================================================================
    print(f"\n{'='*80}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*80}")

    aggregate = {}
    for K in K_values:
        print(f"\n  --- K={K} ---")
        print(f"  {'Strategy':<22s} | {'Mean':>8s} | {'Std':>8s} | {'Max':>8s} | "
              f"{'P25':>8s} | {'Median':>8s} | {'P75':>8s} | {'ImpEW':>8s}")
        print(f"  {'-'*90}")

        k_agg = {}
        eq_mean = None
        for s_name in strategies:
            gaps = all_results[K][s_name]
            if len(gaps) == 0:
                continue
            mean_g = float(onp.mean(gaps))
            std_g = float(onp.std(gaps))
            max_g = float(onp.max(gaps))
            p25_g = float(onp.percentile(gaps, 25))
            median_g = float(onp.median(gaps))
            p75_g = float(onp.percentile(gaps, 75))

            if s_name == 'equal_weight':
                eq_mean = mean_g
            imp = (eq_mean - mean_g) if eq_mean is not None else 0.0

            k_agg[s_name] = {
                'mean_gap': mean_g, 'std_gap': std_g, 'max_gap': max_g,
                'p25_gap': p25_g, 'median_gap': median_g, 'p75_gap': p75_g,
                'improvement_vs_ew': imp, 'n_samples': len(gaps),
            }

            print(f"  {s_name:<22s} | {mean_g:+8.2f} | {std_g:8.2f} | {max_g:+8.2f} | "
                  f"{p25_g:+8.2f} | {median_g:+8.2f} | {p75_g:+8.2f} | {imp:+8.2f}pp")

        aggregate[K] = k_agg

    # ==================================================================
    # Correlation analysis: probe weights vs oracle weights
    # ==================================================================
    print(f"\n{'='*80}")
    print("  CORRELATION: PROBE WEIGHTS vs ORACLE WEIGHTS")
    print(f"{'='*80}")

    correlations = {}
    for K in K_values:
        probe_flat = onp.array([w for vec in correlation_data[K]['probe_w'] for w in vec])
        oracle_flat = onp.array([w for vec in correlation_data[K]['oracle_w'] for w in vec])

        if len(probe_flat) > 2 and onp.std(probe_flat) > 1e-10 and onp.std(oracle_flat) > 1e-10:
            r = float(onp.corrcoef(probe_flat, oracle_flat)[0, 1])
        else:
            r = 0.0

        # Also compute correlation on the weight assigned to the BEST expert
        # (more interpretable than all-weights correlation)
        best_probe = []
        best_oracle = []
        for pv, ov in zip(correlation_data[K]['probe_w'], correlation_data[K]['oracle_w']):
            # Index of max oracle weight
            max_idx = onp.argmax(ov)
            best_oracle.append(ov[max_idx])
            best_probe.append(pv[max_idx])

        if len(best_probe) > 2:
            best_probe_arr = onp.array(best_probe)
            best_oracle_arr = onp.array(best_oracle)
            if onp.std(best_probe_arr) > 1e-10 and onp.std(best_oracle_arr) > 1e-10:
                r_best = float(onp.corrcoef(best_probe_arr, best_oracle_arr)[0, 1])
            else:
                r_best = 0.0
        else:
            r_best = 0.0

        # Rank correlation (Spearman) -- more robust for ordinal comparison
        # For each observation, rank the K weights and compare probe vs oracle ranks
        rank_agreements = []
        for pv, ov in zip(correlation_data[K]['probe_w'], correlation_data[K]['oracle_w']):
            pv_arr = onp.array(pv)
            ov_arr = onp.array(ov)
            # Check if probe picks the same top-1 as oracle
            rank_agreements.append(int(onp.argmax(pv_arr) == onp.argmax(ov_arr)))

        top1_agreement = float(onp.mean(rank_agreements)) if rank_agreements else 0.0

        correlations[K] = {
            'pearson_all_weights': r,
            'pearson_best_weight': r_best,
            'top1_agreement': top1_agreement,
            'n_observations': len(correlation_data[K]['probe_w']),
        }

        print(f"  K={K}: r(all_weights)={r:.4f}  r(best_weight)={r_best:.4f}  "
              f"top1_agree={top1_agreement:.3f}  (n={len(correlation_data[K]['probe_w'])})")

    # ==================================================================
    # Cross-K comparison: does K=3 probe beat K=2 probe?
    # ==================================================================
    print(f"\n{'='*80}")
    print("  CROSS-K COMPARISON")
    print(f"{'='*80}")

    cross_k_comparison = {}
    for K in K_values:
        if K in aggregate and 'ppl_probe_weighted' in aggregate[K]:
            probe_mean = aggregate[K]['ppl_probe_weighted']['mean_gap']
            eq_mean = aggregate[K]['equal_weight']['mean_gap']
            imp = aggregate[K]['ppl_probe_weighted']['improvement_vs_ew']
            cross_k_comparison[K] = {
                'probe_mean_gap': probe_mean,
                'eq_mean_gap': eq_mean,
                'improvement_pp': imp,
            }
            print(f"  K={K}: probe_gap={probe_mean:+.2f}%  eq_gap={eq_mean:+.2f}%  "
                  f"improvement={imp:+.2f}pp")

    # ==================================================================
    # Kill Criteria
    # ==================================================================
    print(f"\n{'='*80}")
    print("  KILL CRITERIA")
    print(f"{'='*80}")

    # K1: probe-oracle correlation at K=3
    k3_corr = correlations.get(3, {}).get('pearson_all_weights', 0.0)
    k1_kill = k3_corr < 0.8
    print(f"\n  K1: PPL-probe vs oracle correlation at K=3 >= 0.8?")
    print(f"      Pearson (all weights): r={k3_corr:.4f}")
    print(f"      Top-1 agreement: {correlations.get(3, {}).get('top1_agreement', 0):.3f}")
    print(f"      STATUS: {'KILL' if k1_kill else 'PASS'} (threshold: r=0.8)")

    # K2: K=3 probe-weighted beats K=2 probe-weighted
    k2_probe_gap = cross_k_comparison.get(2, {}).get('probe_mean_gap', 0.0)
    k3_probe_gap = cross_k_comparison.get(3, {}).get('probe_mean_gap', 0.0)
    k2_kill = k3_probe_gap > k2_probe_gap  # K=3 is worse (higher gap = worse)
    print(f"\n  K2: K=3 probe-weighted better than K=2 probe-weighted?")
    print(f"      K=2 probe mean gap: {k2_probe_gap:+.2f}%")
    print(f"      K=3 probe mean gap: {k3_probe_gap:+.2f}%")
    print(f"      STATUS: {'KILL' if k2_kill else 'PASS'}")

    # Additional: improvement scaling
    k2_imp = cross_k_comparison.get(2, {}).get('improvement_pp', 0.0)
    k3_imp = cross_k_comparison.get(3, {}).get('improvement_pp', 0.0)
    print(f"\n  Improvement scaling:")
    print(f"      K=2 improvement vs equal_weight: {k2_imp:+.2f}pp")
    print(f"      K=3 improvement vs equal_weight: {k3_imp:+.2f}pp")
    if k2_imp > 0 and k3_imp > 0:
        print(f"      Ratio K=3/K=2: {k3_imp/k2_imp:.2f}x")

    overall_kill = k1_kill and k2_kill
    print(f"\n  OVERALL: {'KILL' if overall_kill else 'PASS'}")
    print(f"    K1 (correlation): {'KILL' if k1_kill else 'PASS'}")
    print(f"    K2 (K=3 vs K=2): {'KILL' if k2_kill else 'PASS'}")

    # ==================================================================
    # Save results
    # ==================================================================
    output = {
        'experiment': 'ppl_probe_k3_scaling',
        'config': {
            'd_model': d_model, 'n_heads': n_heads, 'n_layers': n_layers,
            'vocab_size': tok.vocab_size, 'n_domains': n_domains,
            'domains': domains, 'K_values': K_values,
            'rank_per_expert': rank_per_expert,
            'epochs_per_expert': epochs_per_expert,
            'n_train': n_train, 'n_cross_test': n_cross_test,
            'n_seeds': n_seeds, 'n_probe': n_probe,
            'strategies': strategies,
        },
        'aggregate_by_K': {
            str(K): aggregate.get(K, {}) for K in K_values
        },
        'correlations_by_K': {
            str(K): correlations.get(K, {}) for K in K_values
        },
        'cross_k_comparison': {
            str(K): cross_k_comparison.get(K, {}) for K in K_values
        },
        'kill_criteria': {
            'k1_correlation_k3': k3_corr,
            'k1_threshold': 0.8,
            'k1_kill': bool(k1_kill),
            'k2_k3_probe_gap': k3_probe_gap,
            'k2_k2_probe_gap': k2_probe_gap,
            'k2_kill': bool(k2_kill),
            'overall_kill': bool(overall_kill),
        },
        'per_seed_data': all_seed_data,
    }

    output_file = results_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\n  Results saved to {output_file}")
    return output


if __name__ == '__main__':
    if '--fast' in sys.argv:
        run_experiment(n_seeds=2, epochs_per_expert=8, n_train=100,
                       n_cross_test=30)
    else:
        run_experiment()
