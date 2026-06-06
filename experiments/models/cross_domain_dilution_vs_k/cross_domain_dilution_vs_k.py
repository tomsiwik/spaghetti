#!/usr/bin/env python3
"""
Cross-Domain Dilution vs Top-K: Relevance-Weighted Expert Composition

Extends cross_domain_composition.py to test whether weighting experts by
relevance to the query improves multi-expert composition quality.

Parent experiment found: single-expert (-7.0%) beats equal-weight 2-expert (-1.0%)
because the second expert sometimes adds noise. This experiment tests whether
relevance-weighted composition can suppress noisy experts while preserving
the benefit of multi-expert composition.

Composition strategies tested:
  1. equal_weight    -- W_base + (Delta_i + Delta_j) / 2  (parent baseline)
  2. cosine_weighted -- W_base + sum(w_k * Delta_k) where w_k = softmax(cos(flat_delta_k, h_query))
  3. loss_weighted   -- W_base + sum(w_k * Delta_k) where w_k = softmax(-L_k) (oracle)
  4. top1_oracle     -- W_base + Delta_best  (best single expert, oracle)
  5. softmax_sharp   -- cosine_weighted with temperature=0.1 (nearly top-1)
  6. norm_weighted   -- W_base + sum(w_k * Delta_k) where w_k prop to ||Delta_k||

Kill criteria:
  K1: Best weighted strategy shows <2% improvement over equal_weight (aggregate)
  K2: Per-type worst-case (arith_reverse) still >20% gap
"""

import json
import math
import time
import sys
from pathlib import Path

import autograd.numpy as np
from autograd import grad
import numpy as onp

# Import everything from parent experiment
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
    flatten_delta,
    cosine_sim,
    gram_schmidt_deltas,
    _prepare_batch,
)


# ===========================================================================
# Weighted composition strategies
# ===========================================================================

def get_query_embedding(params, test_enc, pad_id):
    """Get mean hidden state of queries as a proxy for query "direction".

    Returns a flattened vector of shape (d,) representing the average
    final-layer hidden state across all test queries. This is used
    to compute relevance of each expert delta to the query distribution.
    """
    # Forward pass to get hidden states (use the base model)
    all_hiddens = []
    batch_size = 32
    for i in range(0, len(test_enc), batch_size):
        batch = test_enc[i:i+batch_size]
        inp, tgt, mask = _prepare_batch(batch, pad_id)
        if onp.sum(mask) == 0:
            continue
        # Get logits and extract final hidden from before head projection
        # We'll use the full forward and project back
        logits = onp.array(forward(params, inp, pad_id))
        # Use the logits projected through pseudo-inverse of W_head as hidden proxy
        # Simpler: just use the mean input embedding as query representation
        cfg = params['_config']
        d = cfg['d']
        B, T = inp.shape
        x = onp.array(params['tok_emb'])[inp] + onp.array(params['pos_emb'])[:T]
        # Mean over tokens and batch
        mean_h = x.mean(axis=(0, 1))
        all_hiddens.append(mean_h)

    if len(all_hiddens) == 0:
        return onp.zeros(params['_config']['d'])
    return onp.mean(all_hiddens, axis=0)


def compute_expert_relevance_activation(expert_deltas, query_embedding):
    """Compute relevance via activation magnitude: ||Delta_k @ h||.

    For each expert, measure how strongly each weight delta transforms
    the query embedding. Sum across all weight matrices.
    """
    d = len(query_embedding)
    scores = {}
    h = query_embedding

    for name, delta in expert_deltas.items():
        total_activation = 0.0
        for key, dw in delta.items():
            if dw.ndim == 2:
                if dw.shape[0] == d:
                    act = onp.linalg.norm(dw.T @ h)
                    total_activation += act
                elif dw.shape[1] == d:
                    act = onp.linalg.norm(dw @ h)
                    total_activation += act
        scores[name] = total_activation

    return scores


def compute_expert_relevance_ppl_proxy(base_params, expert_deltas, test_enc, pad_id,
                                        involved_domains, n_probe=10):
    """Compute relevance via PPL on a small probe subset of the test data.

    This is a cheap proxy for loss_weighted that uses only n_probe examples
    instead of the full test set. It simulates what a production system could
    do with a small buffer of recent queries.
    """
    probe = test_enc[:n_probe]
    scores = {}
    for dom in involved_domains:
        single_p = apply_delta(base_params, expert_deltas[dom])
        loss = eval_loss(single_p, probe, pad_id)
        scores[dom] = -loss  # negative loss = better = higher relevance
    return scores


def compute_expert_relevance_logit_diff(base_params, expert_deltas, test_enc, pad_id,
                                         involved_domains):
    """Compute relevance via logit difference: how much does each expert
    change the base model's predictions on the query?

    Experts that produce larger logit changes are more "activated" by the query.
    This is computable without ground truth labels.
    """
    scores = {}
    # Get base logits
    inp, tgt, mask = _prepare_batch(test_enc[:16], pad_id)
    base_logits = onp.array(forward(base_params, inp, pad_id))

    for dom in involved_domains:
        expert_p = apply_delta(base_params, expert_deltas[dom])
        expert_logits = onp.array(forward(expert_p, inp, pad_id))
        # L2 norm of logit difference, averaged over tokens
        diff = expert_logits - base_logits
        scores[dom] = float(onp.sqrt(onp.mean(diff ** 2)))
    return scores


def softmax_weights(scores, temperature=1.0):
    """Convert relevance scores to weights via softmax with temperature."""
    names = list(scores.keys())
    vals = onp.array([scores[n] for n in names])
    # Temperature scaling
    vals = vals / max(temperature, 1e-10)
    # Stable softmax
    vals = vals - vals.max()
    exp_vals = onp.exp(vals)
    weights = exp_vals / (exp_vals.sum() + 1e-10)
    return {n: float(w) for n, w in zip(names, weights)}


def weighted_merge(base_params, expert_deltas, weights, involved_domains):
    """Merge experts with given weights (only for involved domains).

    Args:
        base_params: base model parameters
        expert_deltas: dict of domain_name -> delta dict
        weights: dict of domain_name -> float weight
        involved_domains: list of domain names to include
    """
    merged = {}
    # Normalize weights for involved domains
    total_w = sum(weights.get(d, 0.0) for d in involved_domains)
    if total_w < 1e-10:
        total_w = 1.0

    for key in expert_deltas[involved_domains[0]]:
        s = sum(
            weights.get(d, 0.0) / total_w * expert_deltas[d][key]
            for d in involved_domains
        )
        merged[key] = s

    return apply_delta(base_params, merged)


# ===========================================================================
# Main experiment
# ===========================================================================

def run_experiment(n_domains=5, rank_per_expert=4, epochs_per_expert=15,
                   d_model=32, n_heads=2, n_layers=2, n_train=200,
                   n_cross_test=50, n_pure_test=50, n_seeds=5,
                   lr=0.001, batch_size=16):
    """Run cross-domain dilution vs top-k experiment.

    Tests 6 composition strategies against the equal-weight baseline.
    """
    domains = list(DOMAIN_GENERATORS.keys())[:n_domains]
    cross_domains = list(CROSS_DOMAIN_GENERATORS.keys())

    results_dir = Path(__file__).parent
    tok = CharTokenizer()

    strategies = [
        'equal_weight',        # baseline: (Delta_i + Delta_j) / 2
        'activation_weighted', # weight by ||Delta_k @ h_query||
        'logit_diff_weighted', # weight by logit change magnitude (no labels)
        'ppl_probe_weighted',  # weight by PPL on 10-example probe (cheap oracle)
        'loss_weighted',       # weight by inverse per-expert loss (full oracle)
        'top1_oracle',         # best single expert (oracle)
        'top1_logit_diff',     # top-1 by logit diff (no labels, production-viable)
    ]

    print("=" * 80)
    print("  CROSS-DOMAIN DILUTION vs TOP-K EXPERIMENT")
    print("=" * 80)
    print(f"  Architecture: d={d_model}, H={n_heads}, L={n_layers}, V={tok.vocab_size}")
    print(f"  Domains: {domains}")
    print(f"  Cross-domain types: {len(cross_domains)}")
    print(f"  Strategies: {strategies}")
    print(f"  Rank per expert: {rank_per_expert}")
    print(f"  Seeds: {n_seeds}")
    print("=" * 80)

    # Per-strategy, per-type, per-seed gaps
    all_gaps = {s: {ct: [] for ct in cross_domains} for s in strategies}
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
            test_data = gen(n_pure_test, onp.random.RandomState(seed + hash(dom_name) % 10000 + 1))
            domain_train_enc[dom_name] = [tok.encode(s) for s in train_data]
            domain_test_enc[dom_name] = [tok.encode(s) for s in test_data]

        combined_train = []
        for dom_name in domains:
            combined_train.extend(domain_train_enc[dom_name])
        rng.shuffle(combined_train)

        cross_test_enc = {}
        cross_test_involved = {}
        for cross_name, (gen_fn, involved) in CROSS_DOMAIN_GENERATORS.items():
            test_data = gen_fn(n_cross_test, onp.random.RandomState(seed + hash(cross_name) % 10000 + 2))
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

        # -- Compute expert norms for norm_weighted strategy --
        expert_norms = {}
        for dom_name, delta in expert_deltas.items():
            flat = flatten_delta(delta)
            expert_norms[dom_name] = float(onp.linalg.norm(flat))

        # -- Evaluate each cross-domain type --
        print(f"  Evaluating {len(cross_domains)} cross-domain types...")
        seed_results = {}

        for cross_name in cross_domains:
            involved = cross_test_involved[cross_name]
            test_enc = cross_test_enc[cross_name]

            # Base loss
            base_loss = eval_loss(base_trained, test_enc, tok.pad_id)

            # Get query embedding for activation-based relevance
            query_emb = get_query_embedding(base_params, test_enc, tok.pad_id)

            # Compute per-expert losses on cross-domain queries (for loss_weighted oracle)
            per_expert_losses = {}
            for dom_name in involved:
                single_p = apply_delta(base_params, expert_deltas[dom_name])
                per_expert_losses[dom_name] = eval_loss(single_p, test_enc, tok.pad_id)

            # Compute various relevance scores
            activation_scores = compute_expert_relevance_activation(
                {d: expert_deltas[d] for d in involved}, query_emb
            )
            logit_diff_scores = compute_expert_relevance_logit_diff(
                base_params, expert_deltas, test_enc, tok.pad_id, involved
            )
            ppl_probe_scores = compute_expert_relevance_ppl_proxy(
                base_params, expert_deltas, test_enc, tok.pad_id, involved, n_probe=10
            )

            # Strategy 1: equal_weight
            equal_delta = {}
            for key in expert_deltas[involved[0]]:
                equal_delta[key] = sum(expert_deltas[d][key] for d in involved) / len(involved)
            equal_params = apply_delta(base_params, equal_delta)
            equal_loss = eval_loss(equal_params, test_enc, tok.pad_id)

            # Strategy 2: activation_weighted
            act_weights = softmax_weights(activation_scores, temperature=1.0)
            act_params = weighted_merge(base_params, expert_deltas, act_weights, involved)
            act_loss = eval_loss(act_params, test_enc, tok.pad_id)

            # Strategy 3: logit_diff_weighted (no labels needed, production-viable)
            logit_weights = softmax_weights(logit_diff_scores, temperature=1.0)
            logit_params = weighted_merge(base_params, expert_deltas, logit_weights, involved)
            logit_loss = eval_loss(logit_params, test_enc, tok.pad_id)

            # Strategy 4: ppl_probe_weighted (10-example probe, cheap oracle)
            probe_weights = softmax_weights(ppl_probe_scores, temperature=1.0)
            probe_params = weighted_merge(base_params, expert_deltas, probe_weights, involved)
            probe_loss = eval_loss(probe_params, test_enc, tok.pad_id)

            # Strategy 5: loss_weighted (full oracle -- uses all test loss)
            loss_scores = {d: -per_expert_losses[d] for d in involved}
            loss_weights = softmax_weights(loss_scores, temperature=1.0)
            loss_params = weighted_merge(base_params, expert_deltas, loss_weights, involved)
            loss_loss = eval_loss(loss_params, test_enc, tok.pad_id)

            # Strategy 6: top1_oracle (best single expert by full loss)
            best_dom = min(involved, key=lambda d: per_expert_losses[d])
            top1_params = apply_delta(base_params, expert_deltas[best_dom])
            top1_loss = eval_loss(top1_params, test_enc, tok.pad_id)

            # Strategy 7: top1_logit_diff (best single by logit change, no labels)
            best_logit_dom = max(involved, key=lambda d: logit_diff_scores[d])
            top1_logit_params = apply_delta(base_params, expert_deltas[best_logit_dom])
            top1_logit_loss = eval_loss(top1_logit_params, test_enc, tok.pad_id)

            # Compute gaps vs base
            strategy_losses = {
                'equal_weight': equal_loss,
                'activation_weighted': act_loss,
                'logit_diff_weighted': logit_loss,
                'ppl_probe_weighted': probe_loss,
                'loss_weighted': loss_loss,
                'top1_oracle': top1_loss,
                'top1_logit_diff': top1_logit_loss,
            }

            strategy_gaps = {}
            for s_name, s_loss in strategy_losses.items():
                if base_loss > 0:
                    gap = (s_loss - base_loss) / base_loss * 100
                else:
                    gap = 0.0
                strategy_gaps[s_name] = gap
                all_gaps[s_name][cross_name].append(gap)

            # Also record weights for analysis
            seed_results[cross_name] = {
                'base_loss': float(base_loss),
                'involved': involved,
                'per_expert_losses': {k: float(v) for k, v in per_expert_losses.items()},
                'activation_scores': {k: float(v) for k, v in activation_scores.items()},
                'logit_diff_scores': {k: float(v) for k, v in logit_diff_scores.items()},
                'ppl_probe_scores': {k: float(v) for k, v in ppl_probe_scores.items()},
                'act_weights': act_weights,
                'logit_weights': logit_weights,
                'probe_weights': probe_weights,
                'loss_weights': loss_weights,
                'strategy_losses': {k: float(v) for k, v in strategy_losses.items()},
                'strategy_gaps': strategy_gaps,
            }

            # Print compact summary
            print(f"    {cross_name:<18s}: base={base_loss:.3f} "
                  f"eq={equal_loss:.3f}({strategy_gaps['equal_weight']:+.1f}%) "
                  f"logit={logit_loss:.3f}({strategy_gaps['logit_diff_weighted']:+.1f}%) "
                  f"probe={probe_loss:.3f}({strategy_gaps['ppl_probe_weighted']:+.1f}%) "
                  f"top1={top1_loss:.3f}({strategy_gaps['top1_oracle']:+.1f}%)")

        all_seed_data.append({
            'seed': seed,
            'cross_results': seed_results,
            'elapsed': time.time() - t0,
        })
        print(f"  Seed {seed} time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Aggregate results
    # ==================================================================
    print(f"\n{'='*80}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*80}")

    # Per-strategy aggregate stats
    print(f"\n  Strategy comparison (mean gap % vs base, across all types and seeds):")
    print(f"  {'Strategy':<20s} | {'Mean':>8s} | {'Std':>8s} | {'Max':>8s} | {'P75':>8s} | {'Imp vs EW':>10s}")
    print(f"  {'-'*78}")

    strategy_agg = {}
    eq_mean = None
    for s_name in strategies:
        all_s_gaps = []
        for ct in cross_domains:
            all_s_gaps.extend(all_gaps[s_name][ct])
        mean_gap = float(onp.mean(all_s_gaps))
        std_gap = float(onp.std(all_s_gaps))
        max_gap = float(onp.max(all_s_gaps))
        p75_gap = float(onp.percentile(all_s_gaps, 75))

        if s_name == 'equal_weight':
            eq_mean = mean_gap

        improvement = (eq_mean - mean_gap) if eq_mean is not None else 0.0

        strategy_agg[s_name] = {
            'mean_gap': mean_gap,
            'std_gap': std_gap,
            'max_gap': max_gap,
            'p75_gap': p75_gap,
            'all_gaps': all_s_gaps,
            'improvement_vs_ew': improvement,
        }

        print(f"  {s_name:<20s} | {mean_gap:+8.2f} | {std_gap:8.2f} | {max_gap:+8.2f} | "
              f"{p75_gap:+8.2f} | {improvement:+10.2f}pp")

    # Per-type comparison for worst case (arith_reverse)
    print(f"\n  Per-type comparison for arith_reverse (worst case in parent):")
    print(f"  {'Strategy':<20s} | {'Mean':>8s} | {'Std':>8s} | {'Max':>8s}")
    print(f"  {'-'*52}")

    arith_reverse_stats = {}
    for s_name in strategies:
        gaps = all_gaps[s_name]['arith_reverse']
        mean_g = float(onp.mean(gaps))
        std_g = float(onp.std(gaps))
        max_g = float(onp.max(gaps))
        arith_reverse_stats[s_name] = {
            'mean': mean_g, 'std': std_g, 'max': max_g,
            'per_seed': [float(g) for g in gaps],
        }
        print(f"  {s_name:<20s} | {mean_g:+8.2f} | {std_g:8.2f} | {max_g:+8.2f}")

    # Per-type comparison for all types (best strategy)
    best_strategy = min(strategies, key=lambda s: strategy_agg[s]['mean_gap'])
    print(f"\n  Best strategy: {best_strategy} (mean gap: {strategy_agg[best_strategy]['mean_gap']:+.2f}%)")

    print(f"\n  Per-type comparison: equal_weight vs {best_strategy}:")
    print(f"  {'Cross-Type':<18s} | {'EW Mean':>8s} | {'Best Mean':>8s} | {'Delta':>8s} | {'EW Max':>8s} | {'Best Max':>8s}")
    print(f"  {'-'*72}")

    per_type_comparison = {}
    for ct in cross_domains:
        ew_gaps = all_gaps['equal_weight'][ct]
        best_gaps = all_gaps[best_strategy][ct]
        ew_mean = float(onp.mean(ew_gaps))
        best_mean = float(onp.mean(best_gaps))
        ew_max = float(onp.max(ew_gaps))
        best_max = float(onp.max(best_gaps))
        delta = ew_mean - best_mean

        per_type_comparison[ct] = {
            'ew_mean': ew_mean, 'best_mean': best_mean,
            'delta': delta, 'ew_max': ew_max, 'best_max': best_max,
        }
        print(f"  {ct:<18s} | {ew_mean:+8.2f} | {best_mean:+8.2f} | {delta:+8.2f} | "
              f"{ew_max:+8.2f} | {best_max:+8.2f}")

    # Weight analysis: do proxy weights correlate with loss weights?
    print(f"\n  Weight correlation analysis (proxy weights vs loss-based oracle):")
    act_w1_list, logit_w1_list, probe_w1_list, loss_w1_list = [], [], [], []
    for seed_data in all_seed_data:
        for ct, cr in seed_data['cross_results'].items():
            inv = cr['involved']
            act_w1_list.append(cr['act_weights'].get(inv[0], 0.5))
            logit_w1_list.append(cr['logit_weights'].get(inv[0], 0.5))
            probe_w1_list.append(cr['probe_weights'].get(inv[0], 0.5))
            loss_w1_list.append(cr['loss_weights'].get(inv[0], 0.5))

    loss_arr = onp.array(loss_w1_list)
    weight_corrs = {}
    for name, arr in [('activation', act_w1_list), ('logit_diff', logit_w1_list),
                       ('ppl_probe', probe_w1_list)]:
        arr = onp.array(arr)
        if onp.std(arr) > 1e-10 and onp.std(loss_arr) > 1e-10:
            corr = float(onp.corrcoef(arr, loss_arr)[0, 1])
        else:
            corr = 0.0
        weight_corrs[name] = corr
        print(f"    {name:<12s} vs loss weights: r={corr:.4f} (std={float(onp.std(arr)):.4f})")
    print(f"    Loss weight std: {float(onp.std(loss_arr)):.4f}")

    # ==================================================================
    # Kill Criteria
    # ==================================================================
    print(f"\n{'='*80}")
    print("  KILL CRITERIA")
    print(f"{'='*80}")

    # Find best non-equal-weight strategy
    best_weighted = min(
        [s for s in strategies if s != 'equal_weight'],
        key=lambda s: strategy_agg[s]['mean_gap']
    )
    improvement = strategy_agg['equal_weight']['mean_gap'] - strategy_agg[best_weighted]['mean_gap']

    k1_kill = improvement < 2.0
    print(f"\n  K1: Best weighted strategy improvement >2% over equal_weight?")
    print(f"      Best weighted: {best_weighted}")
    print(f"      Equal-weight mean gap: {strategy_agg['equal_weight']['mean_gap']:+.2f}%")
    print(f"      Best weighted mean gap: {strategy_agg[best_weighted]['mean_gap']:+.2f}%")
    print(f"      Improvement: {improvement:+.2f}pp")
    print(f"      STATUS: {'KILL' if k1_kill else 'PASS'} (threshold: 2.0pp)")

    # K2: arith_reverse worst-case improves
    ew_ar_mean = arith_reverse_stats['equal_weight']['mean']
    best_ar_mean = arith_reverse_stats[best_weighted]['mean']
    k2_kill = best_ar_mean > 20.0
    print(f"\n  K2: arith_reverse worst-case improves to <20%?")
    print(f"      Equal-weight arith_reverse: {ew_ar_mean:+.2f}%")
    print(f"      {best_weighted} arith_reverse: {best_ar_mean:+.2f}%")
    print(f"      STATUS: {'KILL' if k2_kill else 'PASS'} (threshold: 20%)")

    overall_kill = k1_kill and k2_kill  # Both must fail to kill
    print(f"\n  OVERALL: {'KILL (both criteria fail)' if overall_kill else 'At least one criterion passes'}")
    print(f"    K1 (improvement): {'KILL' if k1_kill else 'PASS'}")
    print(f"    K2 (worst-case): {'KILL' if k2_kill else 'PASS'}")

    # ==================================================================
    # Save results
    # ==================================================================
    output = {
        'experiment': 'cross_domain_dilution_vs_k',
        'config': {
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'vocab_size': tok.vocab_size,
            'n_domains': n_domains,
            'domains': domains,
            'n_cross_domains': len(cross_domains),
            'rank_per_expert': rank_per_expert,
            'epochs_per_expert': epochs_per_expert,
            'n_train': n_train,
            'n_cross_test': n_cross_test,
            'n_seeds': n_seeds,
            'strategies': strategies,
        },
        'strategy_aggregate': {
            s: {
                'mean_gap': strategy_agg[s]['mean_gap'],
                'std_gap': strategy_agg[s]['std_gap'],
                'max_gap': strategy_agg[s]['max_gap'],
                'p75_gap': strategy_agg[s]['p75_gap'],
                'improvement_vs_ew': strategy_agg[s]['improvement_vs_ew'],
            }
            for s in strategies
        },
        'arith_reverse_stats': arith_reverse_stats,
        'per_type_comparison': per_type_comparison,
        'best_strategy': best_strategy,
        'best_weighted_strategy': best_weighted,
        'weight_correlation': weight_corrs,
        'kill_criteria': {
            'k1_improvement_pp': improvement,
            'k1_threshold': 2.0,
            'k1_kill': bool(k1_kill),
            'k2_arith_reverse_best': best_ar_mean,
            'k2_threshold': 20.0,
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
                       n_cross_test=30, n_pure_test=30)
    else:
        run_experiment()
