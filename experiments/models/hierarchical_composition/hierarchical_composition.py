#!/usr/bin/env python3
"""
Hierarchical Expert Composition: Two-Level Hierarchy vs Flat for Clustered Domains

Tests whether a two-level expert hierarchy (foundation LoRA for broad cluster +
specialist LoRA per domain) outperforms flat experts of equivalent total rank
budget, specifically for cross-domain queries within the same cluster.

Architecture comparison (equalized rank budget = 8 per domain):
  - Flat: 5 independent rank-8 experts
  - Hierarchical: 2 foundation rank-4 (per cluster) + 5 specialist rank-4

Cluster assignment:
  - "symbolic": arithmetic, parity (share numerical reasoning)
  - "string": reverse, repeat, sort (share character manipulation)

Kill criteria:
  K1: hierarchical NOT better than flat for within-cluster cross-domain queries
  K2: hierarchy adds >30% complexity for <5% quality improvement
"""

import json
import math
import time
import sys
from pathlib import Path

import numpy as np

# Reuse infrastructure from cross_domain_composition
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

import numpy as onp  # alias for clarity (non-autograd)


# ===========================================================================
# Cluster definitions
# ===========================================================================

CLUSTERS = {
    "symbolic": ["arithmetic", "parity"],
    "string": ["reverse", "repeat", "sort"],
}

# Classify each cross-domain pair as within-cluster or across-cluster
def classify_cross_domains():
    """Classify each cross-domain pair by cluster relationship."""
    domain_to_cluster = {}
    for cname, doms in CLUSTERS.items():
        for d in doms:
            domain_to_cluster[d] = cname

    within_cluster = []
    across_cluster = []

    for cross_name, (gen_fn, involved) in CROSS_DOMAIN_GENERATORS.items():
        c1 = domain_to_cluster.get(involved[0])
        c2 = domain_to_cluster.get(involved[1])
        if c1 == c2:
            within_cluster.append(cross_name)
        else:
            across_cluster.append(cross_name)

    return within_cluster, across_cluster, domain_to_cluster


# ===========================================================================
# Hierarchical composition: extract foundation + specialist
# ===========================================================================

def extract_foundation(expert_deltas, cluster_domains, rank_foundation):
    """Extract shared foundation delta from a cluster of expert deltas via SVD.

    For each weight matrix key, stack the deltas from all cluster domains
    and compute the top-r_f shared directions.

    Returns:
        foundation_delta: dict of key -> (d x d) matrix at rank r_f
        specialist_deltas: dict of domain -> dict of key -> residual at rank r_s
    """
    # Get the keys (weight matrices) from any expert
    sample_dom = cluster_domains[0]
    keys = sorted(k for k in expert_deltas[sample_dom].keys())

    foundation = {}
    residuals = {dom: {} for dom in cluster_domains}

    for key in keys:
        val = expert_deltas[sample_dom][key]
        if val.ndim == 1:
            # For 1-d params (layer norms), average across cluster
            stacked = onp.stack([expert_deltas[d][key] for d in cluster_domains])
            foundation[key] = stacked.mean(axis=0)
            for dom in cluster_domains:
                residuals[dom][key] = expert_deltas[dom][key] - foundation[key]
            continue

        # For 2-d weight matrices: stack and SVD
        # Stack all cluster deltas along first axis: shape (n_cluster * rows, cols)
        stacked = onp.vstack([expert_deltas[d][key] for d in cluster_domains])
        U, S, Vt = onp.linalg.svd(stacked, full_matrices=False)
        r = min(rank_foundation, len(S))

        # Foundation: reconstruct from top-r singular vectors
        # But we need a (rows, cols) matrix. Use the shared right singular vectors
        # weighted by the average projection.
        rows, cols = val.shape
        n_dom = len(cluster_domains)

        # For each domain, its contribution is stacked[i*rows:(i+1)*rows, :]
        # Project each domain delta onto top-r right singular vectors
        Vr = Vt[:r, :]  # (r, cols)

        # Foundation = average of all domain projections onto shared subspace
        projections = []
        for i, dom in enumerate(cluster_domains):
            delta_i = expert_deltas[dom][key]  # (rows, cols)
            # Project: delta_i @ Vr^T @ Vr
            coeff = delta_i @ Vr.T  # (rows, r)
            recon = coeff @ Vr      # (rows, cols)
            projections.append(recon)

        foundation[key] = onp.mean(projections, axis=0)

        # SVD-truncate the foundation to ensure rank-r
        Uf, Sf, Vtf = onp.linalg.svd(foundation[key], full_matrices=False)
        rf = min(rank_foundation, len(Sf))
        foundation[key] = (Uf[:, :rf] * Sf[:rf]) @ Vtf[:rf, :]

        # Residual for each domain
        for dom in cluster_domains:
            residuals[dom][key] = expert_deltas[dom][key] - foundation[key]

    return foundation, residuals


def truncate_specialists(residuals, rank_specialist):
    """SVD-truncate each specialist residual to rank_specialist."""
    specialists = {}
    for dom, delta in residuals.items():
        specialists[dom], _ = svd_truncate_delta(delta, rank_specialist)
    return specialists


# ===========================================================================
# PPL-probe weighting (reuse from dilution experiment)
# ===========================================================================

def softmax_weights(scores, temperature=1.0):
    names = list(scores.keys())
    vals = onp.array([scores[n] for n in names])
    vals = vals / max(temperature, 1e-10)
    vals = vals - vals.max()
    exp_vals = onp.exp(vals)
    weights = exp_vals / (exp_vals.sum() + 1e-10)
    return {n: float(w) for n, w in zip(names, weights)}


def ppl_probe_scores(base_params, expert_deltas, test_enc, pad_id,
                     involved_domains, n_probe=10):
    """Score experts by PPL on small probe subset."""
    probe = test_enc[:n_probe]
    scores = {}
    for dom in involved_domains:
        single_p = apply_delta(base_params, expert_deltas[dom])
        loss = eval_loss(single_p, probe, pad_id)
        scores[dom] = -loss
    return scores


# ===========================================================================
# Composition functions
# ===========================================================================

def compose_flat_equal(base_params, expert_deltas, involved, rank):
    """Flat equal-weight composition."""
    deltas = [expert_deltas[d] for d in involved]
    merged = merge_deltas(deltas, mode='avg')
    return apply_delta(base_params, merged)


def compose_flat_weighted(base_params, expert_deltas, involved, weights):
    """Flat weighted composition."""
    merged = {}
    total_w = sum(weights.get(d, 0.0) for d in involved)
    if total_w < 1e-10:
        total_w = 1.0
    for key in expert_deltas[involved[0]]:
        s = sum(weights.get(d, 0.0) / total_w * expert_deltas[d][key]
                for d in involved)
        merged[key] = s
    return apply_delta(base_params, merged)


def compose_hier_equal(base_params, foundations, specialists,
                       involved, domain_to_cluster):
    """Hierarchical equal-weight composition.

    For within-cluster: add foundation once + average specialists.
    For across-cluster: average foundations + average specialists.
    """
    clusters_involved = set(domain_to_cluster[d] for d in involved)

    # Start with foundation(s)
    merged = {}
    if len(clusters_involved) == 1:
        # Within cluster: add the single foundation at full strength
        cname = list(clusters_involved)[0]
        for key in foundations[cname]:
            merged[key] = foundations[cname][key].copy()
    else:
        # Across cluster: average foundations
        cluster_list = list(clusters_involved)
        for key in foundations[cluster_list[0]]:
            s = sum(foundations[c][key] for c in cluster_list) / len(cluster_list)
            merged[key] = s

    # Add specialist deltas (equal weight)
    for key in specialists[involved[0]]:
        s = sum(specialists[d][key] for d in involved) / len(involved)
        merged[key] = merged.get(key, 0) + s

    return apply_delta(base_params, merged)


def compose_hier_weighted(base_params, foundations, specialists,
                          involved, domain_to_cluster, specialist_weights):
    """Hierarchical weighted composition with PPL-based specialist weights."""
    clusters_involved = set(domain_to_cluster[d] for d in involved)

    merged = {}
    if len(clusters_involved) == 1:
        cname = list(clusters_involved)[0]
        for key in foundations[cname]:
            merged[key] = foundations[cname][key].copy()
    else:
        cluster_list = list(clusters_involved)
        # Weight clusters by how many involved domains they contain
        cluster_weights = {}
        for c in cluster_list:
            count = sum(1 for d in involved if domain_to_cluster[d] == c)
            cluster_weights[c] = count / len(involved)
        for key in foundations[cluster_list[0]]:
            s = sum(cluster_weights[c] * foundations[c][key] for c in cluster_list)
            merged[key] = s

    # Add weighted specialist deltas
    total_w = sum(specialist_weights.get(d, 0.0) for d in involved)
    if total_w < 1e-10:
        total_w = 1.0
    for key in specialists[involved[0]]:
        s = sum(specialist_weights.get(d, 0.0) / total_w * specialists[d][key]
                for d in involved)
        merged[key] = merged.get(key, 0) + s

    return apply_delta(base_params, merged)


# ===========================================================================
# Main experiment
# ===========================================================================

def run_experiment(n_domains=5, rank_flat=8, rank_foundation=4,
                   rank_specialist=4, epochs_per_expert=15,
                   d_model=64, n_heads=4, n_layers=2, n_train=200,
                   n_cross_test=50, n_pure_test=50, n_seeds=5,
                   lr=0.001, batch_size=16):
    """Run hierarchical vs flat composition experiment."""

    domains = list(DOMAIN_GENERATORS.keys())[:n_domains]
    cross_domains = list(CROSS_DOMAIN_GENERATORS.keys())
    within_cluster, across_cluster, domain_to_cluster = classify_cross_domains()

    results_dir = Path(__file__).parent
    tok = CharTokenizer()

    print("=" * 80)
    print("  HIERARCHICAL vs FLAT EXPERT COMPOSITION")
    print("=" * 80)
    print(f"  Architecture: d={d_model}, H={n_heads}, L={n_layers}, V={tok.vocab_size}")
    print(f"  Domains: {domains}")
    print(f"  Clusters: {CLUSTERS}")
    print(f"  Within-cluster pairs: {within_cluster}")
    print(f"  Across-cluster pairs: {across_cluster}")
    print(f"  Rank flat: {rank_flat}, foundation: {rank_foundation}, specialist: {rank_specialist}")
    print(f"  Seeds: {n_seeds}")
    print("=" * 80)

    # Track per-strategy, per-type gaps
    strategies = [
        'flat_equal', 'flat_ppl',
        'hier_equal', 'hier_ppl',
    ]

    all_gaps = {s: {ct: [] for ct in cross_domains} for s in strategies}
    all_seed_data = []
    total_train_time_flat = 0.0
    total_train_time_hier = 0.0

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

        # -- Train domain experts (full rank) --
        print(f"  Training {n_domains} domain experts...")
        t_train_start = time.time()
        full_deltas = {}
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
            full_deltas[dom_name] = compute_delta(base_params, expert_params)
        t_expert_train = time.time() - t_train_start

        # -- FLAT: SVD truncate to rank_flat --
        t_flat_start = time.time()
        flat_deltas = {}
        for dom_name in domains:
            flat_deltas[dom_name], _ = svd_truncate_delta(full_deltas[dom_name], rank_flat)
        t_flat_process = time.time() - t_flat_start
        total_train_time_flat += t_expert_train + t_flat_process

        # -- HIERARCHICAL: extract foundation + specialist --
        t_hier_start = time.time()
        foundations = {}
        all_specialists = {}
        for cname, cdomains in CLUSTERS.items():
            # Extract foundation from full deltas
            foundation, residuals = extract_foundation(full_deltas, cdomains, rank_foundation)
            foundations[cname] = foundation
            # Truncate specialists
            specs = truncate_specialists(residuals, rank_specialist)
            for dom in cdomains:
                all_specialists[dom] = specs[dom]
        t_hier_process = time.time() - t_hier_start
        total_train_time_hier += t_expert_train + t_hier_process

        # -- Measure subspace sharing --
        print(f"  Subspace analysis:")
        for cname, cdomains in CLUSTERS.items():
            if len(cdomains) < 2:
                continue
            flat_vecs = [flatten_delta(flat_deltas[d]) for d in cdomains]
            for i in range(len(cdomains)):
                for j in range(i+1, len(cdomains)):
                    cos = cosine_sim(flat_vecs[i], flat_vecs[j])
                    print(f"    {cdomains[i]} vs {cdomains[j]} (cluster {cname}): cos={cos:.4f}")

        # Cross-cluster
        cross_pairs = []
        for c1_doms in CLUSTERS.values():
            for c2_doms in CLUSTERS.values():
                if c1_doms == c2_doms:
                    continue
                for d1 in c1_doms:
                    for d2 in c2_doms:
                        if (d2, d1) not in cross_pairs:
                            cross_pairs.append((d1, d2))
        for d1, d2 in cross_pairs[:3]:  # sample
            v1, v2 = flatten_delta(flat_deltas[d1]), flatten_delta(flat_deltas[d2])
            cos = cosine_sim(v1, v2)
            print(f"    {d1} vs {d2} (cross-cluster): cos={cos:.4f}")

        # -- Evaluate on cross-domain queries --
        print(f"  Evaluating cross-domain composition...")
        seed_cross_results = {}

        for cross_name in cross_domains:
            involved = cross_test_involved[cross_name]
            test_enc = cross_test_enc[cross_name]

            base_loss = eval_loss(base_trained, test_enc, tok.pad_id)

            # PPL probe weights for flat experts
            probe_scores = ppl_probe_scores(
                base_params, flat_deltas, test_enc, tok.pad_id,
                involved, n_probe=10
            )
            flat_weights = softmax_weights(probe_scores, temperature=1.0)

            # PPL probe weights for specialist experts (apply foundation first)
            # For hierarchical probe: score foundation+specialist combos
            hier_probe_scores = {}
            for dom in involved:
                cname = domain_to_cluster[dom]
                # Build params with foundation + this specialist
                hier_delta = {}
                for key in foundations[cname]:
                    hier_delta[key] = foundations[cname][key] + all_specialists[dom][key]
                single_hier_p = apply_delta(base_params, hier_delta)
                loss = eval_loss(single_hier_p, test_enc[:10], tok.pad_id)
                hier_probe_scores[dom] = -loss
            hier_weights = softmax_weights(hier_probe_scores, temperature=1.0)

            # Strategy 1: flat_equal
            flat_eq_params = compose_flat_equal(base_params, flat_deltas, involved, rank_flat)
            flat_eq_loss = eval_loss(flat_eq_params, test_enc, tok.pad_id)

            # Strategy 2: flat_ppl
            flat_ppl_params = compose_flat_weighted(base_params, flat_deltas, involved, flat_weights)
            flat_ppl_loss = eval_loss(flat_ppl_params, test_enc, tok.pad_id)

            # Strategy 3: hier_equal
            hier_eq_params = compose_hier_equal(
                base_params, foundations, all_specialists,
                involved, domain_to_cluster
            )
            hier_eq_loss = eval_loss(hier_eq_params, test_enc, tok.pad_id)

            # Strategy 4: hier_ppl
            hier_ppl_params = compose_hier_weighted(
                base_params, foundations, all_specialists,
                involved, domain_to_cluster, hier_weights
            )
            hier_ppl_loss = eval_loss(hier_ppl_params, test_enc, tok.pad_id)

            # Gaps vs base
            losses = {
                'flat_equal': flat_eq_loss,
                'flat_ppl': flat_ppl_loss,
                'hier_equal': hier_eq_loss,
                'hier_ppl': hier_ppl_loss,
            }
            gaps = {}
            for sname, sloss in losses.items():
                if base_loss > 0:
                    gaps[sname] = (sloss - base_loss) / base_loss * 100
                else:
                    gaps[sname] = 0.0
                all_gaps[sname][cross_name].append(gaps[sname])

            cluster_type = "within" if cross_name in within_cluster else "across"
            print(f"    {cross_name:<18s} [{cluster_type:>6s}]: "
                  f"flat_eq={flat_eq_loss:.3f}({gaps['flat_equal']:+.1f}%) "
                  f"flat_ppl={flat_ppl_loss:.3f}({gaps['flat_ppl']:+.1f}%) "
                  f"hier_eq={hier_eq_loss:.3f}({gaps['hier_equal']:+.1f}%) "
                  f"hier_ppl={hier_ppl_loss:.3f}({gaps['hier_ppl']:+.1f}%)")

            seed_cross_results[cross_name] = {
                'involved': involved,
                'cluster_type': cluster_type,
                'base_loss': float(base_loss),
                'losses': {k: float(v) for k, v in losses.items()},
                'gaps': gaps,
                'flat_weights': flat_weights,
                'hier_weights': hier_weights,
            }

        all_seed_data.append({
            'seed': seed,
            'cross_results': seed_cross_results,
            'elapsed': time.time() - t0,
        })
        print(f"  Seed {seed} time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Aggregate results
    # ==================================================================
    print(f"\n{'='*80}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*80}")

    # Within-cluster vs across-cluster comparison
    for scope, pair_list in [("WITHIN-CLUSTER", within_cluster),
                              ("ACROSS-CLUSTER", across_cluster),
                              ("ALL", cross_domains)]:
        print(f"\n  {scope} cross-domain pairs ({len(pair_list)} types):")
        print(f"  {'Strategy':<15s} | {'Mean':>8s} | {'Std':>8s} | {'Max':>8s} | {'P75':>8s}")
        print(f"  {'-'*56}")

        scope_agg = {}
        for sname in strategies:
            scope_gaps = []
            for ct in pair_list:
                scope_gaps.extend(all_gaps[sname][ct])
            mean_g = float(onp.mean(scope_gaps)) if scope_gaps else 0.0
            std_g = float(onp.std(scope_gaps)) if scope_gaps else 0.0
            max_g = float(onp.max(scope_gaps)) if scope_gaps else 0.0
            p75_g = float(onp.percentile(scope_gaps, 75)) if scope_gaps else 0.0
            scope_agg[sname] = {
                'mean': mean_g, 'std': std_g, 'max': max_g, 'p75': p75_g,
                'all_gaps': scope_gaps,
            }
            print(f"  {sname:<15s} | {mean_g:+8.2f} | {std_g:8.2f} | {max_g:+8.2f} | {p75_g:+8.2f}")

        if scope == "WITHIN-CLUSTER":
            within_agg = scope_agg
        elif scope == "ACROSS-CLUSTER":
            across_agg = scope_agg
        else:
            all_agg = scope_agg

    # Per-type comparison
    print(f"\n  Per-type: flat_ppl vs hier_ppl (best strategies):")
    print(f"  {'Cross-Type':<18s} | {'Type':>6s} | {'Flat PPL':>9s} | {'Hier PPL':>9s} | {'Delta':>8s}")
    print(f"  {'-'*65}")
    per_type_delta = {}
    for ct in cross_domains:
        flat_gaps = all_gaps['flat_ppl'][ct]
        hier_gaps = all_gaps['hier_ppl'][ct]
        flat_mean = float(onp.mean(flat_gaps))
        hier_mean = float(onp.mean(hier_gaps))
        delta = flat_mean - hier_mean  # positive = hierarchy better
        ct_type = "within" if ct in within_cluster else "across"
        per_type_delta[ct] = {
            'cluster_type': ct_type,
            'flat_ppl_mean': flat_mean,
            'hier_ppl_mean': hier_mean,
            'delta_pp': delta,
        }
        print(f"  {ct:<18s} | {ct_type:>6s} | {flat_mean:+9.2f} | {hier_mean:+9.2f} | {delta:+8.2f}pp")

    # Complexity analysis
    print(f"\n  Complexity:")
    print(f"    Total flat processing time: {total_train_time_flat:.2f}s")
    print(f"    Total hier processing time: {total_train_time_hier:.2f}s")
    overhead = (total_train_time_hier - total_train_time_flat) / max(total_train_time_flat, 0.01)
    print(f"    Overhead: {overhead*100:+.1f}%")

    # ==================================================================
    # Kill Criteria
    # ==================================================================
    print(f"\n{'='*80}")
    print("  KILL CRITERIA")
    print(f"{'='*80}")

    # K1: hierarchy vs flat on within-cluster (using PPL-probe, the best strategies)
    within_flat_mean = within_agg['flat_ppl']['mean']
    within_hier_mean = within_agg['hier_ppl']['mean']
    quality_improvement = within_flat_mean - within_hier_mean  # positive = hier better

    # Also compare equal-weight variants
    within_flat_eq_mean = within_agg['flat_equal']['mean']
    within_hier_eq_mean = within_agg['hier_equal']['mean']
    quality_improvement_eq = within_flat_eq_mean - within_hier_eq_mean

    k1_kill = within_hier_mean >= within_flat_mean
    print(f"\n  K1: Hierarchical better than flat for within-cluster queries?")
    print(f"      Flat PPL within-cluster mean:  {within_flat_mean:+.2f}%")
    print(f"      Hier PPL within-cluster mean:  {within_hier_mean:+.2f}%")
    print(f"      Improvement (flat - hier):     {quality_improvement:+.2f}pp")
    print(f"      (Equal-weight: flat={within_flat_eq_mean:+.2f}%, hier={within_hier_eq_mean:+.2f}%, delta={quality_improvement_eq:+.2f}pp)")
    print(f"      STATUS: {'KILL' if k1_kill else 'PASS'}")

    # Also check across-cluster (informational)
    across_flat_mean = across_agg['flat_ppl']['mean']
    across_hier_mean = across_agg['hier_ppl']['mean']
    across_delta = across_flat_mean - across_hier_mean
    print(f"\n  Across-cluster (informational):")
    print(f"      Flat PPL across-cluster mean:  {across_flat_mean:+.2f}%")
    print(f"      Hier PPL across-cluster mean:  {across_hier_mean:+.2f}%")
    print(f"      Delta: {across_delta:+.2f}pp")

    # K2: complexity vs quality
    k2_kill = overhead > 0.30 and abs(quality_improvement) < 5.0
    print(f"\n  K2: Complexity overhead <30% OR quality improvement >=5%?")
    print(f"      Overhead: {overhead*100:+.1f}%")
    print(f"      Quality improvement: {quality_improvement:+.2f}pp")
    print(f"      STATUS: {'KILL' if k2_kill else 'PASS'}")

    overall_kill = k1_kill  # K1 is the primary criterion
    print(f"\n  OVERALL: {'KILL' if overall_kill else 'PASS'}")
    if overall_kill:
        print(f"  REASON: Hierarchical composition does NOT outperform flat")
    else:
        print(f"  REASON: Hierarchical composition improves within-cluster quality by {quality_improvement:+.2f}pp")

    # Statistical significance (simple t-test on within-cluster gaps)
    from scipy import stats as sp_stats
    within_flat_gaps = within_agg['flat_ppl']['all_gaps']
    within_hier_gaps = within_agg['hier_ppl']['all_gaps']
    if len(within_flat_gaps) > 1 and len(within_hier_gaps) > 1:
        t_stat, p_val = sp_stats.ttest_ind(within_flat_gaps, within_hier_gaps)
        print(f"\n  Statistical test (flat_ppl vs hier_ppl, within-cluster):")
        print(f"    t-statistic: {t_stat:.4f}")
        print(f"    p-value: {p_val:.4f}")
        print(f"    Significant at alpha=0.05: {'YES' if p_val < 0.05 else 'NO'}")
    else:
        t_stat, p_val = 0.0, 1.0

    # ==================================================================
    # Save results
    # ==================================================================
    output = {
        'experiment': 'hierarchical_composition',
        'config': {
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'vocab_size': tok.vocab_size,
            'n_domains': n_domains,
            'domains': domains,
            'clusters': CLUSTERS,
            'rank_flat': rank_flat,
            'rank_foundation': rank_foundation,
            'rank_specialist': rank_specialist,
            'epochs_per_expert': epochs_per_expert,
            'n_train': n_train,
            'n_cross_test': n_cross_test,
            'n_seeds': n_seeds,
        },
        'within_cluster_pairs': within_cluster,
        'across_cluster_pairs': across_cluster,
        'within_cluster_agg': {
            s: {'mean': within_agg[s]['mean'], 'std': within_agg[s]['std'],
                'max': within_agg[s]['max'], 'p75': within_agg[s]['p75']}
            for s in strategies
        },
        'across_cluster_agg': {
            s: {'mean': across_agg[s]['mean'], 'std': across_agg[s]['std'],
                'max': across_agg[s]['max'], 'p75': across_agg[s]['p75']}
            for s in strategies
        },
        'all_agg': {
            s: {'mean': all_agg[s]['mean'], 'std': all_agg[s]['std'],
                'max': all_agg[s]['max'], 'p75': all_agg[s]['p75']}
            for s in strategies
        },
        'per_type_delta': per_type_delta,
        'complexity': {
            'flat_time': total_train_time_flat,
            'hier_time': total_train_time_hier,
            'overhead_pct': overhead * 100,
        },
        'kill_criteria': {
            'k1_within_flat_ppl_mean': within_flat_mean,
            'k1_within_hier_ppl_mean': within_hier_mean,
            'k1_quality_improvement_pp': quality_improvement,
            'k1_kill': bool(k1_kill),
            'k2_overhead_pct': overhead * 100,
            'k2_quality_improvement_pp': quality_improvement,
            'k2_kill': bool(k2_kill),
            'overall_kill': bool(overall_kill),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
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
                       n_cross_test=30, n_pure_test=30, d_model=32,
                       n_heads=2, rank_flat=4, rank_foundation=2,
                       rank_specialist=2)
    else:
        run_experiment()
