#!/usr/bin/env python3
"""
B-Matrix Training Correlation: Does LoRA training create structured B-matrix overlap?

The minimax_grassmannian_packing experiment discovered that the d=256 tail anomaly
(max/mean=9.36x in post-training delta vectors) originates from B-matrix training
dynamics, NOT skeleton geometry. The skeleton has perfect equidistribution (1.00x).

This experiment directly measures:
1. Whether trained B-matrices have higher pairwise |cos| than random B-matrices
2. Whether domain similarity predicts B-matrix correlation (similar > dissimilar)
3. Whether B-matrix correlation increases amplification ratio (safety concern)

The correlated_layer_errors parent proved correlation REDUCES amplification
(amp_ratio=0.074 at rho=1.0), but with synthetic correlation. Here we test
with REAL trained B-matrix correlation patterns.

Kill criteria:
  K1: trained B-matrices show >3x higher pairwise |cos| than random init
      -> structured correlation EXISTS (interesting, not dangerous)
  K2: B-matrix correlation does NOT increase amplification ratio vs uncorrelated
      -> SAFE even if correlated (the desired outcome)

Domain pairs:
  Similar:    (0, 1), (2, 3) -- adjacent domain IDs share transition patterns
  Dissimilar: (0, 50), (2, 100) -- distant domain IDs have independent patterns

CPU only. numpy/scipy on Apple Silicon. Runtime target: < 5 min.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

# Reuse parent infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent / 'grassmannian_expert_init'))
from grassmannian_expert_init import (
    DTYPE, VOCAB_SIZE, CONTEXT_LEN, LORA_RANK, LORA_ALPHA,
    D_CONFIG, welch_bound,
    alternating_projection, random_grassmannian_points,
    MicroMLP, generate_domain_data,
    init_lora_from_frame, init_lora_random_orthonormal,
    lora_delta_vec, train_lora, cosine_sim,
)

# Reuse correlated_layer_errors infrastructure for amplification testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'correlated_layer_errors'))

SEEDS = [42, 137, 777]

# Domain configuration: 4 pairs, 8 experts
# Similar pairs: adjacent domain IDs share transition structure
# Dissimilar pairs: distant domain IDs have independent structure
DOMAIN_PAIRS = {
    'similar_1': (0, 1),
    'similar_2': (2, 3),
    'dissimilar_1': (0, 50),
    'dissimilar_2': (2, 100),
}

# All 8 unique domain IDs
ALL_DOMAINS = sorted(set(d for pair in DOMAIN_PAIRS.values() for d in pair))


# ============================================================================
# B-matrix extraction
# ============================================================================

def extract_b_matrices(B1_list, B2_list):
    """Extract and concatenate all B-matrix parameters into a flat vector.

    B1[l]: (r, d_ff) -- B-matrix for W1 at layer l
    B2[l]: (r, d)    -- B-matrix for W2 at layer l

    Returns flat vector of all B parameters.
    """
    parts = []
    for l in range(len(B1_list)):
        parts.append(B1_list[l].ravel())
        parts.append(B2_list[l].ravel())
    return np.concatenate(parts)


def extract_b_per_layer(B1_list, B2_list):
    """Extract B-matrices per layer as flat vectors.

    Returns list of flat vectors, one per layer (concatenating B1 and B2).
    """
    per_layer = []
    for l in range(len(B1_list)):
        v = np.concatenate([B1_list[l].ravel(), B2_list[l].ravel()])
        per_layer.append(v)
    return per_layer


# ============================================================================
# Amplification ratio computation (simplified from correlated_layer_errors)
# ============================================================================

def compute_amplification_with_real_b(
    model, experts_data, d, d_ff, n_layers, n_inputs=200, seed=42
):
    """Compute amplification ratio using real trained experts.

    experts_data: list of dicts with keys 'A1', 'B1', 'A2', 'B2'
    Returns amplification metrics.
    """
    rng = np.random.RandomState(seed + 9999)
    N = len(experts_data)
    r = LORA_RANK
    scale = LORA_ALPHA / LORA_RANK

    # Compute per-layer deltas: delta_W = scale * A @ B
    all_deltas = []  # [expert][layer] = (d_out, d_in) matrix
    for exp in experts_data:
        layer_deltas = []
        for l in range(n_layers):
            # W1 delta: (d, d_ff)
            dW1 = scale * (exp['A1'][l] @ exp['B1'][l])
            # W2 delta: (d_ff, d)
            dW2 = scale * (exp['A2'][l] @ exp['B2'][l])
            # Concatenate for full layer representation
            layer_deltas.append((dW1, dW2))
        all_deltas.append(layer_deltas)

    # Generate test inputs
    inputs = rng.randn(n_inputs, d).astype(DTYPE) * 0.1

    # Forward pass with all experts composed (additive composition)
    def forward_with_deltas(x, base_model, deltas_per_layer):
        """Forward pass through model with additive LoRA deltas."""
        h = x.copy()
        for l in range(n_layers):
            h_in = h
            W1_eff = base_model.layers[l]['W1'] + deltas_per_layer[l][0]
            z1 = h @ W1_eff
            a1 = np.maximum(z1, 0)
            W2_eff = base_model.layers[l]['W2'] + deltas_per_layer[l][1]
            h = h_in + a1 @ W2_eff
        return h

    # Compose all experts: sum their deltas
    composed_deltas = []
    for l in range(n_layers):
        sum_dW1 = sum(all_deltas[i][l][0] for i in range(N))
        sum_dW2 = sum(all_deltas[i][l][1] for i in range(N))
        composed_deltas.append((sum_dW1, sum_dW2))

    # Forward with all experts
    outputs_all = np.array([
        forward_with_deltas(inp, model, composed_deltas) for inp in inputs
    ])

    # Remove one expert (middle one) and measure deviation
    remove_idx = N // 2
    removed_deltas = []
    for l in range(n_layers):
        sum_dW1 = sum(all_deltas[i][l][0] for i in range(N) if i != remove_idx)
        sum_dW2 = sum(all_deltas[i][l][1] for i in range(N) if i != remove_idx)
        removed_deltas.append((sum_dW1, sum_dW2))

    outputs_removed = np.array([
        forward_with_deltas(inp, model, removed_deltas) for inp in inputs
    ])

    # Compute output deviation
    diff_norms = np.linalg.norm(outputs_all - outputs_removed, axis=1)
    out_norms = np.linalg.norm(outputs_all, axis=1)
    safe_norms = np.maximum(out_norms, 1e-12)
    relative_devs = diff_norms / safe_norms * 100.0

    # Compute per-layer weight-space error (norm of removed expert's delta)
    per_layer_errors = []
    for l in range(n_layers):
        err1 = np.linalg.norm(all_deltas[remove_idx][l][0])
        err2 = np.linalg.norm(all_deltas[remove_idx][l][1])
        # Relative to composed
        comp1_norm = np.linalg.norm(composed_deltas[l][0])
        comp2_norm = np.linalg.norm(composed_deltas[l][1])
        rel_err = (err1 + err2) / max(comp1_norm + comp2_norm, 1e-12) * 100.0
        per_layer_errors.append(rel_err)

    sum_layer_err = sum(per_layer_errors)
    mean_output_dev = float(np.mean(relative_devs))

    if sum_layer_err > 1e-12:
        amp_ratio = mean_output_dev / sum_layer_err
    else:
        amp_ratio = 0.0

    return {
        'mean_output_dev_pct': mean_output_dev,
        'max_output_dev_pct': float(np.max(relative_devs)),
        'sum_layer_err_pct': sum_layer_err,
        'amplification_ratio': amp_ratio,
        'remove_idx': remove_idx,
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment():
    t_start = time.time()

    print("=" * 78)
    print("  EXPERIMENT: B-Matrix Training Correlation")
    print("  K1: trained |cos| > 3x random |cos| (structured correlation)")
    print("  K2: B-matrix correlation does NOT increase amp ratio (safety)")
    print("=" * 78)

    d = 64
    r = LORA_RANK
    N = len(ALL_DOMAINS)  # 8 experts
    nl, d_ff_mult, steps, lr, n_seq, bs = D_CONFIG[d]
    d_ff = d_ff_mult * d

    print(f"\n  Config: d={d}, r={r}, N={N}, layers={nl}, d_ff={d_ff}")
    print(f"  Training: steps={steps}, lr={lr}, n_seq={n_seq}, bs={bs}")
    print(f"  Domain pairs: {DOMAIN_PAIRS}")
    print(f"  All domains: {ALL_DOMAINS}")
    print(f"  Seeds: {SEEDS}")

    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*78}")
        print(f"  SEED {seed}")
        print(f"{'='*78}")

        # ================================================================
        # Phase 1: Build AP skeleton and train experts
        # ================================================================
        rng_ap = np.random.RandomState(seed)
        frames, _ = alternating_projection(N, r, d, n_iter=500, mu_factor=1.2, rng=rng_ap)

        rng_model = np.random.RandomState(seed + d)
        model = MicroMLP(d, nl, d_ff_mult, rng_model)

        # Train experts on each domain with AP-initialized A matrices
        trained_experts = {}  # domain_id -> {'A1', 'B1', 'A2', 'B2', 'loss'}
        for idx, domain_id in enumerate(ALL_DOMAINS):
            x, y = generate_domain_data(domain_id, n_seq)
            A1, B1, A2, B2 = init_lora_from_frame(frames[idx % N], d, d_ff, nl)
            A1, B1, A2, B2, loss = train_lora(model, x, y, A1, B1, A2, B2, steps, lr, bs)
            trained_experts[domain_id] = {
                'A1': A1, 'B1': B1, 'A2': A2, 'B2': B2,
                'loss': float(loss), 'frame_idx': idx % N,
            }
            print(f"    Domain {domain_id}: loss={loss:.4f}")

        # Also train random-init experts (same domains, random orthonormal A)
        random_experts = {}
        for idx, domain_id in enumerate(ALL_DOMAINS):
            x, y = generate_domain_data(domain_id, n_seq)
            rng_lora = np.random.RandomState(seed + d + idx * 31 + 5000)
            A1, B1, A2, B2 = init_lora_random_orthonormal(d, d_ff, nl, rng_lora)
            A1, B1, A2, B2, loss = train_lora(model, x, y, A1, B1, A2, B2, steps, lr, bs)
            random_experts[domain_id] = {
                'A1': A1, 'B1': B1, 'A2': A2, 'B2': B2,
                'loss': float(loss),
            }

        # ================================================================
        # Phase 2: Extract B-matrices and compute pairwise cosines
        # ================================================================

        # Extract B-matrix vectors for AP-trained experts
        ap_b_vecs = {}
        for domain_id in ALL_DOMAINS:
            exp = trained_experts[domain_id]
            ap_b_vecs[domain_id] = extract_b_matrices(exp['B1'], exp['B2'])

        # Extract B-matrix vectors for random-init experts
        rand_b_vecs = {}
        for domain_id in ALL_DOMAINS:
            exp = random_experts[domain_id]
            rand_b_vecs[domain_id] = extract_b_matrices(exp['B1'], exp['B2'])

        # Random B-matrices (untrained, for baseline)
        rng_rand_b = np.random.RandomState(seed + 7777)
        random_b_baseline = {}
        for domain_id in ALL_DOMAINS:
            # Same shape as trained B, but random
            dim = ap_b_vecs[domain_id].shape[0]
            random_b_baseline[domain_id] = rng_rand_b.randn(dim).astype(DTYPE)

        # Also extract full delta vectors (A@B) for comparison
        ap_delta_vecs = {}
        rand_delta_vecs = {}
        for domain_id in ALL_DOMAINS:
            exp = trained_experts[domain_id]
            ap_delta_vecs[domain_id] = lora_delta_vec(exp['A1'], exp['B1'], exp['A2'], exp['B2'])
            exp_r = random_experts[domain_id]
            rand_delta_vecs[domain_id] = lora_delta_vec(exp_r['A1'], exp_r['B1'], exp_r['A2'], exp_r['B2'])

        # ================================================================
        # Phase 3: Measure pairwise B-matrix cosines
        # ================================================================
        print(f"\n  --- B-matrix pairwise cosines ---")

        # Compute ALL pairwise cosines
        ap_b_cosines = {}
        rand_b_cosines = {}
        baseline_b_cosines = {}
        ap_delta_cosines = {}

        domains = sorted(ALL_DOMAINS)
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                d1, d2 = domains[i], domains[j]
                key = (d1, d2)

                ap_cos = abs(cosine_sim(ap_b_vecs[d1], ap_b_vecs[d2]))
                rand_cos = abs(cosine_sim(rand_b_vecs[d1], rand_b_vecs[d2]))
                base_cos = abs(cosine_sim(random_b_baseline[d1], random_b_baseline[d2]))
                delta_cos = abs(cosine_sim(ap_delta_vecs[d1], ap_delta_vecs[d2]))

                ap_b_cosines[key] = ap_cos
                rand_b_cosines[key] = rand_cos
                baseline_b_cosines[key] = base_cos
                ap_delta_cosines[key] = delta_cos

        # Classify pairs by similarity
        similar_pairs = [(0, 1), (2, 3)]
        dissimilar_pairs = [(0, 50), (2, 100)]

        # All pairs
        all_ap_cos = list(ap_b_cosines.values())
        all_rand_cos = list(rand_b_cosines.values())
        all_base_cos = list(baseline_b_cosines.values())
        all_delta_cos = list(ap_delta_cosines.values())

        # Similar-domain B-matrix cosines
        similar_ap_cos = [ap_b_cosines.get(p, 0) for p in similar_pairs
                          if p in ap_b_cosines]
        similar_rand_cos = [rand_b_cosines.get(p, 0) for p in similar_pairs
                            if p in rand_b_cosines]

        # Dissimilar-domain B-matrix cosines
        dissimilar_ap_cos = [ap_b_cosines.get(p, 0) for p in dissimilar_pairs
                             if p in ap_b_cosines]
        dissimilar_rand_cos = [rand_b_cosines.get(p, 0) for p in dissimilar_pairs
                               if p in rand_b_cosines]

        # Per-layer analysis
        ap_per_layer_cos = {l: [] for l in range(nl)}
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                d1, d2 = domains[i], domains[j]
                exp1 = trained_experts[d1]
                exp2 = trained_experts[d2]
                b_layers_1 = extract_b_per_layer(exp1['B1'], exp1['B2'])
                b_layers_2 = extract_b_per_layer(exp2['B1'], exp2['B2'])
                for l in range(nl):
                    cos_l = abs(cosine_sim(b_layers_1[l], b_layers_2[l]))
                    ap_per_layer_cos[l].append(cos_l)

        print(f"\n  All pairwise B-matrix |cos|:")
        print(f"    AP-trained B:      mean={np.mean(all_ap_cos):.6f}, "
              f"max={np.max(all_ap_cos):.6f}, std={np.std(all_ap_cos):.6f}")
        print(f"    Rand-trained B:    mean={np.mean(all_rand_cos):.6f}, "
              f"max={np.max(all_rand_cos):.6f}")
        print(f"    Random baseline:   mean={np.mean(all_base_cos):.6f}, "
              f"max={np.max(all_base_cos):.6f}")
        print(f"    AP delta (A@B):    mean={np.mean(all_delta_cos):.6f}, "
              f"max={np.max(all_delta_cos):.6f}")

        ratio_ap_vs_baseline = np.mean(all_ap_cos) / max(np.mean(all_base_cos), 1e-12)
        ratio_ap_vs_rand_trained = np.mean(all_ap_cos) / max(np.mean(all_rand_cos), 1e-12)

        print(f"\n  Ratios:")
        print(f"    AP-trained / random-baseline:  {ratio_ap_vs_baseline:.2f}x")
        print(f"    AP-trained / rand-trained:     {ratio_ap_vs_rand_trained:.2f}x")

        if similar_ap_cos and dissimilar_ap_cos:
            print(f"\n  Domain similarity effect on B-matrix cosine:")
            print(f"    Similar domains:    mean={np.mean(similar_ap_cos):.6f}")
            print(f"    Dissimilar domains: mean={np.mean(dissimilar_ap_cos):.6f}")
            sim_ratio = np.mean(similar_ap_cos) / max(np.mean(dissimilar_ap_cos), 1e-12)
            print(f"    Similar/dissimilar: {sim_ratio:.2f}x")

        print(f"\n  Per-layer B-matrix cosine (AP-trained):")
        for l in range(nl):
            vals = ap_per_layer_cos[l]
            print(f"    Layer {l}: mean={np.mean(vals):.6f}, max={np.max(vals):.6f}")

        # Print detailed pair table
        print(f"\n  Detailed pair table:")
        print(f"  {'Pair':>12} {'Type':>12} {'B-cos (AP)':>12} {'B-cos (Rand)':>13} "
              f"{'Delta-cos':>11} {'B/Baseline':>11}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*13} {'-'*11} {'-'*11}")
        for (d1, d2), cos_val in sorted(ap_b_cosines.items()):
            pair_type = "similar" if (d1, d2) in similar_pairs else \
                        "dissimilar" if (d1, d2) in dissimilar_pairs else "cross"
            rc = rand_b_cosines.get((d1, d2), 0)
            dc = ap_delta_cosines.get((d1, d2), 0)
            bc = baseline_b_cosines.get((d1, d2), 0)
            ratio = cos_val / max(bc, 1e-12)
            print(f"  ({d1:>3},{d2:>3}) {pair_type:>12} {cos_val:>12.6f} "
                  f"{rc:>13.6f} {dc:>11.6f} {ratio:>11.2f}x")

        # ================================================================
        # Phase 4: Amplification ratio test
        # ================================================================
        print(f"\n  --- Amplification ratio test ---")

        # Test with AP-trained experts (real B-matrix correlation)
        exp_list_ap = [trained_experts[did] for did in ALL_DOMAINS]
        amp_ap = compute_amplification_with_real_b(
            model, exp_list_ap, d, d_ff, nl, n_inputs=200, seed=seed
        )

        # Test with random-trained experts (different A init, different B correlation)
        exp_list_rand = [random_experts[did] for did in ALL_DOMAINS]
        amp_rand = compute_amplification_with_real_b(
            model, exp_list_rand, d, d_ff, nl, n_inputs=200, seed=seed
        )

        # Test with shuffled B-matrices (break correlation, keep magnitudes)
        shuffled_experts = []
        rng_shuffle = np.random.RandomState(seed + 3333)
        perm = rng_shuffle.permutation(N)
        for idx, domain_id in enumerate(ALL_DOMAINS):
            # Take A from this expert, B from a randomly shuffled expert
            src_domain = ALL_DOMAINS[perm[idx]]
            shuffled_experts.append({
                'A1': trained_experts[domain_id]['A1'],
                'B1': trained_experts[src_domain]['B1'],
                'A2': trained_experts[domain_id]['A2'],
                'B2': trained_experts[src_domain]['B2'],
            })
        amp_shuffled = compute_amplification_with_real_b(
            model, shuffled_experts, d, d_ff, nl, n_inputs=200, seed=seed
        )

        print(f"    AP-trained:     amp_ratio={amp_ap['amplification_ratio']:.4f}, "
              f"output_dev={amp_ap['mean_output_dev_pct']:.4f}%")
        print(f"    Rand-trained:   amp_ratio={amp_rand['amplification_ratio']:.4f}, "
              f"output_dev={amp_rand['mean_output_dev_pct']:.4f}%")
        print(f"    Shuffled-B:     amp_ratio={amp_shuffled['amplification_ratio']:.4f}, "
              f"output_dev={amp_shuffled['mean_output_dev_pct']:.4f}%")

        amp_ratio_comparison = amp_ap['amplification_ratio'] / max(amp_shuffled['amplification_ratio'], 1e-12)
        print(f"    AP/shuffled amp ratio: {amp_ratio_comparison:.2f}x")

        # Store results
        seed_result = {
            'seed': seed,
            'config': {
                'd': d, 'r': r, 'N': N, 'n_layers': nl, 'd_ff': d_ff,
                'steps': steps, 'lr': lr,
            },
            'b_matrix_cosines': {
                'ap_trained': {
                    'mean': float(np.mean(all_ap_cos)),
                    'max': float(np.max(all_ap_cos)),
                    'std': float(np.std(all_ap_cos)),
                    'values': {str(k): float(v) for k, v in ap_b_cosines.items()},
                },
                'rand_trained': {
                    'mean': float(np.mean(all_rand_cos)),
                    'max': float(np.max(all_rand_cos)),
                    'std': float(np.std(all_rand_cos)),
                },
                'random_baseline': {
                    'mean': float(np.mean(all_base_cos)),
                    'max': float(np.max(all_base_cos)),
                    'std': float(np.std(all_base_cos)),
                },
                'delta_vectors': {
                    'mean': float(np.mean(all_delta_cos)),
                    'max': float(np.max(all_delta_cos)),
                    'std': float(np.std(all_delta_cos)),
                },
            },
            'ratios': {
                'ap_vs_baseline': ratio_ap_vs_baseline,
                'ap_vs_rand_trained': ratio_ap_vs_rand_trained,
                'similar_vs_dissimilar': float(sim_ratio) if similar_ap_cos and dissimilar_ap_cos else None,
            },
            'domain_effect': {
                'similar_mean': float(np.mean(similar_ap_cos)) if similar_ap_cos else None,
                'dissimilar_mean': float(np.mean(dissimilar_ap_cos)) if dissimilar_ap_cos else None,
            },
            'per_layer_cos': {
                str(l): {
                    'mean': float(np.mean(ap_per_layer_cos[l])),
                    'max': float(np.max(ap_per_layer_cos[l])),
                }
                for l in range(nl)
            },
            'amplification': {
                'ap_trained': amp_ap,
                'rand_trained': amp_rand,
                'shuffled_b': amp_shuffled,
                'ap_vs_shuffled_ratio': amp_ratio_comparison,
            },
            'losses': {
                domain_id: trained_experts[domain_id]['loss']
                for domain_id in ALL_DOMAINS
            },
        }
        all_results.append(seed_result)

    # ================================================================
    # AGGREGATE ANALYSIS
    # ================================================================
    print(f"\n{'='*78}")
    print(f"  AGGREGATE ANALYSIS ({len(SEEDS)} seeds)")
    print(f"{'='*78}")

    # Aggregate B-matrix cosines
    agg_ap_mean = np.mean([r['b_matrix_cosines']['ap_trained']['mean'] for r in all_results])
    agg_ap_max = np.max([r['b_matrix_cosines']['ap_trained']['max'] for r in all_results])
    agg_rand_mean = np.mean([r['b_matrix_cosines']['rand_trained']['mean'] for r in all_results])
    agg_base_mean = np.mean([r['b_matrix_cosines']['random_baseline']['mean'] for r in all_results])
    agg_delta_mean = np.mean([r['b_matrix_cosines']['delta_vectors']['mean'] for r in all_results])

    agg_ratio_vs_baseline = agg_ap_mean / max(agg_base_mean, 1e-12)
    agg_ratio_vs_rand = agg_ap_mean / max(agg_rand_mean, 1e-12)

    print(f"\n  B-matrix |cos| (aggregated across {len(SEEDS)} seeds):")
    print(f"    AP-trained B:      mean={agg_ap_mean:.6f}, max={agg_ap_max:.6f}")
    print(f"    Rand-trained B:    mean={agg_rand_mean:.6f}")
    print(f"    Random baseline:   mean={agg_base_mean:.6f}")
    print(f"    Delta vectors:     mean={agg_delta_mean:.6f}")
    print(f"\n  Ratios:")
    print(f"    AP-trained / random-baseline: {agg_ratio_vs_baseline:.2f}x")
    print(f"    AP-trained / rand-trained:    {agg_ratio_vs_rand:.2f}x")

    # Domain similarity effect
    sim_means = [r['domain_effect']['similar_mean'] for r in all_results
                 if r['domain_effect']['similar_mean'] is not None]
    dissim_means = [r['domain_effect']['dissimilar_mean'] for r in all_results
                    if r['domain_effect']['dissimilar_mean'] is not None]
    if sim_means and dissim_means:
        agg_sim = np.mean(sim_means)
        agg_dissim = np.mean(dissim_means)
        print(f"\n  Domain similarity effect:")
        print(f"    Similar:    mean |cos| = {agg_sim:.6f}")
        print(f"    Dissimilar: mean |cos| = {agg_dissim:.6f}")
        print(f"    Ratio: {agg_sim / max(agg_dissim, 1e-12):.2f}x")

    # Amplification
    agg_amp_ap = np.mean([r['amplification']['ap_trained']['amplification_ratio'] for r in all_results])
    agg_amp_rand = np.mean([r['amplification']['rand_trained']['amplification_ratio'] for r in all_results])
    agg_amp_shuf = np.mean([r['amplification']['shuffled_b']['amplification_ratio'] for r in all_results])
    agg_amp_ratio = agg_amp_ap / max(agg_amp_shuf, 1e-12)

    print(f"\n  Amplification ratio (aggregated):")
    print(f"    AP-trained:   {agg_amp_ap:.4f}")
    print(f"    Rand-trained: {agg_amp_rand:.4f}")
    print(f"    Shuffled-B:   {agg_amp_shuf:.4f}")
    print(f"    AP/shuffled:  {agg_amp_ratio:.2f}x")

    # Statistical test: AP B-cosines vs random baseline
    all_ap_vals = []
    all_base_vals = []
    for r_result in all_results:
        all_ap_vals.extend(r_result['b_matrix_cosines']['ap_trained']['values'].values())
        # baseline is aggregate, use mean as representative
    # Use per-seed means for paired t-test
    ap_means_per_seed = [r['b_matrix_cosines']['ap_trained']['mean'] for r in all_results]
    base_means_per_seed = [r['b_matrix_cosines']['random_baseline']['mean'] for r in all_results]
    if len(ap_means_per_seed) >= 3:
        t_stat, p_val = stats.ttest_rel(ap_means_per_seed, base_means_per_seed)
        print(f"\n  Paired t-test (AP vs baseline, {len(SEEDS)} seeds):")
        print(f"    t={t_stat:.4f}, p={p_val:.4f} {'*' if p_val < 0.05 else 'n.s.'}")

    # ================================================================
    # KILL CRITERIA ASSESSMENT
    # ================================================================
    print(f"\n{'='*78}")
    print(f"  KILL CRITERIA ASSESSMENT")
    print(f"{'='*78}")

    # K1: trained B-matrices show >3x higher pairwise |cos| than random initialization
    print(f"\n  K1: Trained B-matrix |cos| > 3x random baseline?")
    print(f"    AP-trained mean: {agg_ap_mean:.6f}")
    print(f"    Random baseline mean: {agg_base_mean:.6f}")
    print(f"    Ratio: {agg_ratio_vs_baseline:.2f}x (threshold: 3.0x)")

    k1_pass = agg_ratio_vs_baseline > 3.0
    if k1_pass:
        print(f"    K1 PASS: Structured B-matrix correlation EXISTS ({agg_ratio_vs_baseline:.1f}x > 3x)")
    else:
        print(f"    K1 FAIL: B-matrix correlation below threshold ({agg_ratio_vs_baseline:.1f}x < 3x)")

    # Also check rand-trained vs baseline (is it training or init?)
    rand_ratio = agg_rand_mean / max(agg_base_mean, 1e-12)
    print(f"\n    Auxiliary: Rand-trained / baseline: {rand_ratio:.2f}x")
    print(f"    (If >3x, training itself creates correlation regardless of A-init)")

    # K2: B-matrix correlation does NOT increase amplification ratio
    print(f"\n  K2: B-matrix correlation does NOT increase amp ratio?")
    print(f"    AP-trained amp_ratio:  {agg_amp_ap:.4f}")
    print(f"    Shuffled-B amp_ratio:  {agg_amp_shuf:.4f}")
    print(f"    AP/shuffled: {agg_amp_ratio:.2f}x")

    k2_pass = agg_amp_ap <= agg_amp_shuf * 1.5  # within 50% margin
    if k2_pass:
        print(f"    K2 PASS: Correlation does NOT increase amplification (ratio {agg_amp_ratio:.2f}x <= 1.5x)")
        if agg_amp_ap < agg_amp_shuf:
            print(f"    (Correlation actually REDUCES amplification, consistent with parent finding)")
    else:
        print(f"    K2 FAIL: Correlation increases amplification by {agg_amp_ratio:.2f}x (> 1.5x)")

    # Also verify all amp_ratios < 1.0 (absolute safety)
    all_amp_ratios = [r['amplification']['ap_trained']['amplification_ratio'] for r in all_results]
    max_amp = max(all_amp_ratios)
    print(f"\n    Absolute safety: max amp_ratio across all seeds = {max_amp:.4f}")
    print(f"    {'SAFE (< 1.0)' if max_amp < 1.0 else 'UNSAFE (>= 1.0)'}")

    # Overall verdict
    print(f"\n{'='*78}")
    print(f"  OVERALL VERDICT")
    print(f"{'='*78}")

    if k1_pass and k2_pass:
        print(f"  K1 PASS + K2 PASS: B-matrix correlation EXISTS but is SAFE")
        print(f"  Training creates structured B-matrix overlap ({agg_ratio_vs_baseline:.1f}x above random)")
        print(f"  but this correlation does NOT increase amplification ratio.")
        print(f"  Consistent with parent finding: correlation reduces error via rank-1 compressibility.")
        verdict = "proven"
    elif not k1_pass and k2_pass:
        print(f"  K1 FAIL + K2 PASS: No significant B-matrix correlation exists")
        print(f"  B-matrices trained with frozen A do not develop structured overlap.")
        print(f"  The system is safe by default.")
        verdict = "killed_k1"
    elif k1_pass and not k2_pass:
        print(f"  K1 PASS + K2 FAIL: B-matrix correlation EXISTS and is DANGEROUS")
        print(f"  Need B-matrix regularization during training.")
        verdict = "revise"
    else:
        print(f"  K1 FAIL + K2 FAIL: Unexpected result combination.")
        verdict = "revise"
    print(f"{'='*78}")

    # ================================================================
    # Save results
    # ================================================================
    elapsed = time.time() - t_start

    output = {
        'config': {
            'd': d, 'r': r, 'N': N, 'n_layers': nl, 'd_ff': d_ff,
            'steps': steps, 'lr': lr, 'seeds': SEEDS,
            'domain_pairs': {k: list(v) for k, v in DOMAIN_PAIRS.items()},
            'all_domains': ALL_DOMAINS,
        },
        'per_seed': all_results,
        'aggregate': {
            'b_matrix_cos': {
                'ap_trained_mean': agg_ap_mean,
                'ap_trained_max': agg_ap_max,
                'rand_trained_mean': agg_rand_mean,
                'random_baseline_mean': agg_base_mean,
                'delta_vector_mean': agg_delta_mean,
                'ratio_ap_vs_baseline': agg_ratio_vs_baseline,
                'ratio_ap_vs_rand_trained': agg_ratio_vs_rand,
            },
            'amplification': {
                'ap_trained': agg_amp_ap,
                'rand_trained': agg_amp_rand,
                'shuffled_b': agg_amp_shuf,
                'ap_vs_shuffled': agg_amp_ratio,
            },
        },
        'kill_criteria': {
            'k1_structured_correlation_gt_3x': k1_pass,
            'k1_ratio': agg_ratio_vs_baseline,
            'k2_safe_amplification': k2_pass,
            'k2_ratio': agg_amp_ratio,
            'overall_verdict': verdict,
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
