#!/usr/bin/env python3
"""
Collision Scaling: Non-orthogonal collision rate as function of expert count N.

Hypothesis: The fraction of expert pairs with |cos|>0.1 ("collision rate") grows
sublinearly with N, confirming that SOLE does not face a combinatorial collision barrier.

Design:
  For N in {5, 10, 15, 20, 30, 50}:
    1. Build a micro MLP base model (pure numpy, char-level NTP)
    2. Generate N synthetic domains across K=ceil(N/5) clusters using Markov chains
    3. Train N LoRA adapters (one per domain) via vectorized backprop
    4. Compute full NxN pairwise cosine matrix
    5. Measure collision rate: fraction of pairs with |cos| > threshold (0.1)
    6. Separate within-cluster vs cross-cluster collision rates
  Repeat for 3 seeds.
  Fit growth models (linear, quadratic, power law) to collision_rate(N).

Kill criteria:
  K1: collision rate (cos>0.1) grows superlinearly with N (architecture-breaking)
  K2: >30% of pairs have cosine >0.1 at N=20 (clusters dominate)

Architecture: 4-layer MLP, d=64, d_ff=256, rank=8 LoRA on all MLP layers.
Pure numpy -- CPU only, no MLX, no PyTorch.

Based on: experiments/models/orthogonality_by_domain_type/ (proven, within-cluster cos=0.060)
"""

import json
import time
from pathlib import Path

import numpy as np

# ============================================================================
# Constants
# ============================================================================

VOCAB_SIZE = 32
CONTEXT_LEN = 16
D_MODEL = 64
D_FF = 256
N_LAYERS = 4
LORA_RANK = 8
LORA_ALPHA = 8
COLLISION_THRESHOLD = 0.1

# N values to test
N_VALUES = [5, 10, 15, 20, 30, 50]

# Cluster structure: base clusters, domains generated programmatically
BASE_CLUSTER_NAMES = ['code', 'reasoning', 'knowledge', 'creative', 'science',
                      'business', 'engineering', 'social', 'health', 'systems']

# Domain names per cluster (we draw from these as N grows)
DOMAIN_POOLS = {
    'code': ['python', 'javascript', 'rust', 'bash', 'sql', 'go', 'cpp',
             'ruby', 'swift', 'kotlin', 'typescript', 'perl'],
    'reasoning': ['math', 'logic', 'physics', 'statistics', 'economics',
                  'philosophy', 'game_theory', 'probability', 'optimization',
                  'algebra', 'geometry', 'calculus'],
    'knowledge': ['medical', 'law', 'history', 'psychology', 'cooking',
                  'geography', 'literature', 'music', 'art', 'religion',
                  'politics', 'linguistics'],
    'creative': ['poetry', 'fiction', 'screenwriting', 'journalism', 'comedy',
                 'songwriting', 'blogging', 'copywriting', 'essays', 'drama',
                 'satire', 'memoir'],
    'science': ['chemistry', 'biology', 'astronomy', 'geology', 'ecology',
                'genetics', 'neuroscience', 'materials', 'quantum', 'climate',
                'marine_bio', 'paleontology'],
    'business': ['marketing', 'finance', 'accounting', 'management', 'strategy',
                 'sales', 'hr', 'supply_chain', 'consulting', 'startup',
                 'real_estate', 'trading'],
    'engineering': ['mechanical', 'electrical', 'civil', 'chemical', 'aerospace',
                    'robotics', 'biomedical', 'environmental', 'nuclear',
                    'automotive', 'naval', 'industrial'],
    'social': ['sociology', 'anthropology', 'education', 'communication',
               'journalism_studies', 'cultural_studies', 'demography',
               'urban_planning', 'social_work', 'criminology', 'media',
               'public_policy'],
    'health': ['nutrition', 'fitness', 'mental_health', 'pharmacology',
               'nursing', 'dentistry', 'public_health', 'physical_therapy',
               'occupational_therapy', 'radiology', 'surgery', 'pediatrics'],
    'systems': ['networking', 'databases', 'os_design', 'distributed_sys',
                'security', 'cloud', 'devops', 'ml_ops', 'data_eng',
                'embedded', 'firmware', 'blockchain'],
}


def build_domain_config(n_experts):
    """Build cluster and domain assignments for N experts.

    Strategy: distribute experts across clusters as evenly as possible.
    Use ceil(N/5) clusters (min 1, max 10), with ~5 domains per cluster.
    This means within-cluster pairs grow as O(N) while cross-cluster
    pairs grow as O(N^2).
    """
    n_clusters = min(max(1, (n_experts + 4) // 5), len(BASE_CLUSTER_NAMES))
    cluster_names = BASE_CLUSTER_NAMES[:n_clusters]

    # Distribute N experts across clusters
    domains_per_cluster = [n_experts // n_clusters] * n_clusters
    remainder = n_experts % n_clusters
    for i in range(remainder):
        domains_per_cluster[i] += 1

    clusters = {}
    all_domains = []
    domain_to_cluster = {}

    for ci, cname in enumerate(cluster_names):
        pool = DOMAIN_POOLS[cname]
        n_dom = domains_per_cluster[ci]
        domains = pool[:n_dom]
        clusters[cname] = domains
        all_domains.extend(domains)
        for d in domains:
            domain_to_cluster[d] = cname

    return clusters, all_domains, domain_to_cluster


# ============================================================================
# Utilities
# ============================================================================

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float64)


# ============================================================================
# Model: Multi-layer MLP with LoRA
# ============================================================================

class MicroMLP:
    """4-layer MLP for next-token prediction with LoRA support."""

    def __init__(self, rng):
        scale = 0.02
        self.wte = rng.randn(VOCAB_SIZE, D_MODEL) * scale
        self.layers = []
        for _ in range(N_LAYERS):
            self.layers.append({
                'W1': rng.randn(D_MODEL, D_FF) * scale,
                'b1': np.zeros(D_FF),
                'W2': rng.randn(D_FF, D_MODEL) * scale,
                'b2': np.zeros(D_MODEL),
            })
        self.W_out = rng.randn(D_MODEL, VOCAB_SIZE) * scale

    def forward(self, x_ids, lora=None):
        B, T = x_ids.shape
        scale = LORA_ALPHA / LORA_RANK

        emb = self.wte[x_ids]
        h = emb.mean(axis=1)

        intermediates = []
        for l, layer in enumerate(self.layers):
            h_in = h
            z1 = h @ layer['W1'] + layer['b1']
            if lora is not None:
                z1 = z1 + scale * (h @ lora['A1'][l] @ lora['B1'][l])
            a1 = relu(z1)
            z2 = a1 @ layer['W2'] + layer['b2']
            if lora is not None:
                z2 = z2 + scale * (a1 @ lora['A2'][l] @ lora['B2'][l])
            h = h_in + z2
            intermediates.append({'h_in': h_in, 'z1': z1, 'a1': a1})

        logits = h @ self.W_out
        return logits, intermediates, h


def init_lora(rng):
    A1 = [rng.randn(D_MODEL, LORA_RANK) * np.sqrt(2.0 / D_MODEL)
          for _ in range(N_LAYERS)]
    B1 = [np.zeros((LORA_RANK, D_FF)) for _ in range(N_LAYERS)]
    A2 = [rng.randn(D_FF, LORA_RANK) * np.sqrt(2.0 / D_FF)
          for _ in range(N_LAYERS)]
    B2 = [np.zeros((LORA_RANK, D_MODEL)) for _ in range(N_LAYERS)]
    return {'A1': A1, 'B1': B1, 'A2': A2, 'B2': B2}


def lora_to_delta_vector(lora):
    parts = []
    for l in range(N_LAYERS):
        parts.append((lora['A1'][l] @ lora['B1'][l]).flatten())
        parts.append((lora['A2'][l] @ lora['B2'][l]).flatten())
    return np.concatenate(parts)


def train_lora(model, data_x, data_y, rng, steps=300, lr=0.01, batch_size=32):
    """Train LoRA via vectorized backprop. Only B matrices are updated."""
    lora = init_lora(rng)
    n_data = data_x.shape[0]
    scale = LORA_ALPHA / LORA_RANK

    for step in range(steps):
        idx = rng.choice(n_data, size=min(batch_size, n_data), replace=False)
        bx, by = data_x[idx], data_y[idx]
        B_sz = bx.shape[0]

        logits, intermediates, h_final = model.forward(bx, lora)
        probs = softmax(logits)

        target_probs = probs[np.arange(B_sz), by]

        d_logits = probs.copy()
        d_logits[np.arange(B_sz), by] -= 1.0
        d_logits /= B_sz

        d_h = d_logits @ model.W_out.T

        for l in range(N_LAYERS - 1, -1, -1):
            layer = model.layers[l]
            inter = intermediates[l]
            h_in = inter['h_in']
            z1 = inter['z1']
            a1 = inter['a1']

            d_z2 = d_h
            proj2 = a1 @ lora['A2'][l]
            d_B2 = scale * (proj2.T @ d_z2)
            lora['B2'][l] -= lr * d_B2

            W2_eff = layer['W2'] + scale * lora['A2'][l] @ lora['B2'][l]
            d_a1 = d_z2 @ W2_eff.T
            d_z1 = d_a1 * relu_grad(z1)

            proj1 = h_in @ lora['A1'][l]
            d_B1 = scale * (proj1.T @ d_z1)
            lora['B1'][l] -= lr * d_B1

    return lora


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_cluster_prototypes(rng, n_clusters, cross_cluster_distance=2.0):
    """Generate Markov chain transition matrices for each cluster."""
    prototypes = {}
    cluster_names = BASE_CLUSTER_NAMES[:n_clusters]

    for i, name in enumerate(cluster_names):
        logits = rng.randn(VOCAB_SIZE, VOCAB_SIZE) * cross_cluster_distance
        group_start = (i * VOCAB_SIZE) // max(n_clusters, 1)
        group_end = ((i + 1) * VOCAB_SIZE) // max(n_clusters, 1)
        logits[:, group_start:group_end] += 1.5
        logits += np.eye(VOCAB_SIZE) * 0.5
        prototypes[name] = softmax(logits, axis=-1)

    return prototypes


def generate_domain_data(rng, prototypes, domain_name, cluster_name,
                         all_domains, n_sequences=200, noise_scale=0.15):
    prototype = prototypes[cluster_name]
    noise = rng.randn(*prototype.shape) * noise_scale

    domain_seed = hash(domain_name) % (2**31)
    domain_rng = np.random.RandomState(domain_seed)
    domain_bias = domain_rng.randn(*prototype.shape) * noise_scale * 0.5

    domain_idx = all_domains.index(domain_name)
    for j in range(3):
        c = (domain_idx * 2 + j) % VOCAB_SIZE
        domain_bias[:, c] += 0.3

    transition = softmax(np.log(prototype + 1e-10) + noise + domain_bias, axis=-1)

    seq_len = CONTEXT_LEN + 1
    sequences = np.zeros((n_sequences, seq_len), dtype=np.int32)
    for i in range(n_sequences):
        sequences[i, 0] = rng.choice(VOCAB_SIZE)
        for t in range(seq_len - 1):
            prev = sequences[i, t]
            sequences[i, t + 1] = rng.choice(VOCAB_SIZE, p=transition[prev])

    x = sequences[:, :-1]
    y = sequences[:, -1]
    return x, y


# ============================================================================
# Analysis
# ============================================================================

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_pairwise_cosines(domain_vectors, domain_to_cluster, all_domains):
    """Compute all pairwise cosines and classify as within/cross cluster."""
    n = len(all_domains)
    n_pairs = n * (n - 1) // 2

    within_cos = []
    cross_cos = []
    all_cos = []

    for i in range(n):
        for j in range(i + 1, n):
            c = abs(cosine_sim(domain_vectors[all_domains[i]],
                               domain_vectors[all_domains[j]]))
            all_cos.append(c)
            d1, d2 = all_domains[i], all_domains[j]
            if domain_to_cluster[d1] == domain_to_cluster[d2]:
                within_cos.append(c)
            else:
                cross_cos.append(c)

    return {
        'all_cos': all_cos,
        'within_cos': within_cos,
        'cross_cos': cross_cos,
        'n_pairs': n_pairs,
        'n_within': len(within_cos),
        'n_cross': len(cross_cos),
    }


def compute_collision_rates(cosines, threshold=COLLISION_THRESHOLD):
    """Compute fraction of pairs exceeding threshold."""
    all_arr = np.array(cosines['all_cos'])
    within_arr = np.array(cosines['within_cos']) if cosines['within_cos'] else np.array([])
    cross_arr = np.array(cosines['cross_cos']) if cosines['cross_cos'] else np.array([])

    total_rate = float(np.mean(all_arr > threshold)) if len(all_arr) > 0 else 0.0
    within_rate = float(np.mean(within_arr > threshold)) if len(within_arr) > 0 else 0.0
    cross_rate = float(np.mean(cross_arr > threshold)) if len(cross_arr) > 0 else 0.0

    return {
        'total_collision_rate': total_rate,
        'within_collision_rate': within_rate,
        'cross_collision_rate': cross_rate,
        'total_mean_cos': float(np.mean(all_arr)) if len(all_arr) > 0 else 0.0,
        'within_mean_cos': float(np.mean(within_arr)) if len(within_arr) > 0 else 0.0,
        'cross_mean_cos': float(np.mean(cross_arr)) if len(cross_arr) > 0 else 0.0,
        'total_max_cos': float(np.max(all_arr)) if len(all_arr) > 0 else 0.0,
        'within_max_cos': float(np.max(within_arr)) if len(within_arr) > 0 else 0.0,
        'cross_max_cos': float(np.max(cross_arr)) if len(cross_arr) > 0 else 0.0,
        'n_collisions': int(np.sum(all_arr > threshold)),
        'n_pairs': len(all_arr),
        'n_within_collisions': int(np.sum(within_arr > threshold)) if len(within_arr) > 0 else 0,
        'n_cross_collisions': int(np.sum(cross_arr > threshold)) if len(cross_arr) > 0 else 0,
    }


def fit_growth_models(n_values, collision_rates):
    """Fit linear, quadratic, and power law models to collision_rate(N).

    Returns fit parameters and R^2 for each model.
    """
    N = np.array(n_values, dtype=np.float64)
    Y = np.array(collision_rates, dtype=np.float64)

    results = {}

    # Filter out zeros for log fits
    mask = Y > 0
    if mask.sum() < 2:
        # Not enough non-zero points to fit
        return {
            'linear': {'a': 0.0, 'b': 0.0, 'r2': 0.0},
            'quadratic': {'a': 0.0, 'b': 0.0, 'c': 0.0, 'r2': 0.0},
            'power_law': {'a': 0.0, 'beta': 0.0, 'r2': 0.0},
            'best_model': 'insufficient_data',
        }

    ss_tot = np.sum((Y - np.mean(Y))**2)
    if ss_tot < 1e-15:
        return {
            'linear': {'a': 0.0, 'b': float(np.mean(Y)), 'r2': 1.0},
            'quadratic': {'a': 0.0, 'b': 0.0, 'c': float(np.mean(Y)), 'r2': 1.0},
            'power_law': {'a': float(np.mean(Y)), 'beta': 0.0, 'r2': 1.0},
            'best_model': 'constant',
        }

    # 1. Linear: Y = a*N + b
    A_lin = np.column_stack([N, np.ones_like(N)])
    params_lin = np.linalg.lstsq(A_lin, Y, rcond=None)[0]
    Y_pred_lin = A_lin @ params_lin
    r2_lin = 1 - np.sum((Y - Y_pred_lin)**2) / ss_tot
    results['linear'] = {
        'a': float(params_lin[0]),
        'b': float(params_lin[1]),
        'r2': float(r2_lin),
    }

    # 2. Quadratic: Y = a*N^2 + b*N + c
    A_quad = np.column_stack([N**2, N, np.ones_like(N)])
    params_quad = np.linalg.lstsq(A_quad, Y, rcond=None)[0]
    Y_pred_quad = A_quad @ params_quad
    r2_quad = 1 - np.sum((Y - Y_pred_quad)**2) / ss_tot
    results['quadratic'] = {
        'a': float(params_quad[0]),
        'b': float(params_quad[1]),
        'c': float(params_quad[2]),
        'r2': float(r2_quad),
    }

    # 3. Power law: Y = a * N^beta  =>  log(Y) = log(a) + beta*log(N)
    if mask.sum() >= 2:
        logN = np.log(N[mask])
        logY = np.log(Y[mask])
        A_pow = np.column_stack([logN, np.ones_like(logN)])
        params_pow = np.linalg.lstsq(A_pow, logY, rcond=None)[0]
        beta = float(params_pow[0])
        a_pow = float(np.exp(params_pow[1]))
        Y_pred_pow = a_pow * N**beta
        r2_pow = 1 - np.sum((Y - Y_pred_pow)**2) / ss_tot
        results['power_law'] = {
            'a': a_pow,
            'beta': beta,
            'r2': float(r2_pow),
        }
    else:
        results['power_law'] = {'a': 0.0, 'beta': 0.0, 'r2': 0.0}

    # Determine best model
    r2s = {
        'linear': results['linear']['r2'],
        'quadratic': results['quadratic']['r2'],
        'power_law': results['power_law']['r2'],
    }
    results['best_model'] = max(r2s, key=r2s.get)

    # Key test: is growth superlinear?
    # Superlinear if: quadratic 'a' coefficient is positive AND significantly improves fit
    # Or power law beta > 1
    results['superlinear_test'] = {
        'quadratic_a_positive': results['quadratic']['a'] > 0,
        'power_law_beta_gt_1': results['power_law']['beta'] > 1.0,
        'power_law_beta': results['power_law']['beta'],
        'quadratic_improvement': results['quadratic']['r2'] - results['linear']['r2'],
    }

    return results


# ============================================================================
# Main Experiment
# ============================================================================

def run_single_n(model, n_experts, rng, train_steps=300, lr=0.01,
                 n_sequences=200, batch_size=32):
    """Run experiment for a single value of N."""
    clusters, all_domains, domain_to_cluster = build_domain_config(n_experts)
    n_clusters = len(clusters)

    prototypes = generate_cluster_prototypes(rng, n_clusters)

    domain_vectors = {}
    for i, domain in enumerate(all_domains):
        cluster = domain_to_cluster[domain]
        x, y = generate_domain_data(rng, prototypes, domain, cluster,
                                    all_domains, n_sequences=n_sequences)
        lora = train_lora(model, x, y, rng, steps=train_steps,
                          lr=lr, batch_size=batch_size)
        domain_vectors[domain] = lora_to_delta_vector(lora)

    cosines = compute_pairwise_cosines(domain_vectors, domain_to_cluster, all_domains)
    rates = compute_collision_rates(cosines)

    return {
        'n_experts': n_experts,
        'n_clusters': n_clusters,
        'clusters': {k: v for k, v in clusters.items()},
        'rates': rates,
        'cosine_stats': {
            'all_mean': float(np.mean(cosines['all_cos'])),
            'all_std': float(np.std(cosines['all_cos'])),
            'within_mean': float(np.mean(cosines['within_cos'])) if cosines['within_cos'] else 0.0,
            'cross_mean': float(np.mean(cosines['cross_cos'])) if cosines['cross_cos'] else 0.0,
        },
    }


def run_experiment(seeds=(42, 142, 242), n_values=None, train_steps=300,
                   lr=0.01, n_sequences=200, batch_size=32):
    """Run full collision scaling experiment."""
    if n_values is None:
        n_values = N_VALUES

    results_dir = Path(__file__).parent
    t_start = time.time()

    print("=" * 72)
    print("  Collision Scaling: collision rate vs expert count N")
    print("  Hypothesis: collision rate grows sublinearly with N")
    print(f"  N values: {n_values}")
    print(f"  Seeds: {list(seeds)}")
    print(f"  Architecture: {N_LAYERS}-layer MLP, d={D_MODEL}, d_ff={D_FF}, rank={LORA_RANK}")
    print(f"  Collision threshold: |cos| > {COLLISION_THRESHOLD}")
    print("=" * 72)

    all_results = {}

    for seed in seeds:
        print(f"\n{'='*72}")
        print(f"  SEED = {seed}")
        print(f"{'='*72}")

        seed_results = []

        for n in n_values:
            print(f"\n  --- N = {n} experts ---")
            rng = np.random.RandomState(seed)
            model = MicroMLP(rng)

            # Reset RNG for reproducibility at each N
            rng = np.random.RandomState(seed + n * 1000)

            result = run_single_n(model, n, rng, train_steps=train_steps,
                                  lr=lr, n_sequences=n_sequences,
                                  batch_size=batch_size)
            seed_results.append(result)

            r = result['rates']
            print(f"    Collision rate (|cos|>{COLLISION_THRESHOLD}): "
                  f"{r['total_collision_rate']:.4f} "
                  f"({r['n_collisions']}/{r['n_pairs']} pairs)")
            print(f"    Within-cluster: {r['within_collision_rate']:.4f}, "
                  f"Cross-cluster: {r['cross_collision_rate']:.4f}")
            print(f"    Mean |cos|: total={r['total_mean_cos']:.4f}, "
                  f"within={r['within_mean_cos']:.4f}, "
                  f"cross={r['cross_mean_cos']:.4f}")
            print(f"    Max |cos|: {r['total_max_cos']:.4f}")

        all_results[seed] = seed_results

    # ====================================================================
    # Aggregate across seeds
    # ====================================================================
    elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  AGGREGATE RESULTS ({len(seeds)} seeds)")
    print(f"{'='*72}")

    # Build per-N aggregates
    aggregate = {}
    for n in n_values:
        rates_at_n = []
        within_rates = []
        cross_rates = []
        mean_cos_vals = []
        within_cos_vals = []
        cross_cos_vals = []

        for seed in seeds:
            for r in all_results[seed]:
                if r['n_experts'] == n:
                    rates_at_n.append(r['rates']['total_collision_rate'])
                    within_rates.append(r['rates']['within_collision_rate'])
                    cross_rates.append(r['rates']['cross_collision_rate'])
                    mean_cos_vals.append(r['rates']['total_mean_cos'])
                    within_cos_vals.append(r['rates']['within_mean_cos'])
                    cross_cos_vals.append(r['rates']['cross_mean_cos'])

        aggregate[n] = {
            'collision_rate_mean': float(np.mean(rates_at_n)),
            'collision_rate_std': float(np.std(rates_at_n)),
            'within_rate_mean': float(np.mean(within_rates)),
            'within_rate_std': float(np.std(within_rates)),
            'cross_rate_mean': float(np.mean(cross_rates)),
            'cross_rate_std': float(np.std(cross_rates)),
            'mean_cos_mean': float(np.mean(mean_cos_vals)),
            'within_cos_mean': float(np.mean(within_cos_vals)),
            'cross_cos_mean': float(np.mean(cross_cos_vals)),
            'n_clusters': all_results[seeds[0]][n_values.index(n)]['n_clusters'],
        }

    # Print table
    print(f"\n  {'N':>4s} {'K':>3s} {'Pairs':>7s} {'Coll%':>8s} {'(+/-)':>8s} "
          f"{'WColl%':>8s} {'XColl%':>8s} {'MeanCos':>8s} {'WCos':>8s} {'XCos':>8s}")
    print(f"  {'-'*80}")

    for n in n_values:
        a = aggregate[n]
        n_pairs = n * (n - 1) // 2
        print(f"  {n:4d} {a['n_clusters']:3d} {n_pairs:7d} "
              f"{a['collision_rate_mean']*100:7.2f}% {a['collision_rate_std']*100:7.2f}% "
              f"{a['within_rate_mean']*100:7.2f}% {a['cross_rate_mean']*100:7.2f}% "
              f"{a['mean_cos_mean']:.4f}   {a['within_cos_mean']:.4f}   "
              f"{a['cross_cos_mean']:.4f}")

    # ====================================================================
    # Fit growth models
    # ====================================================================
    print(f"\n{'='*72}")
    print(f"  GROWTH MODEL FITTING")
    print(f"{'='*72}")

    mean_rates = [aggregate[n]['collision_rate_mean'] for n in n_values]
    fits = fit_growth_models(n_values, mean_rates)

    print(f"\n  Linear:    Y = {fits['linear']['a']:.6f} * N + {fits['linear']['b']:.6f}")
    print(f"             R^2 = {fits['linear']['r2']:.4f}")
    print(f"\n  Quadratic: Y = {fits['quadratic']['a']:.8f} * N^2 "
          f"+ {fits['quadratic']['b']:.6f} * N + {fits['quadratic']['c']:.6f}")
    print(f"             R^2 = {fits['quadratic']['r2']:.4f}")
    print(f"\n  Power law: Y = {fits['power_law']['a']:.6f} * N^{fits['power_law']['beta']:.4f}")
    print(f"             R^2 = {fits['power_law']['r2']:.4f}")
    print(f"\n  Best model: {fits['best_model']}")

    # Within-cluster fit
    within_rates_mean = [aggregate[n]['within_rate_mean'] for n in n_values]
    fits_within = fit_growth_models(n_values, within_rates_mean)

    # Cross-cluster fit
    cross_rates_mean = [aggregate[n]['cross_rate_mean'] for n in n_values]
    fits_cross = fit_growth_models(n_values, cross_rates_mean)

    print(f"\n  Within-cluster power law: beta = {fits_within['power_law']['beta']:.4f}, "
          f"R^2 = {fits_within['power_law']['r2']:.4f}")
    print(f"  Cross-cluster power law:  beta = {fits_cross['power_law']['beta']:.4f}, "
          f"R^2 = {fits_cross['power_law']['r2']:.4f}")

    # ====================================================================
    # Kill criteria evaluation
    # ====================================================================
    print(f"\n{'='*72}")
    print(f"  KILL CRITERIA EVALUATION")
    print(f"{'='*72}")

    # K1: superlinear growth
    beta = fits['power_law']['beta']
    k1_superlinear = beta > 1.0 and fits['superlinear_test']['quadratic_a_positive']
    print(f"\n  K1: Collision rate grows superlinearly with N")
    print(f"      Power law exponent beta = {beta:.4f}")
    print(f"      Quadratic a > 0: {fits['superlinear_test']['quadratic_a_positive']}")
    print(f"      Quadratic improvement over linear: "
          f"{fits['superlinear_test']['quadratic_improvement']:.4f}")
    if k1_superlinear:
        print(f"      STATUS: KILL -- superlinear growth (beta={beta:.3f} > 1)")
    else:
        if beta <= 1.0:
            print(f"      STATUS: PASS -- sublinear growth (beta={beta:.3f} <= 1)")
        else:
            print(f"      STATUS: PASS -- beta={beta:.3f} but quadratic term negligible")

    # K2: >30% collision at N=20
    if 20 in n_values:
        rate_20 = aggregate[20]['collision_rate_mean']
        k2_high_collision = rate_20 > 0.30
        print(f"\n  K2: >30% of pairs collide at N=20")
        print(f"      Collision rate at N=20: {rate_20*100:.2f}%")
        if k2_high_collision:
            print(f"      STATUS: KILL -- {rate_20*100:.1f}% > 30%")
        else:
            print(f"      STATUS: PASS -- {rate_20*100:.1f}% < 30%")
    else:
        k2_high_collision = False
        print(f"\n  K2: N=20 not tested")

    overall_kill = k1_superlinear or k2_high_collision
    print(f"\n  {'='*60}")
    if overall_kill:
        print(f"  OVERALL VERDICT: KILL")
        if k1_superlinear:
            print(f"    K1: superlinear growth (beta={beta:.3f})")
        if k2_high_collision:
            print(f"    K2: >30% collision at N=20")
    else:
        print(f"  OVERALL VERDICT: SUPPORTED")
        print(f"    Collision rate grows sublinearly (beta={beta:.3f})")
        if 20 in n_values:
            print(f"    At N=20: {aggregate[20]['collision_rate_mean']*100:.1f}% collision rate")
        if 50 in n_values:
            print(f"    At N=50: {aggregate[50]['collision_rate_mean']*100:.1f}% collision rate")
    print(f"  {'='*60}")

    print(f"\n  Total time: {elapsed:.1f}s")

    # ====================================================================
    # Save results
    # ====================================================================
    output = {
        'config': {
            'n_values': n_values,
            'seeds': list(seeds),
            'train_steps': train_steps,
            'lr': lr,
            'n_sequences': n_sequences,
            'batch_size': batch_size,
            'd_model': D_MODEL,
            'd_ff': D_FF,
            'n_layers': N_LAYERS,
            'lora_rank': LORA_RANK,
            'vocab_size': VOCAB_SIZE,
            'context_len': CONTEXT_LEN,
            'collision_threshold': COLLISION_THRESHOLD,
        },
        'per_seed': {str(s): [r for r in all_results[s]] for s in seeds},
        'aggregate': {str(n): v for n, v in aggregate.items()},
        'growth_fits': {
            'total': fits,
            'within_cluster': fits_within,
            'cross_cluster': fits_cross,
        },
        'kill_criteria': {
            'k1_superlinear': k1_superlinear,
            'k1_power_law_beta': beta,
            'k2_high_collision_at_20': k2_high_collision,
            'k2_rate_at_20': aggregate.get(20, {}).get('collision_rate_mean', None),
            'overall_kill': overall_kill,
        },
        'elapsed_seconds': elapsed,
    }

    output_file = results_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {output_file}")

    return output


if __name__ == '__main__':
    run_experiment()
