#!/usr/bin/env python3
"""
Subspace Capacity Empirical: Does empirical expert capacity match N_max = d^2/r^2?

Hypothesis: The theoretical bound N_max = d^2/r^2 (from flattened delta-vector
dimensionality D ~ 2*L*d*d_ff) predicts the empirical capacity -- the N at which
expert signal is destroyed by interference during naive additive composition.

Approach: THREE COMPLEMENTARY MEASURES at increasing N:
  1. GEOMETRIC: Average pairwise |cos| of flattened deltas (from collision_scaling)
  2. SIGNAL RETENTION: fraction of each expert's delta surviving in the merged model
     signal_i = ||proj(merged_delta, expert_delta_i)|| / ||expert_delta_i||
  3. QUALITY: NTP loss ratio merged/expert on domain-specific test data

Design:
  For two rank settings (r=8: N_max=64, r=4: N_max=256) at d=64:
    1. Build a base model and train it for 500 steps on combined data
    2. Fine-tune N LoRA experts on domain-specific data (300 steps each)
    3. Merge ALL experts: W_base + sum(alpha/r * A_i @ B_i)
    4. Measure geometric, signal, and quality metrics vs N
    5. Find capacity cliff via signal retention < 50%
    6. 3 seeds for robustness.

Key insight: the SOLE architecture uses naive addition W + sum(B_i @ A_i).
Each expert contributes 1 delta. When N experts are merged, each expert's
contribution is 1/Nth of the total perturbation (in an idealized equal-norm
case). The question: at what N does cross-expert interference corrupt
individual expert signals beyond recovery?

Kill criteria:
  K1: empirical capacity < 10% of theoretical N_max (theory useless)
  K2: quality degrades smoothly (no sharp cliff)

Architecture: 4-layer MLP, d=64, d_ff=256, rank={8,4}, frozen-A LoRA.
Pure numpy -- CPU only.

Based on: experiments/models/collision_scaling/ (parent experiment, beta=-0.575)
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
LORA_ALPHA = 16  # Higher alpha for stronger signal
COLLISION_THRESHOLD = 0.1

# N values for r=8 (N_max=64)
N_VALUES_R8 = [5, 10, 20, 40, 64, 80]
# N values for r=4 (N_max=256)
N_VALUES_R4 = [5, 10, 20, 40, 80, 128]

BASE_CLUSTER_NAMES = ['code', 'reasoning', 'knowledge', 'creative', 'science',
                      'business', 'engineering', 'social', 'health', 'systems']

DOMAIN_POOLS = {
    'code': [f'code_{i}' for i in range(40)],
    'reasoning': [f'reasoning_{i}' for i in range(40)],
    'knowledge': [f'knowledge_{i}' for i in range(40)],
    'creative': [f'creative_{i}' for i in range(40)],
    'science': [f'science_{i}' for i in range(40)],
    'business': [f'business_{i}' for i in range(40)],
    'engineering': [f'engineering_{i}' for i in range(40)],
    'social': [f'social_{i}' for i in range(40)],
    'health': [f'health_{i}' for i in range(40)],
    'systems': [f'systems_{i}' for i in range(40)],
}


def build_domain_config(n_experts):
    n_clusters = min(max(1, (n_experts + 4) // 5), len(BASE_CLUSTER_NAMES))
    cluster_names = BASE_CLUSTER_NAMES[:n_clusters]
    domains_per_cluster = [n_experts // n_clusters] * n_clusters
    remainder = n_experts % n_clusters
    for i in range(remainder):
        domains_per_cluster[i] += 1
    all_domains = []
    domain_to_cluster = {}
    for ci, cname in enumerate(cluster_names):
        pool = DOMAIN_POOLS[cname]
        n_dom = domains_per_cluster[ci]
        domains = pool[:n_dom]
        all_domains.extend(domains)
        for d in domains:
            domain_to_cluster[d] = cname
    return all_domains, domain_to_cluster


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
# Model
# ============================================================================

class MicroMLP:
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
        emb = self.wte[x_ids]
        h = emb.mean(axis=1)
        for l, layer in enumerate(self.layers):
            h_in = h
            z1 = h @ layer['W1'] + layer['b1']
            if lora is not None:
                scale = LORA_ALPHA / lora['rank']
                z1 = z1 + scale * (h @ lora['A1'][l] @ lora['B1'][l])
            a1 = relu(z1)
            z2 = a1 @ layer['W2'] + layer['b2']
            if lora is not None:
                scale = LORA_ALPHA / lora['rank']
                z2 = z2 + scale * (a1 @ lora['A2'][l] @ lora['B2'][l])
            h = h_in + z2
        logits = h @ self.W_out
        return logits

    def compute_loss(self, x_ids, y_ids, lora=None):
        logits = self.forward(x_ids, lora)
        B = x_ids.shape[0]
        probs = softmax(logits)
        target_probs = probs[np.arange(B), y_ids]
        return -np.mean(np.log(target_probs + 1e-10))

    def forward_merged(self, x_ids, merged_weights):
        B, T = x_ids.shape
        emb = self.wte[x_ids]
        h = emb.mean(axis=1)
        for l, layer in enumerate(self.layers):
            h_in = h
            W1_eff, W2_eff = merged_weights[l]
            z1 = h @ W1_eff + layer['b1']
            a1 = relu(z1)
            z2 = a1 @ W2_eff + layer['b2']
            h = h_in + z2
        logits = h @ self.W_out
        return logits

    def compute_merged_loss(self, x_ids, y_ids, merged_weights):
        logits = self.forward_merged(x_ids, merged_weights)
        B = x_ids.shape[0]
        probs = softmax(logits)
        target_probs = probs[np.arange(B), y_ids]
        return -np.mean(np.log(target_probs + 1e-10))

    def precompute_merged_weights(self, lora_list):
        merged = []
        for l, layer in enumerate(self.layers):
            W1_eff = layer['W1'].copy()
            W2_eff = layer['W2'].copy()
            for lora in lora_list:
                scale = LORA_ALPHA / lora['rank']
                W1_eff += scale * (lora['A1'][l] @ lora['B1'][l])
                W2_eff += scale * (lora['A2'][l] @ lora['B2'][l])
            merged.append((W1_eff, W2_eff))
        return merged


def init_lora(rng, rank):
    A1 = [rng.randn(D_MODEL, rank) * np.sqrt(2.0 / D_MODEL)
          for _ in range(N_LAYERS)]
    B1 = [np.zeros((rank, D_FF)) for _ in range(N_LAYERS)]
    A2 = [rng.randn(D_FF, rank) * np.sqrt(2.0 / D_FF)
          for _ in range(N_LAYERS)]
    B2 = [np.zeros((rank, D_MODEL)) for _ in range(N_LAYERS)]
    return {'A1': A1, 'B1': B1, 'A2': A2, 'B2': B2, 'rank': rank}


def lora_to_delta_vector(lora):
    parts = []
    for l in range(N_LAYERS):
        parts.append((lora['A1'][l] @ lora['B1'][l]).flatten())
        parts.append((lora['A2'][l] @ lora['B2'][l]).flatten())
    return np.concatenate(parts)


def train_lora(model, data_x, data_y, rng, rank, steps=300, lr=0.01, batch_size=32):
    lora = init_lora(rng, rank)
    n_data = data_x.shape[0]
    scale = LORA_ALPHA / rank
    for step in range(steps):
        idx = rng.choice(n_data, size=min(batch_size, n_data), replace=False)
        bx, by = data_x[idx], data_y[idx]
        B_sz = bx.shape[0]
        logits = model.forward(bx, lora)
        probs = softmax(logits)
        d_logits = probs.copy()
        d_logits[np.arange(B_sz), by] -= 1.0
        d_logits /= B_sz
        d_h = d_logits @ model.W_out.T
        intermediates = []
        emb = model.wte[bx]
        h = emb.mean(axis=1)
        for l, layer in enumerate(model.layers):
            h_in = h
            z1 = h @ layer['W1'] + layer['b1']
            z1 += scale * (h @ lora['A1'][l] @ lora['B1'][l])
            a1 = relu(z1)
            z2 = a1 @ layer['W2'] + layer['b2']
            z2 += scale * (a1 @ lora['A2'][l] @ lora['B2'][l])
            h = h_in + z2
            intermediates.append({'h_in': h_in, 'z1': z1, 'a1': a1})
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
# Data Generation
# ============================================================================

def generate_cluster_prototypes(rng, n_clusters, cross_cluster_distance=3.0):
    """Generate distinct Markov transition matrices per cluster.

    Higher cross_cluster_distance = more distinct clusters = stronger signal.
    """
    prototypes = {}
    cluster_names = BASE_CLUSTER_NAMES[:n_clusters]
    for i, name in enumerate(cluster_names):
        logits = rng.randn(VOCAB_SIZE, VOCAB_SIZE) * cross_cluster_distance
        group_start = (i * VOCAB_SIZE) // max(n_clusters, 1)
        group_end = ((i + 1) * VOCAB_SIZE) // max(n_clusters, 1)
        logits[:, group_start:group_end] += 2.0
        logits += np.eye(VOCAB_SIZE) * 1.0
        prototypes[name] = softmax(logits, axis=-1)
    return prototypes


def generate_domain_data(rng, prototypes, domain_name, cluster_name,
                         n_sequences=200, noise_scale=0.2):
    prototype = prototypes[cluster_name]
    domain_seed = hash(domain_name) % (2**31)
    domain_rng = np.random.RandomState(domain_seed)
    domain_bias = domain_rng.randn(*prototype.shape) * noise_scale
    transition = softmax(np.log(prototype + 1e-10) + domain_bias, axis=-1)

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

def compute_pairwise_stats(delta_vectors, max_pairs=5000):
    """Compute pairwise |cos| statistics from delta vectors."""
    n = len(delta_vectors)
    mat = np.array(delta_vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    mat_normed = mat / norms

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        cos_mat = np.abs(mat_normed @ mat_normed.T)
        all_cos = cos_mat[np.triu_indices(n, k=1)]
    else:
        rng = np.random.RandomState(42)
        all_cos = []
        for _ in range(max_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            c = abs(float(mat_normed[i] @ mat_normed[j]))
            all_cos.append(c)
        all_cos = np.array(all_cos)

    return {
        'mean_cos': float(np.mean(all_cos)),
        'max_cos': float(np.max(all_cos)),
        'std_cos': float(np.std(all_cos)),
        'p95_cos': float(np.percentile(all_cos, 95)),
        'collision_rate': float(np.mean(all_cos > COLLISION_THRESHOLD)),
        'n_pairs': len(all_cos),
    }


def compute_signal_retention(delta_vectors):
    """Measure how much of each expert's signal survives in the sum.

    merged_delta = sum(delta_i). For each expert i:
    signal_i = <merged, delta_i> / ||delta_i||^2 * ||delta_i|| / ||merged||

    Ideal: if all deltas are orthogonal and equal norm, signal_i = 1/sqrt(N).
    Capacity cliff: when signal_i drops below theoretical 1/sqrt(N).
    """
    N = len(delta_vectors)
    mat = np.array(delta_vectors)
    merged = mat.sum(axis=0)
    merged_norm = np.linalg.norm(merged)

    norms = np.linalg.norm(mat, axis=1)
    mean_norm = np.mean(norms)

    # Signal retention: projection of merged onto each expert, normalized
    projections = mat @ merged / (merged_norm + 1e-12)
    signal_per_expert = projections / (norms + 1e-12)

    # Theoretical: if orthogonal, projection onto delta_i = ||delta_i||,
    # so signal = ||delta_i|| / ||merged||. With orthogonal equal-norm:
    # ||merged|| = sqrt(N) * ||delta_i||, so signal = 1/sqrt(N).
    theoretical_signal = 1.0 / np.sqrt(N)

    # Actual vs theoretical ratio
    mean_signal = float(np.mean(signal_per_expert))
    ratio = mean_signal / theoretical_signal

    return {
        'mean_signal_retention': mean_signal,
        'min_signal_retention': float(np.min(signal_per_expert)),
        'max_signal_retention': float(np.max(signal_per_expert)),
        'std_signal_retention': float(np.std(signal_per_expert)),
        'theoretical_signal': float(theoretical_signal),
        'retention_ratio': float(ratio),  # >1 = better than theory
        'merged_norm': float(merged_norm),
        'mean_expert_norm': float(mean_norm),
        'norm_ratio': float(merged_norm / (np.sqrt(N) * mean_norm + 1e-12)),
    }


def find_capacity_cliff(n_values, quality_ratios):
    """Find where quality ratio diverges from 1/N trend.

    quality_ratio = merged_loss / expert_loss.
    Expected under dilution: quality_ratio ~ 1 + c*(N-1)/N for some c.
    Cliff: ratio starts diverging above this.
    """
    N = np.array(n_values, dtype=np.float64)
    Q = np.array(quality_ratios, dtype=np.float64)

    # Fit Q = a + b*N from first 3 points (linear in N)
    n_fit = min(3, len(N))
    A_fit = np.column_stack([np.ones(n_fit), N[:n_fit]])
    params = np.linalg.lstsq(A_fit, Q[:n_fit], rcond=None)[0]
    a_fit, b_fit = params

    Q_pred = a_fit + b_fit * N
    excess = Q - Q_pred

    # Cliff: sustained excess > 5%
    cliff_idx = None
    for i in range(n_fit, len(N)):
        if excess[i] > 0.05 and (i + 1 >= len(N) or excess[i + 1] > 0.03):
            cliff_idx = i
            break

    cliff_N = int(N[cliff_idx]) if cliff_idx is not None else None

    return {
        'cliff_N': cliff_N,
        'a_fit': float(a_fit),
        'b_fit': float(b_fit),
        'excess': [float(e) for e in excess],
    }


def find_signal_cliff(n_values, retention_ratios):
    """Find where signal retention drops below 0.5 of theoretical."""
    for i, (n, r) in enumerate(zip(n_values, retention_ratios)):
        if r < 0.5:
            return int(n)
    return None


# ============================================================================
# Main Experiment
# ============================================================================

def run_single_n(model, n_experts, rng, rank, train_steps=300, lr=0.01,
                 n_train=200, n_test=50, batch_size=32):
    all_domains, domain_to_cluster = build_domain_config(n_experts)
    n_clusters = len(set(domain_to_cluster.values()))
    prototypes = generate_cluster_prototypes(rng, n_clusters)

    domain_test = {}
    domain_loras = {}
    delta_vectors = []

    for i, domain in enumerate(all_domains):
        cluster = domain_to_cluster[domain]
        train_rng = np.random.RandomState(rng.randint(2**31))
        test_rng = np.random.RandomState(rng.randint(2**31))
        x_train, y_train = generate_domain_data(
            train_rng, prototypes, domain, cluster, n_sequences=n_train)
        x_test, y_test = generate_domain_data(
            test_rng, prototypes, domain, cluster, n_sequences=n_test)
        domain_test[domain] = (x_test, y_test)

        lora_rng = np.random.RandomState(rng.randint(2**31))
        lora = train_lora(model, x_train, y_train, lora_rng, rank=rank,
                          steps=train_steps, lr=lr, batch_size=batch_size)
        domain_loras[domain] = lora
        delta_vectors.append(lora_to_delta_vector(lora))

    # 1. GEOMETRIC: pairwise cosine stats
    cos_stats = compute_pairwise_stats(delta_vectors)

    # 2. SIGNAL RETENTION: how much of each expert survives merge
    signal_stats = compute_signal_retention(delta_vectors)

    # 3. QUALITY: NTP loss comparison
    all_loras = [domain_loras[d] for d in all_domains]
    merged_weights = model.precompute_merged_weights(all_loras)

    # Evaluate on subsample of domains for speed
    max_eval = min(20, n_experts)
    eval_domains = all_domains[:max_eval]

    base_losses, expert_losses, merged_losses = [], [], []
    for domain in eval_domains:
        x_test, y_test = domain_test[domain]
        base_loss = model.compute_loss(x_test, y_test)
        expert_loss = model.compute_loss(x_test, y_test, domain_loras[domain])
        merged_loss = model.compute_merged_loss(x_test, y_test, merged_weights)
        base_losses.append(float(base_loss))
        expert_losses.append(float(expert_loss))
        merged_losses.append(float(merged_loss))

    mean_base = float(np.mean(base_losses))
    mean_expert = float(np.mean(expert_losses))
    mean_merged = float(np.mean(merged_losses))

    # Quality metrics
    quality_ratio = mean_merged / max(mean_expert, 1e-10)
    expert_improvement = (mean_base - mean_expert) / max(mean_base, 1e-10)
    merged_improvement = (mean_base - mean_merged) / max(mean_base, 1e-10)
    retained_improvement = merged_improvement / max(expert_improvement, 1e-10) \
        if expert_improvement > 0.001 else float('nan')

    # Catastrophic failures
    n_catastrophic = sum(1 for m, e in zip(merged_losses, expert_losses)
                         if m > 2 * e)

    return {
        'n_experts': n_experts,
        'n_clusters': n_clusters,
        'rank': rank,
        'cosine_stats': cos_stats,
        'signal_stats': signal_stats,
        'mean_base_loss': mean_base,
        'mean_expert_loss': mean_expert,
        'mean_merged_loss': mean_merged,
        'quality_ratio': float(quality_ratio),
        'expert_improvement_pct': float(expert_improvement * 100),
        'merged_improvement_pct': float(merged_improvement * 100),
        'retained_improvement_pct': float(retained_improvement * 100)
            if not np.isnan(retained_improvement) else None,
        'n_catastrophic': n_catastrophic,
        'n_eval_domains': max_eval,
    }


def run_experiment(seeds=(42, 142, 242), train_steps=300, lr=0.01,
                   n_train=200, n_test=50, batch_size=32):
    results_dir = Path(__file__).parent
    t_start = time.time()

    D_flat = N_LAYERS * 2 * D_MODEL * D_FF

    rank_configs = [
        {'rank': 8, 'n_max': D_MODEL**2 // 8**2, 'n_values': N_VALUES_R8},
        {'rank': 4, 'n_max': D_MODEL**2 // 4**2, 'n_values': N_VALUES_R4},
    ]

    print("=" * 76)
    print("  SUBSPACE CAPACITY EMPIRICAL")
    print("  Hypothesis: empirical capacity matches N_max = d^2/r^2")
    print(f"  Architecture: {N_LAYERS}-layer MLP, d={D_MODEL}, d_ff={D_FF}")
    print(f"  Delta dim D = {D_flat}")
    print(f"  Seeds: {list(seeds)}, steps={train_steps}")
    print("=" * 76)

    all_results = {}

    for rcfg in rank_configs:
        rank = rcfg['rank']
        n_max = rcfg['n_max']
        n_values = rcfg['n_values']
        rank_key = f"rank_{rank}"

        print(f"\n{'='*76}")
        print(f"  RANK = {rank}, N_max(d^2/r^2) = {n_max}, "
              f"N_max(D/r^2) = {D_flat // rank**2}")
        print(f"{'='*76}")

        all_results[rank_key] = {
            'rank': rank, 'n_max': n_max,
            'n_max_D': D_flat // rank**2, 'per_seed': {}
        }

        for seed in seeds:
            print(f"\n  --- SEED {seed} ---")
            seed_results = []

            for n in n_values:
                t_n = time.time()
                model_rng = np.random.RandomState(seed)
                model = MicroMLP(model_rng)
                exp_rng = np.random.RandomState(seed + n * 1000)

                result = run_single_n(
                    model, n, exp_rng, rank=rank,
                    train_steps=train_steps, lr=lr,
                    n_train=n_train, n_test=n_test, batch_size=batch_size)
                seed_results.append(result)

                dt = time.time() - t_n
                r = result
                sig = r['signal_stats']
                print(f"    N={n:4d}: q_ratio={r['quality_ratio']:.4f}, "
                      f"|cos|={r['cosine_stats']['mean_cos']:.4f}, "
                      f"signal={sig['mean_signal_retention']:.4f} "
                      f"(theory={sig['theoretical_signal']:.4f}, "
                      f"ratio={sig['retention_ratio']:.2f}), "
                      f"expert_imp={r['expert_improvement_pct']:.1f}%, "
                      f"({dt:.1f}s)", flush=True)

            all_results[rank_key]['per_seed'][str(seed)] = seed_results

    # ====================================================================
    # Aggregate
    # ====================================================================
    elapsed = time.time() - t_start

    print(f"\n{'='*76}")
    print(f"  AGGREGATE ({len(seeds)} seeds)")
    print(f"{'='*76}")

    summary = {}

    for rcfg in rank_configs:
        rank = rcfg['rank']
        n_max = rcfg['n_max']
        n_values = rcfg['n_values']
        rank_key = f"rank_{rank}"

        print(f"\n  === RANK {rank}, N_max = {n_max} ===")
        print(f"  {'N':>5s} {'QRatio':>8s} {'StdQ':>7s} {'|cos|':>7s} "
              f"{'Signal':>7s} {'SigTh':>7s} {'SRatio':>7s} {'ExpImp':>7s}")
        print(f"  {'-'*62}")

        agg_per_n = {}
        for n in n_values:
            vals = {'q': [], 'cos': [], 'sig': [], 'sig_th': [], 'sig_ratio': [],
                    'exp_imp': [], 'merged_imp': [], 'cat': []}
            for seed in seeds:
                for r in all_results[rank_key]['per_seed'][str(seed)]:
                    if r['n_experts'] == n:
                        vals['q'].append(r['quality_ratio'])
                        vals['cos'].append(r['cosine_stats']['mean_cos'])
                        vals['sig'].append(r['signal_stats']['mean_signal_retention'])
                        vals['sig_th'].append(r['signal_stats']['theoretical_signal'])
                        vals['sig_ratio'].append(r['signal_stats']['retention_ratio'])
                        vals['exp_imp'].append(r['expert_improvement_pct'])
                        vals['merged_imp'].append(r['merged_improvement_pct'])
                        vals['cat'].append(r['n_catastrophic'])

            agg = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                   for k, v in vals.items()}
            agg_per_n[n] = agg

            print(f"  {n:5d} {agg['q']['mean']:8.4f} {agg['q']['std']:7.4f} "
                  f"{agg['cos']['mean']:7.4f} "
                  f"{agg['sig']['mean']:7.4f} {agg['sig_th']['mean']:7.4f} "
                  f"{agg['sig_ratio']['mean']:7.2f} {agg['exp_imp']['mean']:6.1f}%")

        # Cliff detection on quality ratio
        q_means = [agg_per_n[n]['q']['mean'] for n in n_values]
        q_cliff = find_capacity_cliff(n_values, q_means)

        # Cliff detection on signal retention
        sig_ratios = [agg_per_n[n]['sig_ratio']['mean'] for n in n_values]
        sig_cliff_N = find_signal_cliff(n_values, sig_ratios)

        print(f"\n  Quality cliff: "
              f"{'N=' + str(q_cliff['cliff_N']) if q_cliff['cliff_N'] else 'NONE'}")
        print(f"  Signal cliff (ratio<0.5): "
              f"{'N=' + str(sig_cliff_N) if sig_cliff_N else 'NONE'}")

        summary[rank_key] = {
            'rank': rank,
            'n_max': n_max,
            'aggregate': {str(n): v for n, v in agg_per_n.items()},
            'quality_cliff': q_cliff,
            'signal_cliff_N': sig_cliff_N,
        }

    # ====================================================================
    # Kill Criteria
    # ====================================================================
    print(f"\n{'='*76}")
    print("  KILL CRITERIA")
    print(f"{'='*76}")

    for rcfg in rank_configs:
        rank = rcfg['rank']
        n_max = rcfg['n_max']
        n_values = rcfg['n_values']
        rank_key = f"rank_{rank}"
        s = summary[rank_key]

        print(f"\n  --- RANK {rank}, N_max = {n_max} ---")

        # K1: empirical capacity < 10% of N_max
        # Use signal cliff as capacity measure (more sensitive than quality)
        cliff_N = s['signal_cliff_N'] or s['quality_cliff']['cliff_N']
        if cliff_N:
            ratio = cliff_N / n_max
            k1_kill = ratio < 0.10
            print(f"  K1: capacity < 10%?  cliff={cliff_N}, ratio={ratio:.2f} "
                  f"-> {'KILL' if k1_kill else 'PASS'}")
        else:
            max_tested = max(n_values)
            ratio = max_tested / n_max
            k1_kill = False
            print(f"  K1: no cliff up to N={max_tested}, ratio>={ratio:.2f} -> PASS")

        # K2: smooth degradation?
        excess = s['quality_cliff']['excess']
        max_jump = max(abs(excess[i+1] - excess[i])
                       for i in range(len(excess)-1)) if len(excess) > 1 else 0
        k2_smooth = max_jump < 0.2
        print(f"  K2: smooth? max_jump={max_jump:.4f} -> "
              f"{'SMOOTH' if k2_smooth else 'SHARP CLIFF'}")

    # Overall
    print(f"\n{'='*76}")
    print("  VERDICT")
    print(f"{'='*76}")

    # Compute minimum capacity ratio across both ranks
    ratios = []
    for rcfg in rank_configs:
        rank_key = f"rank_{rcfg['rank']}"
        s = summary[rank_key]
        n_max = rcfg['n_max']
        n_values = rcfg['n_values']
        cliff_N = s['signal_cliff_N'] or s['quality_cliff']['cliff_N']
        if cliff_N:
            ratios.append(cliff_N / n_max)
        else:
            ratios.append(max(n_values) / n_max)

    min_ratio = min(ratios)
    k1_overall = min_ratio < 0.10
    print(f"\n  Min capacity ratio: {min_ratio:.2f}")
    print(f"  K1 (capacity<10%): {'KILL' if k1_overall else 'PASS'}")
    if not k1_overall:
        print(f"  Empirical capacity >= {min_ratio*100:.0f}% of theoretical N_max")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save
    output = {
        'config': {
            'd_model': D_MODEL, 'd_ff': D_FF, 'n_layers': N_LAYERS,
            'vocab_size': VOCAB_SIZE, 'lora_alpha': LORA_ALPHA,
            'seeds': list(seeds), 'train_steps': train_steps,
            'lr': lr, 'n_train': n_train, 'n_test': n_test,
            'D_flat': D_flat,
        },
        'rank_configs': [{'rank': r['rank'], 'n_max': r['n_max'],
                          'n_values': r['n_values']} for r in rank_configs],
        'summary': summary,
        'raw_results': all_results,
        'kill_criteria': {
            'k1_threshold': 0.10,
            'capacity_ratios': {f"rank_{r['rank']}": float(ratios[i])
                                for i, r in enumerate(rank_configs)},
            'min_ratio': float(min_ratio),
            'k1_kill': bool(k1_overall),
        },
        'elapsed_seconds': elapsed,
    }

    output_file = results_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {output_file}")

    return output


if __name__ == '__main__':
    import sys
    if '--fast' in sys.argv:
        run_experiment(seeds=(42,), train_steps=100, n_train=100, n_test=20)
    else:
        run_experiment()
