#!/usr/bin/env python3
"""
Orthogonality by Domain Type: Does domain similarity predict LoRA interference?

Hypothesis: Within-cluster cosine similarity of LoRA expert deltas is higher
than cross-cluster cosine similarity, revealing semantic structure in the
parameter space collision landscape.

Design:
  1. Build a micro MLP base model (pure numpy, char-level next-token prediction)
  2. Generate 15 synthetic domains in 3 semantic clusters using controlled
     character transition matrices (Markov chains)
  3. Train 15 LoRA adapters (one per domain) via vectorized backprop
  4. Measure the full 15x15 cosine similarity matrix of flattened LoRA deltas
  5. Statistical test: within-cluster mean |cos| vs cross-cluster mean |cos|

Cluster design (simulated via character transition matrices):
  Cluster A (code-like):    python, javascript, rust, bash, sql
  Cluster B (reasoning):    math, logic, physics, statistics, economics
  Cluster C (knowledge):    medical, law, history, psychology, cooking

Within-cluster domains share a "cluster prototype" transition matrix with
small per-domain perturbations. Cross-cluster domains use different prototypes.

Architecture: 4-layer MLP, d=64, d_ff=256, rank=8 LoRA on all MLP layers.
This isolates the LoRA orthogonality mechanism without attention overhead.
Pure numpy -- no MLX, no PyTorch, runs in ~2 minutes.

Kill criteria:
  K1: within-cluster |cos| is NOT higher than cross-cluster (no semantic structure)
  K2: no predictable pattern in which domains collide
"""

import json
import time
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 32
CONTEXT_LEN = 16   # short context for speed
D_MODEL = 64
D_FF = 256
N_LAYERS = 4
LORA_RANK = 8
LORA_ALPHA = 8
N_DOMAINS = 15
N_CLUSTERS = 3

# Domain clusters
CLUSTERS = {
    'code': ['python', 'javascript', 'rust', 'bash', 'sql'],
    'reasoning': ['math', 'logic', 'physics', 'statistics', 'economics'],
    'knowledge': ['medical', 'law', 'history', 'psychology', 'cooking'],
}

DOMAIN_TO_CLUSTER = {}
for cluster, domains in CLUSTERS.items():
    for domain in domains:
        DOMAIN_TO_CLUSTER[domain] = cluster

ALL_DOMAINS = []
for cluster in ['code', 'reasoning', 'knowledge']:
    ALL_DOMAINS.extend(CLUSTERS[cluster])


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Model: Multi-layer MLP with LoRA
# ═══════════════════════════════════════════════════════════════════════════════

class MicroMLP:
    """4-layer MLP for next-token prediction.

    Architecture:
      embed(V, d) -> [Linear(d, dff) -> ReLU -> Linear(dff, d)] x N_LAYERS -> Linear(d, V)

    LoRA is applied to the two linear layers in each MLP block:
      W1_adapted = W1 + (alpha/r) * A1 @ B1
      W2_adapted = W2 + (alpha/r) * A2 @ B2

    Input: (B, T) integer token ids
    We use a simple "bag of context" approach: embed each token, average the
    context window, then predict the next token. This is simpler than a
    transformer but sufficient to test whether LoRA deltas trained on similar
    data distributions end up more similar in parameter space.
    """

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
        """Forward pass.
        x_ids: (B, T) integer token ids
        lora: dict with 'A1', 'B1', 'A2', 'B2' lists (one per layer)
        Returns: logits (B, V), intermediates for backprop
        """
        B, T = x_ids.shape
        scale = LORA_ALPHA / LORA_RANK

        # Embed and average context
        emb = self.wte[x_ids]        # (B, T, d)
        h = emb.mean(axis=1)         # (B, d) -- bag of words

        intermediates = []
        for l, layer in enumerate(self.layers):
            h_in = h
            # fc1
            z1 = h @ layer['W1'] + layer['b1']  # (B, dff)
            if lora is not None:
                z1 = z1 + scale * (h @ lora['A1'][l] @ lora['B1'][l])
            a1 = relu(z1)

            # fc2
            z2 = a1 @ layer['W2'] + layer['b2']  # (B, d)
            if lora is not None:
                z2 = z2 + scale * (a1 @ lora['A2'][l] @ lora['B2'][l])

            # Residual
            h = h_in + z2
            intermediates.append({'h_in': h_in, 'z1': z1, 'a1': a1})

        logits = h @ self.W_out  # (B, V)
        return logits, intermediates, h


def init_lora(rng):
    """Initialize LoRA parameters: A (Kaiming), B (zero)."""
    A1 = [rng.randn(D_MODEL, LORA_RANK) * np.sqrt(2.0 / D_MODEL)
           for _ in range(N_LAYERS)]
    B1 = [np.zeros((LORA_RANK, D_FF)) for _ in range(N_LAYERS)]
    A2 = [rng.randn(D_FF, LORA_RANK) * np.sqrt(2.0 / D_FF)
           for _ in range(N_LAYERS)]
    B2 = [np.zeros((LORA_RANK, D_MODEL)) for _ in range(N_LAYERS)]
    return {'A1': A1, 'B1': B1, 'A2': A2, 'B2': B2}


def lora_to_delta_vector(lora):
    """Flatten LoRA deltas (A@B products) into a single vector."""
    parts = []
    for l in range(N_LAYERS):
        parts.append((lora['A1'][l] @ lora['B1'][l]).flatten())
        parts.append((lora['A2'][l] @ lora['B2'][l]).flatten())
    return np.concatenate(parts)


def train_lora(model, data_x, data_y, rng, steps=300, lr=0.01,
               batch_size=32, log_every=100):
    """Train LoRA via vectorized backprop. Only B matrices are updated."""
    lora = init_lora(rng)
    n_data = data_x.shape[0]
    scale = LORA_ALPHA / LORA_RANK
    losses = []

    for step in range(steps):
        idx = rng.choice(n_data, size=min(batch_size, n_data), replace=False)
        bx, by = data_x[idx], data_y[idx]
        B_sz = bx.shape[0]

        # ── Forward ──
        logits, intermediates, h_final = model.forward(bx, lora)
        probs = softmax(logits)  # (B, V)

        # ── Loss ──
        # by: (B,) integer targets
        target_probs = probs[np.arange(B_sz), by]
        loss = -np.mean(np.log(target_probs + 1e-10))
        losses.append(loss)

        if step % log_every == 0:
            print(f"    step {step:4d}: loss={loss:.4f}")

        # ── Backward ──
        # d_logits
        d_logits = probs.copy()  # (B, V)
        d_logits[np.arange(B_sz), by] -= 1.0
        d_logits /= B_sz

        # d_h = d_logits @ W_out.T
        d_h = d_logits @ model.W_out.T  # (B, d)

        # Backward through layers (reverse order)
        for l in range(N_LAYERS - 1, -1, -1):
            layer = model.layers[l]
            inter = intermediates[l]
            h_in = inter['h_in']   # (B, d)
            z1 = inter['z1']       # (B, dff)
            a1 = inter['a1']       # (B, dff)

            # Through residual: d_z2 = d_h, d_h_in_from_residual = d_h
            d_z2 = d_h  # (B, d)

            # Gradient for B2[l]
            # z2_lora = scale * a1 @ A2 @ B2
            # d_B2 = scale * (a1 @ A2).T @ d_z2 = scale * A2.T @ a1.T @ d_z2
            proj2 = a1 @ lora['A2'][l]  # (B, r)
            d_B2 = scale * (proj2.T @ d_z2)  # (r, d)
            lora['B2'][l] -= lr * d_B2

            # d_a1 from fc2: d_a1 = d_z2 @ (W2 + scale * A2 @ B2).T
            W2_eff = layer['W2'] + scale * lora['A2'][l] @ lora['B2'][l]
            d_a1 = d_z2 @ W2_eff.T  # (B, dff)

            # Through ReLU
            d_z1 = d_a1 * relu_grad(z1)  # (B, dff)

            # Gradient for B1[l]
            proj1 = h_in @ lora['A1'][l]  # (B, r)
            d_B1 = scale * (proj1.T @ d_z1)  # (r, dff)
            lora['B1'][l] -= lr * d_B1

            # Propagate through residual
            # d_h_in = d_h (from residual) + d_z1 @ (W1 + scale * A1 @ B1).T
            # But since base is frozen and LoRA is small, we approximate d_h = d_h
            # (skip the fc1 backprop contribution to d_h for speed)

    if losses:
        print(f"    final loss: {losses[-1]:.4f}")
    return lora, losses[-1] if losses else float('inf')


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Data Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cluster_prototypes(rng, cross_cluster_distance=2.0):
    """Generate Markov chain transition matrices for each cluster.

    Each cluster prototype emphasizes different character groups, modeling
    how code/reasoning/knowledge domains use different vocabulary distributions.
    """
    prototypes = {}
    cluster_names = ['code', 'reasoning', 'knowledge']

    for i, name in enumerate(cluster_names):
        logits = rng.randn(VOCAB_SIZE, VOCAB_SIZE) * cross_cluster_distance
        # Cluster-specific character group bias
        group_start = (i * VOCAB_SIZE) // N_CLUSTERS
        group_end = ((i + 1) * VOCAB_SIZE) // N_CLUSTERS
        logits[:, group_start:group_end] += 1.5
        logits += np.eye(VOCAB_SIZE) * 0.5
        prototypes[name] = softmax(logits, axis=-1)

    return prototypes


def generate_domain_data(rng, prototypes, domain_name, cluster_name,
                          n_sequences=200, noise_scale=0.15):
    """Generate (x, y) pairs for next-token prediction.

    x: (n_sequences, CONTEXT_LEN) context tokens
    y: (n_sequences,) next token to predict
    """
    prototype = prototypes[cluster_name]
    noise = rng.randn(*prototype.shape) * noise_scale

    # Domain-specific perturbation
    domain_seed = hash(domain_name) % (2**31)
    domain_rng = np.random.RandomState(domain_seed)
    domain_bias = domain_rng.randn(*prototype.shape) * noise_scale * 0.5

    # Domain signature characters
    domain_idx = ALL_DOMAINS.index(domain_name)
    for j in range(3):
        c = (domain_idx * 2 + j) % VOCAB_SIZE
        domain_bias[:, c] += 0.3

    transition = softmax(np.log(prototype + 1e-10) + noise + domain_bias, axis=-1)

    # Generate sequences
    seq_len = CONTEXT_LEN + 1  # context + target
    sequences = np.zeros((n_sequences, seq_len), dtype=np.int32)
    for i in range(n_sequences):
        sequences[i, 0] = rng.choice(VOCAB_SIZE)
        for t in range(seq_len - 1):
            prev = sequences[i, t]
            sequences[i, t + 1] = rng.choice(VOCAB_SIZE, p=transition[prev])

    x = sequences[:, :-1]  # (n_seq, CONTEXT_LEN)
    y = sequences[:, -1]   # (n_seq,) next token
    return x, y


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_cosine_matrix(domain_vectors):
    domains = list(domain_vectors.keys())
    n = len(domains)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = cosine_sim(domain_vectors[domains[i]],
                                           domain_vectors[domains[j]])
    return matrix, domains


def analyze_clustering(cos_matrix, domains):
    n = len(domains)
    within_cos = []
    cross_cos = []
    pair_details = []

    for i in range(n):
        for j in range(i + 1, n):
            c = abs(cos_matrix[i, j])
            d1, d2 = domains[i], domains[j]
            cl1 = DOMAIN_TO_CLUSTER[d1]
            cl2 = DOMAIN_TO_CLUSTER[d2]
            is_within = cl1 == cl2
            if is_within:
                within_cos.append(c)
            else:
                cross_cos.append(c)
            pair_details.append({
                'domain1': d1, 'domain2': d2,
                'cluster1': cl1, 'cluster2': cl2,
                'cosine': float(cos_matrix[i, j]),
                'abs_cosine': float(c),
                'same_cluster': is_within,
            })

    return {
        'within_cluster': {
            'mean': float(np.mean(within_cos)),
            'std': float(np.std(within_cos)),
            'median': float(np.median(within_cos)),
            'max': float(np.max(within_cos)),
            'min': float(np.min(within_cos)),
            'n_pairs': len(within_cos),
        },
        'cross_cluster': {
            'mean': float(np.mean(cross_cos)),
            'std': float(np.std(cross_cos)),
            'median': float(np.median(cross_cos)),
            'max': float(np.max(cross_cos)),
            'min': float(np.min(cross_cos)),
            'n_pairs': len(cross_cos),
        },
        'ratio': float(np.mean(within_cos) / max(np.mean(cross_cos), 1e-10)),
        'pair_details': pair_details,
    }


def permutation_test(within_cos, cross_cos, n_permutations=10000, rng=None):
    """Permutation test for within > cross cluster cosine."""
    if rng is None:
        rng = np.random.RandomState(42)

    observed = np.mean(within_cos) - np.mean(cross_cos)
    all_cos = np.array(within_cos + cross_cos)
    n_within = len(within_cos)

    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(all_cos))
        stat = np.mean(all_cos[perm[:n_within]]) - np.mean(all_cos[perm[n_within:]])
        if stat >= observed:
            count_ge += 1

    p_value = count_ge / n_permutations
    return {
        'observed_diff': float(observed),
        'p_value': float(p_value),
        'n_permutations': n_permutations,
        'significant_at_005': p_value < 0.05,
        'significant_at_001': p_value < 0.01,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(seed=42, n_seeds=3, train_steps=300, train_lr=0.01,
                   n_sequences=300, batch_size=32):
    """Run the full experiment across multiple seeds."""
    results_dir = Path(__file__).parent
    t_start = time.time()

    print("=" * 72)
    print("  Orthogonality by Domain Type")
    print("  Hypothesis: within-cluster |cos| > cross-cluster |cos|")
    print("  Architecture: 4-layer MLP, d=64, d_ff=256, rank=8 LoRA")
    print(f"  Config: {n_seeds} seeds, {train_steps} steps, lr={train_lr}")
    print("=" * 72)

    all_seed_results = []

    for seed_idx in range(n_seeds):
        current_seed = seed + seed_idx * 100
        print(f"\n{'='*72}")
        print(f"  SEED {seed_idx + 1}/{n_seeds} (seed={current_seed})")
        print(f"{'='*72}")

        rng = np.random.RandomState(current_seed)

        # Build base model
        print("\n  Building base model...")
        model = MicroMLP(rng)

        # Generate cluster prototypes
        print("  Generating cluster prototypes...")
        prototypes = generate_cluster_prototypes(rng)

        # Train domain LoRA experts
        print(f"\n  Training {N_DOMAINS} LoRA experts...")
        domain_vectors = {}
        domain_losses = {}

        for i, domain in enumerate(ALL_DOMAINS):
            cluster = DOMAIN_TO_CLUSTER[domain]
            print(f"\n  [{i+1}/{N_DOMAINS}] {domain} ({cluster})")

            x, y = generate_domain_data(rng, prototypes, domain, cluster,
                                         n_sequences=n_sequences)
            lora, final_loss = train_lora(
                model, x, y, rng,
                steps=train_steps, lr=train_lr, batch_size=batch_size,
                log_every=max(1, train_steps // 3),
            )
            domain_vectors[domain] = lora_to_delta_vector(lora)
            domain_losses[domain] = final_loss

        # Compute cosine matrix
        print(f"\n  Computing 15x15 cosine similarity matrix...")
        cos_matrix, domain_order = compute_cosine_matrix(domain_vectors)
        analysis = analyze_clustering(cos_matrix, domain_order)

        # Results
        print(f"\n  {'='*60}")
        print(f"  RESULTS (seed={current_seed})")
        print(f"  {'='*60}")

        print(f"\n  Within-cluster |cos|:  mean={analysis['within_cluster']['mean']:.6f}, "
              f"std={analysis['within_cluster']['std']:.6f}, "
              f"n={analysis['within_cluster']['n_pairs']}")
        print(f"  Cross-cluster |cos|:   mean={analysis['cross_cluster']['mean']:.6f}, "
              f"std={analysis['cross_cluster']['std']:.6f}, "
              f"n={analysis['cross_cluster']['n_pairs']}")
        print(f"  Ratio (within/cross):  {analysis['ratio']:.2f}x")

        # Permutation test
        within_list = [p['abs_cosine'] for p in analysis['pair_details']
                       if p['same_cluster']]
        cross_list = [p['abs_cosine'] for p in analysis['pair_details']
                      if not p['same_cluster']]
        perm_test = permutation_test(within_list, cross_list, rng=rng)

        print(f"\n  Permutation test:")
        print(f"    observed diff: {perm_test['observed_diff']:.6f}")
        print(f"    p-value:       {perm_test['p_value']:.4f}")
        print(f"    significant (p<0.05): {perm_test['significant_at_005']}")

        # Per-cluster analysis
        print(f"\n  Per-cluster within-cluster mean |cos|:")
        for cname in ['code', 'reasoning', 'knowledge']:
            cpairs = [p['abs_cosine'] for p in analysis['pair_details']
                      if p['same_cluster'] and p['cluster1'] == cname]
            if cpairs:
                print(f"    {cname:12s}: mean={np.mean(cpairs):.6f}, n={len(cpairs)}")

        # Cross-cluster breakdown
        print(f"\n  Cross-cluster mean |cos| by cluster pair:")
        cnames = ['code', 'reasoning', 'knowledge']
        for i_c, c1 in enumerate(cnames):
            for j_c, c2 in enumerate(cnames):
                if j_c <= i_c:
                    continue
                pair_cos = [p['abs_cosine'] for p in analysis['pair_details']
                            if not p['same_cluster'] and
                            ((p['cluster1'] == c1 and p['cluster2'] == c2) or
                             (p['cluster1'] == c2 and p['cluster2'] == c1))]
                if pair_cos:
                    print(f"    {c1:12s} vs {c2:12s}: mean={np.mean(pair_cos):.6f}, "
                          f"n={len(pair_cos)}")

        # Top-10 most similar pairs
        sorted_pairs = sorted(analysis['pair_details'],
                               key=lambda p: p['abs_cosine'], reverse=True)
        print(f"\n  Top 10 most similar pairs:")
        print(f"  {'Pair':<35s} {'|cos|':>10s} {'Same?':>6s}")
        print(f"  {'-'*55}")
        for p in sorted_pairs[:10]:
            tag = "YES" if p['same_cluster'] else "no"
            print(f"  {p['domain1']+' vs '+p['domain2']:<35s} "
                  f"{p['abs_cosine']:>10.6f} {tag:>6s}")

        # Print full cosine matrix (compact)
        print(f"\n  Full cosine matrix (absolute values):")
        short_names = [d[:4] for d in domain_order]
        header = "  " + "".join(f"{n:>7s}" for n in short_names)
        print(header)
        for i_d, d1 in enumerate(domain_order):
            row = f"  {short_names[i_d]:>4s}"
            for j_d in range(len(domain_order)):
                val = abs(cos_matrix[i_d, j_d])
                if i_d == j_d:
                    row += "      ."
                else:
                    row += f" {val:6.4f}"
            print(row)

        # Domain losses
        print(f"\n  Domain final losses:")
        for domain in ALL_DOMAINS:
            cluster = DOMAIN_TO_CLUSTER[domain]
            print(f"    {domain:15s} ({cluster:10s}): {domain_losses[domain]:.4f}")

        all_seed_results.append({
            'seed': current_seed,
            'within_mean': analysis['within_cluster']['mean'],
            'cross_mean': analysis['cross_cluster']['mean'],
            'ratio': analysis['ratio'],
            'p_value': perm_test['p_value'],
            'significant': perm_test['significant_at_005'],
            'analysis': analysis,
            'perm_test': perm_test,
            'domain_losses': domain_losses,
            'cosine_matrix': cos_matrix.tolist(),
            'domain_order': domain_order,
        })

    # ═══════════════════════════════════════════════════════════════════
    # Aggregate across seeds
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  AGGREGATE RESULTS ({n_seeds} seeds)")
    print(f"{'='*72}")

    within_means = [r['within_mean'] for r in all_seed_results]
    cross_means = [r['cross_mean'] for r in all_seed_results]
    ratios = [r['ratio'] for r in all_seed_results]
    p_values = [r['p_value'] for r in all_seed_results]
    n_significant = sum(1 for r in all_seed_results if r['significant'])

    print(f"\n  Within-cluster mean |cos|: {np.mean(within_means):.6f} "
          f"(+/- {np.std(within_means):.6f})")
    print(f"  Cross-cluster mean |cos|:  {np.mean(cross_means):.6f} "
          f"(+/- {np.std(cross_means):.6f})")
    print(f"  Mean ratio:                {np.mean(ratios):.2f}x")
    print(f"  p-values:                  {[f'{p:.4f}' for p in p_values]}")
    print(f"  Seeds significant (p<0.05): {n_significant}/{n_seeds}")

    # Effect size: Cohen's d
    all_within = []
    all_cross = []
    for r in all_seed_results:
        all_within.extend([p['abs_cosine'] for p in r['analysis']['pair_details']
                           if p['same_cluster']])
        all_cross.extend([p['abs_cosine'] for p in r['analysis']['pair_details']
                          if not p['same_cluster']])
    pooled_std = np.sqrt((np.var(all_within) + np.var(all_cross)) / 2)
    cohens_d = (np.mean(all_within) - np.mean(all_cross)) / max(pooled_std, 1e-10)
    print(f"  Cohen's d (effect size):   {cohens_d:.3f}")

    # ── Kill criteria ──
    print(f"\n{'='*72}")
    print(f"  KILL CRITERIA EVALUATION")
    print(f"{'='*72}")

    k1_killed = np.mean(within_means) <= np.mean(cross_means)
    print(f"\n  K1: Within-cluster |cos| NOT higher than cross-cluster")
    print(f"      within={np.mean(within_means):.6f}, cross={np.mean(cross_means):.6f}")
    print(f"      Ratio: {np.mean(ratios):.2f}x")
    if k1_killed:
        print(f"      STATUS: KILL -- no semantic structure in cosine similarity")
    else:
        print(f"      STATUS: PASS -- within-cluster IS higher ({np.mean(ratios):.2f}x)")

    k2_pattern = n_significant >= (n_seeds + 1) // 2  # majority significant
    print(f"\n  K2: No predictable pattern in domain collisions")
    print(f"      {n_significant}/{n_seeds} seeds show significant clustering (p<0.05)")
    if not k2_pattern:
        print(f"      STATUS: KILL -- pattern not reliably detected")
    else:
        print(f"      STATUS: PASS -- clustering is statistically significant")

    overall_kill = k1_killed or not k2_pattern
    print(f"\n  {'='*60}")
    if overall_kill:
        print(f"  OVERALL VERDICT: KILL")
        if k1_killed:
            print(f"    K1 failed: within-cluster cosine not higher than cross-cluster")
        if not k2_pattern:
            print(f"    K2 failed: clustering pattern not statistically significant")
    else:
        print(f"  OVERALL VERDICT: SUPPORTED")
        print(f"    Within-cluster similarity is {np.mean(ratios):.2f}x higher")
        print(f"    Significant in {n_significant}/{n_seeds} seeds")
        print(f"    Cohen's d = {cohens_d:.3f}")
    print(f"  {'='*60}")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save results
    output = {
        'config': {
            'n_seeds': n_seeds,
            'train_steps': train_steps,
            'train_lr': train_lr,
            'n_sequences': n_sequences,
            'batch_size': batch_size,
            'd_model': D_MODEL,
            'd_ff': D_FF,
            'n_layers': N_LAYERS,
            'lora_rank': LORA_RANK,
            'vocab_size': VOCAB_SIZE,
            'context_len': CONTEXT_LEN,
        },
        'clusters': {k: v for k, v in CLUSTERS.items()},
        'seed_results': all_seed_results,
        'aggregate': {
            'within_mean': float(np.mean(within_means)),
            'within_std': float(np.std(within_means)),
            'cross_mean': float(np.mean(cross_means)),
            'cross_std': float(np.std(cross_means)),
            'mean_ratio': float(np.mean(ratios)),
            'p_values': [float(p) for p in p_values],
            'n_significant': n_significant,
            'cohens_d': float(cohens_d),
        },
        'kill_criteria': {
            'k1_within_gt_cross': not k1_killed,
            'k2_pattern_significant': k2_pattern,
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
