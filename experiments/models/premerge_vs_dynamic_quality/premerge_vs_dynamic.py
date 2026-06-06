#!/usr/bin/env python3
"""
Pre-Merge vs Dynamic Routing Quality Comparison.

Hypothesis: Pre-merging all expert deltas into a single model (W_base + (1/N) * sum(delta_i))
degrades quality by >5% compared to dynamic top-k routing at the same expert count,
because 1/N dilution drowns out specialist signal.

Counter-hypothesis: At small N (<20), orthogonal deltas compose cleanly and pre-merge
matches dynamic routing because there is no interference and the 1/N scaling
preserves the relative contribution.

Design:
  1. Pre-train a base MLP on a mixture of all domain data (critical: base model
     must have useful features for LoRA to specialize on top of)
  2. Freeze the base, train N domain-specific LoRA adapters (both A and B)
  3. Compare serving strategies: pre-merge vs dynamic top-k vs oracle
  4. Vary N from 5 to 20
  5. Measure per-domain NTP loss

Architecture: 4-layer MLP, d=64, d_ff=256, rank=8 LoRA.
Pure numpy -- no MLX, no PyTorch. CPU-only.

Kill criteria:
  K1: Pre-merged is >5% worse than dynamic top-k on domain-specific tasks
  K2: Dynamic routing provides zero quality advantage over pre-merge at N<20
"""

import json
import time
from pathlib import Path

import numpy as np


# ===========================================================================
# Constants
# ===========================================================================

VOCAB_SIZE = 32
CONTEXT_LEN = 16
D_MODEL = 64
D_FF = 256
N_LAYERS = 4
LORA_RANK = 8
LORA_ALPHA = 8

N_VALUES = [5, 8, 12, 16, 20]

CLUSTERS = {
    'code':      ['python', 'javascript', 'rust', 'bash', 'sql'],
    'reasoning': ['math', 'logic', 'physics', 'statistics', 'economics'],
    'knowledge': ['medical', 'law', 'history', 'psychology', 'cooking'],
    'creative':  ['poetry', 'fiction', 'journalism', 'comedy', 'lyrics'],
}
N_CLUSTERS = len(CLUSTERS)

DOMAIN_TO_CLUSTER = {}
for cluster, domains in CLUSTERS.items():
    for domain in domains:
        DOMAIN_TO_CLUSTER[domain] = cluster

ALL_DOMAINS = []
for cluster in ['code', 'reasoning', 'knowledge', 'creative']:
    ALL_DOMAINS.extend(CLUSTERS[cluster])


# ===========================================================================
# Utilities
# ===========================================================================

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float64)


# ===========================================================================
# Model
# ===========================================================================

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

    def embed(self, x_ids):
        return self.wte[x_ids].mean(axis=1)

    def forward(self, x_ids, lora=None):
        B, T = x_ids.shape
        scale = LORA_ALPHA / LORA_RANK
        h = self.embed(x_ids)

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

    def forward_with_delta(self, x_ids, delta_weights):
        """Forward pass with pre-merged weight deltas."""
        B, T = x_ids.shape
        h = self.embed(x_ids)

        for l, layer in enumerate(self.layers):
            h_in = h
            W1_eff = layer['W1'] + delta_weights[l]['dW1']
            z1 = h @ W1_eff + layer['b1']
            a1 = relu(z1)

            W2_eff = layer['W2'] + delta_weights[l]['dW2']
            z2 = a1 @ W2_eff + layer['b2']
            h = h_in + z2

        logits = h @ self.W_out
        return logits


def pretrain_base(model, all_x, all_y, rng, steps=2000, lr=0.005,
                  batch_size=64, log_every=500):
    """Pre-train the base model on mixed domain data.

    Trains ALL weights: wte, W1, b1, W2, b2, W_out.
    This is critical: LoRA needs useful base features to specialize on top of.
    """
    n_data = all_x.shape[0]
    print(f"  Pre-training base on {n_data} samples, {steps} steps...")

    for step in range(steps):
        idx = rng.choice(n_data, size=min(batch_size, n_data), replace=False)
        bx, by = all_x[idx], all_y[idx]
        B_sz = bx.shape[0]

        # Forward
        h = model.wte[bx].mean(axis=1)  # (B, d)

        intermediates = []
        for l, layer in enumerate(model.layers):
            h_in = h
            z1 = h @ layer['W1'] + layer['b1']
            a1 = relu(z1)
            z2 = a1 @ layer['W2'] + layer['b2']
            h = h_in + z2
            intermediates.append({'h_in': h_in, 'z1': z1, 'a1': a1, 'z2': z2})

        logits = h @ model.W_out  # (B, V)
        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(B_sz), by] + 1e-10))

        if step % log_every == 0:
            print(f"    step {step:4d}: loss={loss:.4f}")

        # Backward
        d_logits = probs.copy()
        d_logits[np.arange(B_sz), by] -= 1.0
        d_logits /= B_sz

        # W_out gradient
        d_W_out = h.T @ d_logits
        model.W_out -= lr * d_W_out

        d_h = d_logits @ model.W_out.T

        for l in range(N_LAYERS - 1, -1, -1):
            layer = model.layers[l]
            inter = intermediates[l]
            h_in, z1, a1 = inter['h_in'], inter['z1'], inter['a1']

            d_z2 = d_h
            d_W2 = a1.T @ d_z2
            d_b2 = d_z2.sum(axis=0)
            layer['W2'] -= lr * d_W2
            layer['b2'] -= lr * d_b2

            d_a1 = d_z2 @ layer['W2'].T
            d_z1 = d_a1 * relu_grad(z1)

            d_W1 = h_in.T @ d_z1
            d_b1 = d_z1.sum(axis=0)
            layer['W1'] -= lr * d_W1
            layer['b1'] -= lr * d_b1

            # Propagate to h_in (residual)
            d_h = d_h + d_z1 @ layer['W1'].T

        # Embedding gradient (bag-of-words)
        d_emb = d_h  # (B, d)
        for i in range(B_sz):
            for t in range(bx.shape[1]):
                model.wte[bx[i, t]] -= lr * d_emb[i] / bx.shape[1]

    # Final loss
    eval_n = min(256, n_data)
    logits, _, _ = model.forward(all_x[:eval_n])
    probs = softmax(logits)
    final_loss = -np.mean(np.log(
        probs[np.arange(eval_n), all_y[:eval_n]] + 1e-10))
    print(f"    final base loss: {final_loss:.4f}")
    return final_loss


def init_lora(rng):
    A1 = [rng.randn(D_MODEL, LORA_RANK) * np.sqrt(2.0 / D_MODEL)
          for _ in range(N_LAYERS)]
    B1 = [np.zeros((LORA_RANK, D_FF)) for _ in range(N_LAYERS)]
    A2 = [rng.randn(D_FF, LORA_RANK) * np.sqrt(2.0 / D_FF)
          for _ in range(N_LAYERS)]
    B2 = [np.zeros((LORA_RANK, D_MODEL)) for _ in range(N_LAYERS)]
    return {'A1': A1, 'B1': B1, 'A2': A2, 'B2': B2}


def lora_to_delta_weights(lora):
    scale = LORA_ALPHA / LORA_RANK
    deltas = []
    for l in range(N_LAYERS):
        dW1 = scale * (lora['A1'][l] @ lora['B1'][l])
        dW2 = scale * (lora['A2'][l] @ lora['B2'][l])
        deltas.append({'dW1': dW1, 'dW2': dW2})
    return deltas


def lora_to_delta_vector(lora):
    parts = []
    for l in range(N_LAYERS):
        parts.append((lora['A1'][l] @ lora['B1'][l]).flatten())
        parts.append((lora['A2'][l] @ lora['B2'][l]).flatten())
    return np.concatenate(parts)


def train_lora(model, data_x, data_y, rng, steps=500, lr=0.01,
               batch_size=32, log_every=200):
    """Train LoRA: BOTH A and B matrices are updated."""
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
        loss = -np.mean(np.log(target_probs + 1e-10))

        if step % log_every == 0:
            print(f"    step {step:4d}: loss={loss:.4f}")

        # Backward
        d_logits = probs.copy()
        d_logits[np.arange(B_sz), by] -= 1.0
        d_logits /= B_sz
        d_h = d_logits @ model.W_out.T

        for l in range(N_LAYERS - 1, -1, -1):
            layer = model.layers[l]
            inter = intermediates[l]
            h_in, z1, a1 = inter['h_in'], inter['z1'], inter['a1']

            d_z2 = d_h

            # Gradients for B2 and A2
            proj2 = a1 @ lora['A2'][l]  # (B, r)
            d_B2 = scale * (proj2.T @ d_z2)  # (r, d)
            d_A2 = scale * (a1.T @ (d_z2 @ lora['B2'][l].T))  # (dff, r)
            lora['B2'][l] -= lr * d_B2
            lora['A2'][l] -= lr * d_A2

            W2_eff = layer['W2'] + scale * lora['A2'][l] @ lora['B2'][l]
            d_a1 = d_z2 @ W2_eff.T
            d_z1 = d_a1 * relu_grad(z1)

            # Gradients for B1 and A1
            proj1 = h_in @ lora['A1'][l]  # (B, r)
            d_B1 = scale * (proj1.T @ d_z1)  # (r, dff)
            d_A1 = scale * (h_in.T @ (d_z1 @ lora['B1'][l].T))  # (d, r)
            lora['B1'][l] -= lr * d_B1
            lora['A1'][l] -= lr * d_A1

            # Propagate through residual
            W1_eff = layer['W1'] + scale * lora['A1'][l] @ lora['B1'][l]
            d_h = d_h + d_z1 @ W1_eff.T

    # Final loss
    eval_n = min(64, n_data)
    final_logits, _, _ = model.forward(data_x[:eval_n], lora)
    final_probs = softmax(final_logits)
    final_targets = data_y[:eval_n]
    final_loss = -np.mean(np.log(
        final_probs[np.arange(eval_n), final_targets] + 1e-10))
    print(f"    final loss: {final_loss:.4f}")
    return lora, final_loss


# ===========================================================================
# Synthetic Data Generation
# ===========================================================================

def generate_cluster_prototypes(rng, cross_cluster_distance=3.0):
    prototypes = {}
    cluster_names = list(CLUSTERS.keys())
    for i, name in enumerate(cluster_names):
        logits = rng.randn(VOCAB_SIZE, VOCAB_SIZE) * cross_cluster_distance
        group_start = (i * VOCAB_SIZE) // N_CLUSTERS
        group_end = ((i + 1) * VOCAB_SIZE) // N_CLUSTERS
        logits[:, group_start:group_end] += 2.5
        logits += np.eye(VOCAB_SIZE) * 0.8
        prototypes[name] = softmax(logits, axis=-1)
    return prototypes


def generate_domain_data(rng, prototypes, domain_name, cluster_name,
                         n_sequences=400, noise_scale=0.15):
    prototype = prototypes[cluster_name]
    noise = rng.randn(*prototype.shape) * noise_scale
    domain_seed = hash(domain_name) % (2**31)
    domain_rng = np.random.RandomState(domain_seed)
    domain_bias = domain_rng.randn(*prototype.shape) * noise_scale * 0.5
    domain_idx = ALL_DOMAINS.index(domain_name)
    for j in range(3):
        c = (domain_idx * 2 + j) % VOCAB_SIZE
        domain_bias[:, c] += 0.4
    transition = softmax(np.log(prototype + 1e-10) + noise + domain_bias,
                         axis=-1)
    seq_len = CONTEXT_LEN + 1
    sequences = np.zeros((n_sequences, seq_len), dtype=np.int32)
    for i in range(n_sequences):
        sequences[i, 0] = rng.choice(VOCAB_SIZE)
        for t in range(seq_len - 1):
            prev = sequences[i, t]
            sequences[i, t + 1] = rng.choice(VOCAB_SIZE, p=transition[prev])
    return sequences[:, :-1], sequences[:, -1]


# ===========================================================================
# Composition Strategies
# ===========================================================================

def premerge_deltas(delta_list, strategy='average'):
    N = len(delta_list)
    merged = []
    for l in range(N_LAYERS):
        dW1 = sum(d[l]['dW1'] for d in delta_list)
        dW2 = sum(d[l]['dW2'] for d in delta_list)
        if strategy == 'average':
            dW1 = dW1 / N
            dW2 = dW2 / N
        merged.append({'dW1': dW1, 'dW2': dW2})
    return merged


def cosine_route_topk(query_emb, expert_centroids, expert_names, k=1):
    centroid_matrix = np.stack(
        [expert_centroids[n] for n in expert_names])
    q_norm = np.linalg.norm(query_emb) + 1e-10
    c_norms = np.linalg.norm(centroid_matrix, axis=1) + 1e-10
    sims = (query_emb / q_norm) @ (centroid_matrix / c_norms[:, None]).T
    top_indices = np.argsort(sims)[-k:][::-1]
    weight = 1.0 / k
    return [(expert_names[i], weight) for i in top_indices]


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_ntp_loss(model, x, y, lora=None):
    logits, _, _ = model.forward(x, lora)
    probs = softmax(logits)
    B = x.shape[0]
    target_probs = probs[np.arange(B), y]
    return float(-np.mean(np.log(target_probs + 1e-10)))


def evaluate_ntp_loss_with_deltas(model, x, y, delta_weights):
    logits = model.forward_with_delta(x, delta_weights)
    probs = softmax(logits)
    B = x.shape[0]
    target_probs = probs[np.arange(B), y]
    return float(-np.mean(np.log(target_probs + 1e-10)))


def merge_loras(lora_list, weights=None):
    if weights is None:
        weights = np.ones(len(lora_list)) / len(lora_list)
    merged = {
        'A1': [np.zeros_like(lora_list[0]['A1'][l]) for l in range(N_LAYERS)],
        'B1': [np.zeros_like(lora_list[0]['B1'][l]) for l in range(N_LAYERS)],
        'A2': [np.zeros_like(lora_list[0]['A2'][l]) for l in range(N_LAYERS)],
        'B2': [np.zeros_like(lora_list[0]['B2'][l]) for l in range(N_LAYERS)],
    }
    for i, lora in enumerate(lora_list):
        w = weights[i]
        for l in range(N_LAYERS):
            merged['A1'][l] += w * lora['A1'][l]
            merged['B1'][l] += w * lora['B1'][l]
            merged['A2'][l] += w * lora['A2'][l]
            merged['B2'][l] += w * lora['B2'][l]
    return merged


def evaluate_ntp_loss_topk(model, x, y, expert_loras, expert_names,
                           expert_centroids, k=1):
    """NTP loss with dynamic top-k routing per query."""
    B = x.shape[0]
    losses = []
    for i in range(B):
        xi = x[i:i+1]
        yi = y[i:i+1]
        emb = model.embed(xi)[0]
        selected = cosine_route_topk(emb, expert_centroids, expert_names, k=k)
        if k == 1:
            name = selected[0][0]
            loss = evaluate_ntp_loss(model, xi, yi, expert_loras[name])
        else:
            lora_list = [expert_loras[name] for name, _ in selected]
            weights = np.array([w for _, w in selected])
            merged = merge_loras(lora_list, weights)
            loss = evaluate_ntp_loss(model, xi, yi, merged)
        losses.append(loss)
    return float(np.mean(losses))


# ===========================================================================
# Main Experiment
# ===========================================================================

def run_experiment(seed=42, n_seeds=3, pretrain_steps=2000, train_steps=500,
                   pretrain_lr=0.005, train_lr=0.01,
                   n_train_sequences=400, n_test_sequences=100,
                   batch_size=32):
    results_dir = Path(__file__).parent
    t_start = time.time()

    print("=" * 72)
    print("  Pre-Merge vs Dynamic Routing Quality Comparison")
    print("  K1: pre-merge >5% worse than dynamic top-k -> pre-merge bad")
    print("  K2: dynamic provides ZERO advantage at N<20 -> routing moot")
    print(f"  Config: {n_seeds} seeds, base pretrain={pretrain_steps} steps, "
          f"LoRA train={train_steps} steps")
    print(f"  N values: {N_VALUES}")
    print(f"  Architecture: {N_LAYERS}-layer MLP, d={D_MODEL}, d_ff={D_FF}, "
          f"rank={LORA_RANK}")
    print("=" * 72)

    all_seed_results = []

    for seed_idx in range(n_seeds):
        current_seed = seed + seed_idx * 100
        print(f"\n{'='*72}")
        print(f"  SEED {seed_idx + 1}/{n_seeds} (seed={current_seed})")
        print(f"{'='*72}")

        rng = np.random.RandomState(current_seed)
        model = MicroMLP(rng)
        prototypes = generate_cluster_prototypes(rng)

        # Generate data for all 20 domains
        print(f"\n  Generating data for {len(ALL_DOMAINS)} domains...")
        domain_data_train = {}
        domain_data_test = {}
        for domain in ALL_DOMAINS:
            cluster = DOMAIN_TO_CLUSTER[domain]
            total = n_train_sequences + n_test_sequences
            x, y = generate_domain_data(rng, prototypes, domain, cluster,
                                        n_sequences=total)
            domain_data_train[domain] = (x[:n_train_sequences],
                                         y[:n_train_sequences])
            domain_data_test[domain] = (x[n_train_sequences:],
                                        y[n_train_sequences:])

        # Pre-train base model on ALL domain data mixed
        print(f"\n  Phase 1: Pre-training base model...")
        all_train_x = np.concatenate(
            [domain_data_train[d][0] for d in ALL_DOMAINS], axis=0)
        all_train_y = np.concatenate(
            [domain_data_train[d][1] for d in ALL_DOMAINS], axis=0)
        # Shuffle
        perm = rng.permutation(all_train_x.shape[0])
        all_train_x = all_train_x[perm]
        all_train_y = all_train_y[perm]

        base_pretrain_loss = pretrain_base(
            model, all_train_x, all_train_y, rng,
            steps=pretrain_steps, lr=pretrain_lr, batch_size=batch_size * 2)

        # Evaluate base model per domain
        print(f"\n  Base model per-domain losses after pretraining:")
        base_losses = {}
        for domain in ALL_DOMAINS:
            x_test, y_test = domain_data_test[domain]
            base_losses[domain] = evaluate_ntp_loss(model, x_test, y_test)
        mean_base_loss = np.mean(list(base_losses.values()))
        print(f"    Mean base test loss: {mean_base_loss:.4f}")

        # Train LoRA experts per domain (base is frozen)
        print(f"\n  Phase 2: Training {len(ALL_DOMAINS)} LoRA experts...")
        all_loras = {}
        all_deltas = {}
        all_train_losses = {}
        for i, domain in enumerate(ALL_DOMAINS):
            cluster = DOMAIN_TO_CLUSTER[domain]
            print(f"\n  [{i+1}/{len(ALL_DOMAINS)}] {domain} ({cluster})")
            x_train, y_train = domain_data_train[domain]
            lora, final_loss = train_lora(
                model, x_train, y_train, rng,
                steps=train_steps, lr=train_lr, batch_size=batch_size,
                log_every=max(1, train_steps // 3),
            )
            all_loras[domain] = lora
            all_deltas[domain] = lora_to_delta_weights(lora)
            all_train_losses[domain] = final_loss

        # Expert centroids
        expert_centroids = {}
        for domain in ALL_DOMAINS:
            x_train, _ = domain_data_train[domain]
            expert_centroids[domain] = model.embed(x_train).mean(axis=0)

        # Check specialization
        print(f"\n  Checking expert specialization...")
        oracle_losses = {}
        for domain in ALL_DOMAINS:
            x_test, y_test = domain_data_test[domain]
            oracle_losses[domain] = evaluate_ntp_loss(
                model, x_test, y_test, all_loras[domain])
        mean_oracle = np.mean(list(oracle_losses.values()))
        specialization_gap = (mean_base_loss - mean_oracle) / mean_base_loss * 100
        print(f"    Base loss:      {mean_base_loss:.4f}")
        print(f"    Oracle loss:    {mean_oracle:.4f}")
        print(f"    Specialization: {specialization_gap:.1f}% improvement")

        # Test each N value
        n_results = {}
        for N in N_VALUES:
            print(f"\n  {'='*60}")
            print(f"  N = {N} experts")
            print(f"  {'='*60}")

            selected_domains = ALL_DOMAINS[:N]
            selected_loras = {d: all_loras[d] for d in selected_domains}
            selected_deltas = [all_deltas[d] for d in selected_domains]
            selected_centroids = {d: expert_centroids[d]
                                  for d in selected_domains}

            # -- Pre-merge --
            merged_deltas = premerge_deltas(selected_deltas, strategy='average')
            premerge_losses = {}
            for domain in selected_domains:
                x_test, y_test = domain_data_test[domain]
                premerge_losses[domain] = evaluate_ntp_loss_with_deltas(
                    model, x_test, y_test, merged_deltas)
            premerge_mean = np.mean(list(premerge_losses.values()))

            # -- Dynamic top-1 --
            top1_losses = {}
            for domain in selected_domains:
                x_test, y_test = domain_data_test[domain]
                top1_losses[domain] = evaluate_ntp_loss_topk(
                    model, x_test, y_test, selected_loras,
                    selected_domains, selected_centroids, k=1)
            top1_mean = np.mean(list(top1_losses.values()))

            # -- Dynamic top-2 --
            top2_losses = {}
            for domain in selected_domains:
                x_test, y_test = domain_data_test[domain]
                top2_losses[domain] = evaluate_ntp_loss_topk(
                    model, x_test, y_test, selected_loras,
                    selected_domains, selected_centroids, k=min(2, N))
            top2_mean = np.mean(list(top2_losses.values()))

            # -- Oracle --
            oracle_losses_n = {}
            for domain in selected_domains:
                x_test, y_test = domain_data_test[domain]
                oracle_losses_n[domain] = evaluate_ntp_loss(
                    model, x_test, y_test, all_loras[domain])
            oracle_mean = np.mean(list(oracle_losses_n.values()))

            # -- Base --
            base_losses_n = {}
            for domain in selected_domains:
                base_losses_n[domain] = base_losses[domain]
            base_mean = np.mean(list(base_losses_n.values()))

            # Quality gaps
            pm_vs_t1 = (premerge_mean - top1_mean) / top1_mean * 100
            pm_vs_t2 = (premerge_mean - top2_mean) / top2_mean * 100
            pm_vs_oracle = (premerge_mean - oracle_mean) / oracle_mean * 100

            print(f"\n  N={N} SUMMARY (in-domain):")
            print(f"    {'Strategy':<20s} {'Loss':>10s} {'vs Oracle':>12s}")
            print(f"    {'-'*44}")
            print(f"    {'Base':<20s} {base_mean:>10.4f} "
                  f"{(base_mean - oracle_mean) / oracle_mean * 100:>+11.2f}%")
            print(f"    {'Pre-merge (1/N)':<20s} {premerge_mean:>10.4f} "
                  f"{pm_vs_oracle:>+11.2f}%")
            print(f"    {'Dynamic top-1':<20s} {top1_mean:>10.4f} "
                  f"{(top1_mean - oracle_mean) / oracle_mean * 100:>+11.2f}%")
            print(f"    {'Dynamic top-2':<20s} {top2_mean:>10.4f} "
                  f"{(top2_mean - oracle_mean) / oracle_mean * 100:>+11.2f}%")
            print(f"    {'Oracle':<20s} {oracle_mean:>10.4f}       +0.00%")
            print(f"\n    Pre-merge vs top-1: {pm_vs_t1:+.2f}%")
            print(f"    Pre-merge vs top-2: {pm_vs_t2:+.2f}%")

            n_results[N] = {
                'n_experts': N,
                'selected_domains': selected_domains,
                'premerge_in_domain': float(premerge_mean),
                'top1_in_domain': float(top1_mean),
                'top2_in_domain': float(top2_mean),
                'oracle_in_domain': float(oracle_mean),
                'base_in_domain': float(base_mean),
                'premerge_vs_top1_pct': float(pm_vs_t1),
                'premerge_vs_top2_pct': float(pm_vs_t2),
                'premerge_vs_oracle_pct': float(pm_vs_oracle),
                'per_domain_premerge': {k: float(v) for k, v
                                        in premerge_losses.items()},
                'per_domain_top1': {k: float(v) for k, v
                                    in top1_losses.items()},
                'per_domain_oracle': {k: float(v) for k, v
                                      in oracle_losses_n.items()},
            }

        all_seed_results.append({
            'seed': current_seed,
            'n_results': {str(k): v for k, v in n_results.items()},
            'specialization_gap_pct': float(specialization_gap),
            'base_pretrain_loss': float(base_pretrain_loss),
            'mean_base_test_loss': float(mean_base_loss),
            'train_losses': {k: float(v) for k, v in all_train_losses.items()},
        })

    # ===================================================================
    # Aggregate
    # ===================================================================
    elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  AGGREGATE RESULTS ({n_seeds} seeds)")
    print(f"{'='*72}")

    print(f"\n  {'N':>4s} | {'Pre-merge':>12s} | {'Top-1':>12s} | {'Top-2':>12s} | "
          f"{'Oracle':>12s} | {'Base':>12s} | {'PM vs T1':>10s} | {'PM vs T2':>10s}")
    print(f"  {'-'*98}")

    agg_by_n = {}
    for N in N_VALUES:
        pm_vals = [r['n_results'][str(N)]['premerge_in_domain']
                   for r in all_seed_results]
        t1_vals = [r['n_results'][str(N)]['top1_in_domain']
                   for r in all_seed_results]
        t2_vals = [r['n_results'][str(N)]['top2_in_domain']
                   for r in all_seed_results]
        or_vals = [r['n_results'][str(N)]['oracle_in_domain']
                   for r in all_seed_results]
        ba_vals = [r['n_results'][str(N)]['base_in_domain']
                   for r in all_seed_results]
        pm_vs_t1 = [r['n_results'][str(N)]['premerge_vs_top1_pct']
                    for r in all_seed_results]
        pm_vs_t2 = [r['n_results'][str(N)]['premerge_vs_top2_pct']
                    for r in all_seed_results]

        agg_by_n[N] = {
            'premerge_mean': float(np.mean(pm_vals)),
            'premerge_std': float(np.std(pm_vals)),
            'top1_mean': float(np.mean(t1_vals)),
            'top1_std': float(np.std(t1_vals)),
            'top2_mean': float(np.mean(t2_vals)),
            'top2_std': float(np.std(t2_vals)),
            'oracle_mean': float(np.mean(or_vals)),
            'oracle_std': float(np.std(or_vals)),
            'base_mean': float(np.mean(ba_vals)),
            'pm_vs_top1_pct_mean': float(np.mean(pm_vs_t1)),
            'pm_vs_top1_pct_std': float(np.std(pm_vs_t1)),
            'pm_vs_top2_pct_mean': float(np.mean(pm_vs_t2)),
            'pm_vs_top2_pct_std': float(np.std(pm_vs_t2)),
        }

        a = agg_by_n[N]
        print(f"  {N:>4d} | {a['premerge_mean']:>12.4f} | {a['top1_mean']:>12.4f} | "
              f"{a['top2_mean']:>12.4f} | {a['oracle_mean']:>12.4f} | "
              f"{a['base_mean']:>12.4f} | {a['pm_vs_top1_pct_mean']:>+9.2f}% | "
              f"{a['pm_vs_top2_pct_mean']:>+9.2f}%")

    # ===================================================================
    # Kill Criteria
    # ===================================================================
    print(f"\n{'='*72}")
    print(f"  KILL CRITERIA EVALUATION")
    print(f"{'='*72}")

    # K1: Pre-merged >5% worse than dynamic top-k
    k1_worst = max(agg_by_n[N]['pm_vs_top1_pct_mean'] for N in N_VALUES)
    k1_worst_n = max(N_VALUES, key=lambda N: agg_by_n[N]['pm_vs_top1_pct_mean'])
    k1_triggered = k1_worst > 5.0

    print(f"\n  K1: Pre-merged >5% worse than dynamic top-k?")
    for N in N_VALUES:
        gap = agg_by_n[N]['pm_vs_top1_pct_mean']
        flag = " ***" if gap > 5.0 else ""
        print(f"    N={N:>2d}: pre-merge vs top-1 = {gap:+.2f}% "
              f"(+/- {agg_by_n[N]['pm_vs_top1_pct_std']:.2f}){flag}")
    if k1_triggered:
        print(f"  STATUS: TRIGGERED at N={k1_worst_n} ({k1_worst:+.2f}%)")
    else:
        print(f"  STATUS: NOT TRIGGERED -- pre-merge within 5% at all N")

    # K2: Dynamic provides zero advantage at N<20
    k2_n_values = [N for N in N_VALUES if N < 20]
    k2_any_advantage = any(
        agg_by_n[N]['pm_vs_top1_pct_mean'] > 0.5 for N in k2_n_values)
    k2_triggered = not k2_any_advantage

    print(f"\n  K2: Dynamic provides zero advantage at N<20?")
    for N in k2_n_values:
        gap = agg_by_n[N]['pm_vs_top1_pct_mean']
        print(f"    N={N:>2d}: gap = {gap:+.2f}% "
              f"({'dynamic wins' if gap > 0.5 else 'no clear winner'})")
    if k2_triggered:
        print(f"  STATUS: TRIGGERED -- routing unnecessary at N<20")
    else:
        print(f"  STATUS: NOT TRIGGERED -- dynamic has advantage")

    # Verdict
    print(f"\n  {'='*60}")
    if k1_triggered and not k2_triggered:
        verdict = "Pre-merge degrades at high N, dynamic routing needed"
    elif k2_triggered and not k1_triggered:
        verdict = "KILLED -- dynamic routing provides NO advantage at N<20"
    elif k1_triggered and k2_triggered:
        verdict = "Mixed results -- crossover point exists"
    else:
        verdict = ("SUPPORTED -- pre-merge works but dynamic routing provides "
                   "measurable quality gain")
    print(f"  VERDICT: {verdict}")
    print(f"  {'='*60}")

    # Specialization
    spec_gaps = [r['specialization_gap_pct'] for r in all_seed_results]
    mean_spec = np.mean(spec_gaps)
    print(f"\n  Expert specialization: {mean_spec:.1f}% improvement over base")
    if mean_spec < 1.0:
        print(f"  WARNING: Experts did not specialize (<1%).")
        print(f"  Results may be inconclusive.")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save
    output = {
        'config': {
            'n_seeds': n_seeds,
            'pretrain_steps': pretrain_steps,
            'train_steps': train_steps,
            'pretrain_lr': pretrain_lr,
            'train_lr': train_lr,
            'n_train_sequences': n_train_sequences,
            'n_test_sequences': n_test_sequences,
            'batch_size': batch_size,
            'd_model': D_MODEL,
            'd_ff': D_FF,
            'n_layers': N_LAYERS,
            'lora_rank': LORA_RANK,
            'lora_alpha': LORA_ALPHA,
            'vocab_size': VOCAB_SIZE,
            'context_len': CONTEXT_LEN,
            'n_values': N_VALUES,
        },
        'clusters': {k: v for k, v in CLUSTERS.items()},
        'seed_results': all_seed_results,
        'aggregate_by_n': {str(k): v for k, v in agg_by_n.items()},
        'kill_criteria': {
            'k1_premerge_gt_5pct_worse': k1_triggered,
            'k1_worst_gap_pct': float(k1_worst),
            'k1_worst_n': k1_worst_n,
            'k2_dynamic_zero_advantage': k2_triggered,
            'specialization_gap_pct': float(mean_spec),
        },
        'verdict': verdict,
        'elapsed_seconds': elapsed,
    }

    output_file = results_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")

    return output


if __name__ == '__main__':
    run_experiment()
