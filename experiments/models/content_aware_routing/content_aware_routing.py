#!/usr/bin/env python3
"""
Content-Aware Routing: Does content-aware routing outperform hash ring?

Hypothesis: Content-aware routing strategies (tiny MLP classifier, cosine
similarity to expert centroids, keyword/pattern matching) outperform hash ring
routing on domain-specific queries, as measured by expert selection accuracy
and downstream NTP quality.

Design:
  1. Build a micro MLP base model (pure numpy, char-level NTP)
  2. Generate 15 synthetic domains in 3 semantic clusters (Markov chains)
  3. Train 15 LoRA adapters (one per domain)
  4. Implement 5 routing strategies:
     (a) Hash ring (baseline, content-agnostic)
     (b) Tiny MLP classifier (embed -> 1-layer MLP -> softmax)
     (c) Cosine similarity (query embedding vs expert centroids)
     (d) Keyword matching (character frequency pattern matching)
     (e) Oracle routing (perfect domain labels, upper bound)
  5. Measure: routing accuracy, per-query latency, NTP loss with routed expert

Kill criteria:
  K1: Content-aware routing accuracy <60% on domain classification
  K2: Content-aware routing latency >10ms per query
  K3: Hash ring + pre-merge matches content-aware quality (routing unnecessary)

Architecture: 4-layer MLP, d=64, d_ff=256, rank=8 LoRA on all MLP layers.
Pure numpy -- no MLX, no PyTorch.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np


# =============================================================================
# Constants
# =============================================================================

VOCAB_SIZE = 32
CONTEXT_LEN = 16
D_MODEL = 64
D_FF = 256
N_LAYERS = 4
LORA_RANK = 8
LORA_ALPHA = 8
N_DOMAINS = 15
N_CLUSTERS = 3

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

DOMAIN_TO_IDX = {d: i for i, d in enumerate(ALL_DOMAINS)}


# =============================================================================
# Utilities
# =============================================================================

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float64)


# =============================================================================
# Model: Multi-layer MLP with LoRA (identical to orthogonality_by_domain_type)
# =============================================================================

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
        """Get bag-of-words embedding for a batch of sequences.
        x_ids: (B, T) integer token ids
        Returns: (B, d) embedding
        """
        emb = self.wte[x_ids]       # (B, T, d)
        return emb.mean(axis=1)      # (B, d)

    def forward(self, x_ids, lora=None):
        """Forward pass.
        x_ids: (B, T) integer token ids
        lora: dict with 'A1', 'B1', 'A2', 'B2' lists (one per layer)
        Returns: logits (B, V), intermediates, final hidden state
        """
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


def train_lora(model, data_x, data_y, rng, steps=300, lr=0.01,
               batch_size=32, log_every=100):
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

            proj2 = a1 @ lora['A2'][l]
            d_B2 = scale * (proj2.T @ d_z2)
            lora['B2'][l] -= lr * d_B2

            W2_eff = layer['W2'] + scale * lora['A2'][l] @ lora['B2'][l]
            d_a1 = d_z2 @ W2_eff.T
            d_z1 = d_a1 * relu_grad(z1)

            proj1 = h_in @ lora['A1'][l]
            d_B1 = scale * (proj1.T @ d_z1)
            lora['B1'][l] -= lr * d_B1

    final_logits, _, _ = model.forward(data_x[:min(64, n_data)], lora)
    final_probs = softmax(final_logits)
    final_targets = data_y[:min(64, n_data)]
    final_loss = -np.mean(np.log(
        final_probs[np.arange(len(final_targets)), final_targets] + 1e-10))
    print(f"    final loss: {final_loss:.4f}")
    return lora, final_loss


# =============================================================================
# Synthetic Data Generation (from orthogonality_by_domain_type)
# =============================================================================

def generate_cluster_prototypes(rng, cross_cluster_distance=2.0):
    prototypes = {}
    cluster_names = ['code', 'reasoning', 'knowledge']
    for i, name in enumerate(cluster_names):
        logits = rng.randn(VOCAB_SIZE, VOCAB_SIZE) * cross_cluster_distance
        group_start = (i * VOCAB_SIZE) // N_CLUSTERS
        group_end = ((i + 1) * VOCAB_SIZE) // N_CLUSTERS
        logits[:, group_start:group_end] += 1.5
        logits += np.eye(VOCAB_SIZE) * 0.5
        prototypes[name] = softmax(logits, axis=-1)
    return prototypes


def generate_domain_data(rng, prototypes, domain_name, cluster_name,
                          n_sequences=200, noise_scale=0.15):
    prototype = prototypes[cluster_name]
    noise = rng.randn(*prototype.shape) * noise_scale
    domain_seed = hash(domain_name) % (2**31)
    domain_rng = np.random.RandomState(domain_seed)
    domain_bias = domain_rng.randn(*prototype.shape) * noise_scale * 0.5
    domain_idx = ALL_DOMAINS.index(domain_name)
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
    return sequences[:, :-1], sequences[:, -1]


# =============================================================================
# Routing Strategies
# =============================================================================

class HashRingRouter:
    """Consistent hash ring: content-agnostic routing.

    Maps query to a position on the hash ring via FNV1a hash of the raw
    token sequence. Each expert has virtual nodes on the ring. The closest
    virtual node determines the routed expert.
    """

    def __init__(self, expert_names, n_virtual=150, seed=42):
        self.expert_names = expert_names
        self.ring = {}  # position -> expert_name
        for name in expert_names:
            for v in range(n_virtual):
                key = f"{name}:{v}".encode()
                h = hashlib.md5(key).hexdigest()
                pos = int(h, 16) % (2**32)
                self.ring[pos] = name
        self.sorted_positions = sorted(self.ring.keys())

    def route(self, x_ids):
        """Route a single query (T,) int array -> expert name."""
        key = x_ids.tobytes()
        h = hashlib.md5(key).hexdigest()
        pos = int(h, 16) % (2**32)
        # Find closest position on ring
        idx = np.searchsorted(self.sorted_positions, pos)
        if idx >= len(self.sorted_positions):
            idx = 0
        return self.ring[self.sorted_positions[idx]]

    def route_batch(self, x_ids_batch):
        """Route a batch (B, T) -> list of expert names."""
        return [self.route(x_ids_batch[i]) for i in range(x_ids_batch.shape[0])]


class CosineRouter:
    """Route by cosine similarity between query embedding and expert centroids.

    Expert centroids are the mean embedding of that expert's training data,
    computed through the base model's embedding layer.
    """

    def __init__(self, model, expert_names, domain_data):
        self.expert_names = expert_names
        self.centroids = {}  # expert_name -> (d,) centroid
        for name in expert_names:
            x, _ = domain_data[name]
            emb = model.embed(x)  # (N, d)
            self.centroids[name] = emb.mean(axis=0)  # (d,)

    def route(self, x_ids, model):
        """Route a single query (T,) -> expert name."""
        emb = model.embed(x_ids.reshape(1, -1))[0]  # (d,)
        best_name = None
        best_sim = -np.inf
        for name, centroid in self.centroids.items():
            sim = np.dot(emb, centroid) / (
                np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-10)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name

    def route_batch(self, x_ids_batch, model):
        """Route a batch (B, T) -> list of expert names."""
        embs = model.embed(x_ids_batch)  # (B, d)
        centroid_matrix = np.stack(
            [self.centroids[n] for n in self.expert_names])  # (N_exp, d)
        # Normalize
        emb_norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        cent_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-10
        sims = (embs / emb_norms) @ (centroid_matrix / cent_norms).T  # (B, N_exp)
        best_indices = sims.argmax(axis=1)
        return [self.expert_names[i] for i in best_indices]


class MLPRouter:
    """Tiny 1-layer MLP classifier: embed -> Linear(d, N_experts) -> softmax.

    Trained on (embedding, domain_label) pairs from the training data.
    """

    def __init__(self, model, expert_names, domain_data, rng,
                 train_steps=500, lr=0.05, batch_size=64):
        self.expert_names = expert_names
        n_experts = len(expert_names)

        # Collect training data: (embedding, domain_idx)
        all_embs = []
        all_labels = []
        for name in expert_names:
            x, _ = domain_data[name]
            emb = model.embed(x)  # (N, d)
            label = DOMAIN_TO_IDX[name]
            all_embs.append(emb)
            all_labels.extend([label] * emb.shape[0])

        all_embs = np.concatenate(all_embs, axis=0)   # (total, d)
        all_labels = np.array(all_labels, dtype=np.int32)  # (total,)

        # Initialize: W (d, N_experts), b (N_experts,)
        self.W = rng.randn(D_MODEL, n_experts) * 0.01
        self.b = np.zeros(n_experts)

        # Train
        n_data = all_embs.shape[0]
        for step in range(train_steps):
            idx = rng.choice(n_data, size=min(batch_size, n_data), replace=False)
            bx = all_embs[idx]         # (B, d)
            by = all_labels[idx]       # (B,)
            B_sz = bx.shape[0]

            logits = bx @ self.W + self.b  # (B, N_experts)
            probs = softmax(logits)
            loss = -np.mean(np.log(probs[np.arange(B_sz), by] + 1e-10))

            # Backprop
            d_logits = probs.copy()
            d_logits[np.arange(B_sz), by] -= 1.0
            d_logits /= B_sz

            d_W = bx.T @ d_logits          # (d, N_experts)
            d_b = d_logits.sum(axis=0)      # (N_experts,)

            self.W -= lr * d_W
            self.b -= lr * d_b

            if step % 100 == 0:
                acc = (logits.argmax(axis=1) == by).mean()
                print(f"    MLP router step {step}: loss={loss:.4f}, acc={acc:.3f}")

        # Final accuracy
        logits = all_embs @ self.W + self.b
        preds = logits.argmax(axis=1)
        self.train_acc = (preds == all_labels).mean()
        print(f"    MLP router final train acc: {self.train_acc:.4f}")

    def route(self, x_ids, model):
        emb = model.embed(x_ids.reshape(1, -1))[0]
        logits = emb @ self.W + self.b
        return self.expert_names[logits.argmax()]

    def route_batch(self, x_ids_batch, model):
        embs = model.embed(x_ids_batch)
        logits = embs @ self.W + self.b  # (B, N_experts)
        indices = logits.argmax(axis=1)
        return [self.expert_names[i] for i in indices]


class KeywordRouter:
    """Pattern-based routing: character frequency profile matching.

    Each domain has a characteristic frequency profile computed from training
    data. Routing matches the query's character frequencies to the nearest
    profile via L2 distance.
    """

    def __init__(self, expert_names, domain_data):
        self.expert_names = expert_names
        self.profiles = {}
        for name in expert_names:
            x, _ = domain_data[name]
            # Character frequency histogram normalized per sequence
            freqs = np.zeros(VOCAB_SIZE, dtype=np.float64)
            for i in range(x.shape[0]):
                for t in range(x.shape[1]):
                    freqs[x[i, t]] += 1
            freqs /= (freqs.sum() + 1e-10)
            self.profiles[name] = freqs

    def route(self, x_ids):
        freq = np.zeros(VOCAB_SIZE, dtype=np.float64)
        for t in range(x_ids.shape[0]):
            freq[x_ids[t]] += 1
        freq /= (freq.sum() + 1e-10)

        best_name = None
        best_dist = np.inf
        for name, profile in self.profiles.items():
            dist = np.sum((freq - profile) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    def route_batch(self, x_ids_batch):
        profile_matrix = np.stack(
            [self.profiles[n] for n in self.expert_names])  # (N_exp, V)
        B = x_ids_batch.shape[0]
        freqs = np.zeros((B, VOCAB_SIZE), dtype=np.float64)
        for i in range(B):
            for t in range(x_ids_batch.shape[1]):
                freqs[i, x_ids_batch[i, t]] += 1
        freqs /= (freqs.sum(axis=1, keepdims=True) + 1e-10)
        # L2 distance
        dists = np.sum((freqs[:, None, :] - profile_matrix[None, :, :]) ** 2,
                        axis=2)  # (B, N_exp)
        indices = dists.argmin(axis=1)
        return [self.expert_names[i] for i in indices]


class OracleRouter:
    """Perfect routing: knows the true domain label. Upper bound."""

    def route(self, domain_name):
        return domain_name

    def route_batch(self, domain_names):
        return list(domain_names)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_ntp_loss(model, x, y, lora=None):
    """Compute NTP cross-entropy loss on (x, y) data."""
    logits, _, _ = model.forward(x, lora)
    probs = softmax(logits)
    B = x.shape[0]
    target_probs = probs[np.arange(B), y]
    return -np.mean(np.log(target_probs + 1e-10))


def merge_loras(lora_list, weights=None):
    """Weighted average of multiple LoRA adapters (pre-merge)."""
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


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(seed=42, n_seeds=3, train_steps=300, router_train_steps=500,
                   train_lr=0.01, n_sequences=300, batch_size=32,
                   n_test_sequences=100):
    """Run the full content-aware routing comparison."""
    results_dir = Path(__file__).parent
    t_start = time.time()

    print("=" * 72)
    print("  Content-Aware Routing Experiment")
    print("  Hypothesis: content-aware routing outperforms hash ring")
    print(f"  Config: {n_seeds} seeds, {train_steps} train steps, "
          f"{router_train_steps} router train steps")
    print(f"  Architecture: 4-layer MLP, d={D_MODEL}, d_ff={D_FF}, "
          f"rank={LORA_RANK}")
    print("=" * 72)

    all_seed_results = []

    for seed_idx in range(n_seeds):
        current_seed = seed + seed_idx * 100
        print(f"\n{'='*72}")
        print(f"  SEED {seed_idx + 1}/{n_seeds} (seed={current_seed})")
        print(f"{'='*72}")

        rng = np.random.RandomState(current_seed)

        # ── Build base model ──
        print("\n  Building base model...")
        model = MicroMLP(rng)

        # ── Generate data ──
        print("  Generating cluster prototypes...")
        prototypes = generate_cluster_prototypes(rng)

        domain_data = {}  # domain -> (x_train, y_train)
        domain_test = {}  # domain -> (x_test, y_test)
        print(f"\n  Generating data for {N_DOMAINS} domains...")
        for domain in ALL_DOMAINS:
            cluster = DOMAIN_TO_CLUSTER[domain]
            x, y = generate_domain_data(rng, prototypes, domain, cluster,
                                         n_sequences=n_sequences + n_test_sequences)
            domain_data[domain] = (x[:n_sequences], y[:n_sequences])
            domain_test[domain] = (x[n_sequences:], y[n_sequences:])

        # ── Train LoRA experts ──
        print(f"\n  Training {N_DOMAINS} LoRA experts...")
        domain_loras = {}
        domain_train_losses = {}
        for i, domain in enumerate(ALL_DOMAINS):
            cluster = DOMAIN_TO_CLUSTER[domain]
            print(f"\n  [{i+1}/{N_DOMAINS}] {domain} ({cluster})")
            x_train, y_train = domain_data[domain]
            lora, final_loss = train_lora(
                model, x_train, y_train, rng,
                steps=train_steps, lr=train_lr, batch_size=batch_size,
                log_every=max(1, train_steps // 3),
            )
            domain_loras[domain] = lora
            domain_train_losses[domain] = final_loss

        # ── Build routers ──
        print(f"\n  Building routers...")

        print("  [1/4] Hash ring router...")
        hash_router = HashRingRouter(ALL_DOMAINS, seed=current_seed)

        print("  [2/4] Cosine similarity router...")
        cosine_router = CosineRouter(model, ALL_DOMAINS, domain_data)

        print("  [3/4] MLP classifier router...")
        mlp_router = MLPRouter(model, ALL_DOMAINS, domain_data, rng,
                                train_steps=router_train_steps, lr=0.05)

        print("  [4/4] Keyword frequency router...")
        keyword_router = KeywordRouter(ALL_DOMAINS, domain_data)

        oracle_router = OracleRouter()

        # ── Evaluate routing ──
        print(f"\n  Evaluating routing strategies...")

        # Build combined test set with labels
        all_test_x = []
        all_test_y = []
        all_test_domains = []
        for domain in ALL_DOMAINS:
            x_test, y_test = domain_test[domain]
            all_test_x.append(x_test)
            all_test_y.append(y_test)
            all_test_domains.extend([domain] * x_test.shape[0])
        all_test_x = np.concatenate(all_test_x, axis=0)
        all_test_y = np.concatenate(all_test_y, axis=0)
        n_test = all_test_x.shape[0]

        results = {}

        # --- Strategy (a): Hash ring ---
        print("  Evaluating hash ring...")
        t0 = time.time()
        hash_routes = hash_router.route_batch(all_test_x)
        hash_latency = (time.time() - t0) / n_test
        hash_acc = sum(1 for i in range(n_test) if hash_routes[i] == all_test_domains[i]) / n_test
        # Compute NTP loss with hash-routed expert
        hash_losses = []
        for i in range(n_test):
            xi = all_test_x[i:i+1]
            yi = all_test_y[i:i+1]
            routed_expert = hash_routes[i]
            loss = evaluate_ntp_loss(model, xi, yi, domain_loras[routed_expert])
            hash_losses.append(loss)
        hash_mean_loss = np.mean(hash_losses)
        results['hash_ring'] = {
            'accuracy': float(hash_acc),
            'latency_us': float(hash_latency * 1e6),
            'mean_ntp_loss': float(hash_mean_loss),
        }
        print(f"    Hash ring: acc={hash_acc:.4f}, "
              f"latency={hash_latency*1e6:.1f}us, loss={hash_mean_loss:.4f}")

        # --- Strategy (b): MLP classifier ---
        print("  Evaluating MLP classifier...")
        t0 = time.time()
        mlp_routes = mlp_router.route_batch(all_test_x, model)
        mlp_latency = (time.time() - t0) / n_test
        mlp_acc = sum(1 for i in range(n_test) if mlp_routes[i] == all_test_domains[i]) / n_test
        mlp_losses = []
        for i in range(n_test):
            xi = all_test_x[i:i+1]
            yi = all_test_y[i:i+1]
            routed_expert = mlp_routes[i]
            loss = evaluate_ntp_loss(model, xi, yi, domain_loras[routed_expert])
            mlp_losses.append(loss)
        mlp_mean_loss = np.mean(mlp_losses)
        results['mlp_classifier'] = {
            'accuracy': float(mlp_acc),
            'latency_us': float(mlp_latency * 1e6),
            'mean_ntp_loss': float(mlp_mean_loss),
            'train_acc': float(mlp_router.train_acc),
        }
        print(f"    MLP classifier: acc={mlp_acc:.4f}, "
              f"latency={mlp_latency*1e6:.1f}us, loss={mlp_mean_loss:.4f}")

        # --- Strategy (c): Cosine similarity ---
        print("  Evaluating cosine similarity...")
        t0 = time.time()
        cosine_routes = cosine_router.route_batch(all_test_x, model)
        cosine_latency = (time.time() - t0) / n_test
        cosine_acc = sum(1 for i in range(n_test) if cosine_routes[i] == all_test_domains[i]) / n_test
        cosine_losses = []
        for i in range(n_test):
            xi = all_test_x[i:i+1]
            yi = all_test_y[i:i+1]
            routed_expert = cosine_routes[i]
            loss = evaluate_ntp_loss(model, xi, yi, domain_loras[routed_expert])
            cosine_losses.append(loss)
        cosine_mean_loss = np.mean(cosine_losses)
        results['cosine_similarity'] = {
            'accuracy': float(cosine_acc),
            'latency_us': float(cosine_latency * 1e6),
            'mean_ntp_loss': float(cosine_mean_loss),
        }
        print(f"    Cosine sim: acc={cosine_acc:.4f}, "
              f"latency={cosine_latency*1e6:.1f}us, loss={cosine_mean_loss:.4f}")

        # --- Strategy (d): Keyword matching ---
        print("  Evaluating keyword frequency matching...")
        t0 = time.time()
        keyword_routes = keyword_router.route_batch(all_test_x)
        keyword_latency = (time.time() - t0) / n_test
        keyword_acc = sum(1 for i in range(n_test) if keyword_routes[i] == all_test_domains[i]) / n_test
        keyword_losses = []
        for i in range(n_test):
            xi = all_test_x[i:i+1]
            yi = all_test_y[i:i+1]
            routed_expert = keyword_routes[i]
            loss = evaluate_ntp_loss(model, xi, yi, domain_loras[routed_expert])
            keyword_losses.append(loss)
        keyword_mean_loss = np.mean(keyword_losses)
        results['keyword_matching'] = {
            'accuracy': float(keyword_acc),
            'latency_us': float(keyword_latency * 1e6),
            'mean_ntp_loss': float(keyword_mean_loss),
        }
        print(f"    Keyword: acc={keyword_acc:.4f}, "
              f"latency={keyword_latency*1e6:.1f}us, loss={keyword_mean_loss:.4f}")

        # --- Strategy (e): Oracle ---
        print("  Evaluating oracle routing...")
        oracle_routes = all_test_domains  # perfect routing
        oracle_losses = []
        for i in range(n_test):
            xi = all_test_x[i:i+1]
            yi = all_test_y[i:i+1]
            routed_expert = oracle_routes[i]
            loss = evaluate_ntp_loss(model, xi, yi, domain_loras[routed_expert])
            oracle_losses.append(loss)
        oracle_mean_loss = np.mean(oracle_losses)
        results['oracle'] = {
            'accuracy': 1.0,
            'latency_us': 0.0,
            'mean_ntp_loss': float(oracle_mean_loss),
        }
        print(f"    Oracle: acc=1.0000, loss={oracle_mean_loss:.4f}")

        # --- Baseline: no expert (base model only) ---
        print("  Evaluating base model (no expert)...")
        base_losses = []
        for i in range(n_test):
            xi = all_test_x[i:i+1]
            yi = all_test_y[i:i+1]
            loss = evaluate_ntp_loss(model, xi, yi, lora=None)
            base_losses.append(loss)
        base_mean_loss = np.mean(base_losses)
        results['base_model'] = {
            'accuracy': None,
            'latency_us': 0.0,
            'mean_ntp_loss': float(base_mean_loss),
        }
        print(f"    Base model: loss={base_mean_loss:.4f}")

        # --- Baseline: pre-merge all experts ---
        print("  Evaluating pre-merge (all experts averaged)...")
        merged_lora = merge_loras(
            [domain_loras[d] for d in ALL_DOMAINS])
        merge_losses = []
        for i in range(n_test):
            xi = all_test_x[i:i+1]
            yi = all_test_y[i:i+1]
            loss = evaluate_ntp_loss(model, xi, yi, merged_lora)
            merge_losses.append(loss)
        merge_mean_loss = np.mean(merge_losses)
        results['pre_merge_all'] = {
            'accuracy': None,
            'latency_us': 0.0,
            'mean_ntp_loss': float(merge_mean_loss),
        }
        print(f"    Pre-merge all: loss={merge_mean_loss:.4f}")

        # --- Cluster-level accuracy ---
        print("\n  Per-cluster routing accuracy:")
        for strategy_name, routes in [
            ('hash_ring', hash_routes),
            ('mlp_classifier', mlp_routes),
            ('cosine_similarity', cosine_routes),
            ('keyword_matching', keyword_routes),
        ]:
            print(f"    {strategy_name}:")
            for cname in ['code', 'reasoning', 'knowledge']:
                cluster_mask = [DOMAIN_TO_CLUSTER[d] == cname
                                for d in all_test_domains]
                cluster_indices = [i for i, m in enumerate(cluster_mask) if m]
                # Exact domain match
                exact_acc = sum(1 for i in cluster_indices
                                if routes[i] == all_test_domains[i]) / len(cluster_indices)
                # Cluster-level match (correct cluster even if wrong domain)
                cluster_acc = sum(
                    1 for i in cluster_indices
                    if DOMAIN_TO_CLUSTER.get(routes[i], '') == cname
                ) / len(cluster_indices)
                print(f"      {cname:12s}: domain_acc={exact_acc:.3f}, "
                      f"cluster_acc={cluster_acc:.3f}")

        all_seed_results.append({
            'seed': current_seed,
            'results': results,
            'domain_train_losses': {k: float(v)
                                     for k, v in domain_train_losses.items()},
        })

    # =================================================================
    # Aggregate across seeds
    # =================================================================
    elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  AGGREGATE RESULTS ({n_seeds} seeds)")
    print(f"{'='*72}")

    strategies = ['hash_ring', 'mlp_classifier', 'cosine_similarity',
                  'keyword_matching', 'oracle', 'base_model', 'pre_merge_all']

    agg = {}
    for s in strategies:
        accs = [r['results'][s].get('accuracy') for r in all_seed_results
                if r['results'][s].get('accuracy') is not None]
        losses = [r['results'][s]['mean_ntp_loss'] for r in all_seed_results]
        latencies = [r['results'][s]['latency_us'] for r in all_seed_results]
        agg[s] = {
            'mean_accuracy': float(np.mean(accs)) if accs else None,
            'std_accuracy': float(np.std(accs)) if accs else None,
            'mean_ntp_loss': float(np.mean(losses)),
            'std_ntp_loss': float(np.std(losses)),
            'mean_latency_us': float(np.mean(latencies)),
        }

    print(f"\n  {'Strategy':<22s} {'Accuracy':>10s} {'NTP Loss':>10s} "
          f"{'Latency':>12s}")
    print(f"  {'-'*56}")
    for s in strategies:
        a = agg[s]
        acc_str = f"{a['mean_accuracy']:.4f}" if a['mean_accuracy'] is not None else "N/A"
        lat_str = f"{a['mean_latency_us']:.1f}us" if a['mean_latency_us'] > 0 else "0"
        print(f"  {s:<22s} {acc_str:>10s} {a['mean_ntp_loss']:>10.4f} "
              f"{lat_str:>12s}")

    # =================================================================
    # Kill Criteria
    # =================================================================
    print(f"\n{'='*72}")
    print(f"  KILL CRITERIA EVALUATION")
    print(f"{'='*72}")

    # Best content-aware accuracy
    best_ca_name = max(['mlp_classifier', 'cosine_similarity', 'keyword_matching'],
                        key=lambda s: agg[s]['mean_accuracy'] or 0)
    best_ca_acc = agg[best_ca_name]['mean_accuracy']
    best_ca_latency = agg[best_ca_name]['mean_latency_us']
    best_ca_loss = agg[best_ca_name]['mean_ntp_loss']
    hash_loss = agg['hash_ring']['mean_ntp_loss']
    oracle_loss = agg['oracle']['mean_ntp_loss']
    merge_loss = agg['pre_merge_all']['mean_ntp_loss']

    # K1: accuracy < 60%
    k1_killed = best_ca_acc < 0.60
    print(f"\n  K1: Best content-aware routing accuracy <60%")
    print(f"      Best strategy: {best_ca_name} with acc={best_ca_acc:.4f}")
    if k1_killed:
        print(f"      STATUS: KILL -- accuracy {best_ca_acc:.4f} < 0.60")
    else:
        print(f"      STATUS: PASS -- accuracy {best_ca_acc:.4f} >= 0.60")

    # K2: latency > 10ms (10,000 us)
    k2_killed = best_ca_latency > 10000
    print(f"\n  K2: Content-aware routing latency >10ms per query")
    print(f"      Best strategy latency: {best_ca_latency:.1f}us")
    if k2_killed:
        print(f"      STATUS: KILL -- latency {best_ca_latency:.1f}us > 10,000us")
    else:
        print(f"      STATUS: PASS -- latency {best_ca_latency:.1f}us < 10,000us")

    # K3: hash ring + pre-merge matches content-aware quality
    # "matches" = hash ring loss within 5% of content-aware loss
    quality_gap = (hash_loss - best_ca_loss) / best_ca_loss * 100
    k3_killed = abs(quality_gap) < 5.0  # within 5% means routing unnecessary
    print(f"\n  K3: Hash ring + pre-merge matches content-aware quality")
    print(f"      Hash ring loss:       {hash_loss:.4f}")
    print(f"      Best content-aware:   {best_ca_loss:.4f}")
    print(f"      Oracle loss:          {oracle_loss:.4f}")
    print(f"      Pre-merge all loss:   {merge_loss:.4f}")
    print(f"      Quality gap:          {quality_gap:+.2f}%")
    if k3_killed:
        print(f"      STATUS: KILL -- gap only {quality_gap:+.2f}%, "
              f"routing unnecessary")
    else:
        print(f"      STATUS: PASS -- gap {quality_gap:+.2f}%, "
              f"content-aware routing adds value")

    overall_kill = k1_killed or k2_killed or k3_killed
    print(f"\n  {'='*60}")
    if overall_kill:
        killed_by = []
        if k1_killed:
            killed_by.append("K1 (accuracy)")
        if k2_killed:
            killed_by.append("K2 (latency)")
        if k3_killed:
            killed_by.append("K3 (quality gap)")
        print(f"  OVERALL VERDICT: KILL ({', '.join(killed_by)})")
    else:
        print(f"  OVERALL VERDICT: SUPPORTED")
        print(f"    Content-aware routing ({best_ca_name})")
        print(f"    Accuracy: {best_ca_acc:.4f}")
        print(f"    Quality improvement: {quality_gap:+.2f}% vs hash ring")
        print(f"    Latency: {best_ca_latency:.1f}us")
    print(f"  {'='*60}")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save
    output = {
        'config': {
            'n_seeds': n_seeds,
            'train_steps': train_steps,
            'router_train_steps': router_train_steps,
            'train_lr': train_lr,
            'n_sequences': n_sequences,
            'n_test_sequences': n_test_sequences,
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
        'aggregate': agg,
        'kill_criteria': {
            'k1_accuracy_pass': not k1_killed,
            'k2_latency_pass': not k2_killed,
            'k3_quality_gap_pass': not k3_killed,
            'overall_kill': overall_kill,
            'best_content_aware': best_ca_name,
            'best_ca_accuracy': float(best_ca_acc) if best_ca_acc else None,
            'best_ca_latency_us': float(best_ca_latency),
            'quality_gap_pct': float(quality_gap),
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
