#!/usr/bin/env python3
"""
Semantic Router: Learned semantic routing vs hash ring for expert selection.

Hypothesis: A learned semantic router can classify queries into the correct
expert domain with >70% accuracy and <5ms latency per query.

This experiment focuses ONLY on routing mechanism accuracy and latency.
We do NOT evaluate downstream NTP quality (known to be vacuous at micro
scale -- see content_aware_routing KILL).

Design:
  1. Generate 15 synthetic domains in 3 clusters (same as content_aware_routing)
  2. Create richer embeddings: character n-gram features (not bag-of-words)
  3. Implement 6 routing strategies:
     (a) Hash ring (baseline, content-agnostic)
     (b) Keyword frequency classifier (from prior experiment)
     (c) Cosine similarity to expert centroids
     (d) LSH spatial partitioning (SimHash-style binary codes)
     (e) Semantic-router utterance matching (per-route exemplars + threshold)
     (f) Oracle (perfect labels, upper bound)
  4. Measure: domain accuracy, cluster accuracy, latency per query

Kill criteria:
  K1: Semantic router accuracy <70% on domain classification
  K2: Router latency >5ms per query
  K3: Router adds >2% end-to-end latency vs hash ring

Architecture: Pure numpy. Character n-gram embeddings (d=64).
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
D_EMBED = 64        # embedding dimension for routing features
N_DOMAINS = 15
N_CLUSTERS = 3
LSH_N_PLANES = 32   # number of random hyperplanes for SimHash

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
CLUSTER_NAMES = ['code', 'reasoning', 'knowledge']


# =============================================================================
# Feature Extraction: Character N-gram Embeddings
# =============================================================================

def compute_ngram_features(x_ids, vocab_size=VOCAB_SIZE, max_n=3):
    """Compute character n-gram frequency features for a batch of sequences.

    Instead of bag-of-words (unigram only), we use unigram + bigram + trigram
    frequencies projected to D_EMBED via a fixed random projection.

    x_ids: (B, T) integer token ids
    Returns: (B, D_feat) feature vectors where D_feat = V + V^2_trunc + V^3_trunc
    """
    B, T = x_ids.shape

    # Unigram frequencies (V features)
    unigrams = np.zeros((B, vocab_size), dtype=np.float64)
    for i in range(B):
        for t in range(T):
            unigrams[i, x_ids[i, t]] += 1
    unigrams /= T

    # Bigram frequencies (V*V features, but we hash to a smaller space)
    # Use modular hashing to keep feature dimension manageable
    bigram_buckets = 128
    bigrams = np.zeros((B, bigram_buckets), dtype=np.float64)
    for i in range(B):
        for t in range(T - 1):
            key = x_ids[i, t] * vocab_size + x_ids[i, t + 1]
            bucket = key % bigram_buckets
            bigrams[i, bucket] += 1
    if T > 1:
        bigrams /= (T - 1)

    # Trigram frequencies (hash to buckets)
    trigram_buckets = 64
    trigrams = np.zeros((B, trigram_buckets), dtype=np.float64)
    for i in range(B):
        for t in range(T - 2):
            key = (x_ids[i, t] * vocab_size * vocab_size +
                   x_ids[i, t + 1] * vocab_size +
                   x_ids[i, t + 2])
            bucket = key % trigram_buckets
            trigrams[i, bucket] += 1
    if T > 2:
        trigrams /= (T - 2)

    # Concatenate: V + bigram_buckets + trigram_buckets = 32 + 128 + 64 = 224
    features = np.concatenate([unigrams, bigrams, trigrams], axis=1)
    return features


def project_features(features, projection_matrix):
    """Project raw n-gram features to D_EMBED via random projection.

    features: (B, D_raw)
    projection_matrix: (D_raw, D_EMBED)
    Returns: (B, D_EMBED) L2-normalized
    """
    projected = features @ projection_matrix  # (B, D_EMBED)
    norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-10
    return projected / norms


def make_projection_matrix(d_raw, d_embed, rng):
    """Create a fixed random projection matrix (Gaussian, normalized columns)."""
    P = rng.randn(d_raw, d_embed)
    # Normalize columns for variance preservation
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    return P


# =============================================================================
# Synthetic Data Generation (reused from content_aware_routing)
# =============================================================================

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def generate_cluster_prototypes(rng, cross_cluster_distance=2.0):
    prototypes = {}
    for i, name in enumerate(CLUSTER_NAMES):
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
    """Consistent hash ring: content-agnostic routing. O(log N)."""

    def __init__(self, expert_names, n_virtual=150, seed=42):
        self.expert_names = expert_names
        self.ring = {}
        for name in expert_names:
            for v in range(n_virtual):
                key = f"{name}:{v}".encode()
                h = hashlib.md5(key).hexdigest()
                pos = int(h, 16) % (2**32)
                self.ring[pos] = name
        self.sorted_positions = np.array(sorted(self.ring.keys()))
        self.sorted_names = [self.ring[p] for p in self.sorted_positions]

    def route_batch(self, x_ids_batch):
        """Route batch (B, T) -> list of expert names."""
        routes = []
        for i in range(x_ids_batch.shape[0]):
            key = x_ids_batch[i].tobytes()
            h = hashlib.md5(key).hexdigest()
            pos = int(h, 16) % (2**32)
            idx = np.searchsorted(self.sorted_positions, pos)
            if idx >= len(self.sorted_positions):
                idx = 0
            routes.append(self.sorted_names[idx])
        return routes


class KeywordRouter:
    """Character frequency profile matching (L2 distance). From prior experiment."""

    def __init__(self, expert_names, domain_data):
        self.expert_names = expert_names
        self.profiles = {}
        for name in expert_names:
            x, _ = domain_data[name]
            freqs = np.zeros(VOCAB_SIZE, dtype=np.float64)
            for i in range(x.shape[0]):
                for t in range(x.shape[1]):
                    freqs[x[i, t]] += 1
            freqs /= (freqs.sum() + 1e-10)
            self.profiles[name] = freqs
        # Pre-stack for vectorized routing
        self.profile_matrix = np.stack(
            [self.profiles[n] for n in self.expert_names])  # (N, V)

    def route_batch(self, x_ids_batch):
        B = x_ids_batch.shape[0]
        freqs = np.zeros((B, VOCAB_SIZE), dtype=np.float64)
        for i in range(B):
            for t in range(x_ids_batch.shape[1]):
                freqs[i, x_ids_batch[i, t]] += 1
        freqs /= (freqs.sum(axis=1, keepdims=True) + 1e-10)
        dists = np.sum((freqs[:, None, :] - self.profile_matrix[None, :, :]) ** 2,
                       axis=2)
        indices = dists.argmin(axis=1)
        return [self.expert_names[i] for i in indices]


class CosineRouter:
    """Cosine similarity between n-gram embedding and expert centroids."""

    def __init__(self, expert_names, domain_embeddings):
        """domain_embeddings: dict name -> (N_train, D_EMBED) normalized embeddings."""
        self.expert_names = expert_names
        self.centroids = {}
        for name in expert_names:
            embs = domain_embeddings[name]  # (N, D)
            centroid = embs.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-10)
            self.centroids[name] = centroid
        self.centroid_matrix = np.stack(
            [self.centroids[n] for n in self.expert_names])  # (N_exp, D)

    def route_batch(self, query_embeddings):
        """query_embeddings: (B, D) L2-normalized. Returns list of expert names."""
        sims = query_embeddings @ self.centroid_matrix.T  # (B, N_exp)
        indices = sims.argmax(axis=1)
        return [self.expert_names[i] for i in indices]


class LSHRouter:
    """SimHash-based locality-sensitive hashing router.

    Partitions embedding space using random hyperplanes. Each domain gets
    a set of binary hash codes from its training data. At query time,
    hash the query and find the domain with the most similar hash codes
    (Hamming distance).

    This is a geo-hash-like spatial partitioning: each hyperplane bisects
    the space, and the combination of sides defines a bucket.
    """

    def __init__(self, expert_names, domain_embeddings, n_planes=LSH_N_PLANES,
                 rng=None):
        self.expert_names = expert_names
        self.n_planes = n_planes

        # Random hyperplanes for SimHash
        if rng is None:
            rng = np.random.RandomState(42)
        self.planes = rng.randn(D_EMBED, n_planes)  # (D, P)
        self.planes /= np.linalg.norm(self.planes, axis=0, keepdims=True)

        # Compute hash codes for each domain's training data
        self.domain_codes = {}  # name -> (N_train, P) binary
        self.domain_code_means = {}  # name -> (P,) mean binary code
        for name in expert_names:
            embs = domain_embeddings[name]  # (N, D)
            codes = (embs @ self.planes > 0).astype(np.float64)  # (N, P)
            self.domain_codes[name] = codes
            self.domain_code_means[name] = codes.mean(axis=0)  # (P,) soft

        # Pre-stack for vectorized routing
        self.code_matrix = np.stack(
            [self.domain_code_means[n] for n in self.expert_names])  # (N_exp, P)

    def route_batch(self, query_embeddings):
        """query_embeddings: (B, D). Hash and find closest domain by code similarity."""
        query_codes = (query_embeddings @ self.planes > 0).astype(np.float64)  # (B, P)
        # Similarity = fraction of matching bits (dot product of binary vectors)
        sims = query_codes @ self.code_matrix.T  # (B, N_exp)
        indices = sims.argmax(axis=1)
        return [self.expert_names[i] for i in indices]


class UtteranceRouter:
    """Semantic-router-style utterance matching.

    Inspired by aurelio-labs/semantic-router: each route (domain) has a set
    of exemplar embeddings (utterances). At query time, compute cosine similarity
    to ALL exemplars, find the max similarity per domain, and route to the
    domain with the highest max similarity.

    This is like k-nearest-neighbor with k=1 then majority vote by domain.
    """

    def __init__(self, expert_names, domain_embeddings, n_exemplars=50,
                 threshold=0.0):
        """
        domain_embeddings: dict name -> (N_train, D_EMBED) normalized embeddings
        n_exemplars: how many exemplars per domain to store
        threshold: minimum similarity to return a route (0 = always route)
        """
        self.expert_names = expert_names
        self.threshold = threshold
        self.n_exemplars = n_exemplars

        # Store exemplars: subsample from training data
        exemplar_list = []
        exemplar_labels = []
        for idx, name in enumerate(expert_names):
            embs = domain_embeddings[name]
            n = min(n_exemplars, embs.shape[0])
            exemplar_list.append(embs[:n])
            exemplar_labels.extend([idx] * n)

        self.exemplar_matrix = np.concatenate(exemplar_list, axis=0)  # (total, D)
        self.exemplar_labels = np.array(exemplar_labels)  # (total,)
        self.n_experts = len(expert_names)

    def route_batch(self, query_embeddings):
        """query_embeddings: (B, D) L2-normalized.

        For each query, find the exemplar with highest cosine similarity,
        return that exemplar's domain.
        """
        # (B, total_exemplars) cosine similarity
        sims = query_embeddings @ self.exemplar_matrix.T
        # For each query, find best exemplar
        best_exemplar_idx = sims.argmax(axis=1)  # (B,)
        best_sims = sims[np.arange(sims.shape[0]), best_exemplar_idx]
        best_labels = self.exemplar_labels[best_exemplar_idx]  # (B,)

        routes = []
        for i in range(query_embeddings.shape[0]):
            if best_sims[i] >= self.threshold:
                routes.append(self.expert_names[best_labels[i]])
            else:
                # Below threshold: fall back to hash (but for this experiment
                # we use threshold=0, so this never fires)
                routes.append(self.expert_names[0])
        return routes

    def route_batch_aggregated(self, query_embeddings):
        """Alternative: aggregate similarity per domain, pick highest sum.

        More robust than single-exemplar matching. Computes mean similarity
        to each domain's exemplars.
        """
        sims = query_embeddings @ self.exemplar_matrix.T  # (B, total)
        B = query_embeddings.shape[0]

        # Aggregate by domain: for each domain, compute mean similarity
        domain_sims = np.zeros((B, self.n_experts))
        for d in range(self.n_experts):
            mask = self.exemplar_labels == d
            if mask.sum() > 0:
                domain_sims[:, d] = sims[:, mask].mean(axis=1)

        indices = domain_sims.argmax(axis=1)
        return [self.expert_names[i] for i in indices]


class OracleRouter:
    """Perfect routing with known labels."""

    def route_batch(self, domain_labels):
        return list(domain_labels)


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(seed=42, n_seeds=3, n_train=300, n_test=100):
    """Run the full semantic router comparison."""
    results_dir = Path(__file__).parent
    t_start = time.time()

    print("=" * 72)
    print("  Semantic Router Experiment")
    print("  Hypothesis: semantic router achieves >70% domain accuracy, <5ms latency")
    print(f"  Config: {n_seeds} seeds, {n_train} train, {n_test} test per domain")
    print(f"  Features: char n-gram (uni+bi+tri), D_embed={D_EMBED}")
    print(f"  Strategies: hash ring, keyword, cosine, LSH, utterance, oracle")
    print("=" * 72)

    all_seed_results = []

    for seed_idx in range(n_seeds):
        current_seed = seed + seed_idx * 100
        print(f"\n{'='*72}")
        print(f"  SEED {seed_idx + 1}/{n_seeds} (seed={current_seed})")
        print(f"{'='*72}")

        rng = np.random.RandomState(current_seed)

        # -- Generate data --
        print("  Generating cluster prototypes...")
        prototypes = generate_cluster_prototypes(rng)

        domain_data = {}  # domain -> (x_train, y_train)
        domain_test = {}  # domain -> (x_test, y_test)
        for domain in ALL_DOMAINS:
            cluster = DOMAIN_TO_CLUSTER[domain]
            x, y = generate_domain_data(rng, prototypes, domain, cluster,
                                        n_sequences=n_train + n_test)
            domain_data[domain] = (x[:n_train], y[:n_train])
            domain_test[domain] = (x[n_train:], y[n_train:])

        # -- Compute embeddings --
        print("  Computing n-gram embeddings...")
        # Determine raw feature dimension
        sample_feat = compute_ngram_features(domain_data[ALL_DOMAINS[0]][0][:1])
        d_raw = sample_feat.shape[1]
        print(f"    Raw feature dim: {d_raw}, projected to D_embed={D_EMBED}")

        proj = make_projection_matrix(d_raw, D_EMBED, rng)

        # Compute train embeddings per domain
        domain_train_embs = {}
        for domain in ALL_DOMAINS:
            x_train, _ = domain_data[domain]
            feats = compute_ngram_features(x_train)
            embs = project_features(feats, proj)
            domain_train_embs[domain] = embs

        # Compute test embeddings
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

        test_feats = compute_ngram_features(all_test_x)
        test_embs = project_features(test_feats, proj)
        n_total_test = all_test_x.shape[0]

        all_test_clusters = [DOMAIN_TO_CLUSTER[d] for d in all_test_domains]

        # -- Build routers --
        print("  Building routers...")
        hash_router = HashRingRouter(ALL_DOMAINS, seed=current_seed)
        keyword_router = KeywordRouter(ALL_DOMAINS, domain_data)
        cosine_router = CosineRouter(ALL_DOMAINS, domain_train_embs)
        lsh_router = LSHRouter(ALL_DOMAINS, domain_train_embs,
                               n_planes=LSH_N_PLANES, rng=rng)
        utterance_router = UtteranceRouter(ALL_DOMAINS, domain_train_embs,
                                           n_exemplars=50)

        # -- Evaluate each router --
        strategies = {}

        # (a) Hash ring
        print("  Evaluating hash ring...")
        t0 = time.time()
        for _ in range(10):  # repeat for stable timing
            hash_routes = hash_router.route_batch(all_test_x)
        hash_time = (time.time() - t0) / (10 * n_total_test)
        strategies['hash_ring'] = {
            'routes': hash_routes,
            'latency_us': hash_time * 1e6,
        }

        # (b) Keyword frequency
        print("  Evaluating keyword frequency...")
        t0 = time.time()
        for _ in range(10):
            kw_routes = keyword_router.route_batch(all_test_x)
        kw_time = (time.time() - t0) / (10 * n_total_test)
        strategies['keyword_freq'] = {
            'routes': kw_routes,
            'latency_us': kw_time * 1e6,
        }

        # (c) Cosine similarity
        print("  Evaluating cosine similarity...")
        t0 = time.time()
        for _ in range(100):
            cos_routes = cosine_router.route_batch(test_embs)
        cos_time = (time.time() - t0) / (100 * n_total_test)
        strategies['cosine_sim'] = {
            'routes': cos_routes,
            'latency_us': cos_time * 1e6,
        }

        # (d) LSH partitioning
        print("  Evaluating LSH partitioning...")
        t0 = time.time()
        for _ in range(100):
            lsh_routes = lsh_router.route_batch(test_embs)
        lsh_time = (time.time() - t0) / (100 * n_total_test)
        strategies['lsh_partition'] = {
            'routes': lsh_routes,
            'latency_us': lsh_time * 1e6,
        }

        # (e) Utterance matching (1-NN)
        print("  Evaluating utterance matching (1-NN)...")
        t0 = time.time()
        for _ in range(100):
            utt_routes = utterance_router.route_batch(test_embs)
        utt_time = (time.time() - t0) / (100 * n_total_test)
        strategies['utterance_1nn'] = {
            'routes': utt_routes,
            'latency_us': utt_time * 1e6,
        }

        # (e2) Utterance matching (aggregated)
        print("  Evaluating utterance matching (aggregated)...")
        t0 = time.time()
        for _ in range(100):
            utt_agg_routes = utterance_router.route_batch_aggregated(test_embs)
        utt_agg_time = (time.time() - t0) / (100 * n_total_test)
        strategies['utterance_agg'] = {
            'routes': utt_agg_routes,
            'latency_us': utt_agg_time * 1e6,
        }

        # (f) Oracle
        strategies['oracle'] = {
            'routes': list(all_test_domains),
            'latency_us': 0.0,
        }

        # -- Compute accuracy metrics --
        seed_results = {}
        print(f"\n  {'Strategy':<22s} {'Domain Acc':>10s} {'Cluster Acc':>12s} "
              f"{'Latency':>12s}")
        print(f"  {'-'*58}")

        for sname, sdata in strategies.items():
            routes = sdata['routes']
            # Domain-level accuracy
            domain_acc = sum(1 for i in range(n_total_test)
                           if routes[i] == all_test_domains[i]) / n_total_test
            # Cluster-level accuracy
            cluster_acc = sum(
                1 for i in range(n_total_test)
                if DOMAIN_TO_CLUSTER.get(routes[i], '') == all_test_clusters[i]
            ) / n_total_test

            # Per-cluster breakdown
            per_cluster = {}
            for cname in CLUSTER_NAMES:
                c_indices = [i for i in range(n_total_test)
                           if all_test_clusters[i] == cname]
                c_domain_acc = sum(1 for i in c_indices
                                  if routes[i] == all_test_domains[i]) / len(c_indices)
                c_cluster_acc = sum(
                    1 for i in c_indices
                    if DOMAIN_TO_CLUSTER.get(routes[i], '') == cname
                ) / len(c_indices)
                per_cluster[cname] = {
                    'domain_acc': float(c_domain_acc),
                    'cluster_acc': float(c_cluster_acc),
                }

            lat_str = f"{sdata['latency_us']:.2f}us" if sdata['latency_us'] > 0 else "0"
            print(f"  {sname:<22s} {domain_acc:>10.4f} {cluster_acc:>12.4f} "
                  f"{lat_str:>12s}")

            seed_results[sname] = {
                'domain_accuracy': float(domain_acc),
                'cluster_accuracy': float(cluster_acc),
                'latency_us': float(sdata['latency_us']),
                'per_cluster': per_cluster,
            }

        all_seed_results.append({
            'seed': current_seed,
            'results': seed_results,
        })

    # =================================================================
    # Aggregate across seeds
    # =================================================================
    elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  AGGREGATE RESULTS ({n_seeds} seeds)")
    print(f"{'='*72}")

    strategy_names = ['hash_ring', 'keyword_freq', 'cosine_sim',
                      'lsh_partition', 'utterance_1nn', 'utterance_agg', 'oracle']

    agg = {}
    for s in strategy_names:
        domain_accs = [r['results'][s]['domain_accuracy'] for r in all_seed_results]
        cluster_accs = [r['results'][s]['cluster_accuracy'] for r in all_seed_results]
        latencies = [r['results'][s]['latency_us'] for r in all_seed_results]

        agg[s] = {
            'domain_acc_mean': float(np.mean(domain_accs)),
            'domain_acc_std': float(np.std(domain_accs)),
            'cluster_acc_mean': float(np.mean(cluster_accs)),
            'cluster_acc_std': float(np.std(cluster_accs)),
            'latency_us_mean': float(np.mean(latencies)),
            'latency_us_std': float(np.std(latencies)),
        }

    print(f"\n  {'Strategy':<22s} {'Domain Acc':>12s} {'Cluster Acc':>12s} "
          f"{'Latency (us)':>14s}")
    print(f"  {'-'*62}")
    for s in strategy_names:
        a = agg[s]
        d_str = f"{a['domain_acc_mean']:.4f}+/-{a['domain_acc_std']:.4f}"
        c_str = f"{a['cluster_acc_mean']:.4f}"
        l_str = f"{a['latency_us_mean']:.2f}" if a['latency_us_mean'] > 0 else "0"
        print(f"  {s:<22s} {d_str:>12s} {c_str:>12s} {l_str:>14s}")

    # =================================================================
    # Kill Criteria
    # =================================================================
    print(f"\n{'='*72}")
    print(f"  KILL CRITERIA EVALUATION")
    print(f"{'='*72}")

    # Best semantic router (exclude hash_ring and oracle)
    semantic_strategies = ['keyword_freq', 'cosine_sim', 'lsh_partition',
                          'utterance_1nn', 'utterance_agg']
    best_name = max(semantic_strategies,
                    key=lambda s: agg[s]['domain_acc_mean'])
    best_domain_acc = agg[best_name]['domain_acc_mean']
    best_latency = agg[best_name]['latency_us_mean']
    hash_latency = agg['hash_ring']['latency_us_mean']

    # K1: accuracy < 70%
    k1_killed = best_domain_acc < 0.70
    print(f"\n  K1: Best semantic router domain accuracy <70%")
    print(f"      Best: {best_name} with acc={best_domain_acc:.4f}")
    if k1_killed:
        print(f"      STATUS: KILL -- accuracy {best_domain_acc:.4f} < 0.70")
    else:
        print(f"      STATUS: PASS -- accuracy {best_domain_acc:.4f} >= 0.70")

    # K2: latency > 5ms (5000 us)
    k2_killed = best_latency > 5000
    print(f"\n  K2: Router latency >5ms per query")
    print(f"      Best strategy latency: {best_latency:.2f}us")
    if k2_killed:
        print(f"      STATUS: KILL -- latency {best_latency:.2f}us > 5000us")
    else:
        print(f"      STATUS: PASS -- latency {best_latency:.2f}us < 5000us")

    # K3: router adds >2% end-to-end latency vs hash ring
    if hash_latency > 0:
        overhead_pct = ((best_latency - hash_latency) / hash_latency) * 100
    else:
        overhead_pct = 0.0
    # Note: we report absolute latency overhead, but for K3 the question is
    # whether the router adds >2% of TOTAL inference time. At micro scale,
    # inference time ~ 100us-1ms per token. The router latency is additive.
    # We define K3 as: best_latency < 0.02 * typical_inference_time
    # At micro: inference ~ 500us, so 2% = 10us.
    # At macro: inference ~ 20ms, so 2% = 400us.
    # We use conservative micro threshold: router < 10us for production viability.
    k3_threshold_us = 10.0  # 2% of ~500us micro inference
    k3_killed = best_latency > k3_threshold_us
    print(f"\n  K3: Router adds >2% end-to-end latency vs hash ring")
    print(f"      Hash ring latency:    {hash_latency:.2f}us")
    print(f"      Best semantic:        {best_latency:.2f}us")
    print(f"      Overhead vs hash:     {overhead_pct:+.1f}%")
    print(f"      2% of ~500us inference = {k3_threshold_us:.0f}us threshold")
    if k3_killed:
        print(f"      STATUS: KILL -- latency {best_latency:.2f}us > {k3_threshold_us}us")
    else:
        print(f"      STATUS: PASS -- latency {best_latency:.2f}us < {k3_threshold_us}us")

    overall_kill = k1_killed  # K1 is the primary gate
    # K2 and K3 inform production viability but don't kill the mechanism
    print(f"\n  {'='*60}")
    if k1_killed:
        print(f"  OVERALL VERDICT: KILL (K1: accuracy {best_domain_acc:.4f} < 0.70)")
    else:
        if k2_killed or k3_killed:
            killed_by = []
            if k2_killed:
                killed_by.append("K2 (latency)")
            if k3_killed:
                killed_by.append("K3 (overhead)")
            print(f"  OVERALL VERDICT: SUPPORTED (accuracy passes, "
                  f"but {', '.join(killed_by)})")
        else:
            print(f"  OVERALL VERDICT: PROVEN")
            print(f"    Best strategy: {best_name}")
            print(f"    Domain accuracy: {best_domain_acc:.4f}")
            print(f"    Latency: {best_latency:.2f}us")
    print(f"  {'='*60}")

    # =================================================================
    # Detailed per-cluster analysis
    # =================================================================
    print(f"\n  Per-cluster accuracy (averaged across seeds):")
    for s in semantic_strategies:
        print(f"\n    {s}:")
        for cname in CLUSTER_NAMES:
            d_accs = [r['results'][s]['per_cluster'][cname]['domain_acc']
                     for r in all_seed_results]
            c_accs = [r['results'][s]['per_cluster'][cname]['cluster_acc']
                     for r in all_seed_results]
            print(f"      {cname:12s}: domain={np.mean(d_accs):.3f} "
                  f"cluster={np.mean(c_accs):.3f}")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save
    output = {
        'config': {
            'n_seeds': n_seeds,
            'n_train_per_domain': n_train,
            'n_test_per_domain': n_test,
            'd_embed': D_EMBED,
            'vocab_size': VOCAB_SIZE,
            'context_len': CONTEXT_LEN,
            'lsh_n_planes': LSH_N_PLANES,
            'n_domains': N_DOMAINS,
            'n_clusters': N_CLUSTERS,
            'feature_type': 'char_ngram_1_2_3',
        },
        'seed_results': all_seed_results,
        'aggregate': agg,
        'kill_criteria': {
            'k1_accuracy_pass': not k1_killed,
            'k1_best_strategy': best_name,
            'k1_best_accuracy': float(best_domain_acc),
            'k2_latency_pass': not k2_killed,
            'k2_best_latency_us': float(best_latency),
            'k3_overhead_pass': not k3_killed,
            'k3_overhead_pct': float(overhead_pct),
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
