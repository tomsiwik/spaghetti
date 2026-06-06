#!/usr/bin/env python3
"""Routing strategy comparison for SOLE expert selection.

Compares 5 routing strategies on two axes:
  1. Routing LATENCY (empirical, on Apple Silicon)
  2. Expert selection QUALITY (using synthetic quality profiles + oracle comparison)

The predecessor experiment (inference_latency_vs_N) proved:
  - Pre-merge: O(1) latency, +2.6% max overhead
  - Hash ring: 0.5 us/query, O(log N)
  - Dynamic top-k: O(k) not O(N)

Prior killed experiments established:
  - Content-aware routing: killed (experts don't specialize at micro)
  - Semantic router: killed (K1: best domain acc 27.3% < 70%)
  - Pre-merge vs dynamic quality: killed (no specialization)

THIS experiment asks: Given synthetic expert quality profiles that model
real-world specialization, what is the Pareto frontier of routing strategies
for latency vs quality?

We create synthetic "quality matrices" Q[query_type, expert] where Q[i,j] is
the quality gain from expert j on query type i. This models the scenario where
experts DO specialize (as proven at macro scale with 98% win rate).

Strategies:
  (a) Pre-merge all: zero routing cost, quality = mean over all experts
  (b) Hash ring: O(log N) lookup, random expert assignment
  (c) Embedding similarity: O(N) cosine lookup against expert centroids
  (d) Tiny classifier: O(1) after training, MLP on query features
  (e) Hierarchical: cluster routing + hash within cluster

For each strategy, we measure:
  - Routing latency (empirical, us/query)
  - Oracle agreement (% of queries where selected expert matches best expert)
  - Quality capture (fraction of oracle quality achieved)

Usage:
    uv run python -m micro.models.inference_routing_strategies.routing_strategies
"""

import hashlib
import json
import math
import os
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


# ── Synthetic Quality Matrix ──────────────────────────────────────────────

def generate_quality_matrix(n_domains: int, n_experts: int, n_clusters: int,
                            specialization_strength: float = 0.8,
                            rng: np.random.Generator = None) -> dict:
    """Generate a synthetic quality matrix modeling expert specialization.

    Each expert has a "home domain" where it performs best. Quality drops
    with distance from home domain. Experts within the same cluster share
    partial knowledge (inter-cluster quality is lower).

    Args:
        n_domains: Number of distinct query types
        n_experts: Number of experts (>= n_domains)
        n_clusters: Number of domain clusters
        specialization_strength: How much better an expert is at its home domain
            vs random (0 = no specialization, 1 = perfect specialization)
        rng: Random number generator

    Returns:
        dict with:
            Q: quality matrix (n_domains, n_experts) -- higher is better
            domain_to_cluster: mapping from domain to cluster ID
            expert_to_domain: mapping from expert to home domain
            expert_to_cluster: mapping from expert to home cluster
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Assign domains to clusters
    domains_per_cluster = n_domains // n_clusters
    domain_to_cluster = np.array([d // domains_per_cluster for d in range(n_domains)])
    domain_to_cluster = np.minimum(domain_to_cluster, n_clusters - 1)

    # Assign experts to domains (round-robin, then random for extras)
    expert_to_domain = np.zeros(n_experts, dtype=int)
    for i in range(n_experts):
        expert_to_domain[i] = i % n_domains
    expert_to_cluster = domain_to_cluster[expert_to_domain]

    # Build quality matrix
    # Base quality: uniform random in [0, 0.2]
    Q = rng.uniform(0, 0.2, size=(n_domains, n_experts))

    # Home domain bonus: specialization_strength
    for e in range(n_experts):
        home = expert_to_domain[e]
        Q[home, e] += specialization_strength

    # Same-cluster bonus: half of specialization
    for e in range(n_experts):
        home_cluster = expert_to_cluster[e]
        for d in range(n_domains):
            if domain_to_cluster[d] == home_cluster and d != expert_to_domain[e]:
                Q[d, e] += specialization_strength * 0.3

    return {
        "Q": Q,
        "domain_to_cluster": domain_to_cluster,
        "expert_to_domain": expert_to_domain,
        "expert_to_cluster": expert_to_cluster,
    }


# ── Query Generation ─────────────────────────────────────────────────────

def generate_queries(n_queries: int, n_domains: int, embed_dim: int,
                     domain_to_cluster: np.ndarray,
                     rng: np.random.Generator = None) -> dict:
    """Generate synthetic query embeddings with cluster structure.

    Each query belongs to a domain. Embeddings have:
    - Cluster-level structure (queries in same cluster are similar)
    - Domain-level structure (queries in same domain are more similar)
    - Random noise

    Returns:
        dict with:
            embeddings: (n_queries, embed_dim)
            domains: (n_queries,) -- true domain labels
            clusters: (n_queries,) -- true cluster labels
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_clusters = len(set(domain_to_cluster))

    # Cluster centroids: well-separated in embedding space
    cluster_centroids = rng.standard_normal((n_clusters, embed_dim))
    cluster_centroids /= np.linalg.norm(cluster_centroids, axis=1, keepdims=True)
    cluster_centroids *= 3.0  # separation

    # Domain centroids: perturbed from cluster centroid
    domain_centroids = np.zeros((n_domains, embed_dim))
    for d in range(n_domains):
        c = domain_to_cluster[d]
        perturbation = rng.standard_normal(embed_dim) * 0.5
        domain_centroids[d] = cluster_centroids[c] + perturbation

    # Generate queries
    domains = rng.integers(0, n_domains, size=n_queries)
    clusters = domain_to_cluster[domains]

    embeddings = np.zeros((n_queries, embed_dim))
    for i in range(n_queries):
        noise = rng.standard_normal(embed_dim) * 0.3
        embeddings[i] = domain_centroids[domains[i]] + noise

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms

    return {
        "embeddings": embeddings,
        "domains": domains,
        "clusters": clusters,
        "domain_centroids": domain_centroids,
        "cluster_centroids": cluster_centroids,
    }


# ── Routing Strategies ───────────────────────────────────────────────────

class PreMergeRouter:
    """Strategy A: Pre-merge all experts. No routing needed."""

    def __init__(self, Q: np.ndarray):
        # Quality is the mean across all experts (all active)
        self.mean_quality = Q.mean(axis=1)  # (n_domains,)
        self.n_experts = Q.shape[1]

    def route(self, query_embedding: np.ndarray, query_idx: int = 0) -> int:
        """Returns -1 (all experts active, no selection)."""
        return -1  # sentinel: all experts merged

    def quality(self, domain: int) -> float:
        """Quality for a domain query: mean over all experts."""
        return float(self.mean_quality[domain])

    def route_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Batch routing: returns -1 for all queries."""
        return np.full(len(embeddings), -1, dtype=int)


class HashRingRouter:
    """Strategy B: Consistent hash ring. O(log N) per query."""

    def __init__(self, n_experts: int, virtual_nodes: int = 150):
        self.n_experts = n_experts
        self.ring = []
        for e in range(n_experts):
            for vn in range(virtual_nodes):
                h = int(hashlib.md5(f"expert_{e}_vn_{vn}".encode()).hexdigest(), 16)
                self.ring.append((h, e))
        self.ring.sort()
        self.hashes = [h for h, _ in self.ring]
        self.experts = [e for _, e in self.ring]

    def route(self, query_embedding: np.ndarray, query_idx: int = 0) -> int:
        """Hash the query index to select an expert."""
        qh = int(hashlib.md5(f"query_{query_idx}".encode()).hexdigest(), 16)
        idx = bisect_right(self.hashes, qh) % len(self.ring)
        return self.experts[idx]

    def route_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Batch routing via hash ring."""
        results = np.zeros(len(embeddings), dtype=int)
        for i in range(len(embeddings)):
            results[i] = self.route(embeddings[i], i)
        return results


class EmbeddingSimilarityRouter:
    """Strategy C: Cosine similarity to expert centroids. O(N*D) per query."""

    def __init__(self, expert_centroids: np.ndarray):
        """expert_centroids: (n_experts, embed_dim)"""
        self.centroids = expert_centroids
        norms = np.linalg.norm(expert_centroids, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.centroids_norm = expert_centroids / norms

    def route(self, query_embedding: np.ndarray, query_idx: int = 0) -> int:
        """Select expert with highest cosine similarity."""
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
        sims = self.centroids_norm @ q
        return int(np.argmax(sims))

    def route_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Batch cosine similarity routing."""
        # Normalize queries
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        q_norm = embeddings / norms
        # Batch similarity
        sims = q_norm @ self.centroids_norm.T  # (n_queries, n_experts)
        return np.argmax(sims, axis=1)


class TinyClassifierRouter:
    """Strategy D: Small MLP classifier. O(D*H + H*N) per query after training."""

    def __init__(self, n_experts: int, embed_dim: int, hidden_dim: int = 32,
                 rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(42)
        self.n_experts = n_experts
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Two-layer MLP: embed_dim -> hidden_dim -> n_experts
        # Xavier init
        self.W1 = rng.standard_normal((embed_dim, hidden_dim)).astype(np.float32)
        self.W1 *= np.sqrt(2.0 / embed_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden_dim, n_experts)).astype(np.float32)
        self.W2 *= np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(n_experts, dtype=np.float32)

    def train(self, embeddings: np.ndarray, labels: np.ndarray,
              lr: float = 0.01, epochs: int = 100, batch_size: int = 64):
        """Train the classifier on (embedding, best_expert) pairs."""
        n = len(embeddings)
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n)
            total_loss = 0.0
            correct = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                X = embeddings[idx].astype(np.float32)
                y = labels[idx]
                bs = len(idx)

                # Forward
                h = X @ self.W1 + self.b1  # (bs, hidden)
                h_relu = np.maximum(h, 0)  # ReLU
                logits = h_relu @ self.W2 + self.b2  # (bs, n_experts)

                # Softmax + cross-entropy
                logits_max = logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits - logits_max)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                # Loss
                loss = -np.log(np.maximum(probs[np.arange(bs), y], 1e-10)).mean()
                total_loss += loss * bs
                correct += (np.argmax(logits, axis=1) == y).sum()

                # Backward (manual gradient)
                dlogits = probs.copy()
                dlogits[np.arange(bs), y] -= 1
                dlogits /= bs

                dW2 = h_relu.T @ dlogits
                db2 = dlogits.sum(axis=0)

                dh_relu = dlogits @ self.W2.T
                dh = dh_relu * (h > 0).astype(np.float32)

                dW1 = X.T @ dh
                db1 = dh.sum(axis=0)

                # Update
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            if (epoch + 1) % 20 == 0 or epoch == 0:
                acc = correct / n
                avg_loss = total_loss / n

    def route(self, query_embedding: np.ndarray, query_idx: int = 0) -> int:
        """Single query routing."""
        x = query_embedding.astype(np.float32)
        h = np.maximum(x @ self.W1 + self.b1, 0)
        logits = h @ self.W2 + self.b2
        return int(np.argmax(logits))

    def route_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Batch routing."""
        X = embeddings.astype(np.float32)
        h = np.maximum(X @ self.W1 + self.b1, 0)
        logits = h @ self.W2 + self.b2
        return np.argmax(logits, axis=1)


class HierarchicalRouter:
    """Strategy E: Cluster-level routing + hash ring within cluster.

    1. Cosine similarity to cluster centroids: O(C*D) per query
    2. Hash ring within selected cluster: O(log(N/C)) per query
    Total: O(C*D + log(N/C))
    """

    def __init__(self, cluster_centroids: np.ndarray, expert_to_cluster: np.ndarray,
                 n_experts: int, virtual_nodes: int = 150):
        self.n_clusters = len(cluster_centroids)
        norms = np.linalg.norm(cluster_centroids, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.cluster_centroids_norm = cluster_centroids / norms

        # Build per-cluster hash rings
        self.cluster_rings = {}
        for c in range(self.n_clusters):
            experts_in_cluster = np.where(expert_to_cluster == c)[0]
            ring = []
            for e in experts_in_cluster:
                for vn in range(virtual_nodes):
                    h = int(hashlib.md5(f"expert_{e}_vn_{vn}".encode()).hexdigest(), 16)
                    ring.append((h, int(e)))
            ring.sort()
            self.cluster_rings[c] = {
                "hashes": [h for h, _ in ring],
                "experts": [e for _, e in ring],
            }

    def route(self, query_embedding: np.ndarray, query_idx: int = 0) -> int:
        """Route: cluster classification + hash within cluster."""
        # Step 1: cluster assignment via cosine similarity
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
        sims = self.cluster_centroids_norm @ q
        cluster = int(np.argmax(sims))

        # Step 2: hash within cluster
        ring = self.cluster_rings[cluster]
        if not ring["hashes"]:
            return 0
        qh = int(hashlib.md5(f"query_{query_idx}".encode()).hexdigest(), 16)
        idx = bisect_right(ring["hashes"], qh) % len(ring["hashes"])
        return ring["experts"][idx]

    def route_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Batch routing."""
        results = np.zeros(len(embeddings), dtype=int)
        for i in range(len(embeddings)):
            results[i] = self.route(embeddings[i], i)
        return results


# ── Oracle Router ─────────────────────────────────────────────────────────

class OracleRouter:
    """Perfect routing: always selects the best expert for each domain."""

    def __init__(self, Q: np.ndarray):
        self.Q = Q
        self.best_expert = np.argmax(Q, axis=1)  # (n_domains,)

    def route_for_domain(self, domain: int) -> int:
        return int(self.best_expert[domain])

    def quality(self, domain: int) -> float:
        return float(self.Q[domain, self.best_expert[domain]])


# ── Latency Measurement ──────────────────────────────────────────────────

def measure_routing_latency(router, embeddings: np.ndarray,
                            n_warmup: int = 100, n_iters: int = 5000) -> dict:
    """Measure per-query routing latency for a router.

    Returns dict with mean_us, std_us, p50_us, p99_us.
    """
    n = len(embeddings)

    # Warmup
    for i in range(min(n_warmup, n)):
        router.route(embeddings[i % n], i)

    # Measure individual queries
    latencies = []
    for i in range(n_iters):
        idx = i % n
        start = time.perf_counter()
        router.route(embeddings[idx], idx)
        elapsed = (time.perf_counter() - start) * 1e6  # us
        latencies.append(elapsed)

    latencies = np.array(latencies)

    # Also measure batch routing
    batch_start = time.perf_counter()
    router.route_batch(embeddings[:min(1000, n)])
    batch_elapsed = (time.perf_counter() - batch_start) * 1e6
    batch_per_query = batch_elapsed / min(1000, n)

    return {
        "mean_us": float(np.mean(latencies)),
        "std_us": float(np.std(latencies)),
        "p50_us": float(np.median(latencies)),
        "p99_us": float(np.percentile(latencies, 99)),
        "batch_per_query_us": float(batch_per_query),
    }


def measure_premerge_latency(n_iters: int = 5000) -> dict:
    """Pre-merge has zero routing latency (no routing step)."""
    # Measure the overhead of a no-op to establish floor
    latencies = []
    for i in range(n_iters):
        start = time.perf_counter()
        _ = -1  # no-op: "all experts active"
        elapsed = (time.perf_counter() - start) * 1e6
        latencies.append(elapsed)

    latencies = np.array(latencies)
    return {
        "mean_us": float(np.mean(latencies)),
        "std_us": float(np.std(latencies)),
        "p50_us": float(np.median(latencies)),
        "p99_us": float(np.percentile(latencies, 99)),
        "batch_per_query_us": 0.0,
    }


# ── Quality Measurement ──────────────────────────────────────────────────

def measure_quality(router, Q: np.ndarray, query_domains: np.ndarray,
                    query_embeddings: np.ndarray, oracle: OracleRouter) -> dict:
    """Measure routing quality: oracle agreement and quality capture.

    Args:
        router: routing strategy object
        Q: quality matrix (n_domains, n_experts)
        query_domains: true domain for each query
        query_embeddings: query embeddings
        oracle: oracle router for comparison

    Returns:
        dict with:
            oracle_agreement: fraction of queries where selected expert = oracle's choice
            quality_capture: mean(Q[domain, selected]) / mean(Q[domain, oracle_choice])
            cluster_agreement: fraction where selected expert is in correct cluster
    """
    n_queries = len(query_domains)
    n_experts = Q.shape[1]

    oracle_qualities = []
    router_qualities = []
    oracle_agreements = 0

    for i in range(n_queries):
        domain = query_domains[i]

        # Oracle choice
        oracle_expert = oracle.route_for_domain(domain)
        oracle_q = Q[domain, oracle_expert]
        oracle_qualities.append(oracle_q)

        # Router choice
        if hasattr(router, 'route'):
            selected = router.route(query_embeddings[i], i)
        else:
            selected = 0

        if selected == -1:
            # Pre-merge: quality is mean over all experts
            router_q = Q[domain, :].mean()
        else:
            router_q = Q[domain, selected]

        router_qualities.append(router_q)

        if selected == oracle_expert:
            oracle_agreements += 1

    oracle_qualities = np.array(oracle_qualities)
    router_qualities = np.array(router_qualities)

    # Quality capture: what fraction of oracle quality does the router achieve?
    oracle_mean = oracle_qualities.mean()
    router_mean = router_qualities.mean()
    quality_capture = router_mean / oracle_mean if oracle_mean > 0 else 0.0

    # Random baseline quality
    random_quality = Q.mean()
    random_capture = random_quality / oracle_mean if oracle_mean > 0 else 0.0

    return {
        "oracle_agreement": oracle_agreements / n_queries,
        "quality_capture": float(quality_capture),
        "router_mean_quality": float(router_mean),
        "oracle_mean_quality": float(oracle_mean),
        "random_mean_quality": float(random_quality),
        "random_capture": float(random_capture),
        "quality_lift_over_random": float(
            (router_mean - random_quality) / (oracle_mean - random_quality)
            if oracle_mean > random_quality else 0.0
        ),
    }


# ── Expert Centroids ─────────────────────────────────────────────────────

def compute_expert_centroids(expert_to_domain: np.ndarray,
                             domain_centroids: np.ndarray,
                             n_experts: int, embed_dim: int,
                             rng: np.random.Generator = None) -> np.ndarray:
    """Compute expert centroids in embedding space.

    Each expert's centroid is its home domain centroid + small perturbation.
    In production, these would come from aggregating training data embeddings.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    centroids = np.zeros((n_experts, embed_dim))
    for e in range(n_experts):
        home = expert_to_domain[e]
        centroids[e] = domain_centroids[home] + rng.standard_normal(embed_dim) * 0.1

    return centroids


# ── Main Experiment ───────────────────────────────────────────────────────

def run_experiment(n_domains: int = 15, n_experts: int = 30,
                   n_clusters: int = 3, embed_dim: int = 64,
                   n_queries: int = 3000, n_test_queries: int = 2000,
                   specialization_strength: float = 0.8,
                   n_seeds: int = 3,
                   n_latency_iters: int = 5000) -> dict:
    """Run the full routing strategy comparison experiment."""

    print(f"=== Routing Strategy Comparison ===")
    print(f"Domains: {n_domains}, Experts: {n_experts}, Clusters: {n_clusters}")
    print(f"Embed dim: {embed_dim}")
    print(f"Specialization strength: {specialization_strength}")
    print(f"Queries: {n_queries} train + {n_test_queries} test, {n_seeds} seeds")
    print()

    all_results = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(42 + seed)
        print(f"--- Seed {seed} ---")

        # Generate quality matrix
        qm = generate_quality_matrix(n_domains, n_experts, n_clusters,
                                     specialization_strength, rng)
        Q = qm["Q"]

        # Generate queries
        train_queries = generate_queries(n_queries, n_domains, embed_dim,
                                         qm["domain_to_cluster"], rng)
        test_queries = generate_queries(n_test_queries, n_domains, embed_dim,
                                        qm["domain_to_cluster"],
                                        np.random.default_rng(1000 + seed))

        # Compute expert centroids
        expert_centroids = compute_expert_centroids(
            qm["expert_to_domain"], train_queries["domain_centroids"],
            n_experts, embed_dim, rng)

        # Oracle
        oracle = OracleRouter(Q)

        # Train labels for classifier: for each query, what's the best expert?
        train_labels = np.array([
            oracle.route_for_domain(d) for d in train_queries["domains"]
        ])

        # Build routers
        routers = {}

        # (a) Pre-merge
        routers["premerge"] = PreMergeRouter(Q)

        # (b) Hash ring
        routers["hash_ring"] = HashRingRouter(n_experts)

        # (c) Embedding similarity
        routers["embedding_sim"] = EmbeddingSimilarityRouter(expert_centroids)

        # (d) Tiny classifier
        classifier = TinyClassifierRouter(n_experts, embed_dim, hidden_dim=32, rng=rng)
        classifier.train(train_queries["embeddings"], train_labels,
                        lr=0.01, epochs=100, batch_size=64)
        routers["tiny_classifier"] = classifier

        # (e) Hierarchical
        routers["hierarchical"] = HierarchicalRouter(
            train_queries["cluster_centroids"], qm["expert_to_cluster"],
            n_experts)

        # Measure latency and quality for each router
        seed_results = {}
        for name, router in routers.items():
            print(f"  {name}...", end=" ", flush=True)

            # Latency
            if name == "premerge":
                latency = measure_premerge_latency(n_latency_iters)
            else:
                latency = measure_routing_latency(
                    router, test_queries["embeddings"],
                    n_warmup=100, n_iters=n_latency_iters)

            # Quality
            quality = measure_quality(
                router, Q, test_queries["domains"],
                test_queries["embeddings"], oracle)

            print(f"lat={latency['mean_us']:.2f}us, "
                  f"oracle_agr={quality['oracle_agreement']:.3f}, "
                  f"q_capture={quality['quality_capture']:.3f}")

            seed_results[name] = {
                "latency": latency,
                "quality": quality,
            }

        all_results.append(seed_results)

    # Aggregate across seeds
    print("\n=== Aggregated Results ===")
    strategies = list(all_results[0].keys())
    aggregated = {}

    for name in strategies:
        latencies_mean = [r[name]["latency"]["mean_us"] for r in all_results]
        latencies_p99 = [r[name]["latency"]["p99_us"] for r in all_results]
        batch_lat = [r[name]["latency"]["batch_per_query_us"] for r in all_results]
        oracle_agr = [r[name]["quality"]["oracle_agreement"] for r in all_results]
        q_capture = [r[name]["quality"]["quality_capture"] for r in all_results]
        q_lift = [r[name]["quality"]["quality_lift_over_random"] for r in all_results]

        agg = {
            "latency_mean_us": float(np.mean(latencies_mean)),
            "latency_std_us": float(np.std(latencies_mean)),
            "latency_p99_us": float(np.mean(latencies_p99)),
            "batch_per_query_us": float(np.mean(batch_lat)),
            "oracle_agreement_mean": float(np.mean(oracle_agr)),
            "oracle_agreement_std": float(np.std(oracle_agr)),
            "quality_capture_mean": float(np.mean(q_capture)),
            "quality_capture_std": float(np.std(q_capture)),
            "quality_lift_mean": float(np.mean(q_lift)),
            "quality_lift_std": float(np.std(q_lift)),
        }
        aggregated[name] = agg

        print(f"  {name:20s}: lat={agg['latency_mean_us']:8.2f} +/- {agg['latency_std_us']:.2f} us, "
              f"oracle_agr={agg['oracle_agreement_mean']:.3f} +/- {agg['oracle_agreement_std']:.3f}, "
              f"q_capture={agg['quality_capture_mean']:.3f} +/- {agg['quality_capture_std']:.3f}")

    # Kill criteria assessment
    print("\n=== Kill Criteria Assessment ===")

    # K1: best routing strategy is >50ms per query at N=100
    max_latency_us = max(agg["latency_mean_us"] for agg in aggregated.values())
    max_latency_ms = max_latency_us / 1000
    k1_pass = max_latency_ms < 50.0
    print(f"  K1 (best strategy <50ms at N={n_experts}): "
          f"worst={max_latency_ms:.4f}ms {'PASS' if k1_pass else 'KILL'}")

    # K2: routing overhead exceeds expert computation overhead
    # Expert computation at micro: ~1ms forward pass (from predecessor)
    expert_compute_us = 1000.0  # 1ms in microseconds
    routing_overheads = {
        name: agg["latency_mean_us"] / expert_compute_us
        for name, agg in aggregated.items()
    }
    max_routing_ratio = max(routing_overheads.values())
    k2_pass = max_routing_ratio < 1.0
    print(f"  K2 (routing < expert compute): "
          f"worst ratio={max_routing_ratio:.4f} ({'PASS' if k2_pass else 'KILL'})")
    for name, ratio in sorted(routing_overheads.items(), key=lambda x: x[1]):
        print(f"      {name}: {ratio:.4f}x ({aggregated[name]['latency_mean_us']:.2f}us / {expert_compute_us:.0f}us)")

    # K3: no strategy achieves >90% of oracle routing quality
    best_capture = max(agg["quality_capture_mean"] for agg in aggregated.values())
    k3_pass = best_capture > 0.90
    print(f"  K3 (best quality capture >90%): "
          f"best={best_capture:.3f} ({'PASS' if k3_pass else 'KILL'})")

    # Pareto analysis
    print("\n=== Pareto Frontier ===")
    print("  (Lower latency + higher quality capture = better)")

    # Sort by quality capture descending
    sorted_strategies = sorted(
        aggregated.items(),
        key=lambda x: x[1]["quality_capture_mean"],
        reverse=True
    )

    pareto_front = []
    best_lat_so_far = float('inf')
    for name, agg in sorted_strategies:
        if agg["latency_mean_us"] < best_lat_so_far:
            pareto_front.append(name)
            best_lat_so_far = agg["latency_mean_us"]

    print(f"  Pareto-optimal strategies: {pareto_front}")
    for name in pareto_front:
        agg = aggregated[name]
        print(f"    {name}: lat={agg['latency_mean_us']:.2f}us, "
              f"q_capture={agg['quality_capture_mean']:.3f}")

    # Is it EVER worth paying routing overhead?
    premerge_capture = aggregated["premerge"]["quality_capture_mean"]
    print(f"\n  Pre-merge quality capture: {premerge_capture:.3f}")
    print(f"  Quality gap (oracle - premerge): {1.0 - premerge_capture:.3f}")

    for name, agg in sorted(aggregated.items(),
                            key=lambda x: x[1]["quality_capture_mean"], reverse=True):
        if name == "premerge":
            continue
        gain = agg["quality_capture_mean"] - premerge_capture
        cost_us = agg["latency_mean_us"]
        print(f"  {name}: +{gain:.3f} quality for {cost_us:.2f}us routing cost "
              f"({gain / max(cost_us, 1e-6) * 1e6:.1f} quality/ms)")

    # N-scaling projections
    print("\n=== Scaling Projections ===")
    print("  Strategy      | N=30    | N=100   | N=500   | N=1000  ")
    print("  " + "-" * 60)

    for name in strategies:
        if name == "premerge":
            print(f"  {name:15s} | O(1)    | O(1)    | O(1)    | O(1)    ")
        elif name == "hash_ring":
            base = aggregated[name]["latency_mean_us"]
            # O(log N) scaling
            l100 = base * np.log2(100) / np.log2(n_experts)
            l500 = base * np.log2(500) / np.log2(n_experts)
            l1000 = base * np.log2(1000) / np.log2(n_experts)
            print(f"  {name:15s} | {base:.1f}us  | {l100:.1f}us  | {l500:.1f}us  | {l1000:.1f}us  ")
        elif name == "embedding_sim":
            base = aggregated[name]["latency_mean_us"]
            # O(N*D) scaling
            l100 = base * 100 / n_experts
            l500 = base * 500 / n_experts
            l1000 = base * 1000 / n_experts
            print(f"  {name:15s} | {base:.1f}us  | {l100:.1f}us  | {l500:.1f}us  | {l1000:.1f}us  ")
        elif name == "tiny_classifier":
            base = aggregated[name]["latency_mean_us"]
            # O(D*H + H*N): for N >> H, O(H*N)
            # But softmax output layer is the only N-dependent part
            factor_100 = 1.0 + (100 - n_experts) * 0.01  # ~1% per extra expert
            factor_500 = 1.0 + (500 - n_experts) * 0.01
            factor_1000 = 1.0 + (1000 - n_experts) * 0.01
            print(f"  {name:15s} | {base:.1f}us  | {base * factor_100:.1f}us  | {base * factor_500:.1f}us  | {base * factor_1000:.1f}us  ")
        elif name == "hierarchical":
            base = aggregated[name]["latency_mean_us"]
            # O(C*D + log(N/C)): nearly constant with N
            l100 = base * 1.05  # log factor grows very slowly
            l500 = base * 1.15
            l1000 = base * 1.20
            print(f"  {name:15s} | {base:.1f}us  | {l100:.1f}us  | {l500:.1f}us  | {l1000:.1f}us  ")

    results = {
        "config": {
            "n_domains": n_domains,
            "n_experts": n_experts,
            "n_clusters": n_clusters,
            "embed_dim": embed_dim,
            "n_queries_train": n_queries,
            "n_queries_test": n_test_queries,
            "specialization_strength": specialization_strength,
            "n_seeds": n_seeds,
            "n_latency_iters": n_latency_iters,
        },
        "aggregated": aggregated,
        "per_seed": all_results,
        "kill_criteria": {
            "K1_best_under_50ms": bool(k1_pass),
            "K1_worst_latency_ms": float(max_latency_ms),
            "K2_routing_under_expert_compute": bool(k2_pass),
            "K2_worst_ratio": float(max_routing_ratio),
            "K3_best_over_90pct_oracle": bool(k3_pass),
            "K3_best_quality_capture": float(best_capture),
            "all_pass": bool(k1_pass and k2_pass and k3_pass),
        },
        "pareto_frontier": pareto_front,
    }

    return results


# ── Multi-N Scaling Validation ────────────────────────────────────────────

def run_scaling_experiment(n_values: list = None,
                           embed_dim: int = 64,
                           n_clusters: int = 3,
                           seed: int = 42) -> dict:
    """Measure routing latency as function of N for all strategies.

    This validates the O(.) scaling claims empirically.
    """
    if n_values is None:
        n_values = [10, 30, 50, 100]

    print(f"\n=== Routing Latency Scaling with N ===")
    print(f"N values: {n_values}")

    rng = np.random.default_rng(seed)
    scaling_results = {}

    for N in n_values:
        n_domains = max(5, N // 2)  # At least 5 domains
        print(f"\n--- N={N} (domains={n_domains}) ---")

        # Generate quality matrix and queries
        qm = generate_quality_matrix(n_domains, N, n_clusters,
                                     specialization_strength=0.8, rng=rng)
        queries = generate_queries(1000, n_domains, embed_dim,
                                   qm["domain_to_cluster"], rng)

        expert_centroids = compute_expert_centroids(
            qm["expert_to_domain"], queries["domain_centroids"],
            N, embed_dim, rng)

        # Build routers and measure latency
        latencies = {}

        # Pre-merge
        pm = PreMergeRouter(qm["Q"])
        lat = measure_premerge_latency(2000)
        latencies["premerge"] = lat["mean_us"]
        print(f"  premerge: {lat['mean_us']:.2f} us")

        # Hash ring
        hr = HashRingRouter(N)
        lat = measure_routing_latency(hr, queries["embeddings"], n_iters=2000)
        latencies["hash_ring"] = lat["mean_us"]
        print(f"  hash_ring: {lat['mean_us']:.2f} us")

        # Embedding similarity
        es = EmbeddingSimilarityRouter(expert_centroids)
        lat = measure_routing_latency(es, queries["embeddings"], n_iters=2000)
        latencies["embedding_sim"] = lat["mean_us"]
        print(f"  embedding_sim: {lat['mean_us']:.2f} us")

        # Tiny classifier (trained)
        oracle = OracleRouter(qm["Q"])
        train_labels = np.array([oracle.route_for_domain(d) for d in queries["domains"]])
        tc = TinyClassifierRouter(N, embed_dim, hidden_dim=32, rng=rng)
        tc.train(queries["embeddings"], train_labels, epochs=50)
        lat = measure_routing_latency(tc, queries["embeddings"], n_iters=2000)
        latencies["tiny_classifier"] = lat["mean_us"]
        print(f"  tiny_classifier: {lat['mean_us']:.2f} us")

        # Hierarchical
        hier = HierarchicalRouter(
            queries["cluster_centroids"], qm["expert_to_cluster"], N)
        lat = measure_routing_latency(hier, queries["embeddings"], n_iters=2000)
        latencies["hierarchical"] = lat["mean_us"]
        print(f"  hierarchical: {lat['mean_us']:.2f} us")

        scaling_results[N] = latencies

    return scaling_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-experts", type=int, default=30)
    parser.add_argument("--n-domains", type=int, default=15)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--specialization", type=float, default=0.8)
    parser.add_argument("--run-scaling", action="store_true",
                        help="Also run N-scaling experiment")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Main experiment
    results = run_experiment(
        n_domains=args.n_domains,
        n_experts=args.n_experts,
        n_clusters=args.n_clusters,
        embed_dim=args.embed_dim,
        specialization_strength=args.specialization,
        n_seeds=args.n_seeds,
    )

    # Optional scaling experiment
    if args.run_scaling:
        scaling = run_scaling_experiment(
            n_values=[10, 30, 50, 100],
            embed_dim=args.embed_dim,
            n_clusters=args.n_clusters,
        )
        results["scaling"] = scaling

    # Save
    out_dir = Path(__file__).parent
    out_path = Path(args.output) if args.output else out_dir / "results.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
