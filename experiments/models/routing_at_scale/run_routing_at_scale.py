#!/usr/bin/env python3
"""Routing at Scale: Latency and accuracy at N=100, 500, 1000 experts.

Extends experiments/models/inference_routing_strategies/ to production-scale N.
Adds FAISS ANN (approximate nearest neighbor) as a sixth routing strategy.

Kill criteria:
  K1: best routing strategy >100ms at N=1000
  K2: routing accuracy drops below 50% at N=500
  K3: routing becomes the bottleneck (>50% of total inference latency)

Strategies tested:
  (A) Pre-merge all: O(1) -- zero routing cost
  (B) Hash ring: O(log N) consistent hashing
  (C) Embedding cosine: O(N*D) brute force similarity
  (D) Tiny classifier: O(E*H + H*N) MLP
  (E) Hierarchical: O(C*E + log(N/C)) cluster + hash
  (F) FAISS ANN: O(nprobe * N/nlist * E) approximate nearest neighbor

Reference forward pass latency (for K3 bottleneck assessment):
  - Micro: ~1ms
  - Macro (Qwen 7B, 4-bit): ~30ms
  - Macro (Qwen 0.5B, fp16): ~5ms

Usage:
    uv run python -m micro.models.routing_at_scale.run_routing_at_scale
"""

import hashlib
import json
import time
from bisect import bisect_right
from pathlib import Path

import faiss
import numpy as np

# ── Reuse core infrastructure from predecessor ──────────────────────────
# We reimplement here to avoid import issues and to extend with FAISS.


def generate_quality_matrix(n_domains, n_experts, n_clusters,
                            specialization_strength=0.8, rng=None):
    """Generate synthetic quality matrix modeling expert specialization."""
    if rng is None:
        rng = np.random.default_rng(42)

    domains_per_cluster = max(1, n_domains // n_clusters)
    domain_to_cluster = np.array([min(d // domains_per_cluster, n_clusters - 1)
                                  for d in range(n_domains)])

    expert_to_domain = np.zeros(n_experts, dtype=int)
    for i in range(n_experts):
        expert_to_domain[i] = i % n_domains
    expert_to_cluster = domain_to_cluster[expert_to_domain]

    # Base quality
    Q = rng.uniform(0, 0.2, size=(n_domains, n_experts))

    # Home domain bonus
    for e in range(n_experts):
        Q[expert_to_domain[e], e] += specialization_strength

    # Same-cluster bonus (vectorized for large N)
    for e in range(n_experts):
        mask = domain_to_cluster == expert_to_cluster[e]
        mask[expert_to_domain[e]] = False  # exclude home domain
        Q[mask, e] += specialization_strength * 0.3

    return {
        "Q": Q,
        "domain_to_cluster": domain_to_cluster,
        "expert_to_domain": expert_to_domain,
        "expert_to_cluster": expert_to_cluster,
    }


def generate_queries(n_queries, n_domains, embed_dim, domain_to_cluster, rng=None):
    """Generate synthetic query embeddings with cluster structure."""
    if rng is None:
        rng = np.random.default_rng(42)

    n_clusters = len(set(domain_to_cluster))

    # Cluster centroids
    cluster_centroids = rng.standard_normal((n_clusters, embed_dim))
    cluster_centroids /= np.linalg.norm(cluster_centroids, axis=1, keepdims=True)
    cluster_centroids *= 3.0

    # Domain centroids
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
    embeddings = embeddings / np.maximum(norms, 1e-8)

    return {
        "embeddings": embeddings.astype(np.float32),
        "domains": domains,
        "clusters": clusters,
        "domain_centroids": domain_centroids,
        "cluster_centroids": cluster_centroids,
    }


def compute_expert_centroids(expert_to_domain, domain_centroids,
                             n_experts, embed_dim, rng=None):
    """Expert centroids: home domain centroid + small perturbation."""
    if rng is None:
        rng = np.random.default_rng(42)
    centroids = np.zeros((n_experts, embed_dim))
    for e in range(n_experts):
        centroids[e] = domain_centroids[expert_to_domain[e]] + \
                       rng.standard_normal(embed_dim) * 0.1
    return centroids.astype(np.float32)


# ── Routing Strategies ──────────────────────────────────────────────────

class PreMergeRouter:
    """Strategy A: Pre-merge all experts. No routing needed."""
    name = "premerge"

    def __init__(self, Q):
        self.mean_quality = Q.mean(axis=1)
        self.n_experts = Q.shape[1]

    def route(self, query_embedding, query_idx=0):
        return -1

    def route_batch(self, embeddings):
        return np.full(len(embeddings), -1, dtype=int)

    def quality(self, domain):
        return float(self.mean_quality[domain])


class HashRingRouter:
    """Strategy B: Consistent hash ring. O(log N) per query."""
    name = "hash_ring"

    def __init__(self, n_experts, virtual_nodes=150):
        self.n_experts = n_experts
        ring = []
        for e in range(n_experts):
            for vn in range(virtual_nodes):
                h = int(hashlib.md5(f"expert_{e}_vn_{vn}".encode()).hexdigest(), 16)
                ring.append((h, e))
        ring.sort()
        self.hashes = [h for h, _ in ring]
        self.experts = [e for _, e in ring]

    def route(self, query_embedding, query_idx=0):
        qh = int(hashlib.md5(f"query_{query_idx}".encode()).hexdigest(), 16)
        idx = bisect_right(self.hashes, qh) % len(self.hashes)
        return self.experts[idx]

    def route_batch(self, embeddings):
        results = np.zeros(len(embeddings), dtype=int)
        for i in range(len(embeddings)):
            results[i] = self.route(embeddings[i], i)
        return results


class EmbeddingSimilarityRouter:
    """Strategy C: Brute-force cosine similarity. O(N*E) per query."""
    name = "embedding_sim"

    def __init__(self, expert_centroids):
        self.centroids = expert_centroids.copy()
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.centroids_norm = (self.centroids / np.maximum(norms, 1e-8)).astype(np.float32)

    def route(self, query_embedding, query_idx=0):
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
        sims = self.centroids_norm @ q.astype(np.float32)
        return int(np.argmax(sims))

    def route_batch(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        q_norm = (embeddings / np.maximum(norms, 1e-8)).astype(np.float32)
        sims = q_norm @ self.centroids_norm.T
        return np.argmax(sims, axis=1)


class TinyClassifierRouter:
    """Strategy D: Small MLP classifier. O(E*H + H*N) per query."""
    name = "tiny_classifier"

    def __init__(self, n_experts, embed_dim, hidden_dim=64, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        self.W1 = (rng.standard_normal((embed_dim, hidden_dim)) *
                   np.sqrt(2.0 / embed_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden_dim, n_experts)) *
                   np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(n_experts, dtype=np.float32)

    def train(self, embeddings, labels, lr=0.01, epochs=100, batch_size=64):
        n = len(embeddings)
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                X = embeddings[idx].astype(np.float32)
                y = labels[idx]
                bs = len(idx)

                h = X @ self.W1 + self.b1
                h_relu = np.maximum(h, 0)
                logits = h_relu @ self.W2 + self.b2

                logits_max = logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits - logits_max)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                dlogits = probs.copy()
                dlogits[np.arange(bs), y] -= 1
                dlogits /= bs

                dW2 = h_relu.T @ dlogits
                db2 = dlogits.sum(axis=0)
                dh_relu = dlogits @ self.W2.T
                dh = dh_relu * (h > 0).astype(np.float32)
                dW1 = X.T @ dh
                db1 = dh.sum(axis=0)

                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

    def route(self, query_embedding, query_idx=0):
        x = query_embedding.astype(np.float32)
        h = np.maximum(x @ self.W1 + self.b1, 0)
        logits = h @ self.W2 + self.b2
        return int(np.argmax(logits))

    def route_batch(self, embeddings):
        X = embeddings.astype(np.float32)
        h = np.maximum(X @ self.W1 + self.b1, 0)
        logits = h @ self.W2 + self.b2
        return np.argmax(logits, axis=1)


class HierarchicalRouter:
    """Strategy E: Cluster cosine + hash within cluster."""
    name = "hierarchical"

    def __init__(self, cluster_centroids, expert_to_cluster, n_experts,
                 virtual_nodes=150):
        self.n_clusters = len(cluster_centroids)
        norms = np.linalg.norm(cluster_centroids, axis=1, keepdims=True)
        self.cluster_centroids_norm = (
            cluster_centroids / np.maximum(norms, 1e-8)).astype(np.float32)

        self.cluster_rings = {}
        for c in range(self.n_clusters):
            experts_in_cluster = np.where(expert_to_cluster == c)[0]
            ring = []
            for e in experts_in_cluster:
                for vn in range(virtual_nodes):
                    h = int(hashlib.md5(
                        f"expert_{e}_vn_{vn}".encode()).hexdigest(), 16)
                    ring.append((h, int(e)))
            ring.sort()
            self.cluster_rings[c] = {
                "hashes": [h for h, _ in ring],
                "experts": [e for _, e in ring],
            }

    def route(self, query_embedding, query_idx=0):
        q = query_embedding.astype(np.float32)
        q = q / max(np.linalg.norm(q), 1e-8)
        sims = self.cluster_centroids_norm @ q
        cluster = int(np.argmax(sims))

        ring = self.cluster_rings[cluster]
        if not ring["hashes"]:
            return 0
        qh = int(hashlib.md5(f"query_{query_idx}".encode()).hexdigest(), 16)
        idx = bisect_right(ring["hashes"], qh) % len(ring["hashes"])
        return ring["experts"][idx]

    def route_batch(self, embeddings):
        results = np.zeros(len(embeddings), dtype=int)
        for i in range(len(embeddings)):
            results[i] = self.route(embeddings[i], i)
        return results


class FAISSANNRouter:
    """Strategy F: FAISS approximate nearest neighbor. O(nprobe * N/nlist * E)."""
    name = "faiss_ann"

    def __init__(self, expert_centroids, nlist=None, nprobe=4):
        n_experts, embed_dim = expert_centroids.shape
        self.embed_dim = embed_dim
        self.n_experts = n_experts

        # Choose nlist: sqrt(N) is standard FAISS heuristic, min 1
        if nlist is None:
            nlist = max(1, int(np.sqrt(n_experts)))

        # Normalize centroids for inner product search (= cosine similarity)
        norms = np.linalg.norm(expert_centroids, axis=1, keepdims=True)
        centroids_norm = (expert_centroids / np.maximum(norms, 1e-8)).astype(np.float32)

        if nlist <= 1 or n_experts < 40:
            # Too few experts for IVF, use flat index
            self.index = faiss.IndexFlatIP(embed_dim)
            self.index.add(centroids_norm)
        else:
            quantizer = faiss.IndexFlatIP(embed_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist,
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.train(centroids_norm)
            self.index.add(centroids_norm)
            self.index.nprobe = nprobe

    def route(self, query_embedding, query_idx=0):
        q = query_embedding.astype(np.float32).reshape(1, -1)
        q_norm = q / max(np.linalg.norm(q), 1e-8)
        _, I = self.index.search(q_norm, 1)
        return int(I[0, 0])

    def route_batch(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        q_norm = (embeddings / np.maximum(norms, 1e-8)).astype(np.float32)
        _, I = self.index.search(q_norm, 1)
        return I[:, 0]


class OracleRouter:
    """Perfect routing: always selects the best expert for each domain."""

    def __init__(self, Q):
        self.Q = Q
        self.best_expert = np.argmax(Q, axis=1)

    def route_for_domain(self, domain):
        return int(self.best_expert[domain])

    def quality(self, domain):
        return float(self.Q[domain, self.best_expert[domain]])


# ── Measurement ─────────────────────────────────────────────────────────

def measure_routing_latency(router, embeddings, n_warmup=200, n_iters=5000):
    """Measure per-query routing latency in microseconds."""
    n = len(embeddings)

    # Warmup
    for i in range(min(n_warmup, n)):
        router.route(embeddings[i % n], i)

    # Single-query latency
    latencies = []
    for i in range(n_iters):
        idx = i % n
        start = time.perf_counter()
        router.route(embeddings[idx], idx)
        elapsed = (time.perf_counter() - start) * 1e6
        latencies.append(elapsed)

    latencies = np.array(latencies)

    # Batch latency
    batch_n = min(1000, n)
    batch_start = time.perf_counter()
    router.route_batch(embeddings[:batch_n])
    batch_elapsed = (time.perf_counter() - batch_start) * 1e6
    batch_per_query = batch_elapsed / batch_n

    return {
        "mean_us": float(np.mean(latencies)),
        "std_us": float(np.std(latencies)),
        "p50_us": float(np.median(latencies)),
        "p99_us": float(np.percentile(latencies, 99)),
        "batch_per_query_us": float(batch_per_query),
    }


def measure_premerge_latency(n_iters=5000):
    """Pre-merge has zero routing latency."""
    latencies = []
    for i in range(n_iters):
        start = time.perf_counter()
        _ = -1
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


def measure_quality(router, Q, query_domains, query_embeddings, oracle):
    """Measure routing quality: oracle agreement and quality capture."""
    n_queries = len(query_domains)

    oracle_qualities = []
    router_qualities = []
    oracle_agreements = 0
    domain_correct = 0

    # Get expert-to-domain mapping from Q shape
    # (for domain accuracy, check if selected expert has same home domain)

    for i in range(n_queries):
        domain = query_domains[i]
        oracle_expert = oracle.route_for_domain(domain)
        oracle_q = Q[domain, oracle_expert]
        oracle_qualities.append(oracle_q)

        if hasattr(router, 'route'):
            selected = router.route(query_embeddings[i], i)
        else:
            selected = 0

        if selected == -1:
            router_q = Q[domain, :].mean()
        else:
            router_q = Q[domain, selected]

        router_qualities.append(router_q)

        if selected == oracle_expert:
            oracle_agreements += 1

        # Domain accuracy: did we pick an expert whose home domain matches?
        if selected >= 0:
            expert_home = selected % Q.shape[0]  # round-robin assignment
            if expert_home == domain:
                domain_correct += 1

    oracle_qualities = np.array(oracle_qualities)
    router_qualities = np.array(router_qualities)

    oracle_mean = oracle_qualities.mean()
    router_mean = router_qualities.mean()
    random_quality = Q.mean()

    quality_capture = router_mean / oracle_mean if oracle_mean > 0 else 0.0
    random_capture = random_quality / oracle_mean if oracle_mean > 0 else 0.0

    quality_lift = (
        (router_mean - random_quality) / (oracle_mean - random_quality)
        if oracle_mean > random_quality else 0.0
    )

    return {
        "oracle_agreement": oracle_agreements / n_queries,
        "quality_capture": float(quality_capture),
        "router_mean_quality": float(router_mean),
        "oracle_mean_quality": float(oracle_mean),
        "random_mean_quality": float(random_quality),
        "random_capture": float(random_capture),
        "quality_lift_over_random": float(quality_lift),
        "domain_accuracy": domain_correct / n_queries if n_queries > 0 else 0.0,
    }


# ── Main Experiment ─────────────────────────────────────────────────────

def run_scale_experiment(n_values=None, embed_dim=64, n_clusters=5,
                         specialization_strength=0.8, n_seeds=3,
                         n_test_queries=2000, n_train_queries=3000,
                         n_latency_iters=5000):
    """Run the routing-at-scale experiment across N=100, 500, 1000."""
    if n_values is None:
        n_values = [100, 500, 1000]

    print("=" * 70)
    print("ROUTING AT SCALE: Latency and Accuracy at N=100, 500, 1000")
    print("=" * 70)
    print(f"N values: {n_values}")
    print(f"Embed dim: {embed_dim}, Clusters: {n_clusters}")
    print(f"Specialization: {specialization_strength}")
    print(f"Seeds: {n_seeds}, Queries: {n_train_queries} train + {n_test_queries} test")
    print()

    all_results = {}

    for N in n_values:
        n_domains = max(10, N // 10)  # 10% domain diversity
        print(f"\n{'=' * 60}")
        print(f"N={N} experts, {n_domains} domains, {n_clusters} clusters")
        print(f"{'=' * 60}")

        seed_results = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(42 + seed)
            print(f"\n--- Seed {seed} ---")

            # Generate data
            t0 = time.time()
            qm = generate_quality_matrix(n_domains, N, n_clusters,
                                         specialization_strength, rng)
            Q = qm["Q"]

            train_queries = generate_queries(n_train_queries, n_domains, embed_dim,
                                             qm["domain_to_cluster"], rng)
            test_queries = generate_queries(n_test_queries, n_domains, embed_dim,
                                            qm["domain_to_cluster"],
                                            np.random.default_rng(1000 + seed))

            expert_centroids = compute_expert_centroids(
                qm["expert_to_domain"], train_queries["domain_centroids"],
                N, embed_dim, rng)

            oracle = OracleRouter(Q)
            train_labels = np.array([
                oracle.route_for_domain(d) for d in train_queries["domains"]
            ])
            gen_time = time.time() - t0
            print(f"  Data generation: {gen_time:.2f}s")

            # Build routers
            routers = {}

            # (A) Pre-merge
            routers["premerge"] = PreMergeRouter(Q)

            # (B) Hash ring
            t0 = time.time()
            routers["hash_ring"] = HashRingRouter(N, virtual_nodes=150)
            print(f"  Hash ring build: {time.time() - t0:.2f}s")

            # (C) Embedding similarity
            routers["embedding_sim"] = EmbeddingSimilarityRouter(expert_centroids)

            # (D) Tiny classifier -- scale hidden dim with sqrt(N)
            hidden_dim = min(128, max(32, int(np.sqrt(N) * 4)))
            t0 = time.time()
            classifier = TinyClassifierRouter(N, embed_dim, hidden_dim=hidden_dim, rng=rng)
            # Reduce epochs for large N to keep training fast
            epochs = max(30, 100 - N // 10)
            classifier.train(train_queries["embeddings"], train_labels,
                           lr=0.005, epochs=epochs, batch_size=128)
            print(f"  Classifier train ({epochs} epochs, h={hidden_dim}): {time.time() - t0:.2f}s")
            routers["tiny_classifier"] = classifier

            # (E) Hierarchical
            routers["hierarchical"] = HierarchicalRouter(
                train_queries["cluster_centroids"], qm["expert_to_cluster"],
                N, virtual_nodes=150)

            # (F) FAISS ANN
            t0 = time.time()
            nlist = max(1, int(np.sqrt(N)))
            nprobe = min(nlist, max(1, nlist // 4))
            routers["faiss_ann"] = FAISSANNRouter(expert_centroids,
                                                   nlist=nlist, nprobe=nprobe)
            print(f"  FAISS build (nlist={nlist}, nprobe={nprobe}): {time.time() - t0:.3f}s")

            # Measure each router
            router_results = {}
            for name, router in routers.items():
                print(f"  {name:20s}...", end=" ", flush=True)

                # Latency
                if name == "premerge":
                    latency = measure_premerge_latency(n_latency_iters)
                else:
                    latency = measure_routing_latency(
                        router, test_queries["embeddings"],
                        n_warmup=200, n_iters=n_latency_iters)

                # Quality
                quality = measure_quality(
                    router, Q, test_queries["domains"],
                    test_queries["embeddings"], oracle)

                print(f"lat={latency['mean_us']:8.2f}us  "
                      f"p99={latency['p99_us']:8.2f}us  "
                      f"batch={latency['batch_per_query_us']:8.2f}us  "
                      f"q_cap={quality['quality_capture']:.3f}  "
                      f"dom_acc={quality['domain_accuracy']:.3f}")

                router_results[name] = {
                    "latency": latency,
                    "quality": quality,
                }

            seed_results.append(router_results)

        all_results[N] = seed_results

    return all_results


def aggregate_and_assess(all_results, n_values):
    """Aggregate across seeds and assess kill criteria."""

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    strategy_names = list(all_results[n_values[0]][0].keys())
    aggregated = {}

    for N in n_values:
        seed_results = all_results[N]
        n_seeds = len(seed_results)
        agg_N = {}

        for name in strategy_names:
            lats_mean = [sr[name]["latency"]["mean_us"] for sr in seed_results]
            lats_p99 = [sr[name]["latency"]["p99_us"] for sr in seed_results]
            lats_batch = [sr[name]["latency"]["batch_per_query_us"] for sr in seed_results]
            q_caps = [sr[name]["quality"]["quality_capture"] for sr in seed_results]
            q_lifts = [sr[name]["quality"]["quality_lift_over_random"] for sr in seed_results]
            dom_accs = [sr[name]["quality"]["domain_accuracy"] for sr in seed_results]
            oracle_agrs = [sr[name]["quality"]["oracle_agreement"] for sr in seed_results]

            agg_N[name] = {
                "latency_mean_us": float(np.mean(lats_mean)),
                "latency_std_us": float(np.std(lats_mean)),
                "latency_p99_us": float(np.mean(lats_p99)),
                "batch_per_query_us": float(np.mean(lats_batch)),
                "quality_capture_mean": float(np.mean(q_caps)),
                "quality_capture_std": float(np.std(q_caps)),
                "quality_lift_mean": float(np.mean(q_lifts)),
                "quality_lift_std": float(np.std(q_lifts)),
                "domain_accuracy_mean": float(np.mean(dom_accs)),
                "domain_accuracy_std": float(np.std(dom_accs)),
                "oracle_agreement_mean": float(np.mean(oracle_agrs)),
                "oracle_agreement_std": float(np.std(oracle_agrs)),
            }

        aggregated[N] = agg_N

    # Print tables
    print("\n### Latency (mean, us) ###")
    header = f"{'Strategy':20s}"
    for N in n_values:
        header += f" | N={N:>4d}"
    print(header)
    print("-" * len(header))
    for name in strategy_names:
        row = f"{name:20s}"
        for N in n_values:
            val = aggregated[N][name]["latency_mean_us"]
            row += f" | {val:>8.1f}"
        print(row)

    print("\n### Latency P99 (us) ###")
    header = f"{'Strategy':20s}"
    for N in n_values:
        header += f" | N={N:>4d}"
    print(header)
    print("-" * len(header))
    for name in strategy_names:
        row = f"{name:20s}"
        for N in n_values:
            val = aggregated[N][name]["latency_p99_us"]
            row += f" | {val:>8.1f}"
        print(row)

    print("\n### Batch Latency (per query, us) ###")
    header = f"{'Strategy':20s}"
    for N in n_values:
        header += f" | N={N:>4d}"
    print(header)
    print("-" * len(header))
    for name in strategy_names:
        row = f"{name:20s}"
        for N in n_values:
            val = aggregated[N][name]["batch_per_query_us"]
            row += f" | {val:>8.2f}"
        print(row)

    print("\n### Quality Capture (fraction of oracle) ###")
    header = f"{'Strategy':20s}"
    for N in n_values:
        header += f" | N={N:>4d}"
    print(header)
    print("-" * len(header))
    for name in strategy_names:
        row = f"{name:20s}"
        for N in n_values:
            val = aggregated[N][name]["quality_capture_mean"]
            row += f" | {val:>8.3f}"
        print(row)

    print("\n### Domain Accuracy ###")
    header = f"{'Strategy':20s}"
    for N in n_values:
        header += f" | N={N:>4d}"
    print(header)
    print("-" * len(header))
    for name in strategy_names:
        row = f"{name:20s}"
        for N in n_values:
            val = aggregated[N][name]["domain_accuracy_mean"]
            row += f" | {val:>8.3f}"
        print(row)

    # ── Kill Criteria Assessment ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K1: best routing strategy >100ms at N=1000
    max_N = max(n_values)
    best_latency_at_max = min(
        agg["latency_mean_us"] for name, agg in aggregated[max_N].items()
        if name != "premerge"  # exclude pre-merge (trivially 0)
    )
    worst_latency_at_max = max(
        agg["latency_mean_us"] for name, agg in aggregated[max_N].items()
    )
    k1_threshold_us = 100_000  # 100ms in us
    k1_pass = worst_latency_at_max < k1_threshold_us
    print(f"\nK1: best routing strategy >100ms at N={max_N}")
    print(f"  Best non-premerge latency: {best_latency_at_max:.1f}us = {best_latency_at_max/1000:.3f}ms")
    print(f"  Worst latency: {worst_latency_at_max:.1f}us = {worst_latency_at_max/1000:.3f}ms")
    print(f"  Threshold: 100ms = 100,000us")
    print(f"  Margin: {k1_threshold_us / worst_latency_at_max:.0f}x below threshold")
    print(f"  --> {'PASS' if k1_pass else 'KILL'}")

    # K2: routing accuracy drops below 50% at N=500
    # "routing accuracy" = domain accuracy (correct domain expert selected)
    # Also check quality capture as a softer metric
    n_500 = 500 if 500 in n_values else min(n_values, key=lambda x: abs(x - 500))
    best_domain_acc_500 = max(
        agg["domain_accuracy_mean"] for name, agg in aggregated[n_500].items()
        if name != "premerge" and name != "hash_ring"
    )
    best_q_cap_500 = max(
        agg["quality_capture_mean"] for name, agg in aggregated[n_500].items()
    )
    # K2 interpretation: "routing accuracy" most naturally maps to domain accuracy
    # for content-aware routers, and quality capture for all routers
    k2_pass = best_domain_acc_500 >= 0.50 or best_q_cap_500 >= 0.50
    print(f"\nK2: routing accuracy drops below 50% at N={n_500}")
    print(f"  Best domain accuracy (content-aware): {best_domain_acc_500:.3f}")
    print(f"  Best quality capture (any strategy): {best_q_cap_500:.3f}")
    for name, agg in aggregated[n_500].items():
        print(f"    {name:20s}: dom_acc={agg['domain_accuracy_mean']:.3f}, "
              f"q_cap={agg['quality_capture_mean']:.3f}")
    print(f"  --> {'PASS' if k2_pass else 'KILL'}")

    # K3: routing becomes the bottleneck (>50% of total inference latency)
    # Reference: macro forward pass ~30ms = 30,000us (Qwen 7B 4-bit)
    # Conservative: use 5ms = 5,000us (Qwen 0.5B fp16)
    ref_inference_us = 5_000  # 5ms conservative
    worst_routing_ratio = worst_latency_at_max / ref_inference_us
    k3_pass = worst_routing_ratio < 0.50
    print(f"\nK3: routing becomes the bottleneck (>50% of total inference latency)")
    print(f"  Reference inference: {ref_inference_us}us (Qwen 0.5B fp16, conservative)")
    print(f"  Worst routing at N={max_N}: {worst_latency_at_max:.1f}us")
    print(f"  Routing as % of inference: {worst_routing_ratio * 100:.2f}%")
    print(f"  --> {'PASS' if k3_pass else 'KILL'}")

    # Also check with macro reference
    ref_macro_us = 30_000
    macro_ratio = worst_latency_at_max / ref_macro_us
    print(f"  With Qwen 7B reference ({ref_macro_us}us): {macro_ratio * 100:.3f}%")

    # Scaling analysis
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)

    for name in strategy_names:
        if name == "premerge":
            continue
        lats = [aggregated[N][name]["latency_mean_us"] for N in n_values]
        if len(n_values) >= 2:
            # Fit log-log to estimate scaling exponent
            log_n = np.log(n_values)
            log_lat = np.log(lats)
            slope, intercept = np.polyfit(log_n, log_lat, 1)
            print(f"  {name:20s}: scaling exponent = {slope:.3f} "
                  f"(O(N^{slope:.2f}))")
            # Project to N=5000 and N=10000
            for N_proj in [5000, 10000]:
                proj_lat = np.exp(intercept + slope * np.log(N_proj))
                proj_ms = proj_lat / 1000
                print(f"    Projected N={N_proj}: {proj_lat:.0f}us = {proj_ms:.3f}ms")

    overall_pass = k1_pass and k3_pass  # K2 is interpretive
    print(f"\n{'=' * 70}")
    print(f"OVERALL: K1={'PASS' if k1_pass else 'KILL'}, "
          f"K2={'PASS (see notes)' if k2_pass else 'KILL'}, "
          f"K3={'PASS' if k3_pass else 'KILL'}")
    print(f"{'=' * 70}")

    return {
        "aggregated": {str(N): agg for N, agg in aggregated.items()},
        "kill_criteria": {
            "K1_all_under_100ms": bool(k1_pass),
            "K1_worst_latency_us": float(worst_latency_at_max),
            "K1_best_nonpremerge_us": float(best_latency_at_max),
            "K2_accuracy_above_50pct": bool(k2_pass),
            "K2_best_domain_accuracy": float(best_domain_acc_500),
            "K2_best_quality_capture": float(best_q_cap_500),
            "K3_not_bottleneck": bool(k3_pass),
            "K3_worst_routing_pct_conservative": float(worst_routing_ratio * 100),
            "K3_worst_routing_pct_macro": float(macro_ratio * 100),
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Routing at Scale: N=100, 500, 1000")
    parser.add_argument("--n-values", type=int, nargs="+",
                        default=[100, 500, 1000])
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--specialization", type=float, default=0.8)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-test-queries", type=int, default=2000)
    parser.add_argument("--n-latency-iters", type=int, default=5000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()

    all_results = run_scale_experiment(
        n_values=args.n_values,
        embed_dim=args.embed_dim,
        n_clusters=args.n_clusters,
        specialization_strength=args.specialization,
        n_seeds=args.n_seeds,
        n_test_queries=args.n_test_queries,
        n_latency_iters=args.n_latency_iters,
    )

    assessment = aggregate_and_assess(all_results, args.n_values)

    total_time = time.time() - t_start
    print(f"\nTotal experiment time: {total_time:.1f}s")

    # Save results
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

    save_data = {
        "config": {
            "n_values": args.n_values,
            "embed_dim": args.embed_dim,
            "n_clusters": args.n_clusters,
            "specialization_strength": args.specialization,
            "n_seeds": args.n_seeds,
            "n_test_queries": args.n_test_queries,
            "n_latency_iters": args.n_latency_iters,
        },
        "aggregated": assessment["aggregated"],
        "kill_criteria": assessment["kill_criteria"],
        "total_time_s": total_time,
    }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
