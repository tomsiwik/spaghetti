#!/usr/bin/env python3
"""Experiment: Adapter Hot Cache for Pre-Merged Weights on MLX.

Kill criteria:
  K1: Cache hit rate < 50% on domain-balanced traffic
  K2: Cache management overhead > merge savings (net negative latency)

Success criteria:
  S1: Cache hit rate >= 80% with top-20 cached pairs
  S2: Latency improvement >= 2x on cache hits vs on-demand merge
"""

import gc
import json
import time
import math
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

# MLX memory safety
device = mx.device_info()
total = device["memory_size"]
mx.set_memory_limit(total - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# ---- Micro-scale config ----
D_IN = 64
D_OUT = 64
RANK = 4
N_LAYERS = 4
N_TARGETS = 4  # q, k, v, o (attention-only for speed)
LORA_SCALE = 1.0
N_ADAPTERS = 50
TOP_K = 2
N_QUERIES = 5000
CACHE_SIZES = [5, 10, 20, 50, 100, 200]
ZIPF_ALPHAS = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5]
SEED = 42


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ---- LRU Cache Implementation ----
class LRUAdapterCache:
    """LRU cache for pre-merged adapter pair deltas.

    Stores merged delta = sum_a s_a * B_a @ A_a for each (pair_id, layer, target).
    Key: (frozenset of adapter indices) -- canonical pair representation.
    Value: dict of {(layer, target): mx.array delta}
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple) -> Optional[dict]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: tuple, value: dict):
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            if len(self._cache) >= self.capacity:
                self._cache.popitem(last=False)
            self._cache[key] = value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_stats(self):
        self.hits = 0
        self.misses = 0

    def clear(self):
        self._cache.clear()
        self.reset_stats()


# ---- Adapter Generation ----
def generate_adapters(n_adapters: int, seed: int = 42):
    """Generate random LoRA adapters (A, B matrices for each layer/target)."""
    mx.random.seed(seed)
    adapters = {}
    for i in range(n_adapters):
        adapter = {}
        for layer in range(N_LAYERS):
            for target in range(N_TARGETS):
                A = mx.random.normal((RANK, D_IN)) * 0.01
                B = mx.random.normal((D_OUT, RANK)) * 0.01
                mx.eval(A, B)
                adapter[(layer, target)] = (A, B)
        adapters[i] = adapter
    return adapters


# ---- Merge Operations ----
def merge_pair_delta(adapters: dict, pair: tuple, weights: tuple) -> dict:
    """Compute merged delta for an adapter pair with routing weights.

    Returns dict: {(layer, target): delta_matrix}
    """
    i, j = pair
    w_i, w_j = weights
    deltas = {}
    for layer in range(N_LAYERS):
        for target in range(N_TARGETS):
            A_i, B_i = adapters[i][(layer, target)]
            A_j, B_j = adapters[j][(layer, target)]
            delta = w_i * LORA_SCALE * (B_i @ A_i) + w_j * LORA_SCALE * (B_j @ A_j)
            deltas[(layer, target)] = delta
    # Force evaluation to measure actual compute time
    mx.eval(*deltas.values())
    return deltas


def apply_delta_forward(base_weights: dict, delta: dict, x: mx.array) -> mx.array:
    """Simulate forward pass: (W_base + delta) @ x for each layer."""
    h = x
    for layer in range(N_LAYERS):
        for target in range(N_TARGETS):
            W = base_weights[(layer, target)]
            d = delta[(layer, target)]
            h = (W + d) @ h
    mx.eval(h)
    return h


# ---- Traffic Generation ----
def generate_zipf_traffic(n_queries: int, n_domains: int, alpha: float,
                          seed: int = 42) -> list:
    """Generate query traffic with Zipf-distributed domain access.

    Returns list of (pair_tuple, weights_tuple) for each query.
    """
    rng = np.random.default_rng(seed)

    if alpha == 0.0:
        # Uniform distribution
        domain_probs = np.ones(n_domains) / n_domains
    else:
        # Zipf distribution
        ranks = np.arange(1, n_domains + 1, dtype=np.float64)
        domain_probs = 1.0 / (ranks ** alpha)
        domain_probs /= domain_probs.sum()

    # Simulate softmax router: top-2 selection
    # For each query, draw primary domain from Zipf, then select
    # secondary from confusion-based distribution (nearby domains more likely)
    traffic = []
    for _ in range(n_queries):
        primary = rng.choice(n_domains, p=domain_probs)
        # Secondary: prefer "nearby" domains (simulating confusion clusters)
        # Use a distance-based weighting centered on primary
        distances = np.abs(np.arange(n_domains) - primary)
        secondary_probs = 1.0 / (1.0 + distances)
        secondary_probs[primary] = 0  # can't pick same domain twice
        secondary_probs /= secondary_probs.sum()
        secondary = rng.choice(n_domains, p=secondary_probs)

        # Canonical ordering
        pair = (min(primary, secondary), max(primary, secondary))

        # Routing weights from softmax (primary gets higher weight)
        w1 = 0.7 + rng.random() * 0.2  # primary: 0.7-0.9
        w2 = 1.0 - w1                   # secondary: 0.1-0.3
        weights = (w1, w2) if pair[0] == primary else (w2, w1)

        traffic.append((pair, weights))

    return traffic


def generate_domain_balanced_traffic(n_queries: int, n_domains: int,
                                     seed: int = 42) -> list:
    """Domain-balanced: equal queries per domain, but top-2 still cluster."""
    rng = np.random.default_rng(seed)
    traffic = []
    domains_per_query = n_queries // n_domains

    for d in range(n_domains):
        for _ in range(domains_per_query):
            primary = d
            distances = np.abs(np.arange(n_domains) - primary)
            secondary_probs = 1.0 / (1.0 + distances)
            secondary_probs[primary] = 0
            secondary_probs /= secondary_probs.sum()
            secondary = rng.choice(n_domains, p=secondary_probs)
            pair = (min(primary, secondary), max(primary, secondary))
            w1 = 0.7 + rng.random() * 0.2
            w2 = 1.0 - w1
            weights = (w1, w2) if pair[0] == primary else (w2, w1)
            traffic.append((pair, weights))

    # Shuffle
    rng.shuffle(traffic)
    return traffic


# ---- Phase 1: Access Pattern Analysis ----
def phase_access_patterns():
    """Analyze pair access frequency under different distributions."""
    print("\n=== Phase 1: Access Pattern Analysis ===")
    results = {}

    for alpha in ZIPF_ALPHAS:
        traffic = generate_zipf_traffic(N_QUERIES, N_ADAPTERS, alpha, SEED)
        pairs = [t[0] for t in traffic]

        # Count pair frequencies
        from collections import Counter
        pair_counts = Counter(pairs)
        n_unique_pairs = len(pair_counts)
        total = len(pairs)

        # Sort by frequency
        sorted_pairs = pair_counts.most_common()
        freqs = [c for _, c in sorted_pairs]
        cumulative = np.cumsum(freqs) / total

        # Find K for different hit rate thresholds
        k_for_50 = int(np.searchsorted(cumulative, 0.50)) + 1
        k_for_80 = int(np.searchsorted(cumulative, 0.80)) + 1
        k_for_90 = int(np.searchsorted(cumulative, 0.90)) + 1
        k_for_95 = int(np.searchsorted(cumulative, 0.95)) + 1

        # Top-20 hit rate
        top20_rate = float(cumulative[min(19, len(cumulative) - 1)])

        results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "n_unique_pairs": n_unique_pairs,
            "max_possible_pairs": N_ADAPTERS * (N_ADAPTERS - 1) // 2,
            "top1_frequency": float(freqs[0] / total),
            "top5_cumulative": float(cumulative[min(4, len(cumulative) - 1)]),
            "top10_cumulative": float(cumulative[min(9, len(cumulative) - 1)]),
            "top20_cumulative": top20_rate,
            "top50_cumulative": float(cumulative[min(49, len(cumulative) - 1)]),
            "k_for_50pct": k_for_50,
            "k_for_80pct": k_for_80,
            "k_for_90pct": k_for_90,
            "k_for_95pct": k_for_95,
        }
        print(f"  alpha={alpha}: {n_unique_pairs} unique pairs, "
              f"top-20 hit rate={top20_rate:.1%}, "
              f"K for 80%={k_for_80}")

    # Domain-balanced traffic
    traffic_balanced = generate_domain_balanced_traffic(N_QUERIES, N_ADAPTERS, SEED)
    pairs_balanced = [t[0] for t in traffic_balanced]
    from collections import Counter
    pair_counts_balanced = Counter(pairs_balanced)
    n_unique_balanced = len(pair_counts_balanced)
    sorted_balanced = pair_counts_balanced.most_common()
    freqs_balanced = [c for _, c in sorted_balanced]
    cumulative_balanced = np.cumsum(freqs_balanced) / len(pairs_balanced)

    top20_balanced = float(cumulative_balanced[min(19, len(cumulative_balanced) - 1)])
    k80_balanced = int(np.searchsorted(cumulative_balanced, 0.80)) + 1

    results["domain_balanced"] = {
        "alpha": "balanced",
        "n_unique_pairs": n_unique_balanced,
        "top1_frequency": float(freqs_balanced[0] / len(pairs_balanced)),
        "top20_cumulative": top20_balanced,
        "k_for_50pct": int(np.searchsorted(cumulative_balanced, 0.50)) + 1,
        "k_for_80pct": k80_balanced,
        "k_for_90pct": int(np.searchsorted(cumulative_balanced, 0.90)) + 1,
    }
    print(f"  domain_balanced: {n_unique_balanced} unique pairs, "
          f"top-20 hit rate={top20_balanced:.1%}, "
          f"K for 80%={k80_balanced}")

    return results


# ---- Phase 2: Cache Hit Rate vs Size ----
def phase_cache_hit_rate():
    """Measure actual cache hit rate with LRU eviction under different traffic patterns."""
    print("\n=== Phase 2: Cache Hit Rate vs Size ===")
    results = {}

    test_configs = [
        ("zipf_0.0", 0.0),
        ("zipf_0.8", 0.8),
        ("zipf_1.0", 1.0),
        ("zipf_1.5", 1.5),
        ("domain_balanced", None),
    ]

    for name, alpha in test_configs:
        if alpha is not None:
            traffic = generate_zipf_traffic(N_QUERIES, N_ADAPTERS, alpha, SEED)
        else:
            traffic = generate_domain_balanced_traffic(N_QUERIES, N_ADAPTERS, SEED)

        config_results = {}
        for cache_size in CACHE_SIZES:
            cache = LRUAdapterCache(capacity=cache_size)
            for pair, weights in traffic:
                result = cache.get(pair)
                if result is None:
                    # Simulate storing a cached entry (just the key, not actual tensors)
                    cache.put(pair, True)

            config_results[f"K={cache_size}"] = {
                "cache_size": cache_size,
                "hit_rate": round(cache.hit_rate, 4),
                "hits": cache.hits,
                "misses": cache.misses,
            }
            cache.clear()

        results[name] = config_results
        # Print summary for K=20
        k20 = config_results.get("K=20", {})
        print(f"  {name}: K=20 hit rate = {k20.get('hit_rate', 0):.1%}")

    return results


# ---- Phase 3: Latency Measurement ----
def phase_latency_measurement():
    """Measure actual merge latency vs cache lookup latency on MLX."""
    print("\n=== Phase 3: Latency Measurement ===")
    log_memory("pre-latency")

    adapters = generate_adapters(N_ADAPTERS, SEED)
    base_weights = {}
    for layer in range(N_LAYERS):
        for target in range(N_TARGETS):
            W = mx.random.normal((D_OUT, D_IN)) * 0.1
            mx.eval(W)
            base_weights[(layer, target)] = W

    x = mx.random.normal((D_IN,))
    mx.eval(x)

    # Warmup
    for _ in range(5):
        delta = merge_pair_delta(adapters, (0, 1), (0.7, 0.3))
        _ = apply_delta_forward(base_weights, delta, x)

    # --- Measure merge latency (no cache) ---
    n_measure = 100
    merge_times = []
    for trial in range(n_measure):
        pair = (trial % N_ADAPTERS, (trial + 1) % N_ADAPTERS)
        if pair[0] == pair[1]:
            pair = (pair[0], (pair[1] + 1) % N_ADAPTERS)
        pair = (min(pair), max(pair))

        t0 = time.perf_counter()
        delta = merge_pair_delta(adapters, pair, (0.7, 0.3))
        t1 = time.perf_counter()
        merge_times.append((t1 - t0) * 1000)
        del delta

    merge_mean = float(np.mean(merge_times))
    merge_std = float(np.std(merge_times))
    merge_p50 = float(np.percentile(merge_times, 50))
    merge_p95 = float(np.percentile(merge_times, 95))

    print(f"  Merge latency: {merge_mean:.3f} +/- {merge_std:.3f} ms "
          f"(p50={merge_p50:.3f}, p95={merge_p95:.3f})")

    # --- Measure cache lookup latency ---
    cache = LRUAdapterCache(capacity=100)
    # Pre-fill cache with some deltas
    for i in range(min(50, N_ADAPTERS)):
        j = (i + 1) % N_ADAPTERS
        pair = (min(i, j), max(i, j))
        delta = merge_pair_delta(adapters, pair, (0.7, 0.3))
        cache.put(pair, delta)

    lookup_times = []
    for trial in range(n_measure):
        pair = (trial % min(50, N_ADAPTERS),
                (trial + 1) % min(50, N_ADAPTERS))
        if pair[0] == pair[1]:
            pair = (pair[0], (pair[1] + 1) % N_ADAPTERS)
        pair = (min(pair), max(pair))

        t0 = time.perf_counter()
        result = cache.get(pair)
        if result is not None:
            # Simulate applying cached delta (just read it)
            _ = result
        t1 = time.perf_counter()
        lookup_times.append((t1 - t0) * 1000)

    lookup_mean = float(np.mean(lookup_times))
    lookup_std = float(np.std(lookup_times))
    lookup_p50 = float(np.percentile(lookup_times, 50))
    lookup_p95 = float(np.percentile(lookup_times, 95))

    print(f"  Cache lookup: {lookup_mean:.4f} +/- {lookup_std:.4f} ms "
          f"(p50={lookup_p50:.4f}, p95={lookup_p95:.4f})")

    speedup = merge_mean / lookup_mean if lookup_mean > 0 else float('inf')
    print(f"  Speedup (hit vs merge): {speedup:.1f}x")

    # --- Measure cache miss + insert latency ---
    miss_times = []
    miss_cache = LRUAdapterCache(capacity=100)
    for trial in range(n_measure):
        pair = (trial % N_ADAPTERS, (trial + 7) % N_ADAPTERS)
        if pair[0] == pair[1]:
            pair = (pair[0], (pair[1] + 1) % N_ADAPTERS)
        pair = (min(pair), max(pair))

        t0 = time.perf_counter()
        result = miss_cache.get(pair)
        if result is None:
            delta = merge_pair_delta(adapters, pair, (0.7, 0.3))
            miss_cache.put(pair, delta)
        t1 = time.perf_counter()
        miss_times.append((t1 - t0) * 1000)

    miss_mean = float(np.mean(miss_times))
    miss_std = float(np.std(miss_times))

    print(f"  Cache miss (merge + insert): {miss_mean:.3f} +/- {miss_std:.3f} ms")

    # --- Cache management overhead ---
    overhead = miss_mean - merge_mean
    print(f"  Cache overhead (miss - merge): {overhead:.4f} ms")

    results = {
        "merge_latency": {
            "mean_ms": round(merge_mean, 4),
            "std_ms": round(merge_std, 4),
            "p50_ms": round(merge_p50, 4),
            "p95_ms": round(merge_p95, 4),
        },
        "cache_lookup_latency": {
            "mean_ms": round(lookup_mean, 6),
            "std_ms": round(lookup_std, 6),
            "p50_ms": round(lookup_p50, 6),
            "p95_ms": round(lookup_p95, 6),
        },
        "cache_miss_latency": {
            "mean_ms": round(miss_mean, 4),
            "std_ms": round(miss_std, 4),
        },
        "speedup_hit_vs_merge": round(speedup, 2),
        "cache_overhead_ms": round(overhead, 4),
    }

    cleanup(adapters, base_weights, cache, miss_cache)
    log_memory("post-latency")
    return results


# ---- Phase 4: End-to-End Simulation ----
def phase_e2e_simulation():
    """Simulate full serving with cache vs without, measure effective throughput."""
    print("\n=== Phase 4: End-to-End Simulation ===")
    log_memory("pre-e2e")

    adapters = generate_adapters(N_ADAPTERS, SEED)
    base_weights = {}
    for layer in range(N_LAYERS):
        for target in range(N_TARGETS):
            W = mx.random.normal((D_OUT, D_IN)) * 0.1
            mx.eval(W)
            base_weights[(layer, target)] = W

    x = mx.random.normal((D_IN,))
    mx.eval(x)

    n_queries_e2e = 1000
    results = {}

    for scenario_name, alpha in [("zipf_1.0", 1.0), ("domain_balanced", None)]:
        if alpha is not None:
            traffic = generate_zipf_traffic(n_queries_e2e, N_ADAPTERS, alpha, SEED)
        else:
            traffic = generate_domain_balanced_traffic(n_queries_e2e, N_ADAPTERS, SEED)

        # --- No cache baseline ---
        t0 = time.perf_counter()
        for pair, weights in traffic:
            delta = merge_pair_delta(adapters, pair, weights)
            _ = apply_delta_forward(base_weights, delta, x)
            del delta
        t_no_cache = time.perf_counter() - t0

        # --- With cache (K=20) ---
        cache = LRUAdapterCache(capacity=20)
        t0 = time.perf_counter()
        for pair, weights in traffic:
            cached = cache.get(pair)
            if cached is not None:
                _ = apply_delta_forward(base_weights, cached, x)
            else:
                delta = merge_pair_delta(adapters, pair, weights)
                cache.put(pair, delta)
                _ = apply_delta_forward(base_weights, delta, x)
        t_cache_20 = time.perf_counter() - t0
        hit_rate_20 = cache.hit_rate
        cache.clear()

        # --- With cache (K=50) ---
        cache50 = LRUAdapterCache(capacity=50)
        t0 = time.perf_counter()
        for pair, weights in traffic:
            cached = cache50.get(pair)
            if cached is not None:
                _ = apply_delta_forward(base_weights, cached, x)
            else:
                delta = merge_pair_delta(adapters, pair, weights)
                cache50.put(pair, delta)
                _ = apply_delta_forward(base_weights, delta, x)
        t_cache_50 = time.perf_counter() - t0
        hit_rate_50 = cache50.hit_rate
        cache50.clear()

        # --- With cache (K=100) ---
        cache100 = LRUAdapterCache(capacity=100)
        t0 = time.perf_counter()
        for pair, weights in traffic:
            cached = cache100.get(pair)
            if cached is not None:
                _ = apply_delta_forward(base_weights, cached, x)
            else:
                delta = merge_pair_delta(adapters, pair, weights)
                cache100.put(pair, delta)
                _ = apply_delta_forward(base_weights, delta, x)
        t_cache_100 = time.perf_counter() - t0
        hit_rate_100 = cache100.hit_rate
        cache100.clear()

        speedup_20 = t_no_cache / t_cache_20 if t_cache_20 > 0 else 0
        speedup_50 = t_no_cache / t_cache_50 if t_cache_50 > 0 else 0
        speedup_100 = t_no_cache / t_cache_100 if t_cache_100 > 0 else 0

        results[scenario_name] = {
            "n_queries": n_queries_e2e,
            "no_cache_total_ms": round(t_no_cache * 1000, 2),
            "no_cache_per_query_ms": round(t_no_cache / n_queries_e2e * 1000, 3),
            "cache_K20": {
                "total_ms": round(t_cache_20 * 1000, 2),
                "per_query_ms": round(t_cache_20 / n_queries_e2e * 1000, 3),
                "hit_rate": round(hit_rate_20, 4),
                "speedup": round(speedup_20, 2),
            },
            "cache_K50": {
                "total_ms": round(t_cache_50 * 1000, 2),
                "per_query_ms": round(t_cache_50 / n_queries_e2e * 1000, 3),
                "hit_rate": round(hit_rate_50, 4),
                "speedup": round(speedup_50, 2),
            },
            "cache_K100": {
                "total_ms": round(t_cache_100 * 1000, 2),
                "per_query_ms": round(t_cache_100 / n_queries_e2e * 1000, 3),
                "hit_rate": round(hit_rate_100, 4),
                "speedup": round(speedup_100, 2),
            },
        }

        print(f"  {scenario_name}:")
        print(f"    No cache: {t_no_cache/n_queries_e2e*1000:.3f} ms/query")
        print(f"    K=20: {t_cache_20/n_queries_e2e*1000:.3f} ms/query "
              f"(hit={hit_rate_20:.1%}, speedup={speedup_20:.2f}x)")
        print(f"    K=50: {t_cache_50/n_queries_e2e*1000:.3f} ms/query "
              f"(hit={hit_rate_50:.1%}, speedup={speedup_50:.2f}x)")
        print(f"    K=100: {t_cache_100/n_queries_e2e*1000:.3f} ms/query "
              f"(hit={hit_rate_100:.1%}, speedup={speedup_100:.2f}x)")

    cleanup(adapters, base_weights)
    log_memory("post-e2e")
    return results


# ---- Phase 5: Memory Overhead Analysis ----
def phase_memory_overhead():
    """Measure actual memory consumed by cached deltas."""
    print("\n=== Phase 5: Memory Overhead ===")
    log_memory("pre-memory")
    mx.reset_peak_memory()

    adapters = generate_adapters(N_ADAPTERS, SEED)

    mem_before = mx.get_active_memory()
    cache = LRUAdapterCache(capacity=100)
    for i in range(min(100, N_ADAPTERS * (N_ADAPTERS - 1) // 2)):
        a = i % N_ADAPTERS
        b = (i + 1) % N_ADAPTERS
        if a == b:
            b = (b + 1) % N_ADAPTERS
        pair = (min(a, b), max(a, b))
        delta = merge_pair_delta(adapters, pair, (0.7, 0.3))
        cache.put(pair, delta)

    mx.eval()
    mem_after = mx.get_active_memory()
    cache_memory_mb = (mem_after - mem_before) / 1e6

    # Theoretical per-pair memory at micro scale
    theoretical_per_pair_bytes = N_LAYERS * N_TARGETS * D_OUT * D_IN * 4  # float32
    theoretical_per_pair_mb = theoretical_per_pair_bytes / 1e6
    n_cached = len(cache._cache)
    theoretical_total_mb = theoretical_per_pair_mb * n_cached

    # Scale to production
    prod_per_pair_mb = 28 * 7 * 2560 * 2560 * 2 / 1e6  # bf16
    prod_attn_only_mb = 28 * 4 * 2560 * 2560 * 2 / 1e6

    results = {
        "micro_scale": {
            "n_cached_pairs": n_cached,
            "measured_cache_mb": round(cache_memory_mb, 2),
            "theoretical_per_pair_kb": round(theoretical_per_pair_mb * 1000, 2),
            "theoretical_total_mb": round(theoretical_total_mb, 2),
        },
        "production_scale_estimate": {
            "per_pair_all_targets_mb": round(prod_per_pair_mb, 1),
            "per_pair_attn_only_mb": round(prod_attn_only_mb, 1),
            "k20_all_targets_gb": round(20 * prod_per_pair_mb / 1000, 2),
            "k20_attn_only_gb": round(20 * prod_attn_only_mb / 1000, 2),
            "k100_all_targets_gb": round(100 * prod_per_pair_mb / 1000, 2),
            "k100_attn_only_gb": round(100 * prod_attn_only_mb / 1000, 2),
            "fits_in_48gb_budget": 20 * prod_attn_only_mb / 1000 < 30,
        },
    }

    print(f"  Micro: {n_cached} pairs = {cache_memory_mb:.2f} MB "
          f"(theoretical: {theoretical_total_mb:.2f} MB)")
    print(f"  Production: K=20 attn-only = "
          f"{20 * prod_attn_only_mb / 1000:.2f} GB")

    cleanup(adapters, cache)
    log_memory("post-memory")
    return results


# ---- Phase 6: Routing Weight Sensitivity ----
def phase_weight_sensitivity():
    """Test if ignoring routing weight variation in cache invalidates quality."""
    print("\n=== Phase 6: Weight Sensitivity ===")

    adapters = generate_adapters(10, SEED)

    # Compare: exact weights vs approximate (quantized) weights
    pair = (0, 1)
    exact_weights_list = [(0.7, 0.3), (0.75, 0.25), (0.8, 0.2), (0.85, 0.15), (0.9, 0.1)]

    # Compute deltas with different weights
    deltas = []
    for w in exact_weights_list:
        delta = merge_pair_delta(adapters, pair, w)
        deltas.append(delta)

    # Compute L2 distance between deltas
    base_delta = deltas[0]  # reference: (0.7, 0.3)
    distances = []
    for i, (delta, w) in enumerate(zip(deltas, exact_weights_list)):
        total_dist = 0
        total_norm = 0
        for key in delta:
            diff = delta[key] - base_delta[key]
            dist = float(mx.sqrt(mx.sum(diff * diff)).item())
            norm = float(mx.sqrt(mx.sum(base_delta[key] * base_delta[key])).item())
            total_dist += dist
            total_norm += norm
        relative_dist = total_dist / total_norm if total_norm > 0 else 0
        distances.append({
            "weights": w,
            "relative_l2_distance": round(relative_dist, 6),
        })

    # Conclusion: if relative distance is small (<5%), we can cache unweighted
    # deltas and apply weights as scalar multiply at lookup time
    max_relative = max(d["relative_l2_distance"] for d in distances)

    results = {
        "weight_variations": distances,
        "max_relative_l2": round(max_relative, 6),
        "can_cache_unweighted": max_relative < 0.05,
        "recommendation": (
            "Cache individual adapter deltas (B@A), apply routing weights at lookup"
            if max_relative > 0.01
            else "Can cache pair deltas with fixed weights"
        ),
    }

    print(f"  Max relative L2 distance: {max_relative:.6f}")
    print(f"  Recommendation: {results['recommendation']}")

    cleanup(adapters)
    return results


# ---- Main ----
def main():
    t0 = time.time()
    log_memory("start")

    # Phase 1: Access patterns
    access_results = phase_access_patterns()
    log_memory("after-access")

    # Phase 2: Cache hit rates
    hit_rate_results = phase_cache_hit_rate()
    log_memory("after-hit-rate")

    # Phase 3: Latency
    latency_results = phase_latency_measurement()
    log_memory("after-latency")

    # Phase 4: E2E simulation
    e2e_results = phase_e2e_simulation()
    log_memory("after-e2e")

    # Phase 5: Memory
    memory_results = phase_memory_overhead()
    log_memory("after-memory")

    # Phase 6: Weight sensitivity
    weight_results = phase_weight_sensitivity()
    log_memory("after-weight")

    # ---- Kill criteria assessment ----
    # K1: Cache hit rate < 50% on domain-balanced traffic
    balanced_k20_hit = hit_rate_results.get("domain_balanced", {}).get("K=20", {}).get("hit_rate", 0)
    k1_pass = balanced_k20_hit >= 0.50

    # K2: Cache management overhead > merge savings
    cache_overhead = latency_results.get("cache_overhead_ms", 0)
    merge_latency = latency_results.get("merge_latency", {}).get("mean_ms", 1)
    k2_pass = cache_overhead < merge_latency  # overhead less than one merge

    # S1: Cache hit rate >= 80% with K=20
    s1_pass = balanced_k20_hit >= 0.80
    zipf1_k20_hit = hit_rate_results.get("zipf_1.0", {}).get("K=20", {}).get("hit_rate", 0)
    s1_zipf = zipf1_k20_hit >= 0.80

    # S2: Latency improvement >= 2x on cache hits
    speedup = latency_results.get("speedup_hit_vs_merge", 0)
    s2_pass = speedup >= 2.0

    verdict = "SUPPORTED" if k1_pass and k2_pass else "KILLED"

    results = {
        "experiment": "adapter_hot_cache_mlx",
        "config": {
            "d_in": D_IN,
            "d_out": D_OUT,
            "rank": RANK,
            "n_layers": N_LAYERS,
            "n_targets": N_TARGETS,
            "n_adapters": N_ADAPTERS,
            "top_k": TOP_K,
            "n_queries": N_QUERIES,
        },
        "phase1_access_patterns": access_results,
        "phase2_cache_hit_rates": hit_rate_results,
        "phase3_latency": latency_results,
        "phase4_e2e_simulation": e2e_results,
        "phase5_memory": memory_results,
        "phase6_weight_sensitivity": weight_results,
        "kill_criteria": {
            "K1_balanced_hit_rate_50pct": {
                "value": round(balanced_k20_hit, 4),
                "threshold": 0.50,
                "pass": k1_pass,
            },
            "K2_overhead_less_than_merge": {
                "overhead_ms": round(cache_overhead, 4),
                "merge_ms": round(merge_latency, 4),
                "pass": k2_pass,
            },
        },
        "success_criteria": {
            "S1_balanced_hit_rate_80pct_k20": {
                "value": round(balanced_k20_hit, 4),
                "threshold": 0.80,
                "pass": s1_pass,
            },
            "S1_zipf1_hit_rate_80pct_k20": {
                "value": round(zipf1_k20_hit, 4),
                "threshold": 0.80,
                "pass": s1_zipf,
            },
            "S2_speedup_2x": {
                "value": round(speedup, 2),
                "threshold": 2.0,
                "pass": s2_pass,
            },
        },
        "verdict": verdict,
        "total_time_s": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  K1 (balanced hit rate >= 50%): {'PASS' if k1_pass else 'FAIL'} "
          f"({balanced_k20_hit:.1%})")
    print(f"  K2 (overhead < merge): {'PASS' if k2_pass else 'FAIL'} "
          f"({cache_overhead:.4f} ms < {merge_latency:.4f} ms)")
    print(f"  S1 balanced (hit >= 80% K=20): {'PASS' if s1_pass else 'FAIL'} "
          f"({balanced_k20_hit:.1%})")
    print(f"  S1 zipf1.0 (hit >= 80% K=20): {'PASS' if s1_zipf else 'FAIL'} "
          f"({zipf1_k20_hit:.1%})")
    print(f"  S2 (speedup >= 2x): {'PASS' if s2_pass else 'FAIL'} ({speedup:.1f}x)")
    print(f"Total time: {results['total_time_s']}s")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
