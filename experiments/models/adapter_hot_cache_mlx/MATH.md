# MATH.md: Adapter Hot Cache for Pre-Merged Weights

## 1. Mechanism Definition

### Problem Setup

Given N adapters, a softmax router selects top-k (k=2) per query. Each selected
pair (i,j) requires merging their LoRA deltas into the base weight before the
forward pass:

```
W_merged = W_base + sum_{a in selected} s_a * B_a @ A_a
```

where s_a is the routing weight, B_a in R^{d_out x r}, A_a in R^{r x d_in}.

With N=50 adapters and k=2, there are C(50,2) = 1225 possible pairs. The merge
operation for each pair costs:

```
T_merge(k=2) = 2 * (r * d_in * d_out) FLOPs  (two B@A matmuls)
             + d_out * d_in additions          (accumulate into W_base)
```

From exp_benchmark_composition_latency_sweep: merge of N=2 adapters takes
0.83 ms uncompiled, 0.54 ms compiled (single layer, d=2560, r=16).
For 7 layers (full model): 4.57 ms uncompiled.

### Cache Definition

A **hot cache** stores pre-computed merged weight matrices for frequently-used
adapter pairs:

```
cache: Dict[(i,j), W_merged_{i,j}]  where i < j (canonical ordering)
W_merged_{i,j} = W_base + s_i * B_i @ A_i + s_j * B_j @ A_j
```

Cache lookup: O(1) hash table lookup. On hit: skip merge entirely (save T_merge).
On miss: compute merge, insert into cache, evict LRU entry if at capacity.

**Memory per cached pair:** For the full model with L=28 layers, 7 LoRA targets
per layer (q,k,v,o,gate,up,down), each target has d_in x d_out bf16 weights.
But we DON'T cache the full merged weight -- we cache the **merged delta**:

```
Delta_{i,j} = s_i * B_i @ A_i + s_j * B_j @ A_j    (shape: d_out x d_in per target)
```

Per target: d_out * d_in * 2 bytes (bf16) = 2560 * 2560 * 2 = 13.1 MB
Per layer (7 targets): 7 * 13.1 = 91.75 MB
Full model (28 layers): 28 * 91.75 = 2569 MB = 2.51 GB

**THIS IS TOO EXPENSIVE.** At 2.51 GB per cached pair, we can only cache ~16
pairs in the available memory budget (42.9 GB - 1.18 GB base - margins).

**Optimization: Cache only the per-adapter deltas, not pairs.**
Instead of caching C(N,2) pair merges, cache N individual deltas:

```
delta_i = B_i @ A_i    (shape: d_out x d_in per target)
```

Memory per adapter delta (full model): 2.51 GB (same size, but only N entries not C(N,2)).
Still too large for many adapters.

**Key insight from exp_inference_speed_10x:** Runtime LoRA (factored form) is
faster than pre-merge for ternary models. The bottleneck is NOT the B@A matmul
(~0.15 ms/target) but the accumulation into the dense weight. Therefore:

**Revised approach: Cache at the ROUTER DECISION level, not weight level.**

We cache the **routing decisions** and their associated pre-merged deltas for
the specific layer configuration that the router selected. Since the router
uses softmax top-k, and the routing weights are deterministic given the input
embedding, we can cache at the granularity of (pair_id, layer) or even just
track which pairs are hot and pre-compute their deltas lazily.

### Practical Cache Design

For micro-scale experiment (d=64, r=4, L=4 layers, 4 targets/layer):
- Per target delta: 64 * 64 * 4 bytes (float32) = 16 KB
- Per layer: 4 * 16 KB = 64 KB
- Full model delta per pair: 4 * 64 KB = 256 KB
- 100 cached pairs: 25 MB (easily fits)

For production scale (d=2560, r=16, L=28 layers, 7 targets/layer):
- Per target delta: 2560 * 2560 * 2 bytes (bf16) = 13.1 MB
- Per layer: 7 * 13.1 = 91.75 MB
- Full model delta per pair: 28 * 91.75 = 2569 MB
- INFEASIBLE to cache many pairs at this scale

**Therefore: we cache PARTIAL deltas (attention-only: 4 targets/layer) or
use the factored form cache (store B@A per adapter, not per pair).**

Factored delta per adapter:
- Per target: 2560 * 2560 * 2 = 13.1 MB
- Attention-only (4 targets): 4 * 13.1 = 52.4 MB per layer
- Full model: 28 * 52.4 = 1467 MB per adapter

Still large. But storing only the **top-k selected pair's merged delta** for
the CURRENT layer being computed, with LRU eviction, is practical.

## 2. Access Pattern Analysis (Why Caching Works)

### Zipf Distribution Model

In multi-tenant or multi-domain serving, query domains follow a Zipf distribution:

```
P(domain = i) = 1 / (i^alpha * H_{N,alpha})
```

where H_{N,alpha} = sum_{i=1}^{N} 1/i^alpha is the generalized harmonic number,
and alpha is the Zipf exponent (typically 0.8-1.2 for web traffic).

**References:**
- CLONE (arxiv 2506.02847): uses MoE router for dynamic LoRA selection on edge
  devices. Observes that "most queries concentrate on a few adapters."
- S-LoRA (arxiv 2311.03285): multi-tenant LoRA serving with unified paging.
  Reports Zipf-distributed adapter access with alpha ~1.0.
- EdgeMoE (arxiv 2308.14352): on-device inference with expert caching,
  explicitly models Zipf access patterns for cache sizing.

### Pair Access Probability

With top-2 routing and independent domain draws (worst case for caching):

```
P(pair (i,j)) = P(domain=i) * P(domain=j | domain!=i) + P(domain=j) * P(domain=i | domain!=i)
```

But routing is NOT random -- it's deterministic given the input embedding.
The softmax router assigns each query to a domain, then selects top-2.
Adjacent domains in the confusion matrix co-occur frequently.

From exp_softmax_router_scaling, the confusion clusters are:
- Cluster 1: philosophy/history/agriculture/creative_writing/science/environmental/politics/economics (8 domains)
- Cluster 2: education/engineering/sports (3 domains)
- Cluster 3: sociology/linguistics (2 domains)
- Cluster 4: medical/health_fitness (2 domains)
- Cluster 5: legal/finance (2 domains)
- Singletons: code, math, cooking, psychology, cybersecurity (5 domains)

Top-2 selections WITHIN a cluster are far more likely than cross-cluster.
Cluster 1 alone has C(8,2) = 28 pairs, but only ~10 are frequently co-selected
(adjacent in softmax probability space).

### Cache Hit Rate Model

For K cached pairs out of M = C(N,2) possible pairs, under Zipf(alpha) access:

```
hit_rate(K, alpha) = sum_{i=1}^{K} p_i / sum_{i=1}^{M} p_i
                   = H_{K,alpha} / H_{M,alpha}
```

where pairs are ordered by decreasing access frequency.

For alpha=1.0 (standard Zipf):
- K=20, M=1225: hit_rate = H_{20,1} / H_{1225,1} = 3.55 / 7.11 = 49.9%
- K=50, M=1225: hit_rate = H_{50,1} / H_{1225,1} = 4.50 / 7.11 = 63.3%
- K=100, M=1225: hit_rate = H_{100,1} / H_{1225,1} = 5.19 / 7.11 = 73.0%

For alpha=1.5 (concentrated):
- K=20, M=1225: hit_rate = H_{20,1.5} / H_{1225,1.5} = 1.95 / 2.28 = 85.5%
- K=50, M=1225: hit_rate = 2.07 / 2.28 = 90.8%

**For S1 (80% hit rate with K=20):** Requires alpha >= 1.4 under pure Zipf.
With routing clusters, effective alpha is higher because within-cluster pairs
dominate. The experiment will measure empirical alpha.

## 3. What Breaks It

### Flat distribution (alpha < 0.8)
If query domains are uniformly distributed, pair access is nearly uniform
across all 1225 pairs. Cache hit rate degrades to K/M = 20/1225 = 1.6%.

**Kill criterion K1 connection:** Domain-balanced traffic has alpha ~0 by design.
But "domain-balanced" does NOT mean "pair-balanced" -- within each domain, the
router's top-2 is deterministic, creating structural pair concentration even
under uniform domain distribution. The experiment tests this directly.

### Cache invalidation cost
On miss: compute merge (0.83 ms for k=2 uncompiled) + insert into cache
(memory allocation + hash update). If the eviction/insertion overhead exceeds
the merge savings, caching is net negative.

**Kill criterion K2 connection:** Net latency = P(hit) * T_lookup + P(miss) *
(T_merge + T_insert + T_evict) vs baseline T_merge always. Caching is
beneficial when:

```
P(hit) * T_lookup + (1-P(hit)) * (T_merge + T_overhead) < T_merge
P(hit) * (T_merge - T_lookup) > (1-P(hit)) * T_overhead
P(hit) > T_overhead / (T_merge - T_lookup + T_overhead)
```

If T_overhead ~ 0.01 ms (dict lookup + LRU update) and T_merge ~ 0.83 ms:
P(hit) > 0.01 / (0.83 + 0.01) = 1.2%. Extremely low threshold.

The real risk is MEMORY PRESSURE: cache entries consume memory, potentially
causing MLX cache evictions that slow down other operations.

### Routing weight variation
If routing weights s_i, s_j vary per query (not just which pair, but with what
weights), cached deltas are invalid unless we normalize or ignore weights.
Solution: quantize routing weights to a small set, or cache unweighted deltas
and apply weights at lookup time (one scalar multiply, negligible cost).

## 4. Assumptions

1. **Top-k routing produces deterministic pair selections given domain.**
   Justified by softmax router architecture (exp_softmax_router_scaling).
   If wrong: pair space is larger, hit rate drops.

2. **Access patterns have temporal locality.** Justified by real serving
   patterns (S-LoRA, EdgeMoE). If wrong: LRU evicts useful entries.

3. **Cache lookup is O(1) with negligible overhead.** Python dict is O(1)
   amortized. MLX memory operations for tensor retrieval are fast.
   If wrong: K2 fails.

4. **Cached deltas can be reused across queries.** Requires either fixed
   routing weights or weight-independent caching. We use unweighted deltas
   with per-query weight application.

## 5. Complexity Analysis

| Operation | Time | Memory |
|-----------|------|--------|
| Cache lookup | O(1) | O(1) |
| Cache miss merge (k=2) | O(k * r * d_in * d_out) | O(d_out * d_in) per target |
| Cache insertion | O(1) amortized | +1 entry |
| LRU eviction | O(1) with doubly-linked list | -1 entry |
| Total cache memory | - | O(K * L * T * d_out * d_in * dtype_size) |

At micro scale: K=20 pairs, L=4 layers, T=4 targets, d=64, float32:
Cache memory = 20 * 4 * 4 * 64 * 64 * 4 = 5.24 MB

## 6. Worked Example (d=64, N=10, k=2)

M = C(10,2) = 45 possible pairs.

Simulated Zipf(alpha=1.0) domain distribution for 1000 queries:
- Domain frequencies: [263, 132, 88, 66, 53, 44, 38, 33, 29, 26] (approx)
- Top-2 routing creates pairs: most frequent pairs are (0,1), (0,2), (0,3), (1,2)...
- With K=5 cached pairs: covers ~60% of traffic
- With K=10 cached pairs: covers ~75% of traffic

Latency comparison per query:
- No cache: T_merge = 0.83 ms (always merge)
- With cache (75% hit): 0.75 * 0.01 + 0.25 * 0.84 = 0.2175 ms
- Speedup: 0.83 / 0.2175 = 3.8x on average

## 7. Connection to Architecture

The hot cache sits in the serving pipeline between the router and the
forward pass:

```
Input -> Router (softmax top-k) -> pair_id = canonical(selected)
                                 -> Cache.get(pair_id)
                                    HIT:  use cached merged delta
                                    MISS: compute delta, cache it
                                 -> Forward pass with delta applied
```

This is complementary to:
- **Pre-merge for always-on adapters** (instruction adapter): merged once, no cache needed
- **Runtime LoRA for per-token routing**: factored form, no merge needed
- **Hot cache for per-SEQUENCE routing**: merge once per unique pair, cache for reuse

The cache is most valuable in the per-sequence routing regime confirmed by
exp_pointer_routing_no_merge (per-sequence is the correct granularity).

**Production reference:** S-LoRA (arxiv 2311.03285) uses unified paging for
multi-LoRA serving with similar caching concepts at GPU scale. Our approach
adapts this to unified memory (Apple Silicon) where CPU-GPU transfer is free.
EdgeMoE (arxiv 2308.14352) explicitly models LRU expert caching for on-device
serving -- directly analogous to our approach.
