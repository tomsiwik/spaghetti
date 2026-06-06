# Adapter Hot Cache: Research Digest

## Hypothesis

Caching pre-merged adapter pair weights with LRU eviction achieves >= 80% hit
rate with top-20 cached pairs and >= 2x latency improvement, because routing
access patterns under Zipf-distributed traffic concentrate on a small number
of frequently-used pairs.

## Verdict: KILLED

K1 FAIL: Cache hit rate on domain-balanced traffic is 3.9% with K=20 (threshold: 50%).
Even under Zipf(alpha=1.0), hit rate is only 10.8% with K=20.

K2 PASS: Cache management overhead is negative (-0.185 ms), meaning cache
insertion is cheaper than average merge. No management penalty.

S2 PASS: Cache lookup is 2938x faster than on-demand merge (0.0001 ms vs 0.405 ms).
The mechanism works perfectly -- the problem is that hits are too rare.

## What This Experiment Is

An LRU cache for pre-merged LoRA adapter pair deltas in MLX unified memory.
When the softmax router selects top-2 adapters for a query, the pair ID is
looked up in a hash table. On hit, the cached merged delta is used directly
(skipping the merge computation). On miss, the merge is computed and the result
cached with LRU eviction.

## Key Results

### Access Pattern Distribution (N=50 adapters, top-2 routing, 5000 queries)

| Traffic Pattern | Unique Pairs | Top-20 Coverage | K for 80% Coverage |
|-----------------|-------------|-----------------|-------------------|
| Uniform (alpha=0.0) | 1070 / 1225 | 9.4% | 501 |
| Zipf(0.8) | 986 | 21.7% | 408 |
| Zipf(1.0) | 936 | 27.2% | 341 |
| Zipf(1.5) | 701 | 41.4% | 154 |
| Domain-balanced | 1079 | 10.1% | 510 |

**The fundamental problem:** With N=50 adapters, the pair space (1225 possibilities)
is too large relative to any practical cache size. Even under strong Zipf
concentration (alpha=1.5), 80% coverage requires 154 cached pairs.

### LRU Cache Hit Rate (with eviction dynamics)

LRU hit rates are substantially WORSE than static frequency analysis because
of eviction churn:

| Pattern | K=20 | K=50 | K=100 | K=200 |
|---------|------|------|-------|-------|
| Zipf(1.0) | 10.8% | 20.5% | 32.1% | 49.6% |
| Zipf(1.5) | 22.3% | 41.3% | 59.5% | 76.2% |
| Domain-balanced | 3.9% | 7.2% | 13.8% | 25.1% |

### Latency Breakdown (micro scale: d=64, r=4, L=4)

| Operation | Time |
|-----------|------|
| On-demand merge (2 adapters) | 0.405 ms |
| Cache lookup (hit) | 0.0001 ms |
| Cache miss + insert | 0.220 ms |
| Cache overhead (miss - merge) | -0.185 ms |

### End-to-End Speedup

| Scenario | K=20 | K=50 | K=100 |
|----------|------|------|-------|
| Zipf(1.0) | 1.07x | 1.15x | 1.25x |
| Domain-balanced | 1.01x | 1.04x | 1.09x |

Speedup is negligible because hit rates are too low.

### Memory Budget (Production Scale)

| Configuration | Memory Per Pair | K=20 Total |
|---------------|----------------|------------|
| All targets (7/layer, 28 layers) | 2569 MB | 50.2 GB |
| Attention-only (4/layer, 28 layers) | 1468 MB | 29.4 GB |

Production-scale caching is memory-infeasible. Even attention-only at K=20
consumes 29.4 GB -- more than the available budget after the base model.

### Weight Sensitivity

Routing weights vary 0.7-0.9 per query. The relative L2 distance between
cached deltas at different weight ratios is 37.2% -- too large to cache
unweighted deltas. Must either cache individual adapter deltas (not pairs)
and recompute the weighted sum, or cache pair deltas with quantized weights.

## Why It Failed

1. **Pair space combinatorics.** C(50,2) = 1225 pairs. Even concentrated
   Zipf distributions spread across too many pairs for small caches to capture.
   The mathematical bound: for Zipf(alpha=1), 80% coverage requires
   K = M * exp(-0.2 * H_{M,alpha}) which is O(M^0.8) -- sublinear but still
   hundreds of entries.

2. **Top-k routing expands the effective space.** With top-1, only N=50 entries
   exist. With top-2, the space explodes to C(N,2). The Zipf concentration on
   individual domains does NOT translate to proportional pair concentration
   because the secondary selection diffuses.

3. **LRU churn.** Under high-entropy access (many unique pairs), LRU evicts
   useful entries before they can be reused. The working set size exceeds cache
   capacity for any practical K.

4. **Memory at production scale.** Merged deltas are O(d^2) per target. At
   d=2560, each pair delta is 1.5-2.5 GB. Caching even 20 pairs exceeds the
   entire memory budget.

## What Would Fix It

1. **Top-1 routing instead of top-2.** With top-1, cache size = N = 50, and
   Zipf concentration maps directly. At alpha=1.0, top-20 would cover ~60%
   of single-adapter traffic. But top-1 loses composition quality.

2. **Fewer adapters (N < 15).** With N=10, C(10,2) = 45 pairs. K=20 covers
   44% even under uniform access. Practical for small deployments.

3. **Cache at adapter level, not pair level.** Store individual B@A deltas
   (N entries, not C(N,2)). Recompute weighted sum at query time. The sum
   is O(d^2) addition vs O(r*d^2) merge -- but still requires materialized
   deltas in memory.

4. **Runtime LoRA (factored form) eliminates the need for caching entirely.**
   From exp_inference_speed_10x: runtime LoRA (h @ A then result @ B) is
   97 tok/s -- no merge needed, no cache needed. The factored form is already
   an implicit "perfect cache" because A and B are always in memory.

## Key References

- S-LoRA (arxiv 2311.03285): Multi-tenant LoRA serving uses unified paging, not
  per-pair caching. Their approach keeps adapters in factored form.
- EdgeMoE (arxiv 2308.14352): Expert caching for on-device inference. Their
  setting has larger experts (full FFN layers) with fewer options, making
  caching viable. Our setting (small LoRA deltas, many pairs) is the opposite.
- CLONE (arxiv 2506.02847): Edge LoRA serving, focuses on router efficiency
  rather than weight caching.

## Limitations

1. Micro-scale dimensions (d=64 vs d=2560) -- merge times are faster at micro
   scale, reducing potential cache benefit. At production scale, merge is more
   expensive (0.83 ms vs 0.40 ms), but memory cost is also much worse.

2. Simulated routing (random with cluster bias) rather than a trained softmax
   router. A real router might show tighter clustering, but Phase 1 analysis
   shows even perfect Zipf(1.5) is insufficient.

3. Does not test cache warming strategies (pre-compute top-K pairs at startup).
   Warming would help initial hit rate but not steady-state for balanced traffic.

## What Was Learned

**The right caching strategy for multi-adapter serving is NOT pair-level weight
caching.** It is runtime LoRA (factored form), which is an implicit perfect
cache because A and B matrices are always resident. The merge operation that
caching tries to avoid is the wrong level of abstraction -- the factored form
eliminates it entirely.

This confirms and strengthens the finding from exp_inference_speed_10x:
runtime LoRA > pre-merge for ternary serving. The architectural decision is
settled: use runtime LoRA for routed experts, pre-merge only for always-on
adapters (instruction adapter).
