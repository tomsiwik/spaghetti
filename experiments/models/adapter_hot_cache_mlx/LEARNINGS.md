# Learnings: exp_adapter_hot_cache_mlx

## Core Finding

Pair-level adapter weight caching is fundamentally impractical for N >= 15 adapters
with top-2 routing. The combinatorial explosion C(N,2) creates too many unique pairs
for any practical cache size, even under strongly concentrated Zipf access patterns.

## Critical Discoveries

### 1. Pair space defeats caching

With N=50 adapters, C(50,2) = 1225 possible pairs. Under Zipf(alpha=1.0), 80% hit
rate requires caching 341 pairs. Under domain-balanced traffic, 510 pairs. No
reasonable memory budget supports this at production scale (each pair delta is 1.5-2.5 GB).

### 2. Individual domain Zipf does NOT imply pair Zipf

The key mathematical insight: Zipf concentration on individual domains d_i diffuses
when extended to pairs. If P(d_i) ~ 1/i^alpha, then P(pair(i,j)) ~ P(d_i) * P(d_j|d_i).
The secondary selection (proximity-biased in our simulation, confusion-biased in real
routing) spreads each popular domain across multiple pairs, flattening the pair distribution.

Top-20 coverage at Zipf(1.0): 27.2% statically, 10.8% with LRU eviction. Far below
the 80% target.

### 3. LRU eviction makes things worse

Static frequency analysis overestimates cache benefit. LRU eviction under high-entropy
access causes "churn" -- useful entries are evicted before reuse. The gap between static
coverage and LRU hit rate is ~2.5x (27.2% static vs 10.8% LRU at Zipf(1.0), K=20).

### 4. Cache lookup is essentially free

Python dict lookup for cached MLX tensors: 0.0001 ms. The mechanism works perfectly --
the problem is exclusively hit rate, not overhead. K2 passes trivially.

### 5. Memory makes production caching infeasible

At d=2560 (production), each cached pair delta consumes:
- All targets: 2569 MB
- Attention-only: 1468 MB

K=20 attention-only = 29.4 GB. This ALONE exceeds the usable memory after the base model
(~1.2 GB) and adapters (~45 MB each * N=50 = 2.25 GB factored). Total would be ~33 GB,
leaving only ~10 GB for KV cache and system.

### 6. Routing weight variation prevents naive caching

Routing weights vary 0.7-0.9 for the primary adapter. This creates 37% relative L2
difference in the merged delta. Cannot cache "generic" pair deltas -- must either:
(a) cache per-adapter deltas and recompute weighted sum, or
(b) cache with quantized weight bins (adding another combinatorial dimension).

## Design Implications

### Runtime LoRA is the answer, not caching

The experiment definitively proves that pair-level weight caching is a dead end for
multi-adapter serving at scale. The correct architecture:

1. **Always-on adapters** (instruction adapter): pre-merge once into base. No cache.
2. **Routed experts** (per-sequence): runtime LoRA in factored form (h @ A @ B).
   No merge needed, no cache needed. A and B are always in memory (45 MB each).
3. **If top-k > 1**: accumulate factored LoRA outputs (sum of h @ A_i @ B_i * w_i).
   This is O(k * r * d) per token vs O(d^2) for merge. Always faster.

### Cache is useful only at small N

For N <= 10 (C(10,2) = 45 pairs), caching becomes practical:
- K=20 covers ~44% even under uniform access
- Memory: 20 * 256 KB (micro) = 5 MB
- At production scale: still 29 GB, so factored form still wins

### The fundamental asymmetry

Caching trades memory for compute. But LoRA's factored form already provides
the optimal compute-memory tradeoff: O(k*r*d) compute with O(N*r*d) memory.
Materializing the merged delta (O(d^2) memory) is ALWAYS worse than keeping
the factors separate.

## Contradicting Evidence

None. This result is fully consistent with:
- exp_inference_speed_10x: runtime LoRA > pre-merge for ternary
- exp_benchmark_composition_latency_sweep: merge scales linearly with N
- exp_memory_budget_analysis: 853 adapters fit in factored form
