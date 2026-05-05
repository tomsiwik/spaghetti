# MATH.md — Per-(token, layer) Adapter Routing Cache toward 1M context

## Vision

Pierre achieves >1M token context by caching adapter selection per (token, layer)
as a tiny side tensor — like a KV cache but for routing decisions, not attention
keys/values. Combines with Gemma 4's existing sliding-window attention to break
the quadratic context bottleneck for adapter-augmented serving.

## Architecture

```
            PREFILL                                  DECODE
        (1M token document)                  (32 new tokens)
               │                                       │
               ▼                                       ▼
   For each token t in [0, 1M):              Token t at decode step:
     For each layer l in [0, 42):            (1) Lightning Indexer LOOKUP
       (1) Lightning Indexer scores              router_cache[t, l]
       (2) Top-2 selection                  (2) Compose adapters
       (3) Soft Gumbel weights                  weighted by cached scores
       (4) Write to cache:                  (3) Forward pass
            router_cache[t, l] =                base(x) + scale × x @ ΔW_composed
            (idx_top1, w_top1, idx_top2, w_top2)
       
     Cost: ~50µs per (token, layer)         Cost: ~100ns per (token, layer)
     Total: 1M × 42 × 50µs = 35 minutes     Total: 32 × 42 × 100ns = 134µs
                                            (essentially free)
```

## Cache structure

| Field | Type | Bytes |
|---|---|---|
| top-1 adapter index | uint8 (0-7) | 1 |
| top-1 weight | float16 | 2 |
| top-2 adapter index | uint8 | 1 |

Total: 4 bytes per (token, layer) cell

```
1M tokens × 42 layers × 4 bytes = 168 MB
```

Trivially fits alongside Gemma 4 E4B 4-bit (~2GB) + 7 PoLAR adapters (~35MB) on
M5 Pro 48GB unified memory.

## Theoretical grounding

### KV cache analog (DSv4 Section 3.6)

DeepSeek V4 introduces heterogeneous KV cache: state cache (per-request, SWA +
uncompressed tail) + block-allocated CSA/HCA cache for compressed entries. This
experiment introduces a **third tier**: routing cache (per-request, per-(token,
layer)) that stores adapter selection decisions.

The cache is correct under the assumption that adapter selection is a function
only of the token's hidden state at that layer, NOT of subsequent decode tokens.
This is the same assumption that makes attention KV caching work — the past is
fixed; only the present queries it.

### Why this works for 1M context specifically

The bottleneck for adapter-augmented serving at long context is the routing
recompute cost. Per token, computing the gate (M2P-gated softmax) takes ~50µs.
At 1M tokens, full recompute would be 50µs × 1M = 50 seconds JUST for routing —
unacceptable. Cached lookup at 100ns × 1M = 100ms. **500× speedup on routing.**

The actual attention compute at 1M context is handled by Gemma 4's existing
sliding window pattern (Gemma 4 alternates global + local attention). This
experiment doesn't modify attention; it only addresses adapter routing cost.

### Gemma 4's existing infrastructure for long context

Gemma 4 e4b uses sliding window attention in 5 of every 6 layers (window size
4096), with full attention in 1 of 6. For 1M context with W=4096:
- 5/6 layers: O(W·T) = 4M operations per token = manageable
- 1/6 layers: O(T²) — still O(T²) but only 7 layers, not 42

The compose+adapter overhead is what we're addressing here. Base attention's
cost is unchanged.

## Predictions

1. **K1 (memory)**: 1M × 42 × 4B = 168MB. Conservative budget 200MB. PASS expected.

2. **K2 (lookup latency)**: A single memory read on M5 Pro: ~50ns L3 hit, ~100ns
   DRAM. Conservative 100ns. PASS expected.

3. **K3 (cache correctness)**: identical adapter selection on recompute vs cached.
   Test: run prefill, cache; rerun routing on first 1K tokens with cache disabled;
   diff. Should be bit-identical. PASS expected by construction.

4. **K4 (end-to-end latency at 64K)**: baseline at 4K + 50ms. Math: routing
   cache turns per-token routing from 50µs → 100ns; at 64K context, savings =
   64000 × (50µs - 100ns) ≈ 3.2 seconds saved per decode token. With cache,
   per-decode token spends ~100ns × 42 layers = 4.2µs on routing (negligible).

5. **K5 (1M coherent output)**: this is the qualitative gate. Functional check
   that the architecture composes correctly at extreme context — no NaN, no
   degenerate output, no silent corruption.

## Implementation sketch

```python
# scripts/long_context_router.py — extends Lightning Indexer with cache

class CachedLightningRouter:
    def __init__(self, n_adapters: int, n_layers: int, max_tokens: int):
        # Pre-allocate cache
        self.cache_top1_idx = mx.zeros((max_tokens, n_layers), dtype=mx.uint8)
        self.cache_top1_w   = mx.zeros((max_tokens, n_layers), dtype=mx.float16)
        self.cache_top2_idx = mx.zeros((max_tokens, n_layers), dtype=mx.uint8)
        self.cache_top2_w   = mx.zeros((max_tokens, n_layers), dtype=mx.float16)
        self.indexer = LightningIndexer(...)  # from exp_pierre_lightning_indexer_routing

    def route_prefill(self, hidden_states_per_layer: list[mx.array]):
        """Compute and cache routing for prefill tokens."""
        for layer_idx, hs in enumerate(hidden_states_per_layer):
            T = hs.shape[1]
            scores = self.indexer.index_scores(hs)  # (T, n_adapters)
            top2 = mx.topk(scores, k=2, axis=-1)
            weights = nn.softmax(top2.values, axis=-1)
            # Write to cache
            self.cache_top1_idx[:T, layer_idx] = top2.indices[:, 0].astype(mx.uint8)
            self.cache_top1_w  [:T, layer_idx] = weights[:, 0].astype(mx.float16)
            self.cache_top2_idx[:T, layer_idx] = top2.indices[:, 1].astype(mx.uint8)
            self.cache_top2_w  [:T, layer_idx] = weights[:, 1].astype(mx.float16)

    def lookup_decode(self, token_pos: int, layer_idx: int) -> tuple:
        """O(1) lookup for cached routing at (token_pos, layer_idx)."""
        return (
            int(self.cache_top1_idx[token_pos, layer_idx].item()),
            float(self.cache_top1_w  [token_pos, layer_idx].item()),
            int(self.cache_top2_idx[token_pos, layer_idx].item()),
            float(self.cache_top2_w  [token_pos, layer_idx].item()),
        )
```

## Pre-registered KCs

K2158: Memory ≤200MB at 1M tokens × 42 layers × 4B
K2159: Lookup latency ≤100ns per (token, layer) on M5 Pro
K2160: Cache vs recompute → bit-identical adapter selection on 1K validation slice
K2161: 64K-context Pierre latency ≤ 4K-context baseline + 50ms
K2162: 1M-context produces coherent output (qualitative)

## Dependencies

- exp_pierre_lightning_indexer_routing (the routing primitive we cache)
- Finding #831 / `_FusedDeltaLinear` (the canonical composition pattern)
- Pierre's existing `attach.py` infrastructure (proper module replacement)

## Risks

1. **Gemma 4 E4B's sliding window attention may not extend to 1M tokens cleanly.**
   The model was trained at 128K context; 1M is 8× extrapolation. Test #5 is the
   diagnostic — if base output degenerates at 1M (regardless of routing), that's
   a base-model limitation, not a Pierre limitation.

2. **Cache write latency at prefill** dominates in the 1M case (~35min for
   pure routing). For interactive use this matters; for offline/RAG use it's
   acceptable. Mitigation: parallelize prefill routing across layers (each
   independent); current Lightning Indexer is per-layer parallelizable.

3. **Cache invalidation** under streaming context updates: if you add tokens to
   the front of context, all subsequent positions shift. Mitigation: cache is
   per-(absolute-position, layer); new prefix invalidates all. For append-only
   contexts (chat, document), straightforward.

## References

- DeepSeek V4 paper (2026): Section 2.3.1 (Lightning Indexer), Section 3.6 (KV cache)
- Finding #831 (Pierre): canonical composition pattern via _FusedDeltaLinear
- F#58 (research repo): top-2 routing beats uniform 1/N by 13.9%
- F#171 (research repo): SIPS+Gumbel-sigmoid recommended for N>25 routing
- ../talos-vs-macbook: NEON kernel pattern (the indexer key dot products are the
  perfect L1-resident workload — 7 adapters × 128 dims × FP16 = 1.8KB, fits L1)
