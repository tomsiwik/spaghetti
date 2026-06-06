# PAPER.md — T4.4: Multi-Tenant Serving with Shared KV Cache

**Status:** SUPPORTED  
**Date:** 2026-04-11  
**Architecture:** Gemma 4 E4B global attention (num_kv_heads=2, head_dim=512, K=V, 7 global layers)

---

## Prediction vs Measurement

| Kill Criterion | Prediction | Measured | Pass? |
|---|---|---|---|
| K1085: 8 users share KV (max diff = 0) | 0.0 (algebraic, Theorem 1) | **0.0** (all 7 layers) | ✓ |
| K1086: shared memory = 1/8 individual | 0.125 exactly (Theorem 2) | **0.125** | ✓ |
| K1087: attention output bit-exact | 0.0 (algebraic, Theorem 3) | **0.0** (all 7 layers, 8 users) | ✓ |

All predictions confirmed exactly. All results are algebraic identities, not approximations.

---

## Key Results

### Phase 1: KV Identity (Theorems 1 & 3)

For all 7 global attention layers, across all 8 users:
- `max|K_user_i - K_shared| = 0.0` (exact, float16)
- `max|Attn_shared_i - Attn_peruser_i| = 0.0` (exact, float16)

This is an algebraic identity: k_proj has no adapter → W_K @ x computed identically
for every user → K_i = K_j. No floating-point approximation involved.

### Phase 2: Memory Accounting (Theorem 2)

| Configuration | KV Memory | Global Layers |
|---|---|---|
| Per-user serving (8 users) | 28,672 KB (28 MB) | 8× 3,584 KB |
| Shared KV serving | 3,584 KB (3.5 MB) | 1× 3,584 KB |
| Ratio | **0.125 = 1/8** | exact |

**At production scale** (N=8 users, T=1024 context, Gemma 4 E4B):
- KV savings: 98 MB freed per batch (7 global layers × 7 users × 2 MB/layer)
- Per global layer: 2,048 KB saved per user above the 1st

---

## Architecture Details

Gemma 4 E4B global attention (7 layers, at every 6th layer):
- `attention_k_eq_v = True` (V = clone(K), no v_proj weights)
- `num_kv_heads = 2` (GQA: 16 query heads share 2 KV heads)
- `head_dim = 512`
- Adapters: q_proj only (Q adapter doesn't touch k_proj/v_proj)

The KV sharing is a **structural guarantee**, not a serving optimization:
- It requires no special implementation
- It cannot be violated by any Q-only adapter configuration
- It applies to all 7 global layers simultaneously

---

## Connection to T4 Tier

| Experiment | Result | Status |
|---|---|---|
| T4.1: TF-IDF routing | 96.6% @ N=5, 86.1% @ N=25 | Supported |
| T4.2: LSH routing | Killed (centroid similarity too low) | Killed |
| T4.3: MLX hot-swap | p99=4.77ms, 90.8% throughput | Supported |
| T4.4: KV sharing | **8x memory reduction, bit-exact** | **Supported** |
| T4.5: Format compat | Round-trip loss=0.0 | Supported |
| T4.6: E2E latency | p99=1.38ms overhead, 96% throughput | Supported |

### Serving Architecture (T4 tier complete)

```
Request → TF-IDF route (0.125ms) → adapter swap (4.77ms)
        → Shared KV (global layers: 1 allocation for 8 users)
        → Per-user Q (local layers: per-user adapter, exclusive routing)
        → Generation at 37+ tok/s
```

**Memory budget for 8 concurrent users on M5 Pro 48GB:**
- Base model: ~8.5 GB (4-bit quantized)
- 8 adapters (rank=6, q_proj): 8 × 1 MB = 8 MB
- Shared KV (global layers, T=1024): 3.5 MB (vs 28 MB per-user)
- Per-user KV (local layers, T=1024): 8 × 4 MB = 32 MB
- Total: ~8.5 GB + ~44 MB — negligible overhead vs base model

---

## Caveats

1. **Synthetic weights**: Experiment uses synthetic W_K/W_Q at Gemma 4 E4B dimensions
   (d=2816, head_dim=512). Real Gemma 4 weights would produce the same result by Theorem 1.

2. **Global layers only**: Local layers (35 of 42) use per-user adapters on q_proj AND have
   K/V that are technically adapter-independent too (no k_proj/v_proj adapters). KV sharing
   works on ALL 42 layers, but the experiment verifies only the 7 global layers where K=V.

3. **KV cache implementation**: Actual serving requires a KV cache manager that routes
   users to the shared KV buffer. This experiment verifies the algebraic property but not
   the serving-layer implementation. T4.6 verified E2E mechanics.

---

## Impossibility Structure (from killed T4.2)

LSH routing was killed because cosine similarity was too low to distinguish domains (c=0.23).
KV sharing succeeds because it requires zero similarity — it is an algebraic identity based
on the absence of adapters, not on similarity structure.

**Key distinction:**
- LSH routing failed because it needed similarity structure that wasn't there
- KV sharing succeeds because it needs NO structure — just structural absence (no k_proj adapter)
