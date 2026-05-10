# MATH.md — DSA top-k key selection as joint sparse-attention + routing prior

## Hypothesis

DeepSeek Sparse Attention (DSA, V3.2) selects top-k keys per query via a
lightning indexer. The same per-token relevance signal can be **dual-used**:
(a) the sparse attention pattern, (b) the per-token PoLAR routing prior.

Pierre's existing `exp_pierre_kv_cached_layer_routing_1m` design caches
adapter selection per (token, layer). DSA's top-k indices can serve as
that prior — one computation, two purposes.

> **Does a single DSA top-k key-selection mask per layer reused as
> (a) sparse attention pattern AND (b) per-token PoLAR routing prior
> halve routing-cache memory at 128K context while producing routing
> decisions that beat per-token X-LoRA at one-pass cost?**

## Pre-registered Kill Criteria

- **K1 (BEHAVIORAL)** Routing-via-DSA-indices behavioral score ≥ Fisher-Rao K=7 baseline (no regression from sharing the indexer).
- **K2 (MEMORY)** 128K-context routing-cache memory ≤ 50% of independent attention + routing caches.
- **K3 (CACHE-HIT)** ≥60% reuse rate on routing-cache (validates the dual-use thesis).
- **K4 (NO COLLAPSE)** Per-benchmark scores within 5pp of Fisher-Rao K=7 baseline.

## Implementation status

**SPEC ONLY — implementation pending.**

Required engineering:
1. MLX lightning indexer (~80M params, dot-product-based scorer per layer).
2. Top-k selection (k=2048 paper default) integration with attention.
3. Routing-cache plumbing (`exp_pierre_kv_cached_layer_routing_1m` infrastructure).
4. Long-context eval rig at 128K.

This is the **highest-cost SSA experiment** — depends on lightning-indexer
infrastructure that doesn't exist yet. Prerequisite: HiP or similar sparse
attention as the substrate; routing-cache experiment as the consumer.

## References

- DeepSeek Sparse Attention (V3.2 release)
- Sibling: `exp_pierre_kv_cached_layer_routing_1m`, `exp_pierre_lightning_indexer_routing`
- Research agent SSA survey: `a2c10d5138d8eea52`
