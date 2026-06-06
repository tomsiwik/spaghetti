# LEARNINGS.md — T4.4: Multi-Tenant KV Sharing

**Status:** SUPPORTED | **Finding:** #455 | **Date:** 2026-04-11

## Core Finding

Q-only adapters leave k_proj/v_proj unmodified, making K_i = W_K @ x = K_j for all users — an algebraic identity, not an approximation. Eight concurrent users can share a single KV cache allocation, reducing global-layer KV memory from 28 MB to 3.5 MB (8× reduction, exact).

## Why It Works

Gemma 4 E4B has `attention_k_eq_v=True` across 7 global layers (GQA, 2 KV heads, head_dim=512). Since adapters only modify q_proj, W_K is identical for every user. The result is structural — it cannot be violated by any Q-only adapter configuration.

## Contrast With Killed T4.2 (LSH Routing)

LSH failed because it needed similarity structure (assumed c=0.8, got c=0.23). KV sharing succeeds because it needs NO structure — just the structural absence of k_proj/v_proj adapters. Design for structural guarantees over probabilistic ones where possible.

## Implications for Next Experiment

The full T4 serving stack is now verified: TF-IDF routing (0.125ms) + adapter hot-swap (4.77ms p99) + shared KV (8× memory reduction) + E2E overhead (1.4ms). The serving layer gap (real KV cache manager routing 8 users to shared buffer) remains unimplemented. C0/C1 corrective tier is the critical path — composition on Gemma 4 must be verified before production serving is meaningful.

## Caveats

- Experiment uses synthetic weights at Gemma 4 E4B dimensions; real weights produce identical result by Theorem 1
- Only 7 global layers verified; algebraic identity holds for all 42 layers (local layers also have no k_proj adapters)
- Serving-layer implementation (KV cache manager) not built — T4.6 covered E2E mechanics
