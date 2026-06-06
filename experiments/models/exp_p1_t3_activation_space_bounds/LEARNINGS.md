# LEARNINGS.md — T3.3: Activation-Space Interference Power Law

**Status:** SUPPORTED | **Date:** 2026-04-10

---

## Core Finding

Activation-space interference for Gemma 4 q_proj adapters follows a slower power law (alpha=0.15, c=0.061) than predicted from Finding #372 (alpha=0.38), because d_out=2048 provides near-orthogonal subspaces. At N=50 adapters, max pairwise cosine ≈ 0.107 (well below 0.50 threshold). V-norm reduces alpha marginally (delta=-0.013) and is not the structural fix.

## Why

Larger d_out shifts the interference regime: expected pairwise cosine scales as ~1/sqrt(d_out). Gemma 4 q_proj d_out=2048 vs Qwen3-4B fc1 d_out=11008 gives a different effective dimension, lowering alpha from 0.38 to 0.15. This is a dimension-dependent result, not architecture-dependent.

## Critical Discrepancy: Real Adapters

Real fine-tuned adapters produce max_cos=0.596 at N=5 (7.6× higher than synthetic 0.078). Cause: LoRA A-matrices are highly correlated across domains due to shared initialization (Frobenius cos ≈ 0.71-0.83). Random inputs trigger all adapters equally → high cosines → SNR collapse.

## Implications for Next Experiment

Routing is the structural fix — not V-norm, not rank reduction. Matched routing ensures each input hits only its adapter, keeping activation cosines low. T4 experiments should validate that PLE-M2P routing achieves this in practice: routed inputs must show cosine ≪ 0.5; unrouted inputs will exceed 0.5.

## Caveats to Propagate

- Use alpha=0.15 (Gemma 4 q_proj) not 0.38 (Qwen3 fc1) for future interference bounds
- Real-adapter correlation claim (correlated initialization) is post-hoc; validate in T4
- Power law measured on q_proj only; other adapter targets may differ
- K1056 was tautological — avoid in future kill criteria
