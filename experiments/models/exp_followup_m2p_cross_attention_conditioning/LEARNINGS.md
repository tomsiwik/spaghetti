# LEARNINGS — exp_followup_m2p_cross_attention_conditioning

**Verdict:** KILLED (confirmed). Finding #596. Closure refined: `additive-pooled-concat-unpacking-blocks-calibration`.

## Core finding
Swapping mean-pool → cross-attention raised `CV(‖B‖_F)` only from 0.0153 → 0.0200 (1.31× vs ≥3× required, 0.05 absolute required). K1556b passes (regime reproduced), K1556a/c fail. Cross-attention is **necessary but not sufficient** to resolve the K850 CV-collapse from `exp_m2p_scale_calibrated` (Finding #343).

## Why
Lemma 2 bounds Jacobian **rank**, not **magnitude**. Two downstream structures re-pool and dominate the context-sensitivity budget:
1. `B_proj` unpacking head: single linear on `mem.reshape(1, -1)` (flat 512-vec) → output variance capped by `‖B_proj‖_op · ‖Δflat‖`. Small `Δflat` across contexts → small CV regardless of conditioning rank.
2. Self-attention blocks (M2PBlock × 2) after cross-attn are permutation-equivariant over memory tokens — they mix context-modulated slot back into uniform memory before the head sees it.

P4 also failed with **wrong sign** (hard/easy = 0.971 vs ≥1.10 predicted), reinforcing that the magnitude bottleneck binds below the rank bottleneck.

## Implications for next experiment
The conditioning layer is no longer the binding constraint. Two well-motivated follow-ups (both require separate MATH.md + pre-registered KCs — no KC editing):

1. **Per-token unpacking head** (priority): replace single `B_proj` on flattened memory with `N_MEMORY=8` independent heads, each producing `1/N` of `B`. Breaks the "single affine read of flat vector" magnitude bottleneck. Predict: `CV(‖B‖_F) > 0.05` iff at least one slot carries non-trivial context variance.
2. **Cross-attn → `B_proj` skip**: bypass self-attention blocks for the conditioning path. Preserves per-slot variance before permutation-equivariant mixing.

Parent closures C2 (KKT violated) and C3 (L_preserve rigidity) remain untouched by this experiment and still apply to any M2P follow-up at scale.

## Scope caveats
Toy 2-layer GPT, `D_MODEL=256`, single seed (42), apples-to-apples with parent kill. Verdict robust to seed noise (±0.005 cannot cross 0.05 or 3× bar).
