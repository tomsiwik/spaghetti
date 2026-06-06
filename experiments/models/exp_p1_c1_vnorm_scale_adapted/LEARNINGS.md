# LEARNINGS: exp_p1_c1_vnorm_scale_adapted (C1.2: Scale Safety)

**Status:** KILLED | **Finding #443** | **Date:** 2026-04-10

---

## Core Finding

Standard LoRA on Gemma 4 shows 13.3pp variance across scale={5,10,20} (threshold: 5pp).
Post-hoc B normalization makes it catastrophically WORSE (66.7pp variance), because inflating
row norms from 0.357→1.0 multiplies effective scale by 2.80× before the LoRA scale parameter
is even applied. Theorem 3 (unit-norm B reduces sensitivity) is refuted for post-hoc use.

## Why

Training bakes `effective_scale = training_scale × mean_row_norm ≈ 6 × 0.357 = 2.14` into
adapter weights. Post-hoc normalization breaks this learned equilibrium. The fix is constraint
during training (PoLAR, C1.1), not normalization after training (post-hoc). Ref: PoLAR
arxiv 2506.03133, C1.1 Finding #442.

## Key Nuance: Metric Masks Behavior

KC10 passes at 0.0pp degradation (scale=5 vs scale=20), but scale=20 produces garbled
repetition loops on complex problems. The "accuracy metric" is preserved by easy questions
where the correct digit appears in garbled output. This confirms: behavioral testing > accuracy
proxies (CLAUDE.md guardrail #1008).

## Deployment Rule

**Keep inference scale within 2× of training scale.** The "hump" peaks at scale≈10 (1.67×
training scale=6). Both undershoot and overshoot degrade quality, with overshoot causing
garbled repetition loops at scale=20 (effective=3.3×).

## Implications for C1.3

Test PoLAR-trained adapters (C1.1 Finding #442, sr=r=16 exactly) at scale={0.5×, 1×, 2×, 4×}
training scale. If joint Stiefel constraint provides genuine scale invariance, variance should
collapse to < 5pp — this is what Theorem 3 originally predicted, but only holds when constraint
is imposed DURING training.
