# LEARNINGS — LoRA Hub Learned Scalars

## Core Finding
Scalar reweighting of B-only adapters recovers 40% of the Fisher-Rao→DARE gap (+2.7pp to 67.3%), proving the shared-A B-only scalar space is not exhausted. But optimization cost (6.1h / 576 evals) makes it impractical without weight caching.

## Why It Matters
The learned weights reveal adapter interference structure that uniform averaging hides:
- **strategy_integrate is harmful** (weight = -0.67). It actively degrades composition and should be removed or retrained.
- **domain_math is irrelevant** (weight ≈ 0). Math capability routes through strategy_act (weight = 1.35), not the domain adapter. The strategy/domain taxonomy doesn't match how the model actually composes.
- **Strategy > domain** in weight magnitude. The optimizer concentrates on strategy adapters, treating domain adapters as corrections.

## Implication for Pierre
The remaining 4pp gap to DARE requires non-scalar methods (per-adapter A, structure-preserving merges, or per-token routing). Scalar weights are a useful diagnostic but not the composition mechanism. The negative-weight finding for strategy_integrate is immediately actionable — dropping it from the default merge should improve Fisher-Rao by ~1-2pp for free.

## Caveat
9-prompt validation panel likely overfits the learned weights. MedQA dropped from 58% (Fisher-Rao) to 52% under learned scalars — the optimizer traded MedQA for HumanEval gains.
