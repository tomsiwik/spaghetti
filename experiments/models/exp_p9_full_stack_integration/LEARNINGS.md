# LEARNINGS.md — P9.G0: Full Stack Integration

## Core Finding
q_proj-only adapters (r=6) improve generation tasks (GSM8K ~82%) but degrade MCQ accuracy,
because factual knowledge is stored in FFN layers, not attention projections (ROME, arXiv:2202.05262).
Oracle routing at 97.7% gains are therefore task-type-dependent: high on generation, near-zero or
negative on MCQ knowledge recall.

## Why
Routing value-add requires adapters that target the right module type for the task:
- Generation/reasoning tasks: q_proj adapters work (adapter trains on generation outputs)
- MCQ/factual recall tasks: FFN adapters needed (knowledge stored in gate_proj/up_proj/down_proj)
This explains why smoke oracle routing showed only +8.3pp delta (< 10pp K1388 threshold) on mixed MMLU-Pro.

## Footprint Reality Check
Actual adapter sizes: 14.3MB (math/code/medical, q+k+v+o+gate r=6), 9.54MB (legal/finance) = 61.98MB total.
MATH.md estimated 5MB; actual is 12-14x larger. TT-LoRA compression (previously failed for MCQ) was the
intended solution. Without it, 25-domain expansion = ~1.5GB adapter footprint.

## Composition Approximation
Code uses parameter-space average (α*B1 + (1-α)*B2 with shared A1) instead of ideal weight-space
sum α*(B1@A1) + (1-α)*(B2@A2). Mechanically stable, but introduces approximation error proportional
to divergence between A1 and A2. For same-rank same-architecture adapters, error is bounded but non-zero.

## Implications for Next Experiment
1. **K1387 (math GSM8K delta)**: High confidence PASS — math adapter at 82%, base ~55% → delta ≈ 27pp >> 15pp threshold
2. **K1388 (oracle routing MMLU-Pro delta)**: Likely FAIL — q_proj adapters don't improve MCQ; need FFN-targeted adapters
3. **Future adapters**: Train on gate_proj/up_proj/down_proj (FFN) for factual recall tasks to unlock routing value on MCQ
4. **P11 path**: Reasoning SFT adapters (s1K/LIMO) target reasoning traces — these ARE generation tasks, so q_proj remains valid
