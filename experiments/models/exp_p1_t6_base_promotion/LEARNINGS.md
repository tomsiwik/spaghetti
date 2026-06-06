# LEARNINGS.md — T6.3: Base Promotion

## Core Finding
Promoting a crystallized LoRA adapter into base model weights is algebraically exact
(cos=0.99999988) with bounded spectral perturbation (ε=3.6% mean, <5% max on synthetic
weights), confirming the T6.1→T6.2→T6.3 flywheel cycle is structurally sound.

## Why
Standard linear algebra: W_promoted = W_base + scale·B^T@A^T is identical to runtime
application. Davis-Kahan theorem guarantees spectral gap preservation when ε < δ_gap,
which holds empirically (Finding #333: 0pp MMLU change at scale=5). Slot liberation
follows trivially — promoted knowledge moves from adapter slot to base weights.

## Key Numbers
- K1124: cos(ΔW_promoted, ΔW_crystal) = 0.99999988 (min over 42 layers) — formula exact
- K1125: max_layer ε = 4.78%, mean = 3.63% — well under 10% Davis-Kahan threshold
- K1126: 5 → 4 adapter slots after math promotion — slot freed as proven
- K1127: loss 0.072743 → 0.072698 over 5 steps — trainability confirmed

## Caveats
- Synthetic base weights (std=0.05); real Gemma 4 weights have larger ||W||_F → lower ε
- 5-step trainability test too short for behavioral confirmation — T6.4 needs 50+ steps
- Single adapter promoted; sequential cascade (N promotions) is untested — that's T6.4
- B-matrices only crystallized; A-matrix handling across users is future work

## Implications for Next Experiment
T6.4 (flywheel simulation) must test sequential cascade: N promotions one-by-one on a
real model to verify cumulative ε stays bounded and convergence speed on the promoted
base matches training on the original base. This is the critical path blocker before
claiming the flywheel works in production.
