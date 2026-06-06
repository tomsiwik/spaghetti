# LEARNINGS: C1.3 — PoLAR Scale Invariance

## Core Finding

PoLAR's joint Stiefel constraint (BB^T = I_r) makes adapter scale invariance a **structural
guarantee**, not a tuning target. PoLAR variance = 4pp vs LoRA = 12pp across 4× scale range
(scale={3,6,12,24}), with B row norms exactly 1.0000 ± 8.98e-10 at float64 floor.

## Why

Theorem 1: if BB^T = I_r then ||ΔW(s)||_F² = s² · r — effective scale is exactly s × 1.0.
Unconstrained LoRA has mean row norm 0.347 ± 0.093 (layer-dependent), so effective scale is
s × 0.347 with high variance. The Stiefel constraint eliminates this variability structurally.
Gemma 4's QK-norm provides baseline scale protection; PoLAR adds a provable guarantee on top.

## Implications for Next Experiment

C1 tier complete (C1.1: sr=r exactly, C1.2: QK-norm baseline, C1.3: scale invariance guaranteed).
Next: C2 — integrate PoLAR with Grassmannian composition from C0.1. The combination of
joint Stiefel (direction-preserving) + Grassmannian isolation (interference < 1e-4) is the
full P1 adapter stack. Test whether PoLAR adapters compose as cleanly as standard LoRA adapters.

## Caveats

- Accuracy prediction overpredicted (30-50% expected, 4-12% actual): 500 steps / 80 samples
  insufficient for task learning on Gemma 4. Structural finding is valid; behavioral advantage
  of scale invariance requires 2000+ step training to verify.
- KC15 (200% ratio) passes robustly but is near the measurement floor (25 eval samples).
