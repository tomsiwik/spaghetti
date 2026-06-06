# LEARNINGS — exp_p3_b2_fullw_orthogonal_compose

## Core Finding
Full ΔW null-space orthogonalization achieves exact algebraic independence (cos=9.66e-18)
but style compliance degrades 36pp (76%→40%) — WORSE than B-GS P3.B1 (16pp). KILLED.

## Why It Failed
Power equalization confound: full-ΔW norm comparison yields α=4.349× (vs B-norm α=1.369 in P3.B1).
Over-amplifying the personal adapter by 4.349× swamps the style signal even with exact orthogonality.
The per-layer personal amplitude exceeds safe operating range for attention/LayerNorm interaction.

## Key Contrast: B-GS vs Full-ΔW GS
| | P3.B1 (B-GS) | P3.B2 (Full-ΔW GS) |
|---|---|---|
| α | 1.369 | 4.349 |
| style Δ | 16pp | **36pp** (worse) |
| algebraic cos | 2.5e-7 | 9.66e-18 |

Better algebraic orthogonality → WORSE behavioral outcome. The amplification dominates.

## Open Ambiguity
Non-linear interference (attention, LayerNorm) and power over-amplification are confounded.
P3.B3 should test α=1.0 to isolate root cause: if style Δ ≤ 10pp at α=1.0, amplification
was the bug; if still > 10pp, non-linear interference is confirmed.

## Implications for Next Experiment
Option A: P3.B3-alpha — repeat P3.B2 with α=1.0 (quick 1-hour experiment, isolates confound).
Option B: P3.B3-seq — sequential composition (base+domain then personal as hidden-state residual)
which bypasses both amplification and non-linear weight-space interference.
If timeline constrained, jump to sequential (more robust fix, cited arxiv 2402.03513).

## Reference
arxiv 2402.03513 (Null-Space LoRA) — weight-space composition fragile for semantically
distinct adapters trained from same base (common gradient paths create non-random alignment).
