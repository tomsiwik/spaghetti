# LEARNINGS.md — exp_p3_b1_ortho_t2t3_compose

## Status: KILLED — Finding #462

## Core Finding
B-matrix-only Gram-Schmidt orthogonalization eliminates directional interference
(K1172: cosine 0.1607 → 2.5e-7) but fails to preserve personal style compliance
(K1174: 76% → 60%, 16pp > 10pp threshold), because LoRA interference is in ΔW = A×B,
not B alone.

## Why
Even with B_P' ⊥ B_D, the cross-term B_P'^T (A_P^T A_D) B_D ≠ 0 because A-matrices
trained on the same base model share overlapping gradient directions. B-only GS
reduces failure from 100pp (Finding #460) to 16pp — progress, but insufficient.
Root: arxiv 2402.03513 (Null-Space LoRA) identifies full ΔW orthogonalization as the
correct fix.

## Implications for Next Experiment (P3.B2: Full ΔW Orthogonalization)
Compute ΔW_D = A_D × B_D per layer, QR-decompose, project full ΔW_P onto the complement,
then SVD re-factorize into new A_P', B_P'. This guarantees ΔW_P' ⊥ ΔW_D in operator
norm — the algebraic conditions for K1174 (<10pp style loss) will be met if A-matrix
interference is the only remaining source (~16pp measured here).

## Smoke Test Warning
N=5 smoke showed personal-only 100%; full run N=25 showed 76%. Use N≥10 for
style compliance smoke tests to avoid false confidence.
