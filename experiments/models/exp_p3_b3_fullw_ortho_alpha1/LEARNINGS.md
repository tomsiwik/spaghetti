# LEARNINGS — exp_p3_b3_fullw_ortho_alpha1

## Core Finding
Full ΔW null-space orthogonalization with correct scale (PERS_SCALE=4.0 baked in,
natural_ratio=1.087 ≈ equal power) destroys style signal (0% at smoke N=5). KILLED.
Root cause: geometric column-space entanglement, not power imbalance.

## Why It Failed: Column-Space Entanglement

Adapters trained from the same base model (Gemma 4) converge to similar directions in
weight space. The style-encoding direction of ΔW_P lies in col(ΔW_D). After null-space
projection: ΔW_P' = (I - U_D U_D^T) ΔW_P — the style component is removed exactly.

Despite:
- Equal power (natural_ratio=1.087, domain=78.356, personal=72.064)
- Exact algebraic orthogonality (cos=0.0)
Style was completely destroyed (0%). Power preservation without direction preservation
is insufficient for behavioral signal preservation.

## PERS_SCALE Bug Discovery

P3.B2 computed equalization_factor from unscaled personal norms (11.01), giving α=4.349.
P3.B3 with PERS_SCALE=4.0 baked in shows personal norm=72.064 (≈6.5× higher).
P3.B2's α=4.349 was accidentally compensating for the missing PERS_SCALE=4.0
(47.86/11.01 = 4.349 ≈ 4.0/0.92). P3.B2 was running personal adapter at 4.349× scale
(vs intended 1× relative to domain), not at "equal power" as believed.

**Always bake in adapter scale before computing any norm-based metrics.**

## Projection Method Comparison

| Method | Projection Space | Style Δ | Why? |
|---|---|---|---|
| B-GS (P3.B1) | B-matrix row space | 16pp | Removes B-row overlap; col(ΔW_P) partially preserved |
| Full-ΔW GS (P3.B2) | Full ΔW column space | 36pp | Removes col overlap; PERS_SCALE bug amplified 4× |
| Full-ΔW GS α=1.0 (P3.B3) | Full ΔW column space | 100pp | Removes col overlap (style direction); correct scale |

Better algebraic orthogonality → WORSE behavioral preservation. The style IS in col(ΔW_D).

## Impossibility Structure

For any two adapters trained from the same base model:
  col(ΔW_D) ∩ col(ΔW_P) ≠ ∅  (empirically confirmed: style destroyed by projection)
  
For any additive composition using linear projection: f_style is removed by definition.
No amount of power equalization can recover the projected direction.

This means: **Additive weight-space composition via null-space projection is
structurally incompatible with behavioral signal preservation for same-base adapters.**

## Implications for Next Experiment

P3.B4 — Sequential composition:
  output = personal_adapter(domain_adapter(base(x)))
This operates in ACTIVATION space, not weight space. No projection losses. Domain enriches
hidden state; personal adds style on top. The column-space entanglement problem is avoided.

Expected result: style compliance preserved (~76%) while domain knowledge active (≥10%).
If P3.B4 PASS: we have a working composition method (with latency tradeoff).
If P3.B4 FAIL: the non-linearity of transformers makes even sequential composition
insufficient → deeper analysis needed.

## Key Rule for Future Experiments
**Always bake in adapter scale (MATH_SCALE, PERS_SCALE) before computing ΔW**
for any norm-based metric, projection, or comparison.
