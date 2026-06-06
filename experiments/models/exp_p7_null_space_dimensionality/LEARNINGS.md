# LEARNINGS: exp_p7_null_space_dimensionality

**Status**: SUPPORTED | **Finding**: #493

## Core Finding
Gemma 4 E4B local-attention layers (35/42) have exactly 512 structural null dimensions
in q_proj, yielding 85 adapter slots at r=6 — sufficient for 25+ domains with zero
theoretical interference. Global layers' q_proj has zero null space (overdetermined:
4096 input > 2560 output) and must be skipped.

## Why
The null space is a geometric guarantee from dimension mismatch (2560 input - 2048 output
= 512), not a quantization artifact. Matrices are full-rank at ε=1e-4. This confirms
arXiv:2512.15233 (Null-LoRA): null-space isolation is architecturally guaranteed when
the projection is underdetermined, and useless when overdetermined.

## Complete Null-Space Map
| Module     | Local (35 layers)         | Global (7 layers)          |
|------------|---------------------------|----------------------------|
| q_proj     | null=512, slots=85        | null=0, slots=0            |
| k_proj     | null=2048, slots=341      | null=1536, slots=256       |
| v_proj     | null=2048, slots=341      | null=1536, slots=256       |
| o_proj     | null=0, slots=0           | null=1536, slots=256       |

## Key Corrections
- **Retire**: quantization-reduces-rank assumption (wrong; matrices are full rank)
- **K1296 lesson**: specify layer-type population explicitly in future kill criteria
- **Functional vs parametric**: null(W_q) blocks A-matrix interference, but B-matrix
  output still enters attention. P7.A1 must measure functional interference at output.

## Implications for P7.A1 (Null-Space Adapter Quality)
- Primary target: v_proj local layers (341 slots — 13x our 25-domain target)
- Secondary: q_proj local layers (85 slots — 3.4x our 25-domain target)
- Skip global q_proj entirely
- Critical question: does null-space projection preserve adapter's ability to learn
  useful domain features, or does the geometric restriction degrade quality?
- Projection: use right singular vectors V_{2049:2560} to place A-matrices in null(W_q)
