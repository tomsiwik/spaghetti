# LEARNINGS — T1.1: Householder Chain Orthogonality at d=2816

## Core Finding
HRA (Householder Reflection Adapter) achieves float32-floor isometry (2.384e-07) and
algebraically zero cross-adapter interference (3.85e-10) at d=2816, with 2× fewer
parameters than LoRA at the same rank r. All 4 kill criteria PASS.

## Why It Works
Grassmannian initialization of Householder reflection vectors guarantees
⟨H₁-I, H₂-I⟩_F = 0 algebraically (Theorem 2), making multi-domain interference
structurally impossible — not just empirically small. The isometry error (2.384e-07)
matches the float32 machine epsilon floor, identical to Givens (T1.3), confirming
this is a hardware floor, not a method artifact.

## Key Theory Correction
Theorem 3 predicted sr(LoRA) ≈ 1; measurement shows 13.57 ≈ r. Random Gaussian A, B
both have flat singular spectra, so A@B has sr ≈ r. HRA's real advantage: same stable
rank (r) at 2× fewer parameters (r×d vs 2r×d), not higher stable rank.

## Implications for Next Experiment
T1.2 (HRA vs LoRA quality on an actual task) is now unblocked — the structural
properties are verified, behavioral quality comparison is the next unknown.
T1.6 bake-off now has 3 candidates: HRA (GPU-native, zero interference, 0.5× params),
Givens (parallel Metal kernel, 3.14ms), Cayley (exact in float64, CPU-only currently).

## Production Caveat
Build Householder matrices in float64, then cast to float32. numpy BLAS raises
overflow warnings at d=2816 in float32 (results correct but fragile).

## P1 Target
NoPE dims [128:512] → d=384: 6,144 params/layer, isometry 2.384e-07, zero interference.
N_max = 384/r/2 = 12 domains at r=16 via Grassmannian partitioning.
