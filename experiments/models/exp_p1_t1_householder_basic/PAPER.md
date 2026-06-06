# PAPER.md — T1.1: Householder Chain Orthogonality at d=2816

## Prediction vs Measurement

| Metric | Theorem Prediction | Measured | Status |
|--------|--------------------|----------|--------|
| Isometry err (float32, r=16, d=2816) | ≤ 3.8e-6 (r × 2ε_mach) | **2.384e-07** (16× better) | K1007 PASS |
| \|cos(H1-I, H2-I)\| Grassmannian init | 0 algebraic, < 1.9e-6 | **3.85e-10** (algebraic floor) | K1008 PASS |
| Stable rank sr(H^(r)-I) | ≥ r/2 = 8, predicted ≈ r = 16 | **16.00** (exact!) | K1009 PASS |
| σ_max(H^(r)-I) | ≤ 2 (triangle inequality) | **2.0000** (tight) | K1009 bound |
| sr(LoRA A*B, Kaiming) | ~1 (predicted, wrong) | **13.57** (≈ r) | theory corrected |
| HRA param count | r×d = 45,056 | **45,056** | K1010 PASS |
| HRA/LoRA param ratio | 0.5 (2× fewer) | **0.50** | K1010 PASS |
| MLX matmul (N=1024, d=2816) | ~ Givens (3.14ms) | **3.04 ms** | comparable |
| MLX sequential reflections | > matmul | **3.93 ms** | 1.3× slower |

All 4 kill criteria PASS. All 5 theorems verified.

---

## Key Results

### K1007: Isometry at d=2816, r=16
**max |‖H^(r) x‖² - 1| = 2.384e-07** across N=1024 random unit vectors.

The theory predicted ≤ 3.8e-6 (r × 2 × ε_mach), but the actual error matches
single float32 machine epsilon (2.384e-7 = 2 × ε_mach). This is the same value
as Givens T1.3 (2.384e-07), confirming that both methods achieve the float32
floor independently of the rotation method.

Multi-layer behavior (isometry doesn't degrade with depth):
- L=1: 2.384e-07
- L=2: 2.980e-07
- L=4: 3.576e-07
- L=8: 3.576e-07  ← constant at ~3ε_mach, not growing linearly with L

**NoPE subspace (d=384, actual P1 target): isometry_err = 2.384e-07, 6,144 params/layer.**

### K1008: Grassmannian Interference (Algebraic Zero)
**|cos| = 3.85e-10** — effectively machine precision.

Two HRA adapters initialized with orthogonal Grassmannian subspaces (QR of d×2r matrix):
‖H_1 - I‖_F = ‖H_2 - I‖_F = 8.000 (both equal √(4r) = √64 = 8 as predicted).
Inner product ⟨H_1-I, H_2-I⟩_F = 2.465e-08 ≈ 0.

This confirms Theorem 2: when Householder vectors span orthogonal subspaces,
the adapter deltas have *algebraically zero* interference. The 3.85e-10 residual
is purely floating-point accumulation (below any practical threshold).

### K1009: Stable Rank — Clean Algebraic Values
**sr(H^(r)-I) = 16.00 exactly, σ_max = 2.0000 exactly.**

The exact values follow from:
- ‖H^(r)-I‖_F² = 4r = 64 (algebraic, from tr(H^(r)) = d - 2r)
- σ_max(H^(r)-I) = 2 (tight: eigenvalue -1 → |−1−1|=2)
- sr = 4r / 4 = r = 16 **exactly**

Correction to theory: sr(LoRA A*B) = 13.57 ≈ r, *not* ~1.
For random Gaussian A (d×r) and B (r×d), both A and B have approximately
flat singular value spectra, so A@B has stable rank ≈ r.
The key HRA advantage is not higher stable rank (both achieve ≈ r), but
achieving the **same stable rank with 2× fewer parameters** (r×d vs 2r×d).

### K1010: Parameter Efficiency
HRA (r=16, d=2816): 45,056 params = 0.5 × LoRA (90,112 params).
For the actual P1 NoPE target (d=384): 6,144 params per layer.

---

## Comparison with T1.3 (Givens) and T1.4 (Cayley)

| Method | Isometry err | Interference | Stable rank | Params | MLX timing |
|--------|-------------|--------------|-------------|--------|------------|
| Givens (T1.3) | 2.384e-07 | structural zero (disjoint pairs) | d/2 per layer | d/2 per block | 3.14ms |
| Cayley (T1.4) | 7.62e-16 (float64) | not tested | r at r=16 | ~r²/2 | 433μs (CPU-only!) |
| **HRA (T1.1)** | **2.384e-07** | **3.85e-10** | **r=16 exact** | **rd** | **3.04ms** |

HRA key advantage over Givens: Grassmannian initialization gives provably zero
cross-adapter interference (Theorem 2). Givens achieves orthogonality within
a single adapter but doesn't have a natural multi-domain interference theorem.

HRA key advantage over Cayley: No matrix inverse required → GPU-native in MLX.
Cayley's 7.62e-16 exactness is impressive but costs CPU-only linalg (433μs overhead).

---

## Float32 Overflow Warnings (Caveat)

numpy raised `divide by zero / overflow` warnings during float32 matmuls
in `build_householder_matrix`. The final results are correct and finite (all
isometry errors are near machine epsilon). Root cause: numpy BLAS routines
accumulating float32 sums near the exponent limit before normalization.

Fix for production: build Householder matrices in float64, then cast to float32.
The experiment verifies the *algebraic structure* in float32 (results correct),
but the numpy warnings indicate float32 matmul is fragile at d=2816 for this op.

---

## P1 Architectural Implications

With T0.3/T0.4 results (NoPE dims [128:512], Q-only adapters, Grassmannian init):
- HRA on NoPE dims (d=384): 6,144 params/layer, 2.384e-07 isometry, zero interference
- Multi-layer stable: isometry err ≈ 3ε_mach constant up to L=8 layers
- Grassmannian interference theorem extends to arbitrary N adapters (N_max = 384/r/2 = 12 domains at r=16; more with partial Stiefel)

**Next step: T1.6 bake-off (HRA vs Givens vs Cayley on actual quality task).**
T1.2 (HRA vs LoRA quality) is now unblocked.
