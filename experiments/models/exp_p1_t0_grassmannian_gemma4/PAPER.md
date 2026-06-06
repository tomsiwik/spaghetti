# T0.1 Paper: Grassmannian QR on Gemma 4 Weight Shapes

## Prediction vs Measurement

*Smoke test at d=512/d=1024 (rank=4). Gemma 4 dims are analytical corollaries — see below.*

| Kill | Prediction (MATH.md) | Measured | Result |
|------|---------------------|----------|--------|
| K990 | max\|A_i^T A_j\|_F < 1e-6 (d=512, N=10, r=4, f64) | 1.696e-16 | **PASS** |
| K991 | max\|A_i^T A_j\|_F < 1e-6 (d=1024, N=20, r=4, f64) | 1.799e-16 | **PASS** |
| K992 | N_max = floor(d/r) constructions complete | N_max=128 (d=512), 256 (d=1024); max_err=9.19e-16 | **PASS** |
| K993 | Construction time < 1s on GPU | 0.00112s (893× margin) | **PASS** |

## Key Results

**Verified dimensions (smoke test, rank=4):**
- d=512, N=10: max|A_i^T A_j|_F = 1.696e-16 across 45 pairs (f64) — algebraically zero
- d=1024, N=20: max|A_i^T A_j|_F = 1.799e-16 across 190 pairs (f64) — algebraically zero
- Both at ~1× float64 machine epsilon (ε_mach = 2.2e-16)

**Capacity at verified dimensions (rank=4):**
- d=512, r=4: N_max = 128 domains (measured)
- d=1024, r=4: N_max = 256 domains (measured)
- NoPE subspace (d=384, r=4): N_max = 96 domains per layer (measured)

**Gemma 4 capacity at r=16 — ANALYTICAL COROLLARIES (not measured):**

These follow directly from Theorem 1 (algebraic, d-independent) and Corollary 1 (N_max = ⌊d/r⌋):

- d=2816 (26B-A4B q_proj), r=16: N_max = 2816/16 = **176 domains** (7× headroom over 25-domain target)
- d=5376 (31B q_proj), r=16: N_max = 5376/16 = **336 domains**
- NoPE subspace (d=384, r=16): N_max = 384/16 = **24 domains per layer**

The algebraic guarantee is d-independent — QR orthogonality holds for any d, r, N ≤ N_max.
Running at d=512/r=4 verifies the same theorem that governs d=2816/r=16.

**Production float32 interference (verified):**
- d=512, N=10: max|A_i^T A_j|_F (f32) = 8.929e-09 << threshold

**Construction time:** 0.00112s (N=20 adapters, d=1024, GPU)

## Theorem Verification

**Theorem 1 (Grassmannian Partition Construction):** VERIFIED.
- Q^T Q = I is algebraically guaranteed by QR
- Pairwise interference at float64 floor (1.7e-16), not statistical noise
- Theorem 2 bound for N=10, r=4: ≤ 6.95e-14; measured 1.696e-16 (410× below bound)
- Theorem 2 bound for N=20, r=4: ≤ 9.83e-14; measured 1.799e-16 (547× below bound)

## Prior Results Comparison

| Setup | max\|A_i^T A_j\|_F | N_max | Note |
|-------|---------------------|-------|------|
| Qwen3-0.6B, d=1024, r=4 (Finding #393) | 9.50e-08 | 256 | float32 QR |
| Smoke d=512, r=4 (this work) | 1.696e-16 | 128 | **measured** (float64) |
| Smoke d=1024, r=4 (this work) | 1.799e-16 | 256 | **measured** (float64) |
| Gemma 4, d=2816, r=16 | — | 176 | **analytical corollary** |
| Gemma 4, d=5376, r=16 | — | 336 | **analytical corollary** |

## P1 Implications

T0.1 confirms the Grassmannian foundation holds:
1. Algebraic guarantee verified — theorem is d-independent, smoke dims suffice
2. Gemma 4 NoPE (d=384, r=16): N_max = 24 — sufficient for Gemma 4 local layers (analytical)
3. Full q_proj Gemma 4 (d=2816, r=16): N_max = 176 >> 25-domain target (analytical)
4. Construction is fast (~1ms) — adapter initialization is not a bottleneck

**Unblocks:** T1.5 (PoLAR landing field on Gemma 4) — Grassmannian subspace verified algebraically.
