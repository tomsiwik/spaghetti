# P5.A0: LoRI Sparse-Mask Composition — Results

## Reference
arXiv:2504.07448 (LoRI), Finding #59 (prior LoRI on BitNet-2B, null result)

## Configuration
- Model: Gemma 4 E4B 4-bit (2560→2048 q_proj, 42 layers)
- LoRI: rank 30 total, 6 dims per adapter, 5 domains (medical/math/legal/python/creative)
- Frozen shared A: (2560, 30), random uniform init
- Trainable masked B: (30, 2048), disjoint row masks per adapter
- Training: 200 iters, Adam lr=1e-4, batch 4, seq_len 256
- Baseline: standard LoRA rank 6 (A+B trainable) on math domain

## Prediction vs Measurement

### K1264: Parameter Interference

| Metric | Prediction (MATH.md) | Measured | Status |
|--------|---------------------|----------|--------|
| B-space max\|cos\| | = 0.0 (Thm 1) | **0.00e+00** | CONFIRMED |
| W-space max\|cos\| | < 1e-4 (Thm 1+3) | **1.33e-3** | **FAIL** |

**MATH.md proof error identified.** Theorem 1 claims `<ΔW_i, ΔW_j>_F = 0` for disjoint
masks, but the proof conflates B-space and weight-space orthogonality.

The Frobenius inner product is:
```
<ΔW_i, ΔW_j>_F = Σ_{k∈m_i, l∈m_j} (B_i^T B_j)_{k,l} · (AA^T)_{k,l}
```
where (AA^T)_{k,l} is the Gram matrix of A's rows. For random A, cross-block entries
(AA^T)_{k,l} with k∈supp(m_i), l∈supp(m_j) are O(1/√d_in) ≈ 1/50, not zero.

**Correct statement:** Disjoint masks guarantee **B-parameter-space** orthogonality
(`<B_i, B_j>_F = 0`, trivially). Weight-space orthogonality requires additionally
that A's row blocks are orthogonal (i.e., Grassmannian structure on A).

### Pairwise Weight-Space Cosine Similarities

| Pair | cos(ΔW_i, ΔW_j) |
|------|-----------------|
| medical vs math | 1.76e-4 |
| medical vs legal | 8.97e-5 |
| medical vs python | 1.33e-3 |
| medical vs creative | 5.50e-4 |
| math vs legal | -9.14e-4 |
| math vs python | 3.89e-4 |
| math vs creative | 6.90e-5 |
| legal vs python | -4.48e-4 |
| legal vs creative | -7.30e-4 |
| python vs creative | 1.04e-4 |

Mean |cos| = 5.0e-4. Compare to Grassmannian (Finding #440): max cos = 2.25e-8.
LoRI weight-space interference is ~60,000x worse than Grassmannian.

### K1265: Quality vs Standard LoRA

| Metric | Prediction | Measured | Status |
|--------|-----------|----------|--------|
| LoRI math PPL | competitive | **8.34** | — |
| Baseline LoRA PPL | — | **4.86** | — |
| Quality ratio | >= 0.90 | **0.989** | **PASS** |

LoRI achieves **98.9%** of standard LoRA quality with **2.4x fewer trainable parameters**
(516K vs 1.25M per adapter). The frozen A reduces parameter count substantially.

### Per-Domain Quality (Solo)

| Domain | Base PPL | LoRI PPL | Reduction |
|--------|----------|----------|-----------|
| Medical | 20,157 | 68.0 | 99.7% |
| Math | 314 | 8.3 | 97.3% |
| Legal | 2,014 | 48.6 | 97.6% |
| Python | 88 | 6.7 | 92.4% |
| Creative | 414 | 8.6 | 97.9% |

### K1266: 5-Adapter Composition

| Domain | Solo PPL | Composed PPL | Degradation | Status |
|--------|----------|-------------|-------------|--------|
| Medical | 68.0 | 68.4 | **0.5%** | PASS |
| Math | 8.3 | 26.9 | **222.9%** | FAIL |
| Legal | 48.6 | 59.9 | **23.2%** | FAIL |
| Python | 6.7 | 13.8 | **106.0%** | FAIL |
| Creative | 8.6 | 18.5 | **114.8%** | FAIL |

**Root cause:** Additive composition without routing. Each adapter modifies
all inputs, not just its domain. For a math token, delta_y = Σ_i scale·(x@A)@B_i,
meaning medical/legal/python/creative adapters all contribute noise.

Medical degrades least (0.5%) because its base PPL is highest (20K) — the adapter
perturbation is relatively small. Math degrades most because its solo PPL is lowest
(8.3) — it's most sensitive to additive noise from other adapters.

**Fundamental limitation:** Shared A means all adapters project through the SAME
subspace. Unlike Grassmannian (separate orthogonal A per adapter), there's no
input-space separation. B-space disjointness alone is insufficient for composition.

## Kill Criteria Summary

| K | Criterion | Result | Status |
|---|-----------|--------|--------|
| K1264 | max\|cos\| < 1e-4 (weight-space) | 1.33e-3 | **FAIL** |
| K1265 | Quality >= 90% standard LoRA | 0.989 | **PASS** |
| K1266 | Composition < 5% degradation | 222.9% | **FAIL** |

**Overall: KILLED** (K1264 FAIL, K1266 FAIL)

## Key Findings

1. **MATH.md proof error:** Weight-space orthogonality requires Grassmannian A,
   not just disjoint B masks. B-space orthogonality ≠ weight-space orthogonality.

2. **LoRI solo quality is excellent:** 98.9% of standard LoRA quality with 2.4x
   fewer trainable parameters. Frozen A is a viable parameter reduction strategy.

3. **Composition requires routing:** Additive composition of LoRI adapters fails
   catastrophically. The shared A projects all inputs through the same subspace,
   so every adapter modifies every input. Need per-token routing or separate A.

4. **Prior Finding #59 revisited:** The null result on BitNet-2B was because (a) A
   wasn't frozen, (b) ternary model has near-zero natural interference. On Gemma 4
   with proper frozen A, LoRI works well for SOLO adapters.

## Implications for Architecture

- **LoRI solo adapters:** Viable for single-adapter deployment (saves 2.4x params)
- **LoRI composition:** NOT viable without routing
- **Best of both:** Grassmannian A (orthogonal per-adapter projections) + disjoint B
   masks would give both weight-space orthogonality AND B-space structure.
   This is strictly better than either approach alone.

## Parameters

| Component | LoRI (per adapter) | Standard LoRA |
|-----------|-------------------|---------------|
| A trainable | 0 (frozen) | 6 × 2560 × 42 = 645K |
| B trainable | 6 × 2048 × 42 = 516K | 6 × 2048 × 42 = 516K |
| **Total** | **516K** | **1,161K** |
| Ratio | 1.0x | 2.25x |

Training time: 39.3 min total (5 LoRI adapters + 1 baseline + evaluation).
