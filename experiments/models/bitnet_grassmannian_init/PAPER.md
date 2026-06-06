# Grassmannian AP Init + Ternary QAT: Research Digest

## Hypothesis

AP-initialized LoRA experts, after ternary quantization-aware training (QAT)
with STE, produce significantly more orthogonal expert deltas than
random-initialized experts with the same QAT procedure. Specifically, AP-init
should achieve mean |cos| < 0.7x random-init (>30% improvement).

## What This Experiment Is

A controlled comparison of two initialization strategies for ternary LoRA
adapters on a ternary (BitNet-style) base model:

| Condition | A-matrix init | B-matrix init | Training | Quantization |
|-----------|--------------|---------------|----------|--------------|
| **AP-init** | Orthonormal frames from AP skeleton | Zero | Ternary QAT (STE) | Ternary |
| **Random-init** | Gaussian * 0.01 | Zero | Ternary QAT (STE) | Ternary |

Architecture: d=64, r=4, L=2, 5 domains, 3 seeds. Same model, data, and
training procedure as bitnet_ternary_adapter_composition.

## Key References

- Dhillon et al. (2008). "Constructing Packings in Grassmannian Manifolds via AP."
- BitNet b1.58 (arXiv 2402.17764): Ternary LLM, absmean quantization.
- Prior: grassmannian_expert_init (AP packing proven 1.2-1.5x at FP16).
- Prior: bitnet_ternary_adapter_composition (ternary QAT: -19.3% |cos|, -4.4% composed PPL).

## Empirical Results

### Kill Criteria Assessment (3 seeds: 42, 123, 314)

| Criterion | Metric | Value | Threshold | Verdict |
|-----------|--------|-------|-----------|---------|
| **K1** | AP/Rand |cos| ratio | 0.808 | < 0.70 | **FAIL (KILLED)** |
| **K2** | AP/Rand individual PPL ratio | 0.859 | < 1.05 | **PASS** |

### Orthogonality (aggregate, 30 cosine pairs across 3 seeds)

| Condition | Mean |cos| | Std | Improvement |
|-----------|-----------|-----|-------------|
| AP-init | 0.179 | -- | 19.2% lower |
| Random-init | 0.221 | -- | baseline |

Wilcoxon signed-rank test (AP < Random): p = 0.020

Per-seed AP/Random ratios: 0.739, 0.930, 0.762

### Individual Quality (mean PPL across 5 domains)

| Condition | Seed 42 | Seed 123 | Seed 314 | Mean |
|-----------|---------|----------|----------|------|
| AP-init | 4.12 | 4.29 | 4.27 | **4.23** |
| Random-init | 4.63 | 4.98 | 5.15 | 4.92 |
| AP/Rand ratio | 0.889 | 0.861 | 0.828 | **0.859** |

AP-init consistently produces better individual quality (14% lower PPL).

### Composition PPL (1/N averaging, evaluated on all 5 domains)

| Condition | Seed 42 | Seed 123 | Seed 314 | Mean |
|-----------|---------|----------|----------|------|
| AP-init | 27.3 | 38.0 | 26.0 | **30.5** |
| Random-init | 30.6 | 39.7 | 28.2 | 32.9 |
| AP/Rand ratio | 0.892 | 0.957 | 0.923 | **0.927** |

AP-init also improves composition PPL by 7.3%.

## Key Findings

**1. AP-init improves orthogonality by 19.2%, but fails the 30% threshold.**

The improvement is statistically significant (p=0.020) and directionally
correct across all 3 seeds. However, the pre-registered kill criterion required
>30% improvement (ratio < 0.7), and the observed ratio is 0.808. This is a
borderline kill -- the mechanism works, but the effect size is insufficient.

**2. The improvement comes from orthonormal initialization, NOT packing geometry.**

At N=5, r=4, d=64, we have Nr=20 < d=64. The Welch bound is zero -- five
rank-4 subspaces fit perfectly inside R^64 with no mutual interference. AP
packing is trivially solved (all frames are already orthogonal after AP). The
observed improvement is therefore from the orthonormality of A matrices
(a side-effect of AP), not from the Grassmannian packing algorithm itself.

This is a critical confound. The experiment does not test whether AP packing
survives ternary QAT. It tests whether orthonormal-init survives QAT better
than Gaussian-init. To properly test AP packing, one would need Nr > d
(e.g., N=20+, r=4, d=64 giving Nr=80 > d=64).

**3. AP-init unexpectedly improves individual quality by 14%.**

This was not predicted by the hypothesis. Orthonormal A matrices provide
a better starting point for ternary QAT because:
- Orthonormal columns have uniform entry magnitudes (~1/sqrt(d)), which means
  the ternary quantization threshold (alpha = mean|W|) is more stable.
- Gaussian A at scale 0.01 has many near-zero entries that collapse to 0
  under ternary quantization, losing information.

**4. Composition improvement (7.3%) is modest and tracks individual improvement.**

The composed PPL ratio (0.927) roughly matches the individual PPL ratio (0.859)
discounted by the number of experts. This suggests the composition benefit
comes from better individual experts, not from reduced interference per se.

## Comparison to Prior Results

| Experiment | Mechanism | |cos| improvement | Individual quality |
|------------|-----------|-------------------|-------------------|
| grassmannian_expert_init (FP16) | AP packing | 1.23-1.52x (d=128-256) | Not measured |
| bitnet_ternary_composition | Ternary QAT | -19.3% vs FP16 | +2.6% worse |
| **This experiment** | AP-init + ternary QAT | **-19.2% vs random-init** | **14.1% better** |

The orthogonality improvement magnitude is similar to the ternary decorrelation
effect, but the mechanisms are different (initialization structure vs
quantization noise). The quality improvement is the novel finding.

## Limitations

1. **Under-packed regime invalidates AP test.** Nr=20 < d=64 means AP packing
   is trivially achievable. The experiment tests orthonormal-init, not AP
   packing. A proper test needs Nr > d.

2. **Micro scale only.** d=64, r=4, toy domains. At d=4096, the orthonormal-init
   advantage may be smaller because Gaussian random matrices are already
   near-orthogonal at high dimension (concentration of measure).

3. **Confound: init scale.** AP-init A matrices have entries ~1/sqrt(64) = 0.125.
   Random-init has entries ~0.01. This 12.5x scale difference affects ternary
   quantization behavior. A fairer comparison would use random-orthonormal init
   (same as grassmannian_expert_init's three-condition design).

4. **Three seeds only.** Per-seed ratios vary widely (0.739 to 0.930).

5. **High composed PPL.** Both conditions show composed PPL 6-8x worse than
   individual PPL (30.5 vs 4.2), indicating composition is still destructive
   at this scale regardless of initialization.

## What Would Kill This

Already killed by K1 (ratio 0.808 > 0.700 threshold). To salvage the direction:

- **Reframe as orthonormal-init benefit:** The 14% quality improvement is genuine
  and valuable. A follow-up could test orthonormal-init vs Gaussian-init
  directly (without AP), which is a simpler and more useful result.
- **Test in over-packed regime:** N=20, r=4, d=64 (Nr=80 > d=64) would actually
  test whether AP packing geometry survives ternary QAT.
- **Control for init scale:** Use random-orthonormal init as a third condition
  to isolate the packing effect from the orthonormality effect.

## Verdict

**KILLED (K1).** AP-init shows 19.2% lower |cos| than random-init after ternary
QAT (p=0.020), but this falls short of the pre-registered 30% improvement
threshold (ratio 0.808 vs 0.700). The improvement is primarily from orthonormal
initialization, not from Grassmannian packing geometry (which is trivially
achievable at Nr=20 < d=64). K2 passes comfortably (AP-init improves individual
quality by 14%). The 5th consecutive BitNet-track kill.

**Salvageable insight:** Orthonormal LoRA initialization improves ternary QAT
convergence. This is a simple, practical result that does not require the
Grassmannian skeleton infrastructure.
