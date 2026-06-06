# Mixing Ratio Significance: Research Digest

## Hypothesis

The +11.2% quality improvement from 20% synthetic data mixing (reported in
exp_synthetic_vs_real_data) is statistically significant, and the optimal
mixing ratio is stable across random seeds.

**Falsifiable:**
- K1: If Wilcoxon signed-rank p > 0.05 for ratio=0.2 vs ratio=0.0,
  the mixing benefit is indistinguishable from noise.
- K2: If the 95% bootstrap CI for the optimal ratio has width > 0.15,
  the optimal ratio is too unstable to be actionable.

## What This Experiment Is

A statistical follow-up to exp_synthetic_vs_real_data, addressing the
adversarial review's concern that the +11.2% mixing benefit had overlapping
confidence intervals. We scale from 5 to 20 seeds, use paired statistical
tests (Wilcoxon signed-rank on per-seed differences), and find the optimal
ratio with bootstrap confidence intervals on a finer 0.05-step grid.

The experimental setup (data generation, LoRA training, quality metric)
is identical to the parent. The only changes are statistical power (20 seeds)
and resolution (21 ratios instead of 11).

## Key References

- Parent: micro/models/synthetic_vs_real_data/ (status: supported)
- Wilcoxon (1945), "Individual comparisons by ranking methods"
- Efron & Tibshirani (1993), "An Introduction to the Bootstrap"
- Cohen (1988), "Statistical Power Analysis for the Behavioral Sciences"

## Empirical Results

### K1: Mixing Benefit Significance -- KILLED

| Ratio | Mean Diff vs 0.0 | Median Diff | W+:W- | Wilcoxon p | Sig? |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.05 | +0.0002 | +0.0017 | 11:9 | 0.985 | ns |
| 0.10 | -0.0006 | +0.0017 | 11:9 | 0.701 | ns |
| 0.15 | -0.0005 | +0.0042 | 12:8 | 0.784 | ns |
| **0.20** | **-0.0008** | **-0.0012** | **8:12** | **0.571** | **ns** |
| 0.25 | +0.0018 | -0.0002 | 10:10 | 0.729 | ns |
| 0.30 | -0.0037 | -0.0005 | 10:10 | 0.261 | ns |

**K1 KILLED.** No mixing ratio achieves statistical significance vs pure real.
The closest is ratio=0.30 (p=0.261), but even that is far from the 0.05
threshold. At ratio=0.2 specifically, the mean difference is actually
NEGATIVE (-0.0008), with only 8/20 seeds showing improvement. The effect
size is negligible (Cohen's d = -0.057).

The parent's +11.2% finding at ratio=0.2 was noise amplified by small sample
size (5 seeds, ~35% power at the true effect size of ~0).

### K2: Optimal Ratio Stability -- KILLED

| Metric | Value |
|:---:|:---:|
| Global best ratio (pooled mean) | 0.25 |
| Per-seed optimal: mean | 0.215 |
| Per-seed optimal: std | 0.126 |
| Per-seed optimal: range | 0.50 (from 0.05 to 0.55) |
| Bootstrap 95% CI for optimal | [0.05, 0.40] |
| Bootstrap CI width | **0.35** (threshold: 0.15) |

**K2 KILLED.** The optimal ratio is highly unstable. The bootstrap CI
spans from 0.05 to 0.40, a width of 0.35 -- more than 2x the kill
threshold. Different seed subsets yield completely different "optimal"
ratios. Per-seed optima range from 0.05 to 0.55.

### Quality Landscape

The quality curve across mixing ratios is essentially flat from 0.0 to 0.25,
then monotonically decreasing:

| Ratio | Mean Quality | vs Pure Real |
|:---:|:---:|:---:|
| 0.00 | 0.0572 | baseline |
| 0.05 | 0.0574 | +0.3% |
| 0.10 | 0.0567 | -0.9% |
| 0.15 | 0.0567 | -1.0% |
| 0.20 | 0.0564 | -1.5% |
| 0.25 | 0.0590 | +3.2% |
| 0.50 | 0.0544 | -4.9% |
| 1.00 | 0.0260 | -54.5% |

The "improvement" at any low ratio (0.0-0.25) is within noise. The
meaningful finding is that ratios above ~0.5 are genuinely harmful.

### Effect Size Analysis

| Effect | Cohen's d | Interpretation |
|:---:|:---:|:---:|
| ratio=0.2 vs 0.0 | -0.057 | Negligible |
| ratio=0.25 vs 0.0 | +0.118 | Negligible |
| ratio=0.5 vs 0.0 | -0.235 | Small |
| ratio=1.0 vs 0.0 | -2.715 | Very large |

Only the pure-synthetic endpoint (ratio=1.0) has a meaningfully large
effect. The supposed "sweet spot" at 0.1-0.3 has negligible effect size.

## What This Means for SOLE

1. **The 80/20 mixing recommendation from the parent experiment is not
   supported.** There is no statistically significant benefit to mixing
   synthetic data at any ratio. Pure real data is as good as any mix in
   the 0-25% synthetic range.

2. **The parent's K2 (mixing survives) should be downgraded.** The +11.2%
   improvement was a statistical artifact of 5 seeds and high variance.
   With 20 seeds and paired testing, the effect vanishes (d = -0.06).

3. **Synthetic data does not HELP, but small amounts do not HURT.** Up to
   ~25% synthetic, quality is essentially unchanged. This means the current
   Groq-synthetic pipeline is fine IF supplemented with real data -- but the
   synthetic fraction provides no measurable quality benefit.

4. **The pilot-50 distillation (100% synthetic) remains the actual concern.**
   Pure synthetic is genuinely 55% worse than real (highly significant,
   confirmed by both parent and this experiment). The actionable recommendation
   is: add real data, not optimize the mixing ratio.

## Limitations

1. **Linear task at d=64.** The flat quality landscape at low mixing ratios
   may differ for nonlinear tasks. It is possible that mixing helps for
   complex tasks even if it does not help for linear regression.

2. **Single expert per ratio per seed.** Each data point is one LoRA training
   run. Training variance within a seed may obscure small effects. However,
   the paired design controls for the largest variance source (initialization).

3. **The parent used slightly different seed set.** Seeds [42, 123, 456, 789,
   1337] vs our [42-61]. The first seed (42) overlaps. Results are consistent
   with the parent when restricted to 5 seeds -- the difference is statistical
   power, not setup.

4. **Bootstrap CI for optimal ratio is conservative.** With a flat quality
   curve, the argmax is maximally unstable. This is not a failure of the
   bootstrap -- it correctly reflects that there is no clear optimum.

## What Would Kill This

This experiment is already KILLED on both criteria. To REVIVE the mixing
benefit hypothesis, one would need:

- A different task structure (nonlinear) where coverage-quality tradeoff
  behaves differently
- Much larger N (>100 seeds) to detect very small effects (d < 0.2)
- Different quality metric that is more sensitive to the mixing benefit
- Macro-scale validation on real Qwen2.5-7B LoRA training

## Summary

| Question | Answer |
|----------|--------|
| Is the +11.2% mixing benefit significant? | **NO.** p=0.57, d=-0.06. Pure noise. |
| Is there ANY significant mixing ratio? | **NO.** All p > 0.26 across 0.05-0.30. |
| Is the optimal ratio stable? | **NO.** 95% CI = [0.05, 0.40], width 0.35. |
| Does synthetic data hurt? | **Only above ~50%.** 0-25% is neutral. |
| What is the true effect of 20% mixing? | **Zero.** Mean diff = -0.0008, Cohen's d = -0.06. |
| Parent's +11.2% finding? | **Noise** from 5-seed sample (35% power). |
