# Synthetic vs Real Data Quality: Research Digest

## Hypothesis

Synthetic (LLM-generated) training data produces LoRA experts with comparable
quality to real-data experts, and mixing synthetic + real data outperforms either
source alone.

**Falsifiable:**
- K1: If synthetic-only expert is >15% worse than real-data expert on uniform
  held-out evaluation, synthetic data is insufficient for SOLE expert distillation.
- K2: If mixed (synthetic + real) is NOT better than either alone on uniform
  held-out evaluation, mixing provides no benefit.

## What This Experiment Is

A Monte Carlo simulation study comparing three training data regimes for SOLE
expert distillation. Each regime trains LoRA experts (frozen-A, learned-B) on
data generated from calibrated statistical models:

- **Synthetic**: Low label noise (sigma=0.05), concentrated inputs (5 modes,
  Dirichlet alpha=0.5), systematic bias. Models Groq/Phi-1 style "textbook" data.
- **Real**: High label noise (sigma=0.30), diverse inputs (20 modes, Dirichlet
  alpha=2.0), no bias. Models codeparrot-clean style naturally-occurring data.
- **Mixed**: Sweep from 0% to 100% synthetic in 10% increments.

All experts are evaluated on three held-out distributions (uniform, synthetic-like,
real-like) and compared on quality, diversity, orthogonality, and contamination risk.

### Calibration Sources

| Parameter | Calibrated From |
|-----------|----------------|
| Synthetic label noise (0.05) | Phi-1: 50.6% HumanEval with synthetic-only |
| Real label noise (0.30) | codeparrot-clean quality variance |
| Synthetic modes (5) | Shumailov et al. 2024: mode collapse in recursive generation |
| Real modes (20) | GitHub code style diversity (estimated) |
| Benchmark overlap (10% synth, 2% real) | Xu et al. 2024: 5-15% overlap in LLM training data |
| Systematic bias (0.25) | The "LLM accent" -- recognizable style patterns |

## Lineage in the Arena

```
micro/distillation_pilot_50 (supported: 98% win rate, $0.44/expert)
  |
  +-- THIS: micro/synthetic_vs_real_data (data source analysis)
  |     |
  |     +-- Informs: optimal data mixing for production distillation
  |     +-- Informs: contamination risk assessment for benchmark reporting
  |
  +-- exp_pilot50_held_out_eval (active: MMLU/HumanEval eval)
```

## Key References

- Gunasekar et al. 2023, "Textbooks Are All You Need" (Phi-1) -- Synthetic-only
  data achieves 50.6% on HumanEval, competitive with much larger models trained
  on real data. Demonstrates that data QUALITY can compensate for quantity.
- Mukherjee et al. 2023, "Orca" -- Synthetic instruction-following data from GPT-4
  enables small models to match teacher performance on many benchmarks.
- Shumailov et al. 2024, "Model Collapse" -- Recursive training on LLM-generated
  data causes progressive diversity loss and eventual collapse.
- Xu et al. 2024 -- Benchmark contamination analysis showing 5-15% overlap between
  LLM training data and common evaluation benchmarks.

## Empirical Results

### K1: Synthetic vs Real Quality Gap

| Eval Distribution | Synthetic Quality | Real Quality | Gap | vs 15% Threshold |
|:---:|:---:|:---:|:---:|:---:|
| Uniform (unbiased) | 0.0253 +/- 0.0022 | 0.0603 +/- 0.0071 | 58.1% | **KILLED** |
| Synthetic-like | 0.0340 +/- 0.0038 | 0.0580 +/- 0.0092 | 41.4% | KILLED |
| Real-like | 0.0245 +/- 0.0016 | 0.0588 +/- 0.0049 | 58.3% | KILLED |

**K1 KILLED.** Synthetic-only is 58% worse than real-only on uniform evaluation.
The gap is consistent across all three evaluation distributions. Even on
synthetic-like evaluation data (where synthetic experts have home-field advantage),
real-data experts still outperform by 41%.

**Root cause:** Synthetic data concentrates on 5 modes, providing clean gradient
signal in those directions but failing to cover the full input space. Real data,
despite noisy labels, provides gradient signal across 20 modes, enabling broader
generalization. Coverage dominates label quality for LoRA training.

### K2: Mixed vs Pure

| Mixing Ratio | Quality (uniform) | vs Best Pure | Verdict |
|:---:|:---:|:---:|:---:|
| 0.0 (pure real) | 0.0581 | baseline | - |
| 0.1 | 0.0648 | +11.5% | improves |
| **0.2** | **0.0670** | **+15.3%** | **BEST** |
| 0.3 | 0.0576 | -0.9% | neutral |
| 0.5 | 0.0545 | -6.2% | worse |
| 1.0 (pure synth) | 0.0238 | -59.0% | worst |

**K2 SURVIVES.** Mixed at 20% synthetic improves over best pure (real) by +11.2%
(mean across seeds; table shows one illustrative sweep). The sweet spot is
alpha* ~ 0.1-0.3 synthetic fraction.

**Mechanism:** A small synthetic fraction (10-20%) provides clean gradient signal
in high-density regions, acting as a regularizer that reduces noise without
sacrificing coverage. Beyond 30% synthetic, coverage loss begins to dominate.

This aligns with published findings (Genetic Instruct, DPO literature): "accumulate
synthetic alongside real, don't replace" and "quality beats quantity."

### Orthogonality Analysis

| Group | Mean |cos| | Std | Angle (deg) |
|:---:|:---:|:---:|:---:|
| Within synthetic | 0.1064 | 0.0207 | 0.75 |
| Within real | 0.1096 | 0.0099 | 2.22 |
| Within mixed | 0.1119 | 0.0220 | 2.73 |
| Cross synth-real | 0.0951 | 0.0144 | - |

**Key finding: Data source does NOT significantly affect expert orthogonality.**
All |cos| values are near the random baseline at d=64 (sqrt(r/d) ~ 0.35 upper
bound; measured values are 3x below bound). This is consistent with the
structural orthogonality theorem: orthogonality is a property of d and r, not
of training data.

**Nuance:** Synthetic experts have LOWER subspace angles (0.75 deg vs 2.22 deg),
meaning they converge to MORE SIMILAR B subspaces. This is because concentrated
training data constrains optimization to similar solutions. However, the effect
on weight-space cosine is negligible because the delta_W = B @ A product involves
random A, which dominates the alignment.

Cross-regime orthogonality (0.0951) is slightly LOWER than within-regime (~0.107),
suggesting that mixing data sources actually INCREASES inter-expert diversity.

### Contamination Risk

| Regime | P(contaminated) | Expected Overlap | HumanEval Boost |
|:---:|:---:|:---:|:---:|
| Synthetic | 1.000 | 100 examples | 18.3% |
| Real | 1.000 | 20 examples | 3.7% |

Synthetic data carries 5x higher contamination risk. At the pilot-50 scale
(1000 training examples per expert), there is near-certainty of benchmark
contamination in synthetic data. This does NOT invalidate the experts but
requires execution-based evaluation (HumanEval pass@1) rather than
memorization-susceptible metrics.

**Mitigation:** Use HumanEval (execution-based, no contamination risk) rather
than MBPP-style benchmarks. Report contamination estimates alongside results.
Mixed training at 20% synthetic reduces expected boost from 18.3% to 6.6%.

### Effective Rank (Diversity)

| Regime | Effective Rank | vs d=64 max |
|:---:|:---:|:---:|
| Synthetic | 62.7 | 98.0% |
| Real | 63.5 | 99.2% |

Both regimes achieve near-maximal effective rank in 64 dimensions because N=1000
samples from even 5 modes fill the space. The effective rank metric is INSENSITIVE
to mode structure at this scale. The actual diversity difference manifests in
gradient direction concentration, not dimensionality.

At production scale (d=4096, N=1000), the effective rank gap would be much larger
because 5 modes cannot fill a 4096-dimensional space.

## Micro-Scale Limitations

1. **Linear task:** W* is a linear map. Real NLP tasks are nonlinear. The
   coverage-quality tradeoff direction should hold but magnitudes may differ.

2. **d=64 vs d=4096:** At production scale, synthetic coverage gaps would be
   MORE severe (5 modes in 4096D covers ~0.1% vs ~7.8% of the space). The
   quality gap may be even worse than 58%.

3. **Frozen random A:** The SOLE architecture uses Grassmannian-initialized A.
   With optimal A placement, the coverage penalty of synthetic data may be
   partially mitigated.

4. **No actual code:** This simulates abstract feature vectors, not real Python
   code. The textual quality difference between Groq-generated and codeparrot
   code may behave differently.

5. **Quality metric:** Frobenius reconstruction error is a proxy. Real quality
   metrics (HumanEval pass@1, MMLU accuracy) may show different sensitivities
   to coverage vs noise.

## What Would Kill This

### At micro scale (already tested):
- **K1 KILLED:** Synthetic-only is 58% worse. Cannot rely on synthetic data alone.
- **K2 SURVIVES:** 20% synthetic mixing provides +11% improvement.

### At macro scale (would need to test):
- Train Qwen2.5-7B LoRA experts on:
  (a) Groq-only synthetic data (current pipeline)
  (b) codeparrot-clean real data (same domain/size)
  (c) 80/20 real/synthetic mix
- Evaluate on HumanEval pass@1 (execution-based, no contamination)
- Kill if: macro results contradict micro direction (synthetic better than real,
  or mixing does not help)

### Implications for SOLE:
1. **Do not use synthetic-only training for production experts.** The 50-expert
   pilot used synthetic-only, which explains some of the quality concerns.
2. **Mix 10-30% synthetic data with real data** for optimal expert quality.
3. **Use execution-based evaluation** (HumanEval, SWE-bench) to avoid
   contamination inflation. Report contamination estimates.
4. **Orthogonality is unaffected by data source** -- the Grassmannian skeleton
   works regardless of training data composition.
5. **Revise pilot-50 distillation pipeline** to incorporate real data where available.
   Estimated cost increase: 10-20% (real data curation is cheap for code domains
   via HuggingFace datasets).

## Summary

| Question | Answer |
|----------|--------|
| Does synthetic data produce comparable quality? | **NO.** 58% worse on uniform eval. |
| Does synthetic data cause mode collapse? | **YES.** 5x fewer gradient directions. Subspace angles 3x narrower. |
| Does mixing help? | **YES.** 20% synthetic optimal (+11% over pure real). |
| Does data source affect orthogonality? | **NO.** |cos| ~ 0.10 regardless. |
| Contamination risk? | **5x higher** for synthetic (18.3% vs 3.7% boost). |
| Optimal strategy? | **80/20 real/synthetic mix** with execution-based eval. |
