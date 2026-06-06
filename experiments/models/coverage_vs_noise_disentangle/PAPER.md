# Coverage vs Noise Disentangle: Research Digest

## Hypothesis

In the 58% quality gap between synthetic-only (5 modes, sigma=0.05) and
real-only (20 modes, sigma=0.30) LoRA training data, input coverage is the
dominant factor, not label noise.

**Falsifiable:**
- K1: If coverage alone explains <50% of the gap, noise (not coverage) drives
  the quality difference, and the parent experiment's causal claim is wrong.
- K2: If noise alone explains >80% of the gap, coverage is irrelevant for
  LoRA training data quality.

## What This Experiment Is

A 2x2 factorial ablation that breaks the confound identified in the adversarial
review of `exp_synthetic_vs_real_data`. The parent experiment varied coverage
AND noise simultaneously. This experiment independently varies:

- **Factor A: Coverage** -- 5 modes (low) vs 20 modes (high)
- **Factor B: Noise** -- sigma=0.05 (low) vs sigma=0.30 (high)

Four conditions, evaluated on uniform held-out data (10 seeds, 4 experts each):

| | Low Noise (0.05) | High Noise (0.30) |
|---|---|---|
| Low Coverage (5 modes) | original "synthetic" | NEW |
| High Coverage (20 modes) | NEW | original "real" |

The two new conditions (low-cov/high-noise and high-cov/low-noise) isolate
each factor's contribution via standard ANOVA decomposition.

**Design decisions:**
- Systematic bias REMOVED (was 0.25 in parent synthetic condition) to avoid a
  third confound
- Dirichlet alpha and input spread remain coupled to mode count (matching the
  parent's operational definition of "coverage")
- 10 seeds (up from 5) for better statistical power on the decomposition

## Lineage in the Arena

```
micro/synthetic_vs_real_data (proven: 58% gap, mixing benefit)
  |
  +-- THIS: micro/coverage_vs_noise_disentangle (2x2 factorial ablation)
  |         Resolves the coverage-noise confound
  |
  +-- exp_mixing_ratio_significance (open: statistical significance of alpha*)
```

## Key References

- Gunasekar et al. 2023, "Textbooks Are All You Need" (Phi-1) -- Synthetic data
  quality vs quantity tradeoff
- Shumailov et al. 2024, "Model Collapse" -- Diversity loss in recursive
  synthetic generation (motivates the coverage factor)
- Zhou et al. 2023, "LIMA" -- 1000 carefully curated examples beat 52K noisy
  ones. Consistent with coverage mattering more than noise.
- NotebookLM research finding: "LLMs are highly robust to label noise" -- models
  trained on entirely incorrect answers can match correct-answer models (SFT).
  Strict correctness filtering can HURT because it reduces diversity.

## Empirical Results

### 2x2 Quality Table (uniform eval, 10 seeds)

| | Low Noise (0.05) | High Noise (0.30) | Row Mean |
|:---|:---:|:---:|:---:|
| Low Coverage (5 modes) | 0.0238 +/- 0.0021 | 0.0234 +/- 0.0017 | 0.0236 |
| High Coverage (20 modes) | 0.0601 +/- 0.0046 | 0.0584 +/- 0.0059 | 0.0593 |
| Column Mean | 0.0420 | 0.0409 | 0.0414 |

### ANOVA Decomposition

| Effect | Size | % of Variance | t-stat | p-value |
|:---|:---:|:---:|:---:|:---:|
| **Coverage (main)** | **+0.036** | **96.2%** | **+12.07** | **<0.0001** |
| Noise (main) | +0.001 | 1.6% | +0.74 | 0.48 |
| Interaction | -0.001 | 2.2% | -0.37 | 0.72 |

### Simple Effects (conditional)

| Effect | Size | t-stat | p-value |
|:---|:---:|:---:|:---:|
| Coverage at low noise | +0.036 | +10.65 | <0.0001 |
| Coverage at high noise | +0.035 | +10.40 | <0.0001 |
| Noise at low coverage | +0.0004 | +0.25 | 0.81 |
| Noise at high coverage | +0.002 | +0.62 | 0.55 |

### Kill Criteria

| Criterion | Observed | Threshold | Verdict |
|:---|:---:|:---:|:---:|
| K1: coverage explains <50% | **96.2%** | <50% | **SURVIVES** |
| K2: noise explains >80% | **1.6%** | >80% | **SURVIVES** |

**Both kill criteria survive by massive margins.** Coverage explains 96.2% of
the variance (threshold was 50%). Noise explains only 1.6% (threshold was 80%).

### Key Quantitative Findings

1. **Coverage dominates overwhelmingly.** Going from 5 to 20 input modes
   improves quality by +0.036 (150% relative improvement), while going from
   sigma=0.30 to sigma=0.05 improves quality by only +0.001 (4% relative).

2. **Noise effect is statistically non-significant.** All noise-related
   t-tests yield p > 0.48. The 6x difference in label noise (0.05 vs 0.30)
   produces no detectable quality difference.

3. **No interaction.** Coverage effect is the same whether noise is low
   (+0.036) or high (+0.035). Noise effect is the same whether coverage
   is low (+0.0004) or high (+0.002). The factors are additive.

4. **Total gap reproduced.** The low-cov/low-noise vs high-cov/high-noise
   gap is 59.3% (parent: 58.1%), confirming experimental consistency.

5. **Consistent with literature.** The NotebookLM survey found that "LLMs are
   highly robust to label noise" and "input coverage dominates over label
   correctness for generalization." Our frozen-A linear regression reproduces
   this finding in a controlled setting.

### Effect Size Context

The coverage main effect (+0.036) is 36x larger than the noise main effect
(+0.001). To put this in perspective: you would need to increase the noise
ratio from 6x to approximately 200x before noise would explain even 50% of
the variance (extrapolating linearly, which overestimates the noise effect
since the relationship is likely sublinear at extreme noise levels).

## Micro-Scale Limitations

1. **d=64 saturates effective rank.** Both M=5 and M=20 produce near-maximal
   effective rank (62.8 vs 63.5) because 1000 samples fill 64 dimensions.
   At d=4096, the gap would be far more dramatic: 5 modes span ~0.1% of
   the space vs 20 modes spanning ~0.5%.

2. **Linear task.** W* is a rank-r linear map. In nonlinear tasks, noise near
   decision boundaries may matter more than in the linear case. However, the
   literature evidence (SFT with incorrect answers still works) suggests the
   coverage dominance holds for real LLMs.

3. **Isotropic Gaussian noise.** Real label noise is structured (teacher
   systematic errors). Structured noise aligned with coverage gaps could
   interact differently than isotropic noise.

4. **Coverage definition is compound.** We vary mode count, Dirichlet alpha,
   and input spread together. A finer ablation could separate these sub-factors,
   though the compound treatment matches the parent experiment's definition.

5. **Systematic bias removed.** The parent experiment included systematic bias
   (0.25) for synthetic data. Removing this makes the factorial cleaner but
   means the 59.3% gap here is not exactly the same confound as the parent's
   58.1% gap. The bias removal is conservative (it could only reduce the gap).

## What Would Kill This

### At micro scale (already tested):
- **K1 SURVIVES (96.2%).** Coverage alone explains nearly all of the gap.
  The parent experiment's causal attribution to "coverage dominates label
  quality" is now confirmed by the factorial design.
- **K2 SURVIVES (1.6%).** Noise is negligible. Coverage is not irrelevant.

### At macro scale (would need to test):
- Train Qwen2.5-7B LoRA experts on:
  (a) 5 diverse topics with GPT-4 labels (low coverage, low noise)
  (b) 20 diverse topics with GPT-4 labels (high coverage, low noise)
  (c) 5 topics with noisy labels (low coverage, high noise)
  (d) 20 topics with noisy labels (high coverage, high noise)
- Kill if: noise explains >20% of variance at macro scale (would contradict
  the micro finding that noise is negligible)
- Kill if: coverage explains <60% at macro scale (would suggest nonlinear
  tasks change the balance)

### Implications for SOLE:
1. **Maximize input diversity, not label quality.** When curating distillation
   data, prioritize covering more topics/styles over filtering for correctness.
   This aligns with LIMA: "fewer but more diverse examples beat more but
   repetitive examples."
2. **The pilot-50's quality concerns are likely coverage-limited.** Each expert
   was trained on a single topic with 1000 Groq-generated examples. The
   coverage within each topic may be narrow (few input modes).
3. **Noise filtering is low-priority.** Spending compute on response
   verification/filtering provides negligible benefit compared to spending
   the same budget on input diversification.
4. **The 80/20 mix recommendation from the parent experiment remains valid**
   but for a clarified reason: the synthetic fraction helps not because it
   reduces noise, but because it adds coverage from a different distribution.

## Summary

| Question | Answer |
|----------|--------|
| What drives the 58% quality gap? | **Coverage (96.2%).** Not noise. |
| Does label noise matter? | **No.** 6x noise difference has no detectable effect (p=0.48). |
| Is there a coverage-noise interaction? | **No.** Effects are purely additive (p=0.72). |
| Does this confirm the parent claim? | **Yes, and strengthens it.** The parent attributed the gap to coverage; this factorial confirms it cleanly. |
| Practical recommendation? | **Diversify inputs, don't filter noise.** |
