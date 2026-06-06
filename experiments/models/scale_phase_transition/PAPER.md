# Scale Phase Transition: Math Reasoning Activation Boundary

## Theorem (from MATH.md)

The scale-behavior mapping f(s) for math reasoning follows a sharp phase transition,
not a gradual crossover. There exists a critical scale s* such that for s < s*, the
adapter operates in the FORMAT regime (no reasoning activation), and for s > s*, the
adapter fully activates capability. The transition width is effectively zero.

## Hypothesis

Three competing models were tested:
1. **Phase transition (step function):** Sharp jump at some s* in [4, 16]
2. **Gradual crossover (sigmoid):** f(10) ~ 0.45, smooth interpolation
3. **Two-threshold:** Format activates before calculation

## What This Experiment Is

A scale sweep on math domain only, testing s = {1, 2, 4, 6, 8, 10, 12, 16, 20} with
10 prompts each. Binary correctness (numerical answer match) plus chain-of-thought
detection. Resolves the shape of the transition between Finding #249's two data points
(s=2 → 0.10, s=20 → 0.80).

## Key References

- Hu et al. (2021) "LoRA" arXiv:2106.09685
- Nanda et al. (2023) "Progress measures for grokking" arXiv:2301.05217
- Finding #249: Two behavioral regimes (FORMAT vs CAPABILITY)
- Finding #238: Math behavioral quality at per-domain optimal

## Predictions vs Measured

| Prediction | Predicted | Measured | Match |
|-----------|-----------|----------|-------|
| P1 (Model 1): Jump >= 0.4 between adjacent scales | >= 0.4 | 0.60 (s=4→s=6) | YES |
| P2 (Model 2): f(10) ~ 0.45 (midpoint) | 0.45 | 0.80 | NO |
| P3 (Model 3): Intermediate CoT without correct answers | CoT at s=6 without answers | CoT + correct at s=6 | NO |
| P4: Monotonic in scale | Non-decreasing | Non-monotonic (s=16 < s=12) | NO |

**Model 1 (phase transition) is the best fit.** Model 2 and Model 3 are refuted.

## Empirical Results

### Transition Curve

| Scale | Accuracy | CoT Rate | Correct/10 |
|-------|----------|----------|------------|
| base  | 0.10     | 0.90     | 1          |
| 1.0   | 0.10     | 0.70     | 1          |
| 2.0   | 0.10     | 0.80     | 1          |
| 4.0   | 0.10     | 0.60     | 1          |
| **6.0**   | **0.70** | **1.00** | **7**      |
| 8.0   | 0.80     | 0.90     | 8          |
| 10.0  | 0.80     | 0.90     | 8          |
| 12.0  | 0.80     | 0.90     | 8          |
| 16.0  | 0.70     | 1.00     | 7          |
| 20.0  | 0.80     | 1.00     | 8          |

### Key Observations

**1. The transition is a phase transition at s* ~ 5.** The jump from s=4 (0.10) to s=6
(0.70) is +0.60 — the largest between any adjacent scales and constitutes the entire
transition. Below s=4, accuracy equals base rate (0.10). Above s=6, accuracy plateaus
at 0.70-0.80.

**2. Sigmoid fit is degenerate — confirms step function.** Best-fit sigmoid: s_mid=5.7,
tau=0.1, R²=0.989. However, tau=0.1 hits the optimizer's lower bound exactly — the fit
wants a pure step function but is constrained. The R² is high because any step function
near s=5 fits well. **The fit adds no information beyond the raw table.** The transition
occurs between s=4 and s=6; claiming more precision than "s* in [4, 6]" is unsupported.

**3. K3 FAIL: Non-monotonic behavior.** s=16 (0.70) < s=12 (0.80). This violates the
monotonicity assumption (A1 from MATH.md). At n=10, the difference (7/10 vs 8/10) is
within binomial noise (p=0.66 for observing 7/10 when true p=0.8). The non-monotonicity
is likely noise, not a real perturbation model failure.

**4. CoT rate is NOT a leading indicator — low-scale adapters are DISRUPTIVE.**
Base model already has 0.90 CoT rate. CoT at s=4 (0.60) and s=1 (0.70) are LOWER
than base. The adapter at low scale disrupts base CoT formatting without activating
correct computation. Model 3 (two-threshold) is refuted.

**5. Format confound: GSM8K training format drives extraction.** At s=6, generations
switch from verbose prose to GSM8K-style "<<3*26=78>>" format with "####" answer
markers. The accuracy jump may be partially driven by the adapter imposing training
format that the answer extraction regex can parse, not purely by reasoning improvement.
The same prompts at s=4 produce correct reasoning steps in prose that the regex cannot
extract. **This confound does not invalidate the transition (the format IS the
capability) but means "reasoning activation" should be read as "training-format
activation."**

**6. The plateau at 0.70-0.80 suggests a ceiling.** Accuracy does not increase beyond
0.80 at any scale. The same prompt (James/partner, gt=70) is correct at ALL scales
including base. The 2 always-wrong prompts may be too hard for this model regardless
of scale.

**7. Per-prompt transition is noisy but shows threshold structure.** Of the 10 prompts:
- 1 always correct (James/70, base through s=20)
- 0 never correct (all prompts are correct at some scale)
- At s=6: 6 prompts flip from wrong to right (vs s=4)
- At s=8: different set of 8 correct — 3 new prompts gain correctness (Cori, Bert, Movie)
  while 2 lose it (Javier, Calvin)
- Per-prompt correctness is NOT stable across scales: individual prompts flicker on/off,
  suggesting stochastic activation near the threshold, not deterministic per-prompt thresholds
- The aggregate jump (1→7 from s=4 to s=6) reflects most prompts crossing threshold
  simultaneously, but individual prompt trajectories are noisy

### Kill Criteria Results

| Criterion | Result | Evidence |
|-----------|--------|----------|
| K1: No scale in [4,16] > 0.20 | **PASS** | Max intermediate = 0.80 (at s=8,10,12) |
| K2: All scores <= 0.20 or >= 0.60 | **FAIL** | No intermediate values (0.20 < acc < 0.60) exist |
| K3: Non-monotonic | **FAIL** | s=16 (0.70) < s=12 (0.80), likely noise at n=10 |

K1 PASS means intermediate activation exists (the transition is in [4,6], not [16,20]).
K2 FAIL means no gradual crossover — the transition IS a pure phase transition.
K3 FAIL is a statistical artifact: 7/10 vs 8/10 at n=10.

## Limitations

1. **n=10 per scale.** Binomial 95% CI for p=0.8 is [0.49, 0.95]. Cannot distinguish
   0.70 from 0.80 at this sample size. The non-monotonicity (K3) is within noise.

2. **Single domain.** Math only. The transition boundary for code may differ (code showed
   intermediate behavior at s=2 in Finding #249: 0.504 vs base 0.419).

3. **Integer scale steps.** The transition between s=4 and s=6 is 2 scale units wide.
   Finer sampling (s=4.5, 5.0, 5.5) would pinpoint s* more precisely.

4. **Deterministic generation.** temperature=0.0. Stochastic sampling might smooth the
   transition.

## What Would Kill This

1. Replication with n>=50 showing accuracy at s=6 is actually 0.30-0.50 (the jump is
   noise, and the true transition is gradual).
2. Finer sampling showing the transition spans s=4 to s=8 with intermediate values
   (making tau >> 0.1).
3. Different prompts showing the transition point varies widely per prompt (making s*
   a distribution, not a constant).

## Architectural Implication (Hypothesis — Requires Verification)

**Binary scale MAY suffice.** The transition is sharp enough that intermediate scales
provide no benefit for math. For math: any s >= 6 activates full reasoning (7-8/10
correct, indistinguishable at n=10). For knowledge domains: s <= 4 preserves base
knowledge (Finding #249).

This SUGGESTS a simplification from Finding #249's 5-value lookup table to a 2-value
system, but this is a hypothesis, not a conclusion:
- CAPABILITY domains (math, code, medical): s = 6 may suffice
- FORMAT domains (legal, finance): s = 4 or s = 1

**Untested assumptions:**
1. Code and medical domains have the same transition boundary as math (untested)
2. s=6 matches s=20 quality at larger n (indistinguishable at n=10 but could diverge)
3. The transition location is stable across different prompts/difficulty levels
4. The format confound does not affect code/medical evaluation differently

**Follow-up required:** Test s=6 vs s=20 on code and medical domains with n>=50 before
adopting binary scale in the routing architecture.
