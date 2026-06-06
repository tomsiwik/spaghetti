# Channel Capacity Bound for Expert Composition: Research Digest (REVISED)

## Hypothesis

The composition gap between N composed domain experts and joint training follows
Shannon's channel capacity law: gap(N) = (1 - log(1 + SNR/(1+(N-1)*alpha)) / log(1+SNR)) * 100,
where alpha encodes inter-expert interference and SNR encodes expert quality.

**Falsifiable**: The model must predict held-out N values (N=3,4,6,7) with R^2 >= 0.5
after fitting to training data (N=2,5,8).

## What This Model Is

A theoretical framework that models the residual stream of a composed expert model
as a Gaussian Multiple-Access Channel (MAC). Each domain expert is a transmitter
adding its signal to the shared d-dimensional residual stream. Inter-expert
interference (from non-orthogonal expert outputs) acts as noise that degrades
per-expert capacity as N grows.

The framework takes three empirical composition gap measurements (at N=2, 5, 8)
and fits the Shannon channel capacity curve to derive:
1. The effective signal-to-noise ratio (SNR_0)
2. The interference coupling constant (alpha)
3. Predictions for untested N values and the maximum safe N

No new model architecture is introduced. This is an analytical tool.

## Lineage in the Arena

```
  gpt (dense baseline)
    +-- capsule_moe (composition protocol)
          +-- n_expert_scale (N=2,5 gap data)
          +-- n8_identity_boundary (N=8 identity data)
          +-- flat_moe_n8_boundary (N=2,5,8 gap + Jaccard)
                +-- channel_capacity_bound (THIS: theoretical model)
```

## Key References

1. **Shannon, 1948** -- "A Mathematical Theory of Communication". Foundation.
2. **Cover & Thomas, 2006** -- "Elements of Information Theory", Ch. 15: MAC.
3. **Tse & Viswanath, 2005** -- "Fundamentals of Wireless Communications", Ch. 6.
4. **Project-internal**: n_expert_scale (Exp 4), flat_moe_n8_boundary (Exp 15).

## Revision History

**v1**: Fit Shannon model to 3 points (N=2,5,8), reported R^2=0.944, claimed "proven".

**v2 (this revision)**: Collected 4 new data points (N=3,4,6,7), validated out-of-sample,
compared against baseline models, added sensitivity analysis, removed dead code.

## Empirical Results

### Training Data (N=2,5,8 -- used for fitting)

| N | Empirical Gap | Predicted Gap | Error |
|---|--------------|---------------|-------|
| 2 | -0.20% | -0.59% | -0.39% |
| 5 | +1.60% | +2.43% | +0.83% |
| 8 | +5.71% | +5.27% | -0.44% |

**Train R^2 = 0.944**

### Validation Data (N=3,4,6,7 -- held out)

| N | Empirical Gap | Predicted Gap | Error |
|---|--------------|---------------|-------|
| 3 | +5.23% | +0.44% | -4.79% |
| 4 | +5.35% | +1.44% | -3.90% |
| 6 | +6.09% | +3.40% | -2.69% |
| 7 | +4.81% | +4.34% | -0.47% |

**Validation R^2 = -53.2** (catastrophic failure)
**Full-data R^2 = -0.35** (worse than predicting the mean)

### Key Finding: Gap is NOT a Smooth Function of N

The empirical data reveals that the composition gap does not increase
monotonically with N:

| N | Mean Gap | Std |
|---|----------|-----|
| 2 | -0.20% | -- |
| 3 | +5.23% | 0.94% |
| 4 | +5.35% | 1.49% |
| 5 | +1.60% | -- |
| 6 | +6.09% | 1.10% |
| 7 | +4.81% | 0.81% |
| 8 | +5.71% | -- |

The gap jumps from -0.2% at N=2 to +5.2% at N=3, but then is LOWER at N=5 (+1.6%)
than at N=3 or N=4. This non-monotonicity breaks the fundamental assumption of the
channel model (more experts = more interference = higher gap).

The most likely explanation: the domain splitting method creates different-quality
partitions at different N. The quintary split (N=5) produces domains that are more
balanced or more separable than the ternary split (N=3), leading to a lower gap
despite more experts. The composition gap depends on the QUALITY of domain splits,
not just the NUMBER of experts.

### Baseline Model Comparison

All models were fit on N=2,5,8 training data and validated on N=3,4,6,7.

| Model | Params | Train MSE | Val MSE | Val R^2 |
|-------|--------|-----------|---------|---------|
| Shannon | 3 | 0.345 | 11.42 | -53.2 |
| Linear | 2 | 0.296 | 11.68 | -54.4 |
| Power-law | 3 | 0.000 | 14.65 | -68.5 |

All three models fail equally badly on validation. The Shannon model is not
distinguishable from a linear model in predictive power. All achieve negative
R^2 because the validation data is fundamentally non-monotonic.

### Sensitivity Analysis

Perturbing training data by +/- 1%:
- N_max (10% gap) range: [13, 13]
- Prediction is robust to input perturbation

The N_max prediction is insensitive to measurement noise. The problem is not
noisy inputs -- it is that the model's structural assumptions (monotonic
degradation) are wrong.

### Kill Criteria

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Prediction within 2x | 1.52x | <= 2.0x | **PASS** |
| Held-out R^2 | -53.2 | >= 0.50 | **KILL** |
| Full-data R^2 | -0.35 | >= 0.50 | **KILL** |

**VERDICT: KILL. 1/2 kill criteria triggered.**
**Status: KILLED (was "consistent" in revision, now killed by validation data).**

## What Was Learned

1. **The MAC channel analogy is conceptually sound but predictively useless at micro
   scale.** The composition gap is dominated by domain-split quality effects, not by
   the information-theoretic interference predicted by the channel model.

2. **3 parameters on 3 points is not validation.** The original R^2=0.944 was
   completely misleading. With held-out validation, the model performs worse than
   predicting the mean.

3. **Domain split method matters more than N.** The non-monotonic gap pattern
   (N=5 lower than N=3,4) suggests that how you partition domains affects
   composition quality more than how many domains you compose.

4. **The rate-distortion interpretation was a tautology.** It labeled the fitted
   curve but added no predictive power. The "fundamental bound" was a description,
   not a limit.

5. **All simple models fail equally.** Linear, power-law, and Shannon models all
   fail on held-out data, confirming that gap(N) is not a simple function of N.

## What Would Improve This

1. **Control for domain split quality.** Run all N values with the same underlying
   partition (e.g., always 8 domains, then compose subsets of 2, 3, 4, ..., 8).
   This would isolate the N effect from the split effect.

2. **More seeds.** The N=4 std of 1.49% is large. More seeds would narrow
   confidence intervals and might reveal whether the non-monotonicity is real
   or noise.

3. **Measure actual inter-expert correlation (rho) at each N.** The model
   assumes constant rho, but rho likely varies with the split method. Measuring
   rho directly would test whether the interference model holds when rho is known.

## Artifacts

- `channel_capacity_bound.py` -- full analysis code (revised)
- `test_channel_capacity.py` -- unit tests + full experiment runner (revised)
- `MATH.md` -- mathematical derivations (revised)
- `results.json` -- all numerical results (revised)
- `REVIEW-adversarial.md` -- original adversarial review
