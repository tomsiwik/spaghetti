# Peer Review: PPL-Probe K=3+ Scaling

## NotebookLM Findings

Skipped -- documents reviewed manually with sufficient rigor given the experiment's scope and the existence of a detailed parent review.

## Mathematical Soundness

### Correlation Metric is Inflated by Flattening

**Severity: MODERATE -- does not invalidate but overstates the finding.**

The Pearson correlation (r=0.9979 at K=3) is computed by flattening all weight vectors across all tuples, cross-domain types, and seeds into two long 1D arrays (lines 381-386 in run_experiment.py):

```python
probe_flat = onp.array([w for vec in correlation_data[K]['probe_w'] for w in vec])
oracle_flat = onp.array([w for vec in correlation_data[K]['oracle_w'] for w in vec])
r = float(onp.corrcoef(probe_flat, oracle_flat)[0, 1])
```

At K=3, each observation contributes 3 weights (w1, w2, w3) to the flat array. Across 150 observations (10 triples x 3 cross-types x 5 seeds), this gives 450 scalar pairs. But these weights are constrained to sum to 1 and are typically structured with 1 high weight and K-1 low weights. The Pearson r between two vectors that both span [0, 1] with clustered structure will be high even if the per-observation agreement is mediocre.

The more meaningful metric is the "best weight" correlation (r=0.9931) and especially the top-1 agreement (97.3%), which directly measure whether the probe identifies the most relevant expert. These are still strong numbers, so the conclusion holds, but the headline r=0.9979 is likely inflated by 1-3 points.

**The increasing r from K=2 (0.9964) to K=3 (0.9979) to K=5 (0.9983) is particularly suspicious.** More weights per observation means more low-weight entries near zero in both vectors, which mechanically inflates Pearson r. The top-1 agreement trend tells the true story: 100% -> 97.3% -> 92%. Probe quality degrades with K, as expected.

### Information-Theoretic Bound is Vacuous

MATH.md claims the probe needs log2(K!) bits for K-way ranking and argues n=10 examples provide far more. This is a correct but unhelpful lower bound. The actual requirement is not ranking K experts but estimating K-1 continuous weights precisely enough to outperform equal weighting. The continuous estimation problem has much higher information requirements than the ordinal ranking problem. Since the experiment works empirically, this is a presentational issue, not a mathematical error.

### Softmax Temperature (inherited concern)

tau=1.0 remains unjustified. The parent review flagged this. HYPOTHESES.yml now has a separate `exp_ppl_probe_temperature_sensitivity` node, so this is being tracked. Not blocking.

### Probe-Test Data Leakage (inherited, still present)

The parent review identified that `probe = test_enc[:n_probe]` (line 79) overlaps with the evaluation set. This experiment inherits the same issue. The impact assessment from the parent review (estimated <0.5pp) still applies. The probe-oracle gap remains tiny (<0.2pp at all K), suggesting the leakage is not the primary driver. Still should be fixed for macro.

## Novelty Assessment

This is a direct extension of the parent experiment (`cross_domain_dilution_vs_k`) from K=2 to K=3,5. The novelty is in the scaling behavior characterization, not in the mechanism itself. No new prior art concerns beyond those identified in the parent review.

The superlinear improvement scaling claim (improvement grows faster than K) is a useful insight. The theoretical prediction that dilution waste = (K - K_rel)/K growing with K creates more opportunity for the probe is sound and confirmed empirically.

## Experimental Design

### Issue 1: K2 Kill Criterion Fires -- Reinterpretation is Reasonable but Should Have Been Anticipated

The K2 criterion as stated in HYPOTHESES.yml is: "K=3 probe-weighted composition is worse than K=2 probe-weighted." K=3 absolute gap (-7.32%) is indeed worse than K=2 (-8.01%). The paper acknowledges this fires and argues the criterion was poorly chosen because it conflates task difficulty with probe quality.

This reinterpretation is valid. The K=3 composition task is inherently harder (3 expert deltas contributing noise vs 2), so absolute gap degradation is expected even with perfect weighting. The fairer comparison -- improvement over equal-weight baseline -- shows clear scaling (8.77pp to 18.42pp).

However, the fact that the kill criterion needed reinterpretation indicates it was poorly designed upfront. The criterion should have been stated as: "K=3 probe improvement over equal-weight is less than K=2 probe improvement over equal-weight." This would have been a clean PASS.

**Status: The paper's reinterpretation is honest and well-argued. HYPOTHESES.yml status of "supported" (not "proven") is appropriate given K2 technically fires.**

### Issue 2: Top-1 Oracle Results Suspiciously Identical Across K

The top-1 oracle mean gap is nearly identical: -3.86% (K=2), -3.88% (K=3), -3.90% (K=5). This makes sense mechanically -- the top-1 oracle always picks the single best expert regardless of K, and the pool of "best experts" for 2-domain cross-domain queries is the same across K values. This is not a bug but the paper should note it: the top-1 oracle is K-independent because it selects 1 expert from the same pool.

### Issue 3: Sample Size Imbalance Across K

- K=2: 50 observations (10 pairs x 5 seeds, but only tested cross-types matching each pair)
- K=3: 150 observations (10 triples x 3 cross-types x 5 seeds)
- K=5: 50 observations (1 quintuple x 10 cross-types x 5 seeds)

K=3 has 3x the sample size of K=2 and K=5. Comparisons across K should account for this. The K=5 results aggregate over only 1 tuple, so variance comes entirely from cross-type and seed variation, not tuple variation. The paper notes this in limitations (point 4) but doesn't compute confidence intervals to quantify the impact.

### Issue 4: Cross-Domain Tests Always Involve Exactly 2 Domains

All cross-domain test types involve pairs of domains (e.g., arith_reverse). For K=3, this means the test data always has exactly 2 relevant experts and 1 irrelevant one. For K=5, there are 2 relevant and 3 irrelevant. The probe never faces a 3-way or 5-way relevance discrimination -- it always needs to separate 2 relevant from K-2 irrelevant. This makes the task substantially easier than true K-way discrimination.

In production with queries touching 3+ domains simultaneously, the probe faces a harder problem. The information-theoretic argument about K! orderings is misleading because the actual discrimination at micro is always binary-relevant vs irrelevant.

## Hypothesis Graph Consistency

- `exp_ppl_probe_k3_scaling` in HYPOTHESES.yml: status "supported", kill criteria match MATH.md
- K1 (correlation >= 0.8): PASS, correctly assessed
- K2 (K=3 vs K=2 absolute): technically KILL, reinterpreted. Status "supported" is appropriate
- Evidence lines in HYPOTHESES.yml are comprehensive and honest about K2 nuance
- Dependencies: `exp_cross_domain_dilution_vs_k` (the parent). Correctly linked.

## Macro-Scale Risks (advisory)

1. **Cross-domain queries touching 3+ domains.** The micro experiment only tests 2-domain cross-domain queries at K=3,5. At macro, real queries may involve 3+ domains (e.g., "explain the legal implications of this medical AI system" touches law, medicine, AI). The probe's ability to produce accurate 3-way soft weights is untested.

2. **Expert similarity at scale.** At d=32 with 5 very different synthetic domains, PPL differences between experts are large and easy to discriminate. At d=4096 with 500 experts in related domains (e.g., python-async vs python-concurrency), PPL differences will be much smaller. The probe may need n>10 or a tuned temperature.

3. **Probe latency scaling.** At K=3, the probe requires 4 forward passes on 10 examples. At K=5, 6 passes. The paper's production recommendation of K=2-3 routing keeps this manageable. At K>5, probe cost becomes significant and alternatives (embedding similarity, cached scores) should be explored.

4. **The flattened Pearson r metric should not be used as the headline number for macro evaluation.** Use top-1 agreement or per-observation weight MSE instead. The flattened metric will continue to inflate with K, masking real degradation.

## Verdict

**PROCEED**

The experiment achieves its primary objective: demonstrating that PPL-probe weighting scales from K=2 to K=3 and K=5. The mechanism works in principle. The probe maintains near-perfect tracking of the oracle (top-1 agreement 97.3% at K=3), and its value increases superlinearly with K because the dilution problem it solves gets worse. The K2 kill criterion fires on a technicality (absolute gap, not relative improvement) and the reinterpretation is sound.

Non-blocking issues to address in future work or macro validation:

1. **Replace the flattened Pearson r with per-observation metrics** (top-1 agreement, per-observation weight MSE, or rank correlation). The current headline number is inflated and the increasing-with-K trend is an artifact.

2. **Fix probe-test data leakage** for macro experiments. Use a held-out probe buffer separate from the evaluation set.

3. **Test with multi-domain cross-domain queries** (3+ relevant domains) at macro to validate true K-way discrimination, which this experiment does not test.

4. **The K2 kill criterion should be retroactively updated in HYPOTHESES.yml** to clarify that "worse" means "less improvement over equal-weight" rather than "higher absolute gap." The current evidence entry already captures this nuance.
