# Peer Review: capability_benchmark_full_system

## Experiment Type
Claimed: verification (Type 1). Actual: see below.

## Hack Detector
- Fix count: 0 (verification of existing system, no new mechanisms). CLEAN.
- Is MATH.md a proof or a description? **Description dressed in equations.** "Proposition 1" is a linear model with undefined constants (c, delta, f) that are never derived, bounded, or estimated. There is no Theorem/Proof/QED block. The "~" relation is unquantified -- it is a cartoon, not a theorem.
- Metric used as evidence: accuracy (GSM8K), syntax parse rate, F1, MMLU accuracy. These are reasonable behavioral metrics for the claims being made.
- Kill criteria source: Partially derived from prior findings (the +10pp GSM8K prediction from Finding #237). Reasonable.

## Self-Test Audit

1. **One-sentence impossibility property:** "Format-dependency partitioning: by routing to per-domain optimal scales, format-tasks get amplified (s=20) while knowledge-tasks get minimal perturbation (s<=4)." This is a mechanism description, not an impossibility property. An impossibility property would be: "Under X conditions, degradation is mathematically bounded by Y." FLAG -- evasion.

2. **Cited theorems:** LIMA hypothesis (Zhou et al., 2305.11206) -- this is an empirical observation/hypothesis, not a theorem. It has no formal proof and no quantitative bounds. Finding #249, #258, #237 are internal findings, not theorems. No actual mathematical theorem is cited. FLAG.

3. **Predicted numbers:** GSM8K +10pp, Code +15pp, NER +5pp, MMLU -5pp to 0pp, incoherence <5%. These are specific and falsifiable. PASS.

4. **Falsification condition:** "If GSM8K degrades with math adapter at s=20, the format-capability equivalence is wrong." Reasonable and directly targets the proposition. PASS.

5. **Hyperparameter count:** 5 per-domain scales, acknowledged as derived from prior work. PASS.

6. **Hack check:** Clean -- this is a verification, not a new mechanism. PASS.

**Self-Test verdict:** 2 flags (items 1 and 2). The "impossibility property" is a mechanism, and the "theorems" are empirical hypotheses.

## Mathematical Soundness

**There is no proof.** MATH.md contains "Proposition 1" which states:

```
quality(M + s*A, T) - quality(M, T) ~ c*f - delta*(1-f)
```

This is a linear model with three free parameters (c, delta, f) that are never derived from first principles, never bounded, and never estimated before the experiment. The format_dependency f is "defined" in [0,1] but never measured or computed -- it is assigned by intuition (GSM8K: f~0.9, MMLU: f~0.2). With enough free parameters and post-hoc assignment of f values, any linear model can be made to "predict" any direction.

Furthermore:
- The "~" relation has no error bounds, no asymptotic regime, no conditions under which it holds.
- c and delta are not estimated from prior data. The predictions (+10pp, +15pp, etc.) come from prior findings, not from the proposition.
- The proposition is not derived -- it is asserted. There is no proof of why quality should decompose linearly into format and knowledge components.

**This is a description dressed in equations, not a proof.** The predictions are actually just extrapolations from prior empirical findings (#237, #249), which is honest and useful, but it is not a Type 1 verification of a theorem.

**Correct classification:** This is a **Type 2 guided exploration** at best -- testing whether prior in-distribution findings transfer to OOD benchmarks, within the empirical framework of LIMA + prior findings. At that level, it is well-designed.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Credit for this.

| Prediction | Measured | Match |
|---|---|---|
| GSM8K >= +10pp | -15.0pp | NO |
| Code >= +10pp | -10.0pp | NO |
| NER >= +5pp | -7.4pp | NO |
| MMLU -5pp to 0pp | -5.0pp | YES (boundary) |
| Incoherence <5% | 0.0% | YES |

3/5 predictions falsified. The experiment was correctly killed. The analysis in PAPER.md is thorough and identifies the right root cause: in-distribution vs out-of-distribution mismatch.

**Statistical concern:** Sample sizes are very small (20 GSM8K, 10 code, 20 NER). The GSM8K result is 6/20 vs 3/20. A Fisher exact test gives p=0.45 -- this difference is **not statistically significant**. The code result is 9/10 vs 8/10 (p=1.0 by Fisher). The MMLU result is 44/100 vs 39/100 (p=0.56). None of these individual deltas are statistically significant at any reasonable threshold. The experiment was killed on noise.

This is a serious methodological problem. The kill decision may be correct directionally (all deltas are negative), but the evidence does not support confident falsification of the proposition. With these sample sizes, the experiment cannot distinguish "composition hurts" from "no effect + random variation."

## NotebookLM Findings

Skipped -- the mathematical issues are clear enough without external review.

## Novelty Assessment

This is a system verification experiment, not a novelty claim. The key finding (SFT adapters degrade OOD benchmarks) is consistent with known LoRA literature: LoRA adapters overfit to training distribution and can degrade OOD performance, especially at high scaling factors. This is well-documented in the LoRA merging literature (e.g., Yadav et al., "TIES-Merging," NeurIPS 2023).

The insight that NTP vs SFT adapter training methods produce different OOD transfer characteristics is genuinely useful for the project.

## Macro-Scale Risks (advisory)

- At macro scale, OOD degradation from LoRA composition is a well-known problem. The solution space includes: (a) lower scales, (b) task-aware routing that detects OOD queries, (c) adapter merging techniques (TIES, DARE), (d) not applying adapters when the query is OOD.
- The 5 per-domain scale hyperparameters become harder to tune at scale with more domains.

## Verdict

**KILL** (confirmed)

The experiment was correctly killed. The analysis in PAPER.md is honest and insightful. However, the review identifies issues that should be addressed in the finding record:

1. **No proof exists.** MATH.md claims Type 1 (verification) but contains no Theorem/Proof/QED. "Proposition 1" is a linear model with free parameters, not a derivation. Finding #260 should not be recorded as "proof refuted" -- there was no proof to refute. It should be recorded as "empirical hypothesis falsified."

2. **Statistical power is insufficient.** GSM8K 6/20 vs 3/20 is not significant (p=0.45 Fisher exact). Code 9/10 vs 8/10 is not significant. The kill was based on point estimates from underpowered samples. The directional consistency across all benchmarks (all negative) provides weak collective evidence, but no individual result is significant.

3. **Confound: NTP vs SFT adapters.** PAPER.md correctly identifies that Finding #237 (+10pp GSM8K) used NTP adapters while this experiment used SFT adapters. This means the experiment is not a fair test of the "format-capability equivalence" -- it is testing a different adapter type than the one that produced the predictions. The proposition was not tested on the system that generated its predictions.

4. **The real learning is valuable.** The in-distribution vs out-of-distribution distinction, and the NTP vs SFT adapter difference, are genuinely important findings for the project. These should be preserved clearly in the finding.
