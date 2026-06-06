# Peer Review (2nd Pass): Profiling Noise Quantification

## Prior Review Summary

The first review (REVIEW-adversarial.md) gave **PROCEED** with minor issues:
1. Narrative imprecision on why noise fraction is negative
2. Flickering threshold discrepancy (0.05 in code vs 0.005 theoretical)
3. Missing HYPOTHESES.yml entry

This second pass looks deeper at what that review was too lenient on.

---

## 1. Statistical Rigor: n=3 Is Not Enough for This Claim

The paper claims to "definitively address" the noise concern and calls the revival finding "VALIDATED" and "robust." These are strong words for n=3 training seeds.

**No confidence intervals are reported anywhere.** The paper reports means across 3 seeds but never standard deviations for the key noise metrics. Consider the disagreement rate: the paper reports "2.6-3.8%" but never shows per-seed values or variability. With n=3, the standard error of the mean is approximately s/sqrt(3), which for typical variability could be large.

More critically, the **noise fraction** of -6.2% is computed from aggregate means, not from per-seed noise fractions. The correct analysis would compute noise_fraction per seed, then report mean and std. If one seed shows +15% and two show -20%, the mean is -8.3% but the conclusion "noise does not inflate revival" would be wrong for one-third of cases.

Looking at the code (`test_profiling_noise.py` lines 326-363), the aggregation computes `statistics.mean(da_a_vals)` and `statistics.mean(da_c_vals)` separately per interval, then sums the means. This is averaging-then-comparing, not comparing-then-averaging. These give the same point estimate but the former obscures per-seed variance in the noise fraction itself.

**Severity: MEDIUM.** The claim is directional and likely correct, but "definitively addressed" requires either more seeds or explicit per-seed noise fraction reporting with confidence intervals. At n=3, one aberrant seed could flip the sign.

---

## 2. The Consensus Correction Introduces Systematic Bias

The first review noted this subtlety but was too lenient. The consensus dead set is a **strict subset** of the single-run dead set. This creates a systematic bias in the comparison that the paper does not adequately address.

**The core problem:** Consensus removes borderline capsules from the dead set. These borderline capsules are the ones most likely to "revive" (because they were never truly dead -- they have small but nonzero p_i). Removing them from the dead set:

1. Reduces the denominator (fewer dead capsules at S1)
2. Removes the capsules most likely to show D->A transitions (false positives)
3. But ALSO potentially changes the classification at S2 (a capsule might be consensus-dead at S1 but consensus-alive at S2 due to different borderline capsules at each checkpoint)

The paper claims the negative noise fraction means "consensus shows MORE revival." But this is an artifact of the denominator change. Consider: if consensus removes 20 borderline capsules from the dead set at S1, and 5 of those would have shown D->A transitions (the noise), but the smaller dead set at S1 now has fewer total members, the D->A **count** could go up because consensus at S2 also changes, reclassifying some capsules.

**The right comparison would be:** For the SAME set of capsules (e.g., the consensus dead set at S1), compare their fate under single-run vs consensus classification at S2. The paper does not do this. Instead, it compares two different base populations.

The first review identified this ("Consensus cohort tracking compares different base populations") but concluded it "STRENGTHENS the paper's conclusion." That is too generous. It makes the comparison apples-to-oranges, which means the -6.2% noise fraction is not a clean measurement of noise -- it is contaminated by the denominator shift.

**Severity: MEDIUM.** The 2.6-3.8% same-checkpoint disagreement rate is the clean measurement. The -6.2% noise fraction is not. The paper should lead with the disagreement rate as the primary evidence and treat the transition comparison as secondary/illustrative.

---

## 3. Is 640 Samples Actually Sufficient for the Binomial Claim?

The paper's MATH.md Section 3.1 correctly derives that capsules with p_i < 0.005 have appreciable false-dead probability. But the paper then jumps to "only capsules with extremely rare activations (p < 0.005) are susceptible" without asking: **how many capsules have p_i in that range?**

The empirical data shows 22-39% of capsules have 0 < f < 0.05 in at least one run. This is the "flickering" population. But the critical question is how many have 0 < p_i < 0.005 (the noise-susceptible band). The paper cannot measure this directly because p_i is the population parameter, not the sample frequency.

**However**, the same-checkpoint disagreement gives an indirect bound. If disagreement is 2.6-3.8% (13-20 capsules out of 512), and disagreement requires the capsule to be in the noise-susceptible band (p_i such that both (1-p_i)^640 and 1-(1-p_i)^640 are non-negligible), then roughly 13-20 capsules are near the boundary.

The binomial argument holds in the sense that the noise-susceptible population is small. But there is a subtlety the paper misses: **the profiling uses `rng.choices()` (sampling WITH replacement) from a validation set of approximately 3200 names** (a_m domain, 20% val split of ~16K). Over 20 batches of 32, that is 640 draws with replacement from ~3200 items. The expected number of unique items is approximately 3200 * (1 - (1 - 1/3200)^640) = 3200 * (1 - e^(-0.2)) = 3200 * 0.181 = 580 unique items.

So profiling run A sees ~580 unique validation examples and run B sees ~580 different unique examples. The overlap between runs is approximately 3200 * (1 - (1-1/3200)^640)^2 - ... actually, let me compute more carefully. The probability a specific item appears in run A is 1 - (1 - 1/3200)^640 = 0.181. The probability it appears in both runs is 0.181^2 = 0.033. So expected items in both runs: 3200 * 0.033 = 104 items.

**The two runs share only about 104 out of ~580 unique items each.** That is roughly 18% overlap. This means the runs are approximately independent for 82% of their data, which is reasonable. The paper's Assumption 1 ("640 samples represent <50% of available data") is correct but imprecise -- the relevant metric is not percentage of data used but the overlap between runs. The overlap is ~18%, which supports approximate independence.

**Severity: LOW.** The binomial argument holds. But the paper should note that with-replacement sampling from a small validation set introduces correlations that are not captured by the simple binomial model. The empirical disagreement measurement sidesteps this (correctly), but the theoretical Section 3.1 implicitly assumes independent draws from an infinite population.

---

## 4. Edge Case: Capsules Right at f approximately 0.01

The paper's Table in MATH.md shows that at p_i = 0.01, P(false dead) = 0.0016. This means about 1 in 625 profiling runs would falsely classify such a capsule as dead. With 512 capsules, the expected number with p_i near 0.01 is unknown, but if even 10 capsules have p_i in [0.005, 0.015], we expect about 10 * 0.0016 = 0.016 false-dead capsules per run. This is negligible.

**But the analysis misses a different edge case: capsules transitioning through the boundary.** A capsule whose true p_i changes from 0.003 at S=100 to 0.008 at S=200 is genuinely transitioning, but the profiling protocol will see this as noisy at S=100 (52.7% miss probability at p=0.003 * (1-0.003)^640... wait, at p=0.003: (0.997)^640 = 0.147, so 14.7% chance of being missed) and fairly reliable at S=200 (p=0.008: (0.992)^640 = 0.006, so 0.6% miss). This capsule is genuinely reviving, but the profiling protocol would correctly classify it as dead at S=100 with 85% probability and alive at S=200 with 99.4% probability. So most of the time, this D->A transition would be correctly detected as genuine.

The real edge case is capsules with p_i approximately 0.002-0.003 at BOTH checkpoints. These would show noise-driven transitions (flickering). The same-checkpoint disagreement of 13-20 capsules bounds this population. Of these ~15 borderline capsules, how many also appear as D->A between checkpoints? This is at most 15 (and likely far fewer, since a borderline capsule at S1 would need to be non-borderline at S2 to produce a transition that is NOT caught by the dual profiling).

**Severity: LOW.** The edge cases are bounded by the disagreement measurement. The paper's conclusion holds.

---

## 5. The Cohort Analysis Has a Subtle Flaw

The cohort analysis (PAPER.md "Cohort Analysis: Single-Run vs Consensus") tracks capsules dead at S=100 through later checkpoints. The single-run cohort uses `masks_a[100]` as the reference, while the consensus cohort uses `consensus_masks[100]`.

The problem: at later checkpoints, the single-run cohort checks `masks_a[S]` (run A at that checkpoint), while the consensus cohort checks `consensus_masks[S]` (consensus at that checkpoint). This means:

- **Single-run cohort:** A capsule dead in run A at S=100, checked against run A at S=3200
- **Consensus cohort:** A capsule dead in BOTH runs at S=100, checked against BOTH runs at S=3200

A capsule in the consensus cohort is classified as "revived" at S=3200 if it is NOT dead in both runs at S=3200. This means it only needs to be alive in ONE of the two runs to be classified as revived. This is a LOWER bar for revival than the single-run case (where it must be alive in run A specifically).

This asymmetry biases the consensus revival rate upward relative to the single-run rate. The paper does not note this.

Looking at the code (`test_profiling_noise.py` lines 206-226):

```python
dead_at_100_consensus = {i for i, d in enumerate(consensus_masks[100]) if d}
# ...
dead_s_c = {i for i, d in enumerate(consensus_masks[S]) if d}
# ...
cohort_consensus[S] = {
    "pct_revived": len(dead_at_100_consensus - dead_s_c) / len(dead_at_100_consensus) * 100,
```

A capsule leaves `dead_s_c` (consensus dead at S) if it is alive in EITHER run A or run B at S. So being alive in just one run counts as "revived" under consensus tracking. Under single-run tracking, being alive in run A specifically is required.

For a borderline capsule at S=3200 (p_i approximately 0.003), the probability of being alive in at least one of two runs is 1 - (1-p)^(2*640) = 1 - (0.997)^1280 = 1 - 0.022 = 97.8%, compared to being alive in run A specifically: 1 - (0.997)^640 = 85.3%. So consensus is substantially more likely to classify borderline capsules as "revived."

**This explains part of why the consensus revival rate (28.2%) exceeds the single-run rate (26.7%).** It is not because "noise underestimates revival" -- it is because the consensus definition of "alive at S2" is more permissive (alive in either run, not a specific run).

**Severity: MEDIUM.** This does not invalidate the core finding (profiling is reliable, disagreement is low) but it means the narrative that "consensus shows MORE revival, proving noise underestimates revival" is misleading. The consensus comparison is not apples-to-apples for two reasons (different base population AND different revival criterion), and both biases push consensus revival upward.

---

## 6. What the First Reviewer Missed

### 6.1 No Permutation Test or Bootstrap

With n=3 seeds and discrete counts (13-20 disagreements out of 512), a permutation test or bootstrap would establish whether the disagreement rate is significantly below 20%. At 3.8% vs 20% threshold, it clearly passes, but the margin matters for the strength of the conclusion. The paper reports no p-values or confidence intervals for any metric.

### 6.2 The "total_positions" Denominator

In `profile_activations`, `total_positions = B * T` summed over batches. Each "position" is a (batch_item, token_position) pair. With batch_size=32, block_size (T) presumably around 16-32 for character-level names, total_positions = 20 * 32 * T. If T=16, that is 10,240 "positions" per profiling run, not 640.

The MATH.md says N_prof = 640 (samples), but the actual number of positions used for frequency estimation is much larger (10K+). Each of the 640 "samples" (names) contributes T token positions. This means the effective sample size for the binomial model is ~10K, not 640. The false-dead probabilities in Section 3.1 are therefore **overstated** -- the real false-dead probability at p_i=0.001 with N_prof=10240 is (0.999)^10240 = 0.000036, not 0.527.

**This is a significant error in the math.** The binomial model uses N_prof=640 (number of names), but the activation frequency is computed over (name, token) positions, which is ~16x larger. The profiling is ~16x more reliable than the paper claims.

Does this change the conclusion? No -- it makes the conclusion stronger (less noise). But the MATH.md analysis is wrong about the effective sample size, and the "critical insight" about capsules with p_i < 0.005 being susceptible is wrong. With N_eff = 10K, only capsules with p_i < 0.0003 would have appreciable false-dead probability.

This also means the same-checkpoint disagreement of 2.6-3.8% is HIGHER than the binomial model would predict with N_eff=10K. This suggests the disagreement is not purely binomial sampling noise but may reflect other sources of variability (different names activate different patterns, not just random sampling).

**Severity: HIGH for mathematical accuracy, LOW for conclusions.** The math is wrong about the sample size, but this makes the profiling MORE reliable than claimed, so the conclusion still holds. However, the discrepancy between predicted (near-zero) and observed (2.6-3.8%) disagreement at N_eff=10K suggests the simple binomial model is insufficient.

### 6.3 The Transition Counting Bug Risk

In `test_profiling_noise.py` lines 183-201, transitions are computed between consecutive checkpoints using DIFFERENT profiling seeds at each checkpoint. Specifically:

- Checkpoint S1: masks_a[S1] uses profiling seed 1000
- Checkpoint S2: masks_a[S2] uses profiling seed 1000

Both use seed 1000, so the profiling RNG is reseeded to 1000 at each checkpoint. This means runs A at S1 and S2 use the SAME random batches (same seed). This is actually correct for the single-run comparison (consistent profiling methodology), and the first reviewer verified this. No bug here.

---

## 7. Kill Criteria Assessment

The three kill criteria are reasonable and pre-registered. My concerns are about the strength of evidence, not the thresholds:

| Criterion | Value | Threshold | Margin | Confidence |
|-----------|-------|-----------|--------|------------|
| Same-checkpoint disagreement | 3.8% max | >20% | 5.3x | HIGH (direct measurement, clean) |
| Noise-attributable D->A | -6.2% | >50% | N/A (wrong sign) | LOW (apples-to-oranges comparison, see Section 5) |
| Noise-corrected revival rate | 17.4% | <5% | 3.5x | MEDIUM (consensus definition is more permissive) |

The first criterion is the strongest and cleanest evidence. The second is methodologically flawed (as discussed). The third is inflated by the consensus revival definition asymmetry.

---

## 8. Hypothesis Graph Consistency

The experiment is listed as item 12 in VISION.md but not in HYPOTHESES.yml. The first reviewer noted this. The kill criteria in the code match MATH.md Section 7. The experiment correctly targets Exp 18's adversarial review Finding 2 (borderline capsule flickering).

---

## 9. Macro-Scale Risks (advisory)

All points from the first review stand. Additional risk:

**The N_eff discrepancy becomes worse at macro scale.** With longer sequences (T=512+ instead of T=16), the effective sample count becomes enormous (640 * 512 = 327K positions per run). At that scale, profiling noise would be negligible for any capsule with p_i > 0.00001. The 2.6-3.8% disagreement observed at micro scale would drop to near zero, making this entire experiment irrelevant for macro. Conversely, if disagreement persists at macro scale, it would indicate a non-binomial noise source that the current framework cannot explain.

---

## Verdict

**PROCEED** -- but with required revisions to the paper's claims and math.

The core finding (profiling disagreement is low, revival is not dominated by noise) is sound. The same-checkpoint disagreement of 2.6-3.8% is the cleanest evidence and strongly supports the conclusion. But the paper has mathematical and methodological issues that overstate certainty and misattribute effects.

### Required Revisions (not blocking for micro, but must be addressed before citing this result)

1. **Fix the effective sample size in MATH.md.** N_prof is not 640 (names); the activation frequency is computed over ~10K+ (name, token) positions. Recompute the false-dead probabilities with the correct N_eff. Acknowledge that the observed 2.6-3.8% disagreement exceeds what the corrected binomial model predicts, suggesting non-binomial sources of variability.

2. **Report per-seed noise fractions with standard deviations.** The -6.2% aggregate tells us nothing about seed-level variability. If even one seed shows positive noise fraction, the "definitively addressed" language is too strong.

3. **Acknowledge the consensus comparison asymmetry.** The consensus cohort uses a more permissive revival criterion (alive in either run at S2) vs the single-run cohort (alive in specific run A at S2). This biases consensus revival upward independently of noise correction. The paper should either (a) use the same base population for both comparisons, or (b) clearly state this caveat.

4. **Downgrade language from "definitively" to "directionally."** At n=3 without confidence intervals, no claim is definitive. The evidence is strong and directional. The same-checkpoint disagreement of <4% is the robust finding.

5. **Lead with the clean metric.** The same-checkpoint disagreement (2.6-3.8%) is the methodologically clean measurement. The noise fraction (-6.2%) and consensus cohort comparison are secondary and flawed. Restructure the paper to lead with disagreement as the primary evidence.
