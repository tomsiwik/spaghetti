# Peer Review: Generation Quality with Per-Domain Optimal Scales

## Experiment Type
Guided exploration (correctly identified in MATH.md)

## Hack Detector
- Fix count: 1 (per-domain scale selection). Not a hack -- applies an empirically-determined correction from a prior finding. CLEAN.
- Is MATH.md a proof or a description? **Description, honestly labeled.** MATH.md explicitly states "No formal theorem is proven" and the Self-Test answers "there is none" for the impossibility property. This is appropriate for a guided exploration.
- Metric used as evidence: Domain-specific composite scores (factual recall, syntax validity, math correctness). These are behavioral proxies, not perplexity. Acceptable for a generation quality test.
- Kill criteria source: Derived from the original killed experiment's failure mode (3/5 worse). Reasonable threshold.

## Self-Test Audit
1. **One-sentence impossibility property:** "Honest answer: there is none." -- Correct and honest for a guided exploration. PASS.
2. **Cited theorems:** LIMA (2305.11206) and Finding #217. LIMA is real. Finding #217 is an internal empirical result, not a theorem. Correctly framed as empirical. PASS.
3. **Predicted numbers:** H1-H5 are specific and falsifiable. PASS.
4. **Falsification condition:** "If scale-aware routing is STILL worse on >= 3/5 domains, the problem is not scale but something deeper." Clear and well-targeted. PASS.
5. **Hyperparameter count:** Claims 0 new -- uses scales from Finding #217. Technically correct (the scales are determined externally). PASS.
6. **Hack check:** "No. This is a retest with corrected methodology." Honest. PASS.

Self-Test: all 6 items addressed. No blanks or evasions.

## Mathematical Soundness

No formal proof is claimed, so there is nothing to verify step-by-step. The mathematical framework is:
- Finding #217 identified per-domain optimal scales empirically
- This experiment applies those scales and measures behavioral outcomes
- The predictions (H1-H5) follow logically from Finding #217's results

The framework is internally consistent. The one subtlety: Finding #217 measured PPL ratios, while this experiment measures generation quality via behavioral metrics. The transfer from "optimal PPL scale" to "optimal generation quality scale" is an assumption, not a proven equivalence. MATH.md does not flag this gap. This is a minor omission, not a blocking issue.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table (lines 13-19). Verification:

| Prediction | PAPER.md Claim | Verified Against results.json | Correct? |
|------------|---------------|-------------------------------|----------|
| H1: >=4/5 improve over base | 5/5 | perscale_better_than_base: 5 | YES |
| H2: >=3/5 beat uniform | 2/5 (REFUTED) | perscale_better_than_uniform: 2 | YES |
| H3: Legal flips to >=0% | +1.7% | perscale_adv_pct: +1.68% | YES |
| H4: Finance flips to >=0% | +1.4% | perscale_adv_pct: +1.35% | YES |
| H5: Math >=+100% | +700% | perscale_adv_pct: +700% | YES |

All numbers in PAPER.md match results.json. H2 refutation is correctly reported.

## Critical Issues Found

### Issue 1 (MEDIUM): Different Adapters Than Original Killed Test

The original killed experiment (`generation_quality_test`) used NTP adapters from `real_data_domain_experts/adapters/`. This experiment uses SFT adapters from `bitnet_sft_generation_v3/sft_adapters/`. These are different adapter sets trained with different objectives.

PAPER.md frames this as a "retest" of the original experiment and directly compares numbers ("was -31.6%, now +1.7%"). The uniform condition in THIS experiment shows medical at +17.9%, but the original killed experiment showed medical at -6.9%. This confirms the conditions are not identical.

**Mitigating factor:** The internal comparison (base vs uniform vs per-scale within this experiment) is valid. The scale sweep (Finding #217) used the same SFT adapters, so the optimal scales are correctly matched. The claim "per-domain scale fixes the TWO-WORLD problem" is supported by this experiment's internal evidence.

**Required fix:** PAPER.md should note that the adapters differ from the original killed test, and that the uniform baseline numbers differ as a result. The comparison to the original test is suggestive but not a controlled retest.

### Issue 2 (MEDIUM): Legal and Finance Improvements Are Statistically Null

Legal: delta = 0.0016, SE = 0.023. Effect size = 0.07 SE. Not significant.
Finance: delta = 0.0024, SE = 0.063. Effect size = 0.04 SE. Not significant.

PAPER.md does acknowledge this (Finding 2, lines 86-89), calling them "within noise" and stating "the adapters are not significantly HELPING these domains -- they are just no longer HURTING." This is honest.

However, H3 and H4 ("flips from degradation to >=0%") are claimed as CONFIRMED when the measured values are statistically indistinguishable from zero. A more rigorous framing: the adapters at low scale are neutral (not harmful, not helpful). The "improvement" over base is measurement noise.

**Required fix:** The prediction table should note that H3 and H4 are confirmed only in the point-estimate sense. At n=10, we cannot distinguish "slight improvement" from "no change." The finding should be framed as "low-scale adapters do not degrade knowledge-dependent domains" rather than "low-scale adapters improve them."

### Issue 3 (LOW): Only 2/5 Domains Actually Tested the Hypothesis

Medical, code, and math use s=20 in both conditions. Their results are mathematically identical (confirmed by checking every raw score). The entire experiment's discriminative power rests on 2 domains x 10 prompts = 20 data points. This is acknowledged in Limitation #5 but deserves more prominence.

### Issue 4 (LOW): Same Prompts as Scale Sweep

The prompts come from the same `valid.jsonl` files used in the scale sweep that determined the optimal scales. Finding #217's optimal scales were tuned on data that overlaps with this test set. This is a form of data leakage -- the scales were optimized for these exact prompts. MATH.md assumption #1 mentions this but calls it "should transfer directly." It is not transfer; it is evaluation on training distribution.

**Required fix:** Add explicit limitation: "The per-domain optimal scales were determined using prompts from the same distribution. An independent test set would strengthen the finding."

## Novelty Assessment

This is not a novel finding per se -- it is a necessary validation step. The novel finding was #217 (domain-dependent scale). This experiment confirms that applying those scales in a generation setting produces the expected behavioral outcome. Its value is as a validation experiment, not as new science.

No prior art issues. This is a straightforward application of an internal finding.

## Macro-Scale Risks (advisory)

1. At larger scales (7B+), the base model may have sufficient domain knowledge that knowledge-dependent domains benefit from higher adapter scales. The scale profile may need recalibration per base model.
2. Oracle routing is an upper bound. Learned routing errors will compound with scale sensitivity -- routing to the wrong adapter at the wrong scale could be worse than base.
3. Per-domain scale assumes homogeneity within a domain. Hard prompts may need different scales than easy ones within the same domain.

## Verdict

**REVISE**

The experiment is well-designed and the internal evidence is sound. The code is clean, seeds are controlled, eval metrics are behavioral, and results match claims. However, three documentation fixes are needed:

1. **PAPER.md must note the adapter difference.** The SFT adapters used here differ from the NTP adapters in the original killed test. Add a sentence in "Key References" or "What This Experiment Is" noting: "This experiment uses SFT adapters (bitnet_sft_generation_v3), not the NTP adapters used in exp_generation_quality_test. The uniform baseline results differ as a consequence (e.g., medical is +17.9% here vs -6.9% in the original)."

2. **H3/H4 framing.** In the prediction table, add a column or footnote noting the effect is <0.1 SE and statistically indistinguishable from zero. Reframe Finding 2 header from "Improvement on Knowledge-Dependent Domains is Real but Small" to "Knowledge-Dependent Domains Are Preserved (Not Degraded) at Low Scale." The word "improvement" overclaims.

3. **Data leakage note.** Add to Limitations: "Per-domain optimal scales were determined on prompts from the same validation distribution used here. This is in-distribution confirmation, not out-of-distribution generalization."

These are documentation fixes, not re-run requirements. The underlying experiment is sound.
