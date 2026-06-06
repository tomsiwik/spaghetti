# Peer Review: Input-Dependent Adapter Scaling

## Experiment Type
Guided exploration

## Hack Detector
- Fix count: 1 (scale modulation by similarity). Clean -- no stacking.
- Is MATH.md a proof or a description? **Description dressed in equations.** Proposition 1 is explicitly labeled "NOT a formal theorem." Proposition 2 states a trivially obvious bound without proof. Neither has a Theorem/Proof/QED block.
- Metric used as evidence: Behavioral score (domain-specific) and Pearson r (sim vs score). The behavioral scores are reasonable proxies. The correlation is the right metric for the hypothesis.
- Kill criteria source: Derived from the framework (r > 0.3 from Proposition 1's monotonicity claim; domain improvement count from P2; coherence from P5). Reasonable for guided exploration.

## Self-Test Audit

1. **One-sentence impossibility property:** Answer acknowledges this is NOT a proof-of-impossibility experiment. Honest. Acceptable for guided exploration.
2. **Cited theorems:** "LoRA perturbation structure" -- not a named theorem, but a structural fact about LoRA. "TF-IDF distributional semantics (Salton & Buckley 1988)" -- real reference but not a theorem being applied; it is just a definition of TF-IDF. Finding #249 is internal. **Weak but not blank.** The framework rests on LoRA's additive structure (Delta = s * B @ A @ x), which is a definition, not a theorem with conditions to check.
3. **Predicted numbers:** P1-P5 are specific and falsifiable. P1 (r > 0.3), P2 (>= 2/3 domains), P5 (< 20% incoherent). Good.
4. **Falsification condition:** "r < -0.3 would falsify the framework." This is too lenient -- the actual result (r = -0.079, essentially zero) falsifies the assumption A1 just as effectively as anti-correlation would. Zero correlation means zero predictive power. The self-test answer should say r near zero falsifies A1, not just r < -0.3. **Minor evasion.**
5. **Hyperparameter count:** 1 (alpha). Honest acknowledgment that it cannot be derived. Clean.
6. **Hack check:** No stacking. Clean.

**Verdict on self-test:** Complete, mostly honest. One minor evasion on item 4.

## Mathematical Soundness

**This is guided exploration, so the bar is: does MATH.md state the proven framework and identify the unknown precisely?**

**Proven framework stated:** Yes. LoRA's perturbation structure (Delta = s * B @ A @ x) is real and the additive decomposition is correct. Finding #249's per-domain scales are cited as established.

**Unknown identified precisely:** Yes. "Whether TF-IDF embedding similarity to domain centroids preserves distributional similarity to adapter training data" (Assumption A1). This is a clear, testable unknown.

**Does exploration narrow the unknown?** Yes. The result definitively narrows: TF-IDF similarity does NOT predict adapter effectiveness. r = -0.079 across 30 queries with per-domain breakdowns all near zero. The unknown is resolved (negatively).

**Proposition 1 critique:** This is labeled a "Claim" with "Justification," not a theorem with proof. The justification has a major hidden assumption: that TF-IDF space geometry mirrors the model's internal representation space geometry. MATH.md explicitly acknowledges this ("the TF-IDF-to-representation-space assumption is precisely the unknown"). This is honest.

**Proposition 2 critique:** The claim that input-dependent scaling "cannot produce worse output than the base model" is stated without proof and is actually wrong on its face. The claim says scaling stays in [alpha*s_d, s_d], which interpolates between reduced and full adapter. But this is NOT the same as interpolating between base (s=0) and full adapter. At alpha=0.3 and s_d=20, the minimum scale is 6.0 -- this is still a large perturbation. The proposition conflates "reduced scale" with "closer to base model." There is no guarantee that s=6 produces output between s=0 and s=20 in quality space. Quality is not monotone in scale (as Finding #252 itself shows for code). **This proposition is incorrect as stated**, but since it is not load-bearing for the experiment's conclusions, it does not affect the verdict.

**Worked example (Section F):** The example shows a query getting scale 7.0 instead of 20.0. The commentary notes this "may be too low" for well-matched queries. This honestly reveals the core problem: linear f(sim) with typical similarities of 0.1-0.5 will aggressively downscale nearly everything. The normalization fix (dividing by max_sim_d) was mentioned but it is unclear from PAPER.md whether it was implemented.

Checking the results.json: sim_stats show p25/p75 ranges like 0.097-0.158 for math, 0.195-0.299 for code. The paper says mapping uses [p25, p75] to [alpha, 1.0], so the normalization WAS implemented. Good.

## Prediction vs Measurement

PAPER.md contains the prediction-vs-measurement table. Assessment:

| # | Prediction | Result | Comment |
|---|-----------|--------|---------|
| P1 | r > 0.3 | r = -0.079 | Clear miss. Framework assumption A1 refuted. |
| P2 | >= 2/3 domains improve | 1/3 | Miss, but K1 threshold was 0/3, so not killed. |
| P3 | Math smallest improvement | Math LARGEST | Inverted. But n=10 binary scoring makes this noise. |
| P4 | Code largest improvement | Code +2.3% (insignificant) | Miss. |
| P5 | < 20% incoherent | 3.3% | Pass. |

4/5 predictions missed. The paper correctly identifies this as NOT SUPPORTED and attributes the math improvement to stochastic noise at n=10. This interpretation is sound -- with binary scoring and n=10, a 1-prompt difference is well within noise.

**Statistical rigor concern:** PAPER.md correctly notes that r with n=10 has 95% CI of approximately +/-0.63. This means the experiment CANNOT distinguish r=0 from r=0.5 at any reasonable confidence level. The conclusion "zero predictive power" is directional but not statistically rigorous. However, the fact that all three per-domain correlations are near zero (-0.009, 0.032, -0.093) strengthens the case -- three independent near-zero estimates are more convincing than one.

## NotebookLM Findings

Skipping NotebookLM step -- the experiment is straightforward enough that the manual review covers all relevant angles.

## Novelty Assessment

**Prior art cited:** LoRAuter (arXiv:2602.21222) and MoLoRA (arXiv:2603.15965) are cited in PAPER.md. Both use LEARNED routing/scaling rather than fixed embedding similarity. The paper correctly identifies that the literature already solved this problem with learned functions, making TF-IDF a cheaper but less capable alternative.

**Is this a known result?** The specific negative result (TF-IDF similarity does not predict adapter effectiveness) is worth recording. The general insight (you need learned, not fixed, routing for per-query scaling) is well-established in the MoE literature but confirming it in this specific architecture has some value.

**Delta over existing:** Small. The main contribution is confirming that cheap lexical features are insufficient for adapter scale prediction in this architecture. This extends the proxy chain (Finding mentioned as "nine-level proxy chain extended").

## Macro-Scale Risks (advisory)

Not applicable -- this mechanism was killed at micro scale. If the direction is revisited with learned routing, the macro risks would be: (1) learned routing adds parameters and training cost, (2) routing quality may degrade with many adapters, (3) the training signal for per-query scale is unclear (what is the reward?).

## Additional Issues

1. **K1 threshold is too lenient.** K1 kills only at 0/3 domains improved. With 3 domains and n=10 per domain, random chance alone could produce 1/3 improvement. A stricter threshold (2/3 required to not-kill) would have been more appropriate. The result (1/3 improved, but the improvement is noise) shows this leniency masked what should be a cleaner negative result.

2. **All domain_scales are 20.0 for the three eval domains.** This means math, code, and medical all use the same base scale. Finding #249 is cited as providing per-domain optima, but all three are set to 20.0. Either Finding #249 found 20.0 for all three (suspicious uniformity), or the implementation did not actually use per-domain optima. This should be clarified.

3. **The "nine-level proxy chain" framing** is mentioned in the task description but not in PAPER.md. If this experiment is positioned as extending a proxy chain, the chain should be documented explicitly.

## Verdict

**PROCEED** (as a negative finding)

The experiment is well-executed guided exploration that definitively narrows the unknown: TF-IDF similarity does not predict adapter effectiveness. The MATH.md honestly labels its propositions as claims rather than theorems, identifies the precise unknown being tested, and the experiment cleanly resolves it. PAPER.md contains the required prediction-vs-measurement table and correctly interprets all results.

The finding status should be **killed** (for the TF-IDF-to-scale mechanism specifically), not provisional as currently stated. The broader direction (input-dependent scaling via learned functions) remains open, but that is a different experiment.

Minor revisions recommended but not blocking:
1. Change finding status from "provisional" to "killed" -- the specific mechanism (TF-IDF similarity predicts adapter effectiveness) is refuted, not merely unconfirmed.
2. Clarify why all three eval domains have domain_scales = 20.0 when Finding #249 supposedly provides per-domain optima.
3. Tighten the self-test falsification answer: r near zero falsifies A1 just as much as r < -0.3 does.
