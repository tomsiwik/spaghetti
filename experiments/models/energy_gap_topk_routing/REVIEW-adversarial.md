# Peer Review: Energy Gap Top-k Routing

## Experiment Type
Guided exploration (Type 2). MATH.md correctly identifies the proven framework (Neyman-Pearson optimality of energy gap ranking, Finding #182) and the unknown (whether ranking translates to generation quality improvement).

## Hack Detector
- Fix count: 1 (argmin over energy gaps). Clean, no stacking.
- Is MATH.md a proof or a description? **Description dressed in equations, with one attempt at a theorem.** Section C is a mechanism description. "Theorem 1" in Section D is closer to a proof but has issues (see below).
- Metric used as evidence: Keyword-density composite score (prose), math answer correctness (math). The math correctness metric is meaningful. The prose domain scores (0.44 vs 0.45) are noise on a crude proxy.
- Kill criteria source: K575 (80% accuracy) is loosely derived from the AUC^4 bound (~79%). K576 (top-1 > uniform) follows from p >> 1/N. K577 (10% overhead) is stated in MATH.md's worked example to be ~18%, meaning the proof itself predicted failure. Partially derived from proof, partially arbitrary.

## Self-Test Audit

1. **One-sentence impossibility property:** "Selecting the adapter with maximum NLL reduction guarantees p >> 1/N when AUC >> 0.5." This is a valid single property. PASS.

2. **Cited theorems:** Neyman-Pearson lemma and data processing inequality. Neyman-Pearson is real and correctly applied in spirit (energy gap as likelihood ratio). However, the data processing inequality citation is incorrect: the DPI states that processing cannot increase mutual information. Saying "argmin over gaps preserves ranking information" is not what the DPI says. The DPI would say you LOSE information by going from the full gap vector to a single index. The correct justification is simply that argmin is a deterministic function of the sufficient statistics and preserves the MAP decision. FLAG: misapplied theorem.

3. **Predicted numbers:** Top-1 accuracy >= 79% on math, overall >= 70%, overhead < 10%. The first two are specific and falsifiable. The overhead prediction contradicts MATH.md's own worked example (Section G) which calculates 18%. MATH.md simultaneously predicts <10% (Section E) and ~18% (Section G). This is internally inconsistent.

4. **Falsification condition:** "If top-1 accuracy significantly below AUC^(N-1)" or "if generation quality doesn't improve despite correct selection." These target the proof's assumptions. PASS.

5. **Hyperparameter count:** 0. Correct for the routing mechanism itself. However, LORA_SCALE=20.0 and the quality scoring weights (0.45/0.25/0.10/0.20 for prose, 0.5/0.5 for math/code) are implicit hyperparameters of the evaluation, not the mechanism. Acceptable for a guided exploration.

6. **Hack check:** Clean. Single mechanism replacing uniform weights. PASS.

## Mathematical Soundness

**Section C (AUC^(N-1) bound):**
The claim that P(correct top-1) >= a^(N-1) for pairwise AUC = a requires independence of pairwise comparisons. This is acknowledged ("lower bound because pairwise AUCs are not independent") but the direction of the bias is asserted without proof. The claim that dependence makes the bound conservative (actual accuracy higher) is plausible but not proven. In the legal/finance case, the correlated errors actually hurt: the same prompts that confuse legal also confuse finance, leading to systematic rather than random errors. The bound is useful as a rough estimate but should not be called a theorem.

**Section D ("Theorem 1"):**
This is labeled a theorem but is actually an inequality derivation with an informal argument. The "proof" establishes that top-1 beats uniform when p > 1/N AND "the correct adapter's quality dominates the average." The second condition is assumed, not proven. The QED is premature -- it is a valid argument sketch but not a formal proof. For a Type 2 guided exploration, this level of rigor is acceptable but should not be labeled "Theorem/Proof/QED."

**Overhead prediction inconsistency:**
MATH.md Section E predicts "<10% overhead" while Section G's worked example calculates 18%. The 10% figure is aspirational (with prefix optimization), while the 18% is the actual prediction without optimization. The experiment measured 29.5% (inflated by disk loading). The honest prediction from the math is 18%, not 10%.

**Code domain prediction failure:**
MATH.md predicted ~50% accuracy on code (AUC=0.5 from Finding #182), but measured 100%. PAPER.md explains this as "code AUC improved with BitNet data." This means the AUC values from Finding #182 are not the ones operative in this experiment. The proof's predictions are based on stale data. This does not invalidate the mechanism, but it means the quantitative predictions cannot be verified because the input parameters changed.

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table. Assessment:

| Prediction | Measured | Honest? |
|-----------|----------|---------|
| Top-1 accuracy >= 79% on math | 100% | YES, passes bound. But 100% on 10 samples is not very informative (95% CI: 72-100%). |
| Top-1 accuracy ~50% on code | 100% | NO, prediction wrong. Explained post-hoc. |
| Overall accuracy >= 70% | 88% | YES, passes bound. |
| Math top-1 > uniform | +100.6% | YES, strong. |
| Overhead < 10% | 29.5% | FAIL, honestly reported. But MATH.md's own calculation predicted 18%, so the 10% threshold was already wrong. |

The table is mostly honest. The code domain surprise is acknowledged. The overhead failure is correctly attributed to implementation, though the fundamental overhead (~18% from Section G) would also exceed the 10% kill criterion.

**Missing from table:** Variance estimates. N=10 per domain with single seed means routing accuracy of 70% on legal/finance has a 95% CI of roughly 35-93% (binomial). The 88% overall has 95% CI of roughly 76-95%. These are wide enough that the 80% kill criterion could plausibly fail with different prompts.

## Novelty Assessment

The mechanism (argmin NLL reduction for adapter selection) is not novel. It is a standard approach in mixture-of-experts literature (router-free MoE via loss-based selection). MoLoRA (arxiv 2603.15965) is cited. The contribution here is applying it to LoRA adapters with energy gap as the routing signal, validated on a specific BitNet setup.

For the project's purposes, the novelty question is less important than whether it works as a building block. It does, with caveats.

## Key Concerns

1. **Legal/finance confusion is structural, not statistical.** The energy gap difference is 0.041 nats (legal vs finance adapter on legal queries). This is within noise for any practical deployment. With N > 5 domains, adding more similar domains (e.g., compliance, tax, insurance) would create a cluster of confused domains. The 88% accuracy is inflated by 3 perfectly separable domains and 2 confused ones. This is Finding #186 and is acknowledged, but the implications for scaling are underweighted.

2. **Quality metric for prose is not validated.** Keyword density is acknowledged as crude. But the PAPER.md claims like "-3.0% legal" and "-7.0% finance" for top-1 vs uniform are based on this crude metric. These numbers should not be cited as evidence of quality degradation. They are noise. The only solid quality result is math answer correctness (70% vs 30%), which is genuine and compelling.

3. **The "results.json verdict says KILLED" anomaly.** The JSON file contains `"verdict": "KILLED"` at the analysis level, but the paper reports SUPPORTED. This looks like the verdict was set before the analysis was complete, or the experiment runner has a bug in verdict logic. Minor, but sloppy.

4. **Forward pass cost is O(N) and not amortizable.** Each query requires N+1 forward passes on the prompt. For N=25 adapters (the proven capacity), this is 26 forward passes per query just for routing. Even with cached models, this is ~26x prompt processing cost. The math in Section G calculates this honestly for N=5 but the scaling to N=25 is not addressed.

## Macro-Scale Risks (advisory)

1. Energy gap routing cost scales linearly with N. At N=25, prompt overhead becomes dominant. Need prefix-only evaluation or learned routing to scale.
2. Legal/finance confusion pattern will worsen with semantically adjacent domains.
3. Single-adapter routing is brittle for cross-domain queries (acknowledged in Limitation #4).
4. The approach reloads models from disk per adapter for energy computation. Production would need all adapters in memory simultaneously, which conflicts with memory budgets at large N.

## Verdict

**PROCEED**

Justification:

The experiment is correctly typed as guided exploration and meets the Type 2 requirements: it states the proven framework (energy gap ranking optimality) and identifies the unknown (whether ranking improves generation). The unknown was narrowed: ranking DOES improve generation on structured tasks (math: +133% answer correctness), and the routing accuracy (88%) exceeds the predicted lower bound.

Specific issues that should be noted but are not blocking:

1. The "Theorem 1" is an argument sketch, not a formal proof. Acceptable for Type 2 but should be relabeled as "Proposition" or "Claim." Not blocking because Type 2 does not require a complete proof -- it requires operating within a proven framework, which it does (Neyman-Pearson).
2. The data processing inequality citation is incorrect but the conclusion (argmin preserves ranking) is true by simpler reasoning.
3. The overhead kill criterion (10%) was contradicted by MATH.md's own calculation (18%). The criterion should have been set at 20%. The measured 29.5% includes disk loading overhead. Not blocking because the core finding is about routing accuracy and quality, not latency.
4. N=10 per domain is small. Variance is unquantified. The math result (70% vs 30% correctness) is compelling despite small N; the prose results are inconclusive noise.

The finding (Finding #185) should remain at status `supported`, not `conclusive`, because: (a) no formal proof was verified, (b) variance is unquantified, and (c) the prose domain results are inconclusive. The math domain result alone justifies SUPPORTED.
