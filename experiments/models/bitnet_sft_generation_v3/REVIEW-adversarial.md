# Peer Review: bitnet_sft_generation_v3

## Experiment Type
Verification (claims to verify composition of three independently-proven mechanisms)

## Hack Detector
- Fix count: 3 (SFT masking + energy gap routing + execution-based eval). FLAG -- this is fix #3 on the generation quality problem (v1 killed, v2 presumably killed, now v3 killed).
- Is MATH.md a proof or a description? **Description dressed in equations.** All three "theorems" either restate definitions or cite empirical findings as proof steps. No genuine deductive reasoning anywhere.
- Metric used as evidence: Composite score blending keyword F1 and execution metrics. PAPER.md itself admits keyword density "does not predict task quality" (Finding #179), yet 3/5 domains are evaluated solely by keyword density.
- Kill criteria source: K602/K604 derived from predictions, which is correct. But predictions themselves are derived from empirical findings, not from proofs.

## Self-Test Audit

1. **One-sentence impossibility property:** "SFT masking sets instruction-token gradients exactly to zero by chain rule." This is a definition, not an impossibility property. The response mask excludes instruction tokens from the loss by construction -- there is no mathematical content here beyond "we defined the loss this way." **EVASION.**

2. **Cited theorems:** Chain rule of calculus (valid but trivially applied), Neyman-Pearson lemma (MISAPPLIED -- see Mathematical Soundness below). **FLAG on Neyman-Pearson.**

3. **Predicted numbers:** P1-P5 are specific and falsifiable. **PASS.**

4. **Falsification condition:** "If SFT adapters produce HIGHER NTP loss than NTP adapters." This targets only Theorem 1 (the trivial one). It does not target the actual failure mode that killed the experiment: routing collapse when one adapter dominates. **WEAK.**

5. **Hyperparameter count:** Claims 0 new, all inherited. This is correct. **PASS.**

6. **Hack check:** Claims "each component was proven independently." But this is the third generation quality experiment to be killed. The self-test should have flagged: "Why do independently-proven components fail when composed?" That is the real question, and it was not addressed. **EVASION.**

## Mathematical Soundness

### Theorem 1 (SFT Convergence): TRIVIALLY TRUE / NOT A THEOREM

The "proof" restates the definition of the SFT loss. If you define a loss that sums only over set R, then of course the gradient is zero for tokens not in R. This is not a mathematical insight -- it is the definition. Calling it a "theorem" and writing "QED" does not make it one.

The actual question Theorem 1 should answer is: "Does response-only masking produce adapters that are better at generating responses than NTP-trained adapters?" That requires a convergence analysis showing that the restricted gradient still reaches a useful optimum. No such analysis is provided.

### Theorem 2 (Routing Optimality): MISAPPLICATION OF NEYMAN-PEARSON

The Neyman-Pearson lemma states that the likelihood ratio test is the most powerful test of a simple null hypothesis H0 against a simple alternative H1 at a given significance level. The proof invokes this for routing but:

1. **Wrong setting.** NP applies to binary hypothesis testing, not N=5 multiclass selection. The extension to multiple classes requires additional structure (e.g., pairwise comparison with Bonferroni correction, or a different theorem entirely).

2. **No null/alternative hypotheses specified.** The proof jumps from "likelihood ratio is optimal" to "argmin energy gap is optimal" without establishing that the energy gap IS a likelihood ratio, or what the null hypothesis is.

3. **Empirical numbers substituted for proof.** "From Finding #185: empirical accuracy = 88%" is a measurement, not a deduction. The "proof" is: we measured 88% before, therefore we predict >=80% now. This is induction, not a theorem.

4. **The prediction was catastrophically wrong.** Predicted >=80%, measured 36%. This alone should have triggered a deeper examination of the proof's assumptions rather than attributing the failure to "different NLL profiles."

### Theorem 3 (Composition Correctness): UNSOUND

Chains two unsound theorems and adds Finding #203 as a proof step. "Finding #203 shows wrong adapters still capture ~87% of benefit" is an empirical observation from a different experiment with different adapters (NTP, not SFT). Using it as a proof step is invalid.

The actual failure mode -- code adapter dominating all routing decisions -- was not considered in any theorem or assumption. The proof assumed "each adapter reduces NLL most on its own domain" without stating this as an explicit assumption, let alone proving it. This hidden assumption is the one that broke.

### Summary of Mathematical Issues

The MATH.md contains zero genuine proofs. It contains:
- One definition restated as a theorem (Thm 1)
- One empirical observation dressed as a theorem with a misapplied citation (Thm 2)
- One composition of the above with additional empirical observations (Thm 3)

## Prediction vs Measurement

PAPER.md contains the prediction-vs-measurement table. Credit for that.

| Prediction | Measured | Match |
|-----------|---------|-------|
| P1: SFT loss < NTP loss | Yes | PASS |
| P2: Routing >= 80% | 36% | FAIL (2.2x below threshold) |
| P3: >= 4/5 domains beat base | 2/5 | FAIL |
| P4: Math >= 0.40 | 0.70 | PASS (but via wrong adapter) |
| P5: Code F1 >= 5% | +37% | PASS |

P2 and P3 are the core predictions and both failed catastrophically. P4 passed but for the wrong reason (code adapter routed to math, not math adapter). P5 passed. The proof's main predictions are refuted.

## NotebookLM Findings

Skipped -- the mathematical issues are clear from direct reading. NotebookLM review would not change the verdict.

## Novelty Assessment

The experiment itself is a valid engineering integration test. However:

- SFT response masking is standard practice (Ouyang et al. 2022), correctly acknowledged
- Energy gap routing was already tested in Finding #185 under NTP; the transfer to SFT is the novel question
- The discovery that one SFT adapter can dominate routing is genuinely useful -- but it is an observation, not a proof-verified finding

The real novel contribution is the empirical discovery: **SFT training breaks the energy-gap routing assumption because one adapter can become a universal improver.** This is a valuable killed finding. It should be recorded as such without pretending the "proofs" predicted it.

## Critical Structural Issue: The TWO-WORLD Problem

Three generation quality experiments have now been killed. The pattern is clear:

1. **Adapters work individually.** SFT adapters converge, reduce loss, improve task metrics.
2. **Routing fails when adapters are composed.** Energy gap routing assumes separable NLL profiles, which breaks when one adapter dominates.

This is not a bug in the routing threshold or the evaluation metric. It is a structural property: SFT training optimizes for general instruction-following, which makes adapters more similar (not more distinct). Energy gap routing requires distinctness. These goals are in tension.

No amount of re-running with different hyperparameters will fix this. The next experiment needs to either:
(a) Prove that a different routing mechanism works when adapters are similar, or
(b) Prove that a different training objective preserves adapter distinctness under SFT.

## Macro-Scale Risks (advisory)

1. If one adapter dominates at N=5, the problem worsens at N=25 or N=50.
2. The "all adapters are general-purpose improvers" finding (Finding #203) is a double-edged sword: it means routing may be unnecessary (just use the best adapter), which undermines the entire composable-experts architecture.
3. Keyword density metrics for prose domains remain unvalidated. Three experiments have been run without solving the prose evaluation problem.

## Verdict

**KILL**

The experiment is correctly killed by its own pre-registered criteria (K602 FAIL, K604 FAIL). The review confirms:

1. **MATH.md contains no genuine proofs.** Theorem 1 restates a definition. Theorem 2 misapplies Neyman-Pearson and substitutes empirical measurements for deduction. Theorem 3 chains the above. None would survive formal review.

2. **Core predictions catastrophically failed.** Routing accuracy predicted >=80%, measured 36%. Domain improvement predicted >=4/5, measured 2/5. The proof framework did not predict the actual failure mode (single-adapter dominance).

3. **Third generation kill confirms structural problem.** The TWO-WORLD pattern (adapters work individually, routing fails at composition) is not addressable by iterating on the same mechanisms. The next attempt needs new mathematics, specifically a proof that addresses adapter similarity under SFT training.

**Before any v4 attempt, the following must exist in MATH.md:**
1. A formal theorem (not a definition restatement) proving that the proposed routing mechanism is correct even when one adapter has lower NLL than all others on all domains.
2. An explicit assumption about adapter NLL profile separability, with a quantitative bound on the minimum inter-adapter energy gap required for correct routing.
3. A proof that SFT training preserves (or a mechanism that restores) this minimum gap.

Without these, a fourth generation quality experiment would be repeating a dead end.
