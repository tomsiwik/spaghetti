# Peer Review: task_accuracy_evolve_signal

## NotebookLM Findings

Skipped -- the experiment documents are sufficiently self-contained for direct review. The MATH.md pre-registration, PAPER.md results, and code are all consistent and well-structured.

## Mathematical Soundness

**Kendall tau computation: correct.** The code uses `scipy.stats.kendalltau` which handles ties via midrank. The gold accuracy vector and subset accuracy vector are passed directly, which is the right approach. No manual rank conversion errors.

**MDAD derivation: correct but misapplied in one place.** The MATH.md MDAD formula is the standard two-proportion z-test margin. The table at p=0.70 giving MDAD=40.2pp for k=10 is correct:

  MDAD = 1.96 * sqrt(2 * 0.70 * 0.30 / 10) = 1.96 * sqrt(0.042) = 1.96 * 0.2049 = 0.402

The paper then correctly notes this is for *significance testing*, not ranking, and pivots to the Polo et al. empirical 6pp threshold. This is honest and well-handled.

**Expected tau formula: approximately correct.** The arcsin approximation for expected tau under Gaussian noise is a standard result from the rank correlation literature. The predicted tau values (0.17 at k=10, 0.56 at k=50) are close to the observed (0.302, 0.580), with the discrepancy plausibly explained by per-subject variance in adapter spread.

**PPL tau computation: one subtle issue.** The code computes PPL tau by correlating gold accuracy with negative PPL (`-p`), which is correct (lower PPL = better). However, the PPL is computed per-subject per-adapter from the *same* gold-standard 100 questions used for accuracy. This means the PPL ranking and accuracy ranking are computed on the same data, making the K3 comparison (accuracy-tau vs PPL-tau) internally consistent but not an independent validation. The paper does not flag this. It is a minor issue because K3 is a PASS either way, and the real question (K1) is unaffected.

**Hidden assumption: "first 100 questions" as gold.** The code uses `gold_indices = list(range(min(GOLD_SIZE, len(rows))))` -- i.e., the first 100 questions in dataset order, not a random sample. If the MMLU dataset has any ordering structure (e.g., easier questions first, grouped by subtopic), this introduces a systematic bias. The subsets are random draws *from* these 100, so relative comparisons are internally valid, but the gold-standard ranking itself could differ from a ranking on a random 100-question sample. This is a minor concern; MMLU test sets are generally shuffled.

## Novelty Assessment

**Low novelty, but that is appropriate.** This is an engineering validation experiment, not a mechanism invention. The question "can small subsets rank models?" has been studied by Polo et al. (2024) and EssenceBench. The specific application to adapter ranking in SOLE is new. The experiment correctly cites and builds on the prior art.

**The main contribution is the negative result.** Confirming that MMLU cannot discriminate between domain-specialized adapters (spread < MDAD) is useful for the project roadmap, and the pre-registration of the expected failure in MATH.md is good practice.

## Experimental Design

**Strength: pre-registration.** MATH.md predicted tau~0.17 at k=10 before running the experiment. The actual result (0.302) is in the same ballpark. This is excellent scientific practice and increases trust in the analysis.

**Strength: sweep over subset sizes.** Testing k=10, 25, 50 provides a dose-response curve rather than a single data point.

**Weakness 1: N=5 adapters is critically low for Kendall tau.** With 5 adapters, there are only C(5,2)=10 concordant/discordant pairs. A single swap changes tau by 0.2. The paper acknowledges this in Limitations but does not quantify the confidence interval on tau. For N=5, the standard error of tau under H0 is approximately sqrt(2(2*5+5) / (9*5*(5-1))) = sqrt(15/180) = 0.289. This means a tau of 0.302 is barely 1 standard deviation from zero -- statistically indistinguishable from random. The K1 KILL is still correct (you cannot claim tau >= 0.7 with this evidence), but the paper should be more explicit that even the *gold-standard ranking* is fragile at N=5.

**Weakness 2: only 5 adapters tested out of "up to 20" planned.** The SPEC says "up to 20 adapters." The HYPOTHESES.yml entry says "5-20." Only 5 were available. With 20 adapters, there would be 190 pairs, making tau much more stable. The experiment should note that the N=5 result is a lower bound on the discriminative power -- with more adapters showing wider spread, the method might work. Currently, the paper partially addresses this but could be clearer.

**Weakness 3: SPEC mandated vLLM, experiment used HF.** The SPEC explicitly says "Must NOT use sequential HF generate() -- vLLM batch inference only." The paper explains this was due to RunPod environment corruption, which is a legitimate practical constraint. However, this means K2 timing numbers (10.3s/domain) are from HF sequential inference, not vLLM batch. The PAPER correctly notes the timings are inflated, so K2 PASS is even more clearly passed. No impact on rankings or K1/K3.

**Weakness 4: 0-shot MMLU.** Standard MMLU evaluation is 5-shot. The paper notes this in Limitations but does not discuss whether 0-shot vs 5-shot changes the adapter spread. With 5-shot prompting, the base model typically scores higher and the adapters might differentiate more (or less). This is an uncontrolled variable.

**Design question: are the subsets truly nested?** The draws at k=10, k=25, k=50 are independent random samples from the gold set, not nested (i.e., the 10-question subset is not necessarily a subset of the 25-question subset). This is fine for the stated analysis but means you cannot directly attribute tau improvement to "more questions" -- it could also be "different questions." The paper does not claim a causal relationship, so this is acceptable.

## Hypothesis Graph Consistency

The HYPOTHESES.yml kill criteria are:
1. "10-question held-out accuracy does NOT rank adapters consistently (Kendall tau < 0.7)" -- **tested and triggered** (tau=0.302). Correct KILL.
2. "per-domain evaluation cost exceeds 60s/adapter" -- **tested, PASS** (10.3s).
3. "accuracy ranking disagrees with PPL ranking AND gold-standard ranking" -- **tested, PASS** (accuracy tau 0.302 > PPL tau 0.235).

The kill criteria are correctly tested. K1 is clearly triggered. The hypothesis status should be updated to "killed" with the caveat that the kill is specific to MMLU (out-of-domain for these adapters), not to task-accuracy-based ranking in general.

**Concern: the PAPER concludes "not a project-level kill" but the hypothesis status is still "active."** This needs to be resolved. The hypothesis as stated ("Small held-out benchmark reliably ranks adapter quality at macro") is killed for MMLU. If domain-specific benchmarks are expected to work, that should be a new hypothesis, not a continuation of this one.

## Macro-Scale Risks (advisory)

1. **Domain-specific benchmarks are untested.** The paper recommends HumanEval, MATH-500, MedQA as alternatives. None have been tested for adapter ranking. The assumption that "in-domain deltas >> 6pp" is plausible but unverified. From individual_expert_held_out, the python adapter gained +9.1pp on HumanEval -- which IS above the MDAD threshold. This is encouraging but is a single adapter on a single benchmark.

2. **LOO PPL was proven on N=5 with 1/N scaling.** The paper recommends LOO PPL as the primary Evolve signal. At N=50, LOO evaluation requires N forward passes, each with N-1 adapters composed. The cost scales as O(N^2) which may become prohibitive. The paper does not discuss this.

3. **The "hybrid signal" recommendation is reasonable but vague.** LOO PPL for cross-domain + domain-specific accuracy for within-domain is architecturally sound. But the decision boundary ("when to use which signal") is undefined. This is acceptable as a recommendation; it does not need to be solved in this experiment.

## Verdict

**PROCEED** -- with required status update.

The experiment is well-designed, honestly pre-registered, correctly executed, and reaches sound conclusions. The K1 KILL is real and clearly evidenced. The limitations are properly acknowledged. The code implements the analysis correctly.

**Required action (not blocking for PROCEED):**

1. Update HYPOTHESES.yml: change `exp_task_accuracy_evolve_signal` status from "active" to "killed" with a note specifying the kill is for MMLU-based ranking and recommending a new hypothesis for domain-specific benchmark ranking.

2. The PAPER should add one sentence to the Analysis section quantifying the Kendall tau standard error at N=5 (~0.29 under H0), making explicit that the observed tau values are statistically noisy. This strengthens rather than weakens the KILL verdict -- even the point estimate is far below 0.7, and the uncertainty makes it even less reliable.

**Advisory (non-blocking):**

3. Consider filing a follow-up hypothesis: "Domain-specific accuracy (HumanEval, MATH-500, MedQA) at k=10-50 reliably ranks domain clones (tau >= 0.7)." The HumanEval +9.1pp result suggests this is feasible for code adapters.

4. The K3 comparison (accuracy-tau vs PPL-tau on the same 100 questions) could be noted as a same-data comparison, not an independent validation. This does not change the verdict but improves intellectual honesty.
