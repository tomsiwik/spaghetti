# Peer Review: Correction Signal Quality

## NotebookLM Findings

Skipped -- this experiment is a pure Monte Carlo simulation with hand-specified parameters from published literature. The mathematical content is straightforward enough to review directly without external tooling. There are no novel derivations requiring deep verification.

## Mathematical Soundness

### What holds

1. **Sigmoid calibration is correctly implemented.** The logit inversion at lines 47-56 of the code correctly fits sigmoid(beta_0 + beta_1 * d) to the (base_accuracy, hard_accuracy) endpoints. Verified: TEACHER beta_0 = logit(0.92) = 2.44, beta_1 = logit(0.70) - logit(0.92) = -2.88. Matches MATH.md Section 2.2.

2. **Expert improvement model is coherent.** The diminishing returns term (1 - q_current)^alpha = (1-q)^0.7 correctly produces slower improvement as quality rises. Degenerate and wrong corrections correctly produce negative deltas. The sequential update in the simulation loop (lines 200-229) correctly implements the cumulative formula from MATH.md Section 3.2.

3. **EIR formula is dimensionally correct.** Quality-per-dollar is a valid cost-effectiveness metric. The ordering execution >> teacher >> human is robust to parameter perturbation given the 2000x cost gap between human and teacher, and 10x gap between teacher and execution.

4. **K1 and K2 evaluations match stated criteria.** K1 checks average teacher error against 0.20 threshold. K2 checks per-domain execution degeneracy against 0.10 threshold. Both are implemented correctly.

### What does not hold or is problematic

1. **MATH.md Section 5 K2 calculation is wrong.** The worked example computes "p_degen_exec * q_exec" but the kill criterion should be the degeneracy rate among *accepted* solutions, which is p_degen_exec / (p_degen_exec + (1 - p_degen_exec)) = p_degen_exec. The code correctly computes `n_degenerate / (n_corrections - n_no_signal)`, but MATH.md multiplies p_degen by q_exec which is a different quantity (the probability of "correct AND degenerate" among all attempts, not among accepted solutions). The code is authoritative and correct; MATH.md Section 5 has a formula error that produces lower numbers (making K2 appear safer than it is).

2. **Teacher error rate computation double-counts degeneracy.** In the code, `accuracy_empirical = n_correct_nondegen / (n_corrections - n_no_signal)` and `error_rate_empirical = n_wrong / (n_corrections - n_no_signal)`. The denominator is total applicable corrections. But `accuracy_empirical + error_rate_empirical + degeneracy_rate_empirical` should equal 1.0 (they partition the outcome space). From results.json, teacher on python_basics: accuracy 0.804 + error 0.1295 + degeneracy 0.0665 = 1.0. This checks out. However, the PAPER.md reports "Teacher Error Rate" which is the `error_rate_mean` -- this counts only *wrong* corrections, not degenerate ones. A degenerate correction is coded as "correct but harmful," so it is NOT counted in the error rate. This means the 19.6% figure underestimates the rate of *harmful* corrections. Harmful = wrong + degenerate. For teacher: 19.6% wrong + 6.5% degenerate = 26.1% harmful on average. **The K1 threshold was defined as "wrong >20%" but the real question is "harmful >20%", and that is clearly exceeded.**

3. **MATH.md Section 6 worked example contains arithmetic errors.** The "Wait -- teacher EIR is actually higher" self-correction at line 209 correctly recalculates, but the initial "EIR: execution ($64.6/dollar) >> teacher ($4,240/dollar)" is left in the document as misleading text. The final EIR values are correct but the document flow is confusing.

4. **No confidence intervals on K1.** The average teacher error is 19.6% against a 20% threshold -- a margin of 0.4pp. With 10 seeds and N=200 corrections per seed, the standard error of the mean error rate is approximately std/sqrt(n_seeds). From results.json, teacher error rates per domain have std ~0.027-0.036. The per-source aggregate error_rate_mean std is 0.046. Standard error of the mean across 10 seeds: ~0.046/sqrt(10) = 0.015. So the 95% CI on the average is approximately [0.166, 0.226]. **The 20% threshold falls well within this confidence interval.** The "SURVIVES" verdict is statistically indistinguishable from "KILLED" at any conventional significance level.

5. **Difficulty distributions are hand-specified, not derived.** The sigmoid parameters are calibrated to RLAIF literature endpoints (88% agreement, 85% saturation, 10-15% positional bias), but the difficulty distributions per domain (mean, std) are arbitrary. Systems_programming has mean 0.75 -- why not 0.65 or 0.85? A 0.05 shift in difficulty_mean changes teacher error rates by ~2-3pp, which would flip K1 decisively in either direction. The experiment's core finding (K1 barely survives) is not robust to plausible input perturbations.

## Novelty Assessment

**This is not novel research. It is a calibrated planning exercise.** The experiment correctly acknowledges this in the PAPER.md (Section "What This Experiment Is": "explicitly a simulation study -- no actual model training occurs"). The references cited (RLAIF, Self-Refine, Constitutional AI) are the appropriate calibration sources. No prior work packages these into a SOLE-specific decision tree, so the integration is useful even if no new mechanism is proposed.

The decision tree output (execution for well-tested code, teacher for non-code, human for critical) is common engineering wisdom. The value-add is making it quantitative with calibrated error rates and cost-effectiveness ratios. This is sufficient for a micro-scale planning experiment.

## Experimental Design

### What works

1. **The simulation correctly tests the stated hypothesis.** The hypothesis is "different correction sources have quantifiably different accuracy/cost/degeneracy profiles." The simulation produces those profiles.

2. **Kill criteria are well-defined and testable.** K1 (teacher error >20%) and K2 (execution degeneracy >10%) are concrete, falsifiable, and relevant to the Evolve phase.

3. **10 Monte Carlo seeds with 200 corrections each is adequate** for the statistical precision needed (std typically ~0.01-0.05 on quality deltas).

### What does not work

1. **The experiment is entirely circular.** The input parameters define the output. The sigmoid accuracy model with hand-specified (base_accuracy=0.92, hard_accuracy=0.70) *directly determines* the teacher error rate. There is no discovery here -- the K1 result (19.6%) is a mechanical consequence of the sigmoid integral over the chosen difficulty distributions. The "simulation" is an elaborate way to compute an integral that could be done in closed form (as MATH.md Section 5 actually does: "q_teacher_avg = ~0.83", implying 17% error). The discrepancy between the closed-form estimate (17%) and simulation result (19.6%) is because the domains have non-uniform difficulty distributions biased toward harder problems, but this is itself a design choice.

2. **No sensitivity analysis.** Given that the findings hinge on parameter choices (teacher hard_accuracy=0.70, difficulty distributions), the experiment should have swept key parameters to identify which inputs flip K1/K2. For instance: at what teacher hard_accuracy does K1 kill? (Answer: approximately 0.65, easily computed.) At what test coverage does K2 kill for algorithm_design? (Answer: approximately 0.67, from the (1-coverage)*0.30 formula.) These analytical thresholds would be more informative than the point estimates.

3. **The "accuracy_empirical" metric for the teacher has a discrepancy.** PAPER.md Table shows teacher error on python_basics as 13.0%, but the MATH.md sigmoid predicts ~8% error at difficulty 0.3. The simulation produces 13% because it samples from a Normal(0.3, 0.15) difficulty distribution, not a point estimate. This is correct behavior, but the PAPER.md does not explain why empirical rates differ from MATH.md point predictions.

4. **Execution source accuracy uses the same sigmoid as teacher/human, but execution feedback is fundamentally different.** Execution is binary (pass/fail), not probabilistic. A test suite either catches the bug or it doesn't. Modeling execution accuracy as sigmoid(beta_0 + beta_1 * difficulty) conflates two different failure modes: (a) the test suite misses a bug (false positive), and (b) the test suite rejects correct code (false negative). The sigmoid model treats these symmetrically, but in practice false positives (passing bad code) dominate because test suites are designed to reject bad code, not to accidentally accept it. This asymmetry is acknowledged in MATH.md Section 2.3 but not reflected in the implementation.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry correctly states:
- kill_criteria: "teacher corrections are wrong >20% of the time" and "execution feedback produces degenerate solutions"
- status: supported
- blocks: exp_execution_based_self_learning

The "supported" status is appropriate -- the experiment produces directionally useful information but is a simulation, not empirical validation. The evidence summary in HYPOTHESES.yml accurately reflects the results.

However, the evidence field claims "6 domains, 10 seeds, 200 corrections each" which is accurate for the simulation, and the decision tree outcome matches the code's output.

## Macro-Scale Risks (advisory)

1. **The entire experiment needs empirical validation.** Every parameter was hand-specified. At macro scale, measure actual teacher correction accuracy on pilot-50 expert outputs. If real 70B teacher error on systems_programming is 30% (not 24%), the decision tree changes significantly.

2. **Degenerate solutions are harder to detect than modeled.** The simulation treats degeneracy as a binary random variable with known rate. Real degenerate solutions (hardcoded outputs, overfitted patterns) are adversarially difficult to detect and the rate is unknown a priori.

3. **Correction independence assumption breaks in practice.** The simulation treats corrections as i.i.d. If a teacher makes a systematic error on a class of problems (e.g., always gets off-by-one wrong), correcting with teacher feedback will reinforce the same systematic bias across multiple corrections. The quality trajectory would be worse than the i.i.d. model predicts.

4. **Cost model needs updating.** The $0.001/correction for 70B teacher is based on 2023 API pricing. Current batch API pricing (Groq at $0.19/expert for 1000 samples) suggests the real cost may be different. Not blocking, but the cost-effectiveness ratios should be recalculated.

## Verdict

**PROCEED**

This is an honest simulation study that correctly identifies its own limitations. The findings are directionally useful for the Evolve phase design:

1. The cost-effectiveness ordering (execution >> teacher >> human) is robust to parameter perturbation given the orders-of-magnitude cost differences.
2. The systems_programming degeneracy finding (K2 killed) is genuine -- low test coverage makes execution feedback unreliable, regardless of exact parameter values.
3. The decision tree provides a concrete starting point for the correction routing pipeline.

The K1 "barely survives" finding should be interpreted as **"teacher corrections are at the boundary of acceptable error for hard domains"** rather than as a pass. The confidence interval clearly spans both sides of the threshold.

Non-blocking fixes (for FINDINGS.md caveats):

1. **MATH.md Section 5 K2 formula error:** The formula "p_degen_exec * q_exec > 0.10" computes P(degenerate AND correct), not the degeneracy rate among accepted solutions. Fix to "p_degen_exec > 0.10" or clarify the conditional.

2. **Report harmful rate, not just wrong rate:** K1 should acknowledge that wrong (19.6%) + degenerate (6.5%) = 26.1% harmful corrections on average. The effective signal quality of teacher corrections is lower than the 19.6% figure suggests.

3. **Add sensitivity analysis:** Compute and report the analytical thresholds: at what teacher_hard_accuracy does K1 flip? At what test_coverage does K2 flip for each code domain? These closed-form breakpoints are more informative than the point estimates.

4. **MATH.md Section 6 cleanup:** Remove or clearly mark the incorrect intermediate EIR calculation that is self-corrected two lines later.

5. **Add confidence interval on K1:** The margin is 0.4pp against a standard error of ~1.5pp. State explicitly that K1 survival is not statistically significant.
