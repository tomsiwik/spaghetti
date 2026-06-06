# Peer Review: exp_pro_benchmark_suite

## Experiment Type
Verification -- replicating Finding #329-330 (scale=5 preserves MMLU on BitNet-2B) at pro scale (Qwen3-4B-4bit).

## Hack Detector
- Fix count: 1 mechanism (NRE composition at scale=5). No stacking of hacks. Clean.
- Is MATH.md a proof or a description? **Description dressed in equations.** MATH.md explicitly says "Argument (not a proof -- perturbation bound is approximate)." The Davis-Kahan citation is correct but no formal theorem/proof/QED block exists. The "predictions" are empirical calibrations from prior findings, not derived from the bound.
- Metric used as evidence: accuracy on 3 benchmarks (MMLU, GSM8K, Code). MMLU logit-based is standard. GSM8K and Code are generation-based with extraction heuristics.
- Kill criteria source: K822 is a reasonable behavioral gate (composed must not lose ALL 3). S81 is a quantitative threshold (within 5pp on 2/3). Both are sensible for a replication experiment.

## Self-Test Audit
MATH.md has a "Self-Test" section with 3 falsification conditions. This is not the standard 6-item self-test (one-sentence impossibility, cited theorems, predicted numbers, falsification condition, hyperparameter count, hack check). However, the 3 conditions listed are:

1. ">5pp MMLU degradation falsifies Conjecture 1" -- specific and falsifiable. OK.
2. "Single adapters degrade more than composed" -- tests NRE averaging claim. OK.
3. "GSM8K/Code degrade >10pp while MMLU preserved" -- tests uniform perturbation. OK.

**Missing:** No hack check, no hyperparameter count, no cited-theorem conditions check. The self-test is incomplete by the standard template. However, for a replication experiment with a single hyperparameter (scale=5, already determined), this is minor.

## Mathematical Soundness

MATH.md is honest about its status: "Argument (not a proof)." The Davis-Kahan citation is correctly applied in spirit -- perturbation of eigenvectors bounded by spectral norm of perturbation divided by spectral gap. The linear scaling claim (sin(theta) proportional to alpha) follows directly from the bound.

**Issues:**

1. **The bound is vacuous without knowing delta (spectral gap).** MATH.md never estimates delta for Qwen3-4B weight matrices. The argument is purely relative: "alpha=5 is 4x smaller perturbation than alpha=20." This is valid as a relative argument but makes no absolute prediction.

2. **Linear scaling assumes fixed B-matrices.** The Davis-Kahan bound applies to a fixed Delta_W. When alpha changes, the B-matrices are the same (no retraining), so this holds. Correct.

3. **NRE composition changes the perturbation structure.** The composition formula averages B-matrices with norm rescaling. The proof treats this as a single perturbation, which is correct for the composed weight update.

4. **The "predictions" in the table are not derived from the proof.** "<5pp degradation" for MMLU is calibrated from Finding #329-330 (empirical), not derived from Davis-Kahan. This is fine for a replication experiment but the MATH.md should not frame empirical calibration as theoretical prediction.

**Verdict on math:** Adequate for a replication experiment. The framework is sound, the relative scaling argument is valid, and the honest "not a proof" label is appropriate.

## Prediction vs Measurement

PAPER.md contains the prediction-vs-measurement table. Assessment of each:

| Metric | Predicted | Measured | My Assessment |
|--------|-----------|----------|---------------|
| MMLU composed | <5pp deg | 0pp (12/12) | **CANNOT CONFIRM** (ceiling, see below) |
| GSM8K composed | <10pp deg | -16.6pp (1/6) | **NOT CONFIRMED** but n=6 is noise |
| Code composed | <10pp deg | +83.3pp (5/6) | **CONFOUNDED** (prompt format) |

### BLOCKING Issue 1: MMLU Answer-Position Bias (data integrity failure)

**11 of 12 MMLU questions have answer "B".** The single exception is question 4 (answer "A": habeas corpus).

This is a devastating confound. A model with ANY tendency to prefer token "B" in the answer position -- which instruction-tuned models often have, since "B" is a common next-token after "Answer:" -- will score near-perfect regardless of whether it "knows" the answers. The 100% score across ALL configurations (base, single, composed, DARE) is consistent with all configurations simply having a B-bias at the logit level.

To verify this is not just answer knowledge:
- If the model genuinely knows all 12 answers, the benchmark is too easy (ceiling effect, already noted in PAPER.md).
- If the model has a B-position bias, the benchmark is broken (not merely "too easy").

**Either way, the MMLU result is uninformative.** It cannot distinguish between "scale=5 preserves knowledge" and "the model always says B." The 12/12 score is consistent with both hypotheses.

### BLOCKING Issue 2: Sample sizes make GSM8K and Code unfalsifiable

PAPER.md correctly computes the binomial CIs and acknowledges the overlap. But the implication is stronger than acknowledged: **at n=6, the experiment cannot detect the predicted effects.**

For GSM8K at n=6:
- Predicted: <10pp degradation
- To detect a 10pp effect at alpha=0.05 with 80% power requires n >= 200 (binomial test).
- At n=6, the minimum detectable effect is approximately 40pp.
- The measured -16.6pp (1 question) has a two-sided p-value of 1.0 by Fisher exact test (the CIs overlap completely). This is pure noise.

For Code at n=6:
- Base scores 0/6. ANY correct answer is "improvement."
- The 0% base rate means the benchmark has zero sensitivity to degradation. You cannot detect "<10pp degradation" from a 0% baseline -- there is nothing to degrade.
- The +83pp is a prompt-format artifact (correctly identified in PAPER.md).

**The experiment has no statistical power to verify ANY of its predictions.**

### Issue 3: Code benchmark measures prompt format compliance, not capability

PAPER.md correctly identifies this: base Qwen3-4B expects `<|im_start|>user...` format, but the experiment uses `### Instruction:` format. SFT adapters were trained on this format. The base model's 0/6 is not "cannot write code" -- it is "does not respond to this prompt format."

The 0/6 -> 5/6 jump is evidence that adapters change the model's prompt-format sensitivity. This is a behavioral outcome -- but it is NOT evidence about benchmark preservation. It confounds the S81 criterion (within 5pp of base) because the baseline itself is an artifact.

### Issue 4: Composed = DARE (identical results)

Composed and composed+DARE produce identical results: (12/12, 1/6, 5/6). PAPER.md interprets this as "DARE has no effect at scale=5." But given the sample sizes, this could also mean: both configurations produce the same prompt-format behavior, and the benchmarks cannot detect any difference. With n=6, the probability of identical results by chance (even if true accuracies differ by 15pp) is substantial.

## Kill Criteria Evaluation

**K822 (composed does not lose ALL 3 benchmarks): PASS.** This is evaluated correctly. The criterion is extremely lenient -- the model only needs to NOT degrade on all three simultaneously. Given the Code prompt-format confound (base scores 0%), K822 would pass for virtually any adapter configuration that responds to SFT format.

**S81 (within 5pp on 2/3 benchmarks): FAIL.** Evaluated correctly. 1/3 within 5pp (MMLU only). PAPER.md's reframing ("2/3 not degraded") is editorial -- the pre-registered criterion says "within 5pp," and +83pp is not within 5pp. The criterion design is indeed questionable (it penalizes improvement), but post-hoc reframing of criteria is not sound practice.

**The real problem:** K822 is so lenient that it provides almost no information. The experiment is designed to pass K822 and would only fail if the composition were catastrophically broken across ALL benchmarks simultaneously. This is a weak kill criterion for a replication experiment.

## Novelty Assessment

This is explicitly a replication experiment (Finding #329-330 at pro scale). No novelty claim is made or needed. The value is in confirming cross-model generalization.

However, the replication **does not actually replicate** the key finding. Finding #329-330 showed 0pp MMLU degradation at scale=5 on BitNet-2B with n=50 questions. This experiment shows 12/12 MMLU on Qwen3-4B with n=12 questions -- but those 12 questions are 92% answer-B, making the result uninterpretable. The replication is not comparable.

## Composition Architecture Concern (non-blocking)

The `attach_adapter` function uses `domain_idx=0` for composed adapters, meaning the NRE-averaged B-matrix is paired with domain 0's A-matrix from the Grassmannian skeleton. This is consistent across the codebase and is a deliberate design choice (single-A composition rather than sum-of-per-domain-adapters). The mathematical implication is that composition projects through a single subspace rather than the union of domain subspaces. This is an architectural simplification worth noting but is not a bug specific to this experiment.

## Macro-Scale Risks (advisory)
- If MMLU is fixed (balanced answers, n >= 50), the result may change.
- Prompt format choice will dominate Code and GSM8K results. Use Qwen3 chat template for fair comparison.
- DARE identity at scale=5 may not hold at larger N (e.g., N=24).

## Verdict

**REVISE**

The experiment is honestly reported and the PAPER.md analysis is thoughtful (especially the prompt-format and ceiling-effect discussions). However, the data does not support the central claim. Specific blocking fixes:

### Blocking Fixes

1. **Replace MMLU questions with balanced answer distribution.** The current set has 11/12 answers = "B". This is a data-integrity failure that makes the MMLU result uninterpretable. Use the standard MMLU format with roughly uniform A/B/C/D answer distribution, or sample directly from the published MMLU dataset (at least 50 questions).

2. **Increase GSM8K to at least n=30.** At n=6, no comparison can reach significance. The prior tiny experiment used n=30 and correctly noted CIs were still wide. At n=30, a 20pp effect becomes marginally detectable.

3. **Use Qwen3 chat template for Code benchmark (or evaluate both).** The `### Instruction:` format is not the model's native format. The base model's 0/6 is a format artifact. Either: (a) use `<|im_start|>user...` for all configurations, or (b) run both formats and report both.

4. **Acknowledge in PAPER.md verdict that the replication is inconclusive, not "SUPPORTED."** The current evidence is: MMLU untestable (ceiling + answer bias), GSM8K noise at n=6, Code confounded by prompt format. "SUPPORTED (provisional)" overstates what the data shows. The honest verdict is INCONCLUSIVE pending fixes 1-3.

### Non-Blocking Observations

5. K822 is too lenient to be informative. Consider strengthening to "composed N=5 within 10pp of base on at least 2/3 benchmarks" (i.e., make K822 what S81 was, with wider tolerance).

6. The `results.json` field `"all_pass": true` is misleading -- it reflects K822 pass but S81 fail. The overall assessment should not be "all pass."

7. MATH.md is honest about "not a proof." For a replication, this is acceptable. For a finding upgrade to "conclusive," a formal theorem would be needed.
