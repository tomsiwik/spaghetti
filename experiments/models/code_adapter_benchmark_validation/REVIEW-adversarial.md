# Peer Review: Code Adapter Benchmark Validation

## Experiment Type
Guided exploration (Type 2)

## Hack Detector
- Fix count: 0. Pure evaluation experiment, no mechanisms or losses added.
- Is MATH.md a proof or a description? **Description dressed in equations.** There is no Theorem/Proof/QED block. MATH.md cites two empirical papers (LIMA, Aryabumi et al.) and derives predictions from them, but these are predictions from empirical observations, not from proven mathematical theorems. LIMA is an empirical finding, not a theorem. "SFT teaches format not knowledge" is a hypothesis, not a proven mathematical property.
- Metric used as evidence: Benchmark accuracy (MMLU, GSM8K, HumanEval). These are well-established external metrics, which is a strength over custom evals.
- Kill criteria source: Derived from the framework's predictions. K614 from LIMA, K615 from Code Reasoning Transfer. Reasonable.

## Self-Test Audit

1. **One-sentence impossibility property:** "The Superficial Alignment Hypothesis predicts SFT cannot erase pre-trained knowledge." This is an empirical hypothesis, not a mathematical impossibility property. A mathematical impossibility would be something like "orthogonality of LoRA subspaces guarantees zero interference." However, for a Type 2 guided exploration, citing the empirical framework is acceptable. **MARGINAL PASS.**

2. **Cited theorems:** LIMA (2305.11206) and Aryabumi et al. (2405.20535). Neither is a theorem -- both are empirical papers. LIMA is a widely-cited empirical observation. Aryabumi et al. showed code training helps reasoning in their specific experimental setup. The conditions under which these hold (model scale, data quality, fine-tuning method) are not analyzed. The experiment itself ended up falsifying the Aryabumi prediction, which is actually a valuable result. **PASS with caveat:** these are empirical references, not theorems.

3. **Predicted numbers:** MMLU |delta| <= 2pp, GSM8K >= +5pp, HumanEval >= +10pp, base MMLU >= 25%. These are specific and falsifiable. **PASS.**

4. **Falsification condition:** "If code SFT degrades MMLU by >5pp, LIMA is wrong for ternary models." This targets the framework, not just the experiment. Good. **PASS.**

5. **Hyperparameter count:** 0 -- using existing adapter. **PASS.**

6. **Hack check:** No fixes stacked, pure evaluation. **PASS.**

## Mathematical Soundness

There is no formal proof to verify. This is a guided exploration operating within an empirical framework (LIMA + code reasoning transfer). The predictions are derived logically from the cited empirical results, and the derivation is sound:

- If LIMA holds, SFT should not degrade MMLU. Prediction: |delta| <= 2pp.
- If code reasoning transfer holds, code SFT should improve GSM8K/HumanEval. Prediction: +5pp/+10pp.

The thresholds are reasonable but ultimately arbitrary (why 5pp and not 3pp?). The 10pp threshold for HumanEval is justified by the "direct domain match" argument, which is sensible.

**Key concern:** The framework makes no prediction about what happens when LoRA scale is 20.0 on a ternary model. LIMA studied full fine-tuning. Aryabumi et al. used standard training, not LoRA. The gap between "SFT teaches format" and "rank-16 LoRA at scale=20.0 with STE quantization on ternary weights teaches format" is substantial. The experiment found this gap matters (degradation on GSM8K/HumanEval), which is the most valuable result.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table. Results:

| Prediction | Measured | Match |
|---|---|---|
| MMLU |delta| <= 2pp | +8pp | PARTIAL (overshot positively) |
| GSM8K >= base + 5pp | -18pp | NO |
| HumanEval >= base + 10pp | -15pp | NO |
| Base MMLU >= 25% | 38% | YES |

The directional signals are clear and large enough to be meaningful even at n=50. A -18pp swing on GSM8K (29/50 to 20/50) is a 9-question difference, well outside the noise band for n=50 (roughly +/-6pp at 95% CI via binomial). The -15pp on HumanEval (12/20 to 9/20) is only a 3-question difference at n=20, which is within the noise band (+/-10pp at 95% CI). **The HumanEval result alone would not be statistically significant.** However, the GSM8K + HumanEval results together provide a coherent directional signal.

The MMLU +8pp overshoot is interesting. PAPER.md explains this as format compliance (the adapter helps the model output single letters instead of verbose explanations). This is plausible but untested -- a simple format-only analysis (how many base responses were verbose vs. single-letter) would have strengthened this claim.

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that document-level review is sufficient.

## Novelty Assessment

This is not novel research; it is a validation experiment that tests a prior finding (#208) against external benchmarks. Its value is in killing a false positive, which is exactly what good science looks like. The finding that "custom evals measured format compliance, not capability" is an important methodological insight for the project.

**Prior art considerations:**
- The format-compliance vs. capability distinction is well-established in the SFT literature. This experiment rediscovered it empirically.
- The observation that LoRA with high scaling factor can degrade base model capabilities is consistent with known LoRA scaling issues (Hu et al., 2021 noted that alpha/r ratio matters).

## Experimental Design Concerns

1. **Same prompt format for base and adapter.** Both use "### Instruction / ### Response" format. This is fair for the adapter comparison, but the base model might perform better with its native prompt format or no wrapper at all. The MMLU +8pp could be an artifact: the adapter is better at THIS format, while the base model would score higher with a different format. PAPER.md acknowledges this in Limitations but does not test it.

2. **LoRA scale=20.0 not ablated.** This is the most important confound. At scale=20.0, the LoRA perturbation dominates the base model's representations. The degradation could be entirely due to over-scaling. A single additional run at scale=1.0 would have been very informative. PAPER.md acknowledges this.

3. **n=20 for HumanEval.** The 3-question difference (12 to 9) is not statistically significant on its own. The conclusion relies on GSM8K + HumanEval together.

4. **5.2x slowdown is unexplained.** The adapter evaluation took 2659s vs 513s for base. This is far more than the expected ~1% FLOPs overhead from rank-16 LoRA. The STE quantization in TernaryLoRALinear (lines 163-168) likely causes this -- the quantize-round-dequantize path prevents hardware-accelerated computation. This is not discussed in PAPER.md but is relevant to the project's serving story.

5. **Confound: was the adapter actually loaded correctly?** The degradation is so severe that one should verify the adapter is functioning as intended. A sanity check (e.g., running the adapter on the same custom eval from Finding #208 to confirm it still shows improvement there) would rule out a loading bug. The code path (lines 186-216) loads skeleton A-matrices and trained B-matrices separately, which is complex.

## Macro-Scale Risks (advisory)

- The LoRA scale=20.0 issue will compound at scale. Need scale ablation before any macro experiment.
- The 5.2x inference slowdown from TernaryLoRALinear makes runtime composition impractical. This needs to be solved (e.g., merge adapter into base weights before serving).
- The format-compliance finding suggests that the entire adapter evaluation pipeline (not just this experiment) may be measuring the wrong thing.

## Verdict

**PROCEED** (as a killed finding)

Justification: This is a well-executed validation experiment that correctly kills Finding #208's claim of "universal code adapter superiority." The experiment design is sound, kill criteria are well-defined and properly evaluated, results are clearly reported with a prediction-vs-measurement table, and the interpretation is appropriately cautious.

Specific strengths:
1. Uses external standardized benchmarks rather than custom evals -- exactly the right validation strategy
2. Kill criteria derived from the framework's predictions, not arbitrary
3. Root cause analysis (format-compliance vs. capability) is insightful and well-argued
4. Honest about limitations (small n, no scale ablation, single format)
5. The finding that base BitNet-2B-4T is surprisingly strong (58% GSM8K, 60% HumanEval) is independently valuable

Minor issues that do not block PROCEED:
- MATH.md has no formal proof (acceptable for Type 2 guided exploration of empirical framework)
- HumanEval n=20 is not independently significant (but GSM8K n=50 is, and both point the same direction)
- Missing sanity check that adapter loads correctly (the MMLU improvement provides some evidence it is active)
- 5.2x slowdown deserves investigation but is orthogonal to the finding

The finding status should be **killed** for Finding #208, with the methodological insight recorded as a separate **supported** finding about format-compliance vs. capability in SFT evaluation.
