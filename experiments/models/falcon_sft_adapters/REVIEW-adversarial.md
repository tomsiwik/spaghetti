# Peer Review: falcon_sft_adapters (Second Pass)

## Experiment Type
Guided exploration (Type 2) -- testing whether SFT loss resolves NTP adapter degradation within the proven LoRA composition framework.

## Fix Verification (6 Required Fixes from First Review)

| # | Required Fix | Status | Evidence |
|---|-------------|--------|----------|
| 1 | K1 recorded as FAIL for routed/individual | FIXED | PAPER.md lines 115-119: "Verdict: FAIL. Individual SFT adapters degrade base on 5/6 benchmarks." results.json k1_562.result = "fail" |
| 2 | "Two diseases" reframed as HYPOTHESIS | FIXED | PAPER.md lines 98, 105, 109 use "We HYPOTHESIZE" with explicit "This was not tested" disclaimers |
| 3 | Statistical significance added | FIXED | PAPER.md lines 66-82: full z-test table, p=0.41 for GSM8K base->SFT, p=0.10 for NTP->SFT, explicit "NOT statistically significant" |
| 4 | Status downgraded to PROVISIONAL | FIXED | PAPER.md line 27: "Verdict: PROVISIONAL" with justification citing 3/4 prediction failures |
| 5 | MATH.md "Theorem" relabeled to "Property" | FIXED | MATH.md line 47: "Property (Response-only gradient under SFT loss)" |
| 6 | lora_scale=20 semantics verified | FIXED | PAPER.md lines 142-169: MLX source code quoted, confirmed raw 20x multiplier, impact analysis included |

All 6 fixes properly applied.

## Hack Detector
- Fix count: 1 (SFT masking). Clean single-mechanism intervention. No flags.
- Is MATH.md a proof or a description? Description of a known property (chain rule applied to masked loss). Now honestly labeled "Property" rather than "Theorem." Acceptable for Type 2 guided exploration.
- Metric used as evidence: GSM8K accuracy, MMLU accuracy. Reasonable proxies, and the paper no longer overclaims them as direct evidence.
- Kill criteria source: Derived from predictions P1-P4. K1 now honestly recorded as FAIL.

## Self-Test Audit
1. One-sentence impossibility property -- PASS. Now includes the important caveat about shared attention weights.
2. Cited theorems -- PASS. Chain rule, Ouyang et al. 2022, Hu et al. 2022. No longer inflated as novel theorems.
3. Predicted numbers -- PASS for Type 2. P1-P4 are directional thresholds, appropriate for guided exploration.
4. Falsification condition -- PASS. Correctly targets root cause hypothesis.
5. Hyperparameter count -- PASS. Zero new hyperparameters.
6. Hack check -- PASS. Single root-cause intervention.

## Mathematical Soundness

The first review identified that MATH.md's gradient claim was subtly wrong: "instruction-processing pathways receive zero gradient" conflates "no loss at instruction positions" with "no gradient through shared parameters that attend to instructions." The revised MATH.md (lines 57-63) now correctly states:

> "This does NOT mean instruction-processing pathways receive zero gradient. [...] The correct statement is narrower: SFT eliminates gradient signal from the task of predicting instruction tokens themselves, but response-token gradients still flow through shared parameters that process instructions as context."

This is an important correction that the researcher handled well. The mathematical framework is now accurately stated.

One remaining minor issue: MATH.md line 79 still says "because instruction-processing gradients are zero" in prediction P1. This is the old, imprecise language that Section C (lines 57-63) explicitly corrects. The prediction should say "because instruction-token-prediction gradients are zero." This is cosmetic -- the correct statement is present in the document -- but the inconsistency within the same file is worth noting.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table (lines 10-16). The table is honest: 3/4 predictions marked NO, P3 marked DIRECTIONAL with the p-value caveat. The additional row (SFT composed > NTP composed) is appropriately separated as a non-predicted observation.

The PROVISIONAL status is correctly justified given:
- 3/4 quantitative predictions failed
- The one passing prediction (P3, GSM8K) is not statistically significant (p=0.41)
- The directional observation (SFT > NTP on 4/6 composed) is interesting but unconfirmed

## Remaining Issues (Non-Blocking)

### 1. MATH.md Prediction P1 Wording Inconsistency (MINOR)
Line 79: "because instruction-processing gradients are zero" contradicts the corrected language in lines 57-63. Should read "because instruction-token-prediction gradients are zero." The correct understanding IS present in the document, so this does not affect the finding.

### 2. "BestSingle" Column in Benchmark Table (MINOR)
PAPER.md line 59: BestSingle GSM8K = 0.620 (code adapter). This is interesting -- the code adapter gets 62% on GSM8K vs 50% for the math adapter. This suggests the adapters are not learning domain-specific knowledge but rather general perturbations modulated by lora_scale=20. The paper does not comment on this anomaly. At scale=20x, the adapters may be functioning more as random perturbations than domain experts. This reinforces the lora_scale ablation as the critical next step.

### 3. NTP Baseline From Different Experiment (MINOR, Previously Noted)
The NTP composed scores are from exp_falcon_e3b_composition. The base model was re-evaluated (providing cross-validation), but the NTP adapters were not re-trained with identical settings. This is acknowledged implicitly but could be stated more explicitly.

## Novelty Assessment

Unchanged from first review. SFT with response-only masking is standard practice. The contribution is the empirical diagnosis within this project's adapter composition framework. No external novelty, but useful internal finding for the research program.

## Macro-Scale Risks (advisory)

1. The lora_scale=20 confound is the dominant concern. Until ablated, no conclusion about SFT vs NTP can be trusted at any scale.
2. The code adapter outperforming the math adapter on GSM8K (0.62 vs 0.50) suggests adapters at scale=20 are not learning domain-specific features. This would undermine the entire composition architecture if it persists at standard scale values.
3. With standard lora_scale (1.0-2.0), the individual adapter degradation may disappear entirely, making the SFT vs NTP distinction moot.

## Verdict

**PROCEED**

All 6 required fixes from the first review have been properly applied. The paper is now honest about:
- K1 failure (individual adapters degrade)
- Statistical insignificance of the headline result (p=0.41)
- PROVISIONAL status given 3/4 prediction failures
- The lora_scale=20 confound as the highest-priority next investigation
- The "two diseases" hypothesis as untested speculation

The remaining issues are minor (wording inconsistency in one prediction, unexplained cross-domain GSM8K anomaly). The experiment correctly identifies the lora_scale ablation as the critical next step, which is the right prioritization.

This is an honest PROVISIONAL finding: SFT composed directionally outperforms NTP composed, but the evidence is not statistically significant and confounded by extreme lora_scale. The experiment's primary value is identifying the lora_scale=20 confound as the likely root cause of adapter degradation across multiple prior experiments.
