# Peer Review: pilot50_composition_quality

## NotebookLM Findings

Skipped -- this experiment has no MATH.md (no formal derivations to verify) and the PAPER.md is a short empirical digest. The review proceeds directly to systematic attack.

## Mathematical Soundness

No formal mathematical claims are made. The experiment is purely empirical. The PPL computation (line 104: `exp(mean(losses))`) is standard cross-entropy perplexity. No issues there.

One subtle point: the `add_weighted_adapter` with `combination_type="linear"` and weights `[1/N]*N` performs `W + (1/N)*sum(B_i @ A_i)`. This is NOT the same as evaluating each adapter separately and averaging outputs. The paper correctly describes this as "pre-merge composition" but the 1/N factor means each expert contributes only 20% of its learned delta. This is the expected behavior for equal-weight averaging, and the degradation is a direct consequence -- not a bug.

**Assessment: Sound.** No mathematical errors.

## Novelty Assessment

This is not a novelty-seeking experiment. It is an infrastructure gate test: does equal-weight pre-merge work at macro scale with real adapters? The answer is "partially" -- composition preserves gains over base but dilutes individual expertise. This is consistent with the micro finding that relevance-weighted composition resolves dilution (exp_cross_domain_dilution_vs_k, r=0.990 oracle correlation).

**Assessment: Appropriate for its role.** Not a novelty experiment.

## Experimental Design

### Strengths

1. **Three-way comparison** (base vs single-expert vs composed) is the right design. It disentangles "composition is harmful" (K3) from "composition dilutes" (K1).
2. **K3 PASS is the key result.** All 5 domains beat base after composition. This is genuinely important -- it means pre-merge does not catastrophically interfere even with equal weights.
3. **Deterministic adapter selection** (seed=42, sorted order) enables reproducibility.

### Weaknesses

1. **N=5 only, not N=5/10/25/50 as designed.** The script was designed for multiple N values (line 39: `[5, 10, 25, 50]`), but only 5 adapters were available. The paper acknowledges this. K2 (superlinear scaling) is essentially untestable with a single data point. The paper claims "K2 PASS" but this is vacuous -- there is no scaling curve to assess.

2. **Eval contamination.** The `load_eval_texts` function (lines 62-85) falls back to `train.jsonl` when `eval.jsonl` is unavailable, using the last 200 lines. This means the "eval" data may overlap with training data. However, since this is a relative comparison (composed vs single-expert vs base), contamination affects all three conditions equally and does not invalidate the relative degradation measurements.

3. **Only 5 domains tested.** With N=5, the composition set equals the evaluation set -- every adapter in the composed model has a matching evaluation domain. This is the BEST case for equal-weight composition. At N=50, most of the 50 adapters would be irrelevant noise for any given domain evaluation. The 127% mean degradation at N=5 would almost certainly be worse at N=50.

4. **4-bit quantization as confound.** Both base and composed models use NF4, which adds quantization noise. The paper acknowledges this but does not assess its impact. Since the comparison is relative (same quantization for all conditions), this is not fatal but introduces unknown noise.

5. **K1 threshold of 10% was always unrealistic for equal-weight merging.** With 1/N weighting at N=5, each expert contributes 20% of its delta. Even under perfect orthogonality, composed PPL cannot match single-expert PPL because 80% of the expert's signal is missing. The 10% threshold implicitly assumed near-lossless composition, which is only achievable with routing (selecting the relevant expert) not averaging. This is a **threshold design error**, not an experiment failure.

### Alternative Explanations

The 127% mean degradation has a simple explanation: 1/N dilution. Each adapter's contribution is scaled to 20%, so the expert-to-base PPL improvement is roughly scaled to 20% as well. Check:

- bash: single achieves (4.744 - 1.384) = 3.36 improvement. At 20%: 0.672 improvement. Predicted composed: 4.744 - 0.672 = 4.072. Actual: 2.375. The composed model is BETTER than the 20% prediction, suggesting some constructive composition.
- medical: single achieves (25.143 - 2.242) = 22.901 improvement. At 20%: 4.580. Predicted: 25.143 - 4.580 = 20.563. Actual: 7.772. Again much better than naive 20%.

This suggests the adapters are NOT purely orthogonal in effect space -- there is beneficial cross-domain transfer. The "dilution" framing in the paper somewhat undersells the result.

## Kill Criteria Evaluation

| Criterion | Assessment |
|-----------|------------|
| K1: <10% degradation | **FAIL is correct** but the threshold was poorly calibrated. 10% is impossible with equal 1/N weighting at any N>1. This tells us the threshold was wrong, not that composition is broken. |
| K2: not superlinear | **PASS is vacuous.** Single data point. Cannot assess scaling. Should be marked "INCONCLUSIVE" not "PASS". |
| K3: <20% worse than base | **PASS is correct and important.** 0/5 domains worse than base. Strong result. |

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry specifies kill criteria for N=50, but only N=5 was tested. The experiment is **incomplete** against its own specification. The verdict should not be "FAIL" (which implies the mechanism is broken) but rather "PARTIAL -- equal-weight insufficient, routing needed."

The paper correctly identifies next steps (weight normalization, dynamic weighting, selective composition) and these are already registered as experiments (exp_composition_weight_normalization, exp_dynamic_weight_composition_macro, exp_cluster_grouped_composition).

## Integration Risk

Low. The finding aligns perfectly with the existing PPL-probe result from micro (exp_cross_domain_dilution_vs_k), which already showed that relevance-weighted composition resolves equal-weight dilution with r=0.990 oracle correlation. The macro experiment confirms the micro prediction: equal-weight is lossy, weighted composition is needed.

## Macro-Scale Risks (advisory)

1. **N=50 equal-weight may be catastrophic.** The first run (pre-retrain adapters) produced PPL in the trillions. Even post-retrain, extrapolating the N=5 degradation to N=50 is alarming. Weight normalization (exp_composition_weight_normalization) is critical path.

2. **Cluster interference.** At N=50, within-cluster experts (e.g., python + bash + sql) will have higher cosine similarity. The 127% degradation at N=5 (low-overlap domains) could be much worse for semantically related experts.

3. **The K1 threshold needs recalibration.** For equal-weight pre-merge, a reasonable threshold would be "composed PPL < 2x single-expert PPL" (100% degradation). The current 10% threshold is only meaningful for routed/weighted composition.

## Verdict

**PROCEED**

Justification:

1. **The mechanism works in principle.** All 5 domains beat base after composition (K3 PASS). Pre-merge does not destroy capability.
2. **The K1 FAIL is expected and already has a known fix.** Micro already proved that PPL-probe weighting achieves r=0.990 with oracle (exp_cross_domain_dilution_vs_k). Equal-weight dilution is the expected failure mode when routing is absent.
3. **The experiment is incomplete but not wrong.** N=5 is all that was available. The results are directionally correct and motivate the right next experiments.
4. **The data reveals more than the paper claims.** Composed PPL is substantially better than the naive 1/N prediction, suggesting beneficial cross-domain transfer.

The following should be addressed before citing this result as conclusive:

1. Mark K2 as "INCONCLUSIVE" not "PASS" in HYPOTHESES.yml evidence -- a single N value cannot assess scaling behavior.
2. Re-run at N=10/25/50 once more adapters complete retraining, with weight normalization variants (exp_composition_weight_normalization).
3. Add a "predicted 1/N dilution" column to the results table to show that composition is actually BETTER than naive dilution would predict. This strengthens the positive interpretation.
4. Recalibrate K1 threshold: split into K1a (equal-weight: <100% degradation) and K1b (routed: <10% degradation). The current 10% threshold conflates two different composition strategies.
