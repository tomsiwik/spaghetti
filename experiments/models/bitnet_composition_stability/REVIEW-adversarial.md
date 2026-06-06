# Peer Review: BitNet Composition Stability (Re-review)

**Date:** 2026-03-19
**Scope:** Re-review after 4 fixes applied from initial review.
**Documents reviewed:** MATH.md, PAPER.md (revised), HYPOTHESES.yml, results.json, bitnet_composition_stability.py, VISION.md, SOLE_ADVERSARIAL_REVIEW.md

---

## Previous Review Fixes -- Verification

All 4 required fixes from the initial review have been applied:

1. **Absolute PPL comparison table added.** PAPER.md now includes a per-domain table (seed 42) showing FP16 wins 3/5 domains in absolute composed PPL. This was the most important fix -- it prevents the reader from mistaking the ratio advantage for an absolute quality advantage.

2. **1.60x stability claim reframed.** The asterisk footnote explicitly states: "The 1.60x ratio improvement is driven by the ternary base having ~1.6x higher PPL (the denominator), not by the composed model having lower absolute PPL." Clear and honest.

3. **"What This Actually Shows" section added.** Four numbered points correctly frame the result as a non-catastrophe finding, identify the mechanism as quantization recovery, flag the limitation for natively-trained BitNet, and note the FP16 baseline is also non-catastrophic at micro.

4. **HYPOTHESES.yml evidence updated.** The evidence entry now includes: "Kill criteria thresholds designed for interference-based catastrophe (100x, 10x) are not informative for the quantization-recovery mechanism observed." This is the correct framing.

Non-blocking observation (max cosine 0.83) is also now mentioned in the Diagnostic Metrics section.

## Mathematical Soundness

### What holds

1. **Quantization implementation is correct.** `ternary_quantize` at line 149 implements `RoundClip(W / mean(|W|), -1, 1)` faithfully. The scale factor is applied back (`W_t * alpha`) for computation. Verified against MATH.md Section 2.1.

2. **Composition arithmetic is standard.** Equal-weight merge via `sum(delta_i) / N` (line 418-419) matches the SOLE default. The code correctly implements both averaged and un-averaged composition.

3. **PPL computation is correct.** Cross-entropy loss with proper masking, exponentiated for PPL. Ratio computation (composed/base) is straightforward division.

4. **Diagnostic metrics are sound.** Delta norm CV, absolute cosine similarity with proper normalization, max/min ratio -- all standard computations verified against results.json.

### What does not hold

1. **MATH.md Section 3.3 contains theory refuted by the experiment's own data.** The claim that ternary base makes adapters "tend to have more uniform magnitude" is stated as theoretical reasoning, then refuted in Section 4.1 (CV 0.199 vs 0.107). The MATH.md acknowledges this refutation, which is good practice. However, the section header "Composition Interference Analysis" oversells what is actually speculation. This is a minor issue -- the self-correction is present.

2. **MATH.md Section 3.2 "signal decomposition" is trivially true.** Writing `y = y_base + y_adapter` is just linearity of matrix multiplication. The predictability claim for the ternary base signal ignores that adapter signals are FP16 and unconstrained. Not wrong, just not insightful.

3. **No formal bound is derived.** The MATH.md title "Mathematical Foundations" suggests more rigor than is present. The document is an empirical report with notation, not a mathematical proof. Acceptable for micro scope, but noted.

## Novelty Assessment

### Prior art

- **LoTA-QAF (2505.18724):** Tests lossless ternary adapter merging -- ternary adapters on ternary base. This experiment tests FP16 LoRA on post-quantized ternary base, a different configuration. No duplication.

- **MoTE (2506.14435):** Learned routing over ternary experts. This experiment tests equal-weight composition (no routing). Different composition strategy.

- **references/BITNET_SOLE_RESEARCH.md:** Contains the exact hypothesis being tested. The experiment is a faithful implementation of the project's own research plan.

### Delta over existing work

The experiment's actual contribution is a **negative mechanistic finding**: the hypothesized interference reduction mechanism is not observed. Instead, the favorable ratio is a denominator effect from quantization recovery. This negative result has clear value for the project -- it tells downstream experiments (exp_bitnet_orthogonality_trained, exp_bitnet_adapter_magnitude_analysis) what NOT to look for, and correctly flags that natively-trained BitNet may not show the same effect.

## Experimental Design

### Strengths

1. **Three seeds, consistent results.** The ternary composed ratio is 0.64, 0.65, 0.62 across seeds -- low variance, reproducible.

2. **Sum composition control.** Both FP16 and ternary fail catastrophically without 1/N averaging (PPL in thousands to millions). This isolates the averaging mechanism as essential.

3. **Rich diagnostics.** Cosine similarities, delta norm CVs, per-domain breakdowns, single-adapter baselines. The experiment tests multiple mechanistic hypotheses, not just the headline metric.

4. **Honest reporting.** The revised paper does not hide any inconvenient finding. The absolute PPL comparison, the denominator effect explanation, and the mechanism mismatch are all prominently stated.

### Remaining concerns (non-blocking)

1. **Post-quantization vs native BitNet.** The paper correctly flags this as Limitation 1. This is the experiment's fundamental limitation and cannot be resolved at micro scale. The downstream experiments (exp_bitnet_composition_stability at macro with microsoft/bitnet-b1.58-2B-4T) will address this.

2. **FP16 micro baseline does not reproduce macro catastrophe.** FP16 R=1.01 at micro vs PPL-in-trillions at macro. The paper correctly flags this as Limitation 2. Since both bases are non-catastrophic at micro, the comparison has limited predictive value for macro. This is inherent to micro-scale testing of a scale-dependent phenomenon.

3. **The ternary composed-vs-single ratio is worse than FP16 on 2/5 domains.** For "repeat," ternary composed/single = 2.74 vs FP16 composed/single = 1.77 (mean across seeds). For "arithmetic," ternary = 2.00 vs FP16 = 1.60. This means ternary composition degrades more relative to each adapter's individual quality, even though the ratio relative to base PPL looks better. The paper does not highlight this. It is a secondary metric but worth noting: the ternary "advantage" in the base ratio is entirely because the base is worse, and the composition actually damages individual adapter quality more on ternary than on FP16 for some domains.

4. **W_head kept ternary.** The output projection (W_head) is quantized to ternary along with all other weight matrices. At macro scale, this matrix is most responsible for logit-scale distribution. The interaction between ternary W_head and FP16 LoRA deltas on W_head could be the dominant source of composition issues at scale. Worth flagging for downstream work.

## Hypothesis Graph Consistency

- **Status "supported" is correct.** The kill criteria pass but the mechanism differs from the hypothesis. "Proven" would be inappropriate.

- **Kill criteria match the code.** K1 (100x base), K2 (10x single on >50% domains), K3 (convergence) are all correctly implemented and evaluated.

- **Blocking relationships are correct.** exp_bitnet_ternary_adapter_composition, exp_bitnet_orthogonality_trained, and exp_mote_sole_architecture all depend on this experiment. The non-catastrophic result is a valid gate-pass for proceeding to those experiments.

- **Downstream experiment notes should be updated.** exp_bitnet_adapter_magnitude_analysis tests "adapter weight norm variance on BitNet >= variance on FP16" as a kill criterion. This experiment already shows ternary delta norm CV is HIGHER (0.199 vs 0.107), meaning that downstream experiment would be KILLED before running. The HYPOTHESES.yml notes for that experiment should mention this preliminary evidence.

## Macro-Scale Risks (advisory)

1. **Quantization recovery mechanism disappears with native BitNet.** The primary driver of R < 1 (adapters recovering quantization loss) does not exist for natively-trained BitNet-2B. Macro validation must use microsoft/bitnet-b1.58-2B-4T, not a post-quantized model.

2. **The macro FP16 catastrophe is caused by specific adapter magnitude outliers (SQL).** This experiment's micro-scale FP16 shows no outlier adapters (max/min norm ratio 1.2-1.5x). The ternary experiment does not test whether ternary prevents the specific SQL-style outlier, because the micro scale does not produce outliers.

3. **1/N averaging is doing all the work.** Both bases catastrophically fail with sum composition. The stability observed is entirely a property of averaging, not of the base weight structure. At macro, if PPL-probe routing assigns non-uniform weights, the averaging protection may partially disappear.

## Verdict

**PROCEED**

The revised paper is honest, complete, and correctly framed. All 4 fixes from the initial review have been applied. The key claims are now appropriately scoped:

- "Ternary base composition does not catastrophically fail at micro" -- supported by data
- "The mechanism is quantization recovery, not interference reduction" -- supported by diagnostics
- "This does not predict behavior on natively-trained BitNet" -- correctly flagged
- "The 1.60x ratio advantage is a denominator effect" -- explicitly stated

The experiment provides a valid gate-pass for the downstream BitNet experiments. The negative mechanistic finding (no magnitude bounding, no orthogonality improvement) is valuable for directing those experiments.

### Non-blocking recommendations for downstream work

1. exp_bitnet_adapter_magnitude_analysis should note in its HYPOTHESES.yml that this experiment already shows ternary delta norm CV is higher (0.199 vs 0.107), providing preliminary evidence against the magnitude bounding hypothesis.

2. The macro BitNet composition experiment should use natively-trained BitNet (microsoft/bitnet-b1.58-2B-4T), not post-quantization, to avoid the quantization recovery confound.

3. Consider reporting ternary composed-vs-single ratios alongside composed-vs-base ratios in future experiments, since the former reveals actual composition damage while the latter is confounded by the denominator effect.
