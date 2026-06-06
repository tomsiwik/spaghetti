# Peer Review: removal_safety_complete_bound

## NotebookLM Findings

Skipped (NotebookLM not configured). Manual deep review performed against parent experiment data, code, and results.

## Mathematical Soundness

### What holds

1. **The core formula D = sum_eps * alpha_total is correctly derived and validated.** The paper correctly identifies that alpha_combined = 0.022 (from residual_layernorm_error_dynamics at L=24, pre_rmsn, d=64) already encompasses depth dampening, so there is no double-counting with the feedforward alpha=0.25 from multilayer_removal_cascade. This is the most important conceptual step and it is done right.

2. **The alpha=0.022 value is verified against parent data.** I checked the parent experiment's results.json: pre_rmsn at L=24, d=64, N=8 gives amplification ratios of 0.0205, 0.0198, and 0.0259 across three seeds, averaging 0.0220. The current experiment reproduces this at d=64, N=8 (0.0210, 0.0188, 0.0253, mean=0.0217) and extends it to d=128/256 and N=50, finding CV=10.6% across all conditions. This is a legitimate validation that the constant transfers.

3. **K1 arithmetic is correct.** D_direct = 4.82% * 0.022 = 0.106%. Empirical = 0.098%. Both are well below 1%. Margin is 10x.

4. **K2 arithmetic is correct.** Ratio = 0.098/0.106 = 0.93x, within the [0.5, 2.0] acceptance band.

5. **Dimension scaling power law fit is sound.** Three points (d=64, 128, 256) fit d^(-1.17) with R^2=0.9999. The exponent is steeper than the parent's -1.016, which is explained by N=50 vs N=8 providing more averaging. This is a plausible mechanism.

### What does not hold or needs qualification

6. **MATH.md Section 3.2 conflates tested and projected values.** The section computes "sum_epsilon ~ 0.72% total weight error" for N=50 at d=256 WITH Grassmannian skeleton decorrelation (cos=0.025). But the experiment does NOT use the skeleton -- it uses random initialization. The actual measured sum_epsilon is 4.82%, which is 6.7x larger. The paper does note this in Limitation 6, but MATH.md presents the 0.72% calculation as if it were the primary derivation, creating a misleading impression. The validated bound is D = 4.82% * 0.022 = 0.106%, not D = 0.72% * 0.022 = 0.016%.

7. **MATH.md Section 3.4 states "D = 0.016%" as the combined bound.** This is the projected value with skeleton decorrelation, not the empirically validated value. The paper's Table (Test 2) correctly shows 0.106% as the direct bound, but the mathematical derivation section arrives at a number that was never tested. This is a clarity issue, not a soundness issue, but it could mislead readers.

8. **The "analytical bound" is calibrated from parent data, not independently derived.** D_analytical = 31.4 * d^(-1.016) comes from the parent experiment's power law fit at N=8. When applied to N=50, it gives 0.112% at d=256, which happens to be close to empirical (0.098%). But this is not a prediction from first principles -- it is a fitted curve from the same family of experiments. The "direct bound" (sum_eps * alpha) is genuinely predictive and should be presented as the primary result.

9. **The 1/sqrt(L) scaling factor in the forward pass is not standard.** Both this experiment and the parent use `scale = 1/sqrt(L)` in the Pre-RMSNorm forward pass. While this is inspired by initialization scaling (like in GPT-2 and some residual scaling approaches), Qwen2.5 and Llama do NOT use 1/sqrt(L) scaling in their actual architectures. They use unscaled residual connections. This means the alpha=0.022 measured here may NOT transfer to production models, because the 1/sqrt(L) factor artificially suppresses error propagation through depth. Without this scaling, the feedforward amplification ratio at L=24 was 0.25 (not 0.022), which means the production alpha could be significantly higher than 0.022.

   This is the most significant concern in the entire paper. The PAPER.md Section on "What Would Kill This" mentions "learned normalization parameters" but does not identify the 1/sqrt(L) scaling as a potential confound. The paper claims alpha is an "architectural constant" but it is actually a constant of the specific (non-production) architecture being tested.

10. **Extrapolation to SOLE cosines (90x lower) is unvalidated.** The claim "d=896 with SOLE cosines ~0.00025%" assumes linear scaling of output deviation with cosine, which is not tested. The relationship between inter-expert cosine and output deviation is mediated by the GS correction term, which has second-order (cos^2) behavior. The extrapolation direction is correct (lower cosines = lower deviation) but the magnitude is speculative.

## Novelty Assessment

This experiment is a synthesis, not a novel mechanism. It combines five parent experiments into one predictive formula. The novelty is in:

1. Correctly identifying that alpha_combined subsumes alpha_depth (avoiding double-counting)
2. Showing that alpha is stable across d and N (architectural constant claim)
3. Providing a validated, tight bound (0.93x ratio) rather than a loose inequality

This is appropriate for a capstone experiment. No prior art concerns -- this is internal synthesis of the project's own results.

## Experimental Design

### Strengths

- **Multi-seed validation (3 seeds per config).** Standard for micro experiments.
- **Scale sweep (d=64, 128, 256; N=8, 50).** Tests the bound across the range that matters.
- **Both "direct" and "analytical" predictions.** The direct bound is properly derived from measured quantities.
- **Ground truth comparison.** Full GS recompute for N-1 experts provides unambiguous ground truth.

### Weaknesses

- **Only one expert removed (middle index).** Does removal position matter? If the GS ordering affects orthogonalization quality, removing expert 0 vs expert 49 could give different results. This is unlikely to be large but is not tested.

- **n_inputs decreases at larger d.** At d=64, n_inputs=300; at d=256, n_inputs=200. This increases estimator variance at the target scale. The std_dev (0.008%) is already small enough that this does not affect conclusions, but it is an unnecessary limitation.

- **No ablation of the 1/sqrt(L) scaling.** As noted in item 9, this is the most important design gap. Running the same experiment with scale=1.0 would quantify how much of the alpha=0.022 result comes from the artificial scaling vs genuine architectural dampening.

### Controls

- Parent baseline (d=64, N=8) reproduces known results from the parent experiment -- this is a good sanity check.
- GS recompute ground truth is the correct comparison target.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node:
- K1: "combined bound predicts <1% output deviation at d=256, L=24, N=50" -- tested and passed (0.106%)
- K2: "empirical measurement matches combined bound within 2x" -- tested and passed (0.93x)
- Dependencies on 5 parent experiments are correctly listed
- Status "proven" is justified by both kill criteria passing

## Macro-Scale Risks (advisory)

1. **The 1/sqrt(L) scaling issue (critical).** If the production model does not use this scaling, alpha at L=24 could be 5-10x higher than 0.022. Even at 0.22 (10x), the bound would be ~1.06%, which barely passes K1. The first macro experiment should measure the amplification ratio on a real Qwen2.5-0.5B model to calibrate this constant.

2. **Learned RMSNorm gamma parameters.** The micro model uses gamma=1.0 (no learnable scale). If trained gamma values are systematically >1 in certain layers, they act as error amplifiers. This needs measurement on a real checkpoint.

3. **PPL sensitivity.** The bound is on relative L2 deviation of hidden states, not on perplexity. At 0.1% deviation, PPL impact is likely negligible, but this assumption should be validated at macro scale.

4. **Attention layer interactions.** The micro model has no attention mechanism. The attention_self_repair experiment (KILLED at 2.1%) tested a different question (repair) than error propagation through QKV projections. The amplification ratio through real attention could differ from the MLP-only measurement.

## Verdict

**PROCEED**

The core result is sound: D = sum_eps * alpha is a validated, tight bound (0.93x of empirical) that passes both kill criteria with large margins. The synthesis of five parent experiments is done carefully, with correct handling of the double-counting risk between depth dampening and architecture dampening.

However, the paper should be revised for clarity on two points before the result is cited in downstream work:

1. **MATH.md Section 3.2-3.4 should clearly distinguish tested vs projected values.** The 0.016% "combined bound" uses the untested skeleton decorrelation. The validated bound is 0.106% (without skeleton). The current text makes it easy to confuse these.

2. **The 1/sqrt(L) scaling limitation should be explicitly flagged as a macro validation requirement.** The paper's "What Would Kill This" section should include: "If production transformers do not use 1/sqrt(L) residual scaling, the amplification ratio may be significantly higher than 0.022. The first macro validation must measure alpha on a real model checkpoint."

These are documentation fixes, not experimental re-runs. The mechanism works in principle, the math is sound within its stated architecture, and the kill criteria are genuinely passed. The 1/sqrt(L) question is a macro-scale risk, not a micro-scale failure.
