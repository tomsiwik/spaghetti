# Peer Review: Correlated Layer Errors

## NotebookLM Findings

Skipped -- the experiment is sufficiently contained (single-file, 675 lines, pure numpy) that manual review covers the full scope without external tooling.

## Mathematical Soundness

### Correlation model (Section 2.2 of MATH.md): Correct

The mixing formula `delta_{k,l} = rho * d_k + sqrt(1 - rho^2) * n_{k,l}` correctly produces unit-norm vectors with the desired pairwise cosine equal to rho between any two layers for the same expert. The code faithfully implements this: random_component is projected orthogonal to expert_direction, then the two are mixed with coefficients (rho, sqrt(1 - rho^2)). The resulting vector has unit norm because the two components are orthogonal and the coefficients satisfy alpha^2 + beta^2 = 1. No issues here.

### Concern about the "why" explanations: Partially speculative

The paper offers three post-hoc explanations (consistent masking, GS correction coherence, rank-1 spectral dampening) for why correlation reduces error. These are plausible but not independently verified.

**Specific issue with Mechanism A (consistent masking).** The paper claims "suppression ~ (1 - mask_rate)^L = 0.5^24 ~ 6e-8 (if same dims masked)." This is wrong as stated. GELU does not produce a fixed binary mask -- it is a smooth function. A dimension with a small negative input gets suppressed but not zeroed. More importantly, the mask depends on the pre-activation value (base_weights @ h + delta @ h), which changes every layer as h evolves. Even with the same delta direction, the set of near-zero dimensions shifts across layers. The 0.5^24 calculation assumes a fixed, binary, layer-independent mask, which is false.

However, this error is in the explanatory narrative, not in the experimental measurements. The measured amplification ratios stand on their own regardless of which mechanistic story is correct.

**The linear activation result undermines the masking story anyway.** Test 5 shows that the linear activation (no masking at all) produces the same 0.41x ratio as GELU. The paper acknowledges this and pivots to "rank-1 spectral dampening" as the primary mechanism. This is the right conclusion but it contradicts the prominence given to Mechanism A in MATH.md. The paper should lead with the spectral argument.

### The d=256 anomaly: Honestly reported but under-analyzed

At d=256, the correlated/independent ratio is 1.82x -- correlation is WORSE, not better. The paper dismisses this because both absolute values are small (0.096% vs 0.175%). But the claim "correlation never worsens amplification" (Section 5.1: "amp_ratio(rho) <= amp_ratio(rho=0) <= 1.0") is stated as a universal inequality, when the data clearly violates it at d=256. The inequality should be qualified: it holds for d <= 128 in the tested range, with convergence to ratio ~1.0 at high d.

This matters because the extrapolation argument to production d=896 rests on "both converge to negligible." That is likely true, but the paper should not claim a universal inequality when the data show a counterexample. The direction of the effect could change at intermediate scales.

### Amplification ratio definition: Consistent with parent

amp_ratio = mean_output_dev / sum_per_layer_error. This is the same metric as the parent, and the interpretation (< 1.0 means sub-additive) is correct. The core result -- amp_ratio is well below 1.0 at all tested correlation levels -- is robust across 3 seeds.

### The GS correction coherence claim: Needs caution

The paper notes sum_per_layer_error drops from 17.8% to 9.0% as rho goes from 0 to 1. This is interesting but confounds two effects: (a) the weight-space error from GS removal genuinely decreases because coherent deltas have more predictable cross-projections, and (b) the amp_ratio itself changes. Since amp_ratio = output_dev / weight_error, a decrease in weight_error with roughly proportional decrease in output_dev could yield a flat amp_ratio -- which is almost what the data shows (0.088 vs 0.074). So the "correlation helps" story is partly driven by the weight-space error being smaller, not just by the forward pass being more forgiving.

## Novelty Assessment

This is a well-motivated follow-up to the parent experiment. The parent explicitly flagged correlated errors as an open risk. The experiment directly addresses that risk.

I am not aware of prior work that specifically studies how inter-layer correlation in LoRA expert deltas affects error propagation under Gram-Schmidt composition. The closest related work would be general perturbation theory for deep networks (e.g., sensitivity to weight perturbations), but the specific GS composition context is novel to this project.

## Experimental Design

### Does it test what it claims? Yes, with one gap.

The experiment tests controlled inter-layer correlation at 8 levels, across 5 depths, 4 dimensions, 3 activations, and with a double-adversarial condition. This is thorough for a micro experiment. The kill criteria are binary and clearly evaluated.

### Controls are adequate.

Three seeds per condition. The rho=0 baseline reproduces the parent's results. The actual measured inter-layer correlations are verified against targets.

### One methodological concern: the intra-layer cosine adjustment (Test 4).

When `intra_layer_cos` is set, the code applies `alpha_intra = sqrt(intra_layer_cos)` and `beta_intra = sqrt(1 - intra_layer_cos)`. This means the actual achieved cosine between experts is `intra_layer_cos` (since the shared and unique components are orthogonal, cos(i,j) = alpha_intra^2 = intra_layer_cos). However, this adjustment is applied AFTER the inter-layer correlation mixing, which means it partially disrupts the carefully controlled inter-layer correlation. The extent of this disruption is not measured or discussed. The Test 4 results should be interpreted with the caveat that the actual inter-layer correlation may differ from the target when intra_layer_cos is also set.

### Missing: statistical significance tests.

The paper reports a regression with p=0.43, correctly noting the trend is not significant. Good. But no confidence intervals or significance tests are provided for the key comparison (0.41x ratio). With only 3 seeds, the standard error could be substantial. A permutation test or bootstrap CI would strengthen the claim.

## Macro-Scale Risks (advisory)

1. **Residual connections.** The paper acknowledges this. In a real transformer, h_{l+1} = h_l + f(h_l). The residual stream creates a highway for correlated errors to propagate without passing through activations. This could negate the spectral contraction mechanism that the linear-activation test identified as the primary driver. This is the single biggest risk for macro validation.

2. **Attention layers.** The experiment uses only dense layers (W @ h). Real transformers interleave attention and FFN. Attention computes data-dependent weights that could selectively amplify perturbations aligned with high-attention subspaces. A correlated perturbation (same direction at every layer) might be more likely to trigger such amplification than a random one.

3. **Structured base weights.** Pre-trained model weights have highly non-random spectral structure. The "spectral contraction" mechanism may behave differently when the base weight spectrum has a few dominant singular values (as is typical in trained models) versus the isotropic random spectrum used here.

4. **Real LoRA correlation structure.** The uniform-rho model (same correlation at every layer pair) may not match real expert correlation patterns. Some experts may show high correlation in embedding layers but low correlation in output layers, or vice versa.

None of these are blocking for micro -- they are exactly what macro validation should test.

## Verdict

**PROCEED**

The experiment is well-designed, the code is correct, the results are clearly presented, and the kill criteria are appropriately evaluated. The core finding -- sub-additivity is robust to inter-layer correlation -- is supported by the data at d=32-128 and plausible at higher d.

Two items to address in a documentation pass (not blocking):

1. Remove or qualify the universal inequality claim in MATH.md Section 5.1 ("amp_ratio(rho) <= amp_ratio(rho=0) <= 1.0 for all tested configurations"). The d=256 data violates this for output deviation ratio. The amp_ratio itself is flat-to-slightly-decreasing, but the strong "for all" phrasing is inaccurate for the dev ratio.

2. Downgrade the consistent masking explanation (Mechanism A) and lead with the spectral/rank-1 argument, since the linear activation test shows masking is not the primary driver. The 0.5^24 calculation in MATH.md is misleading.

The experiment successfully resolves the open risk flagged by the parent. Status PROVEN is justified: K2 triggered, K1 not triggered, sub-additivity is robust.
