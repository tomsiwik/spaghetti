# Peer Review: Residual + LayerNorm Error Dynamics

## NotebookLM Findings

Skipped -- experiment is straightforward enough that manual review suffices. The mathematical claims are standard transformer analysis and the code is short enough to verify directly.

## Mathematical Soundness

### What holds

1. **Architecture definitions are correct.** The seven forward functions accurately implement the stated equations. Pre-LN, Pre-RMSNorm, Post-LN, and residual all match their standard definitions.

2. **LayerNorm/RMSNorm implementations are correct.** Both are standard (no learnable parameters, which is acknowledged as a limitation).

3. **Jacobian analysis (Section 3.2) is qualitatively correct.** For the residual case, J_l^{res} = I + (1/sqrt(L)) * J_l^{ff} is accurate, and the eigenvalue argument (eigenvalues near 1 + lambda_i/sqrt(L)) follows. The claim that this reduces per-layer perturbation to O(eps/sqrt(L)) is sound.

4. **Dimension scaling analysis is methodologically sound.** Log-log regression on 4 data points (d=32,64,128,256) with R^2 > 0.98 across all architectures. Exponents in [-1.16, -0.92] are consistent with ~1/d scaling. K2 correctly killed.

5. **K1 assessment is correct in direction.** Residual connections do change amplification by >50%, and the change is favorable. The paper correctly notes this strengthens (not weakens) the parent safety claim.

### Issues found

**Issue 1 (minor): The 1/sqrt(L) scaling is non-standard and biases the comparison.**

The code applies explicit `1/sqrt(L)` scaling to the residual branch output. This is NOT what production transformers do. Production models handle depth scaling through weight initialization (GPT-2: 1/sqrt(2L) on output projection init, Qwen/Llama: similar strategies). The explicit 1/sqrt(L) in the forward pass artificially suppresses the branch contribution at depth 24 by a factor of ~4.9x, which directly reduces the amplification ratio.

However, the paper acknowledges this in Assumptions (item 2) and argues the qualitative effect is the same. This is fair -- the point is that production models DO have mechanisms that scale down the branch contribution. But the specific amplification ratios (0.022, 0.045, etc.) should not be taken as precise predictions of production behavior. They are directionally correct: residual architectures dampen error relative to feedforward.

**Severity: Low.** The directional claim survives. The precise numbers are specific to this scaling choice.

**Issue 2 (minor): Power law fit on 4 points is fragile.**

Four data points (d=32,64,128,256) for dimension scaling regression. While R^2 values are high (>0.98), 4 points fitting a 2-parameter model (C, alpha) leaves only 2 degrees of freedom. The confidence intervals on the exponents are not reported. With standard errors from `scipy.stats.linregress` available in the code but not reported, this is a missed opportunity to show whether the exponents are statistically distinguishable from -1.0.

**Severity: Low.** The directional claim (error decreases with d) is robust. The exact exponent matters less than the sign and rough magnitude.

**Issue 3 (medium): The "90x below random" SOLE cosine correction is unjustified in this paper.**

The extrapolation `dev(SOLE) ~ 0.033% / 90 ~ 0.0004%` applies a 90x correction factor from a different experiment (structural_orthogonality_proof). The linearity of this correction is assumed but not demonstrated. Error propagation through nonlinear layers (activation + normalization) means that reducing the per-layer weight perturbation by 90x does NOT necessarily reduce output deviation by 90x. The relationship could be sub-linear or super-linear depending on the operating point.

The paper should cite the specific source of the 90x factor and note that the linear extrapolation is an assumption. At the micro scale, the actual SOLE-cosine regime was not tested in this experiment.

**Severity: Medium.** The qualitative conclusion (production is very safe) likely holds, but the specific 0.0004% number is weakly supported.

**Issue 4 (negligible): Depth scaling R^2 values for sub-additive trend are low.**

The depth scaling regressions (amp_ratio vs L) have R^2 values of 0.55-0.74 for the residual architectures. This is adequate for showing a negative trend but the "sub-additive" characterization is noisy. The paper does not over-claim here, so this is just a note.

### Mathematical gaps (acceptable at micro scale)

- No formal error bounds, only empirical measurements. This is fine for micro.
- The Jacobian analysis in Section 3.3 (LayerNorm projection) uses the correct formula but does not formally prove the amplification mechanism. The empirical data (amp_ratio=3.41) speaks for itself.

## Novelty Assessment

**Low novelty, but that is acceptable.** The individual facts (residual connections stabilize training, Pre-LN is better than Post-LN, LayerNorm without residual is unstable) are well-known in the transformer literature (Xiong et al. 2020, He et al. 2016). The paper correctly cites these.

The specific contribution is applying these known facts to the expert removal error propagation question. This is an incremental but necessary step in the SOLE research program. The parent experiment (multilayer_removal_cascade) used feedforward-only, and this experiment validates that the parent's conclusions were conservative when extended to production architectures.

**No reinvention detected.** The paper builds on the parent experiment's framework and extends it with standard transformer components.

## Experimental Design

**Strengths:**
1. Seven architectures tested systematically with the same framework -- good controlled comparison.
2. Three random seeds for each configuration.
3. Both near-orthogonal and clustered (cos~0.3) expert regimes tested.
4. Dimension scaling test (4 values) directly addresses K2.
5. Depth scaling (7 values) provides depth-amplification curves.

**Weaknesses:**

1. **Only one remove_idx tested (N//2 = 4).** The parent experiment likely tested multiple removal indices. Removing expert 4 (middle of the GS ordering) may not be representative. Early experts (index 0) are less modified by GS orthogonalization; late experts (index 7) are more modified. This could affect the amplification ratio. However, this is inherited from the parent experiment's design.

2. **No variance/CI reported in summary tables.** The paper reports mean values across 3 seeds but does not report standard deviations in the summary tables (only in the aggregate analysis section). The summary table in PAPER.md (Test 1) appears to show means but this is not explicitly stated.

3. **No statistical test for architecture differences.** With 3 seeds, a t-test or bootstrap CI comparing architectures would strengthen the K1 claim. The differences are large enough (11.5x) that this is unlikely to change the verdict, but it would be more rigorous.

**Does the experiment test what it claims?**

Yes. K1 asks whether residual connections change the amplification ratio by >50%. The experiment directly measures amplification ratios across architectures. K2 asks whether LayerNorm breaks 1/d scaling. The experiment directly tests dimension scaling with and without LayerNorm. Both kill criteria are cleanly addressed.

**Could a simpler mechanism explain the result?**

The 1/sqrt(L) scaling alone would reduce amplification by ~4.9x at L=24. The residual architecture shows 5.6x reduction (0.254 -> 0.045), which is very close to the 1/sqrt(L) effect. Pre-RMSNorm shows 11.5x reduction, so the additional ~2x from RMSNorm normalization is a real but smaller effect. The paper's narrative somewhat overstates the role of normalization vs. the simpler 1/sqrt(L) scaling factor.

However, the 1/sqrt(L) scaling IS part of the residual architecture's design (whether explicit or via initialization), so attributing the improvement to "residual connections" is fair if you include the scaling as part of the package.

## Hypothesis Graph Consistency

- **Status: proven** -- appropriate. K1 triggered (favorably) demonstrating that dynamics ARE different. K2 killed, demonstrating 1/d scaling is preserved. Both kill criteria resolved.
- **blocks: []** -- correct. This is a terminal safety validation, not a blocker for other experiments.
- **depends_on: exp_multilayer_removal_cascade** -- correct lineage.

## Macro-Scale Risks (advisory)

1. **Attention mechanism.** This experiment uses linear layers only. Attention's softmax nonlinearity has different Lipschitz properties. The paper acknowledges this but claims the residual connection around attention blocks provides the same benefit. This is plausible but unverified.

2. **Learned gamma/beta parameters.** Production LayerNorm has learnable scale and shift. If gamma values are large (>1), they amplify the branch output, partially counteracting the 1/sqrt(L) scaling. This is worth checking at macro scale.

3. **Multi-head attention + FFN sub-blocks.** Real transformer layers have TWO residual connections (one around attention, one around FFN). This provides MORE identity shortcuts than the single-residual model tested here, likely making production even safer.

4. **The 0.0004% extrapolation.** As noted above, the linear correction from SOLE cosines is an assumption. Macro experiments should measure actual output deviation when removing a real expert from a composed Qwen model, not extrapolate from micro.

## Verdict

**PROCEED**

The experiment is well-designed, the code correctly implements all seven architectures, and both kill criteria are cleanly resolved. The core finding -- that production transformer architectures (Pre-LN, Pre-RMSNorm) are significantly safer than the feedforward model used in the parent experiment -- is sound and directionally robust.

The specific amplification ratios depend on the 1/sqrt(L) scaling convention, but the ordering and the directional conclusion are reliable. The 1/d scaling preservation across all architectures (K2) is convincingly demonstrated.

Minor improvements that do not block PROCEED:
1. Report confidence intervals on the dimension scaling exponents (data is available from linregress).
2. Note that the "90x SOLE cosine correction" to 0.0004% is an untested linear extrapolation, not a measured result.
3. Clarify that the 1/sqrt(L) scaling contributes significantly to the amplification reduction (it accounts for ~5x of the 11.5x improvement for Pre-RMSNorm).
