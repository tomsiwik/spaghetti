# Peer Review: RMSNorm Gamma Non-Uniformity

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that a manual review suffices. The core question (does per-dimension gamma break alpha scale-invariance?) has a clean theoretical answer and the empirical tests are direct.

## Mathematical Soundness

### What holds

1. **The cancellation argument is correct in the linear regime.** If the network were purely linear, gamma would be a fixed linear transformation applied identically to all three forward passes (all-experts, naive-removed, GS-removed). The ratio ||u_L|| / ||h_L|| would be exactly invariant to gamma. This is a clean argument and the derivation in MATH.md Section 3.3 is correct.

2. **The Jacobian analysis is correct.** The derivative of RMSNorm(h; gamma) with respect to h (MATH.md line 75) is standard and verified:
   d/dh[gamma * h / rms(h)] = diag(gamma) * (1/rms(h)) * (I - h h^T / (d * rms(h)^2))
   This is the standard RMSNorm Jacobian with a diagonal gamma factor.

3. **The first-order perturbation analysis is sound.** Both signal and perturbation propagate through the same Jacobian at the same operating point. Since gamma is input-independent, it appears identically in both paths. The residual 1.43x effect from GELU nonlinearity is a legitimate second-order correction.

### What is questionable

4. **Section 3.4's argument against systematic gamma-perturbation correlation is hand-wavy.** The claim that "the perturbation direction depends on the weight-space error, not on gamma" is true, but the claim that "gamma is learned to optimize training loss, not to selectively amplify error directions" is not a proof -- it is an appeal to intent. In production models, gamma values are correlated with the learned weight structure, and LoRA adapters may preferentially modify dimensions that the base model emphasizes (which gamma encodes). The paper acknowledges this in Assumption 4 and Limitations Section 2, which is appropriate, but the PAPER.md's claim that "the adversarial concern is fully resolved" (Section 2) is too strong. The concern about non-uniform gamma is partially resolved; the correlation concern remains open.

5. **The proof structure elides the nonlinearity.** The theoretical argument says gamma cancels exactly in the linear case, then waves at GELU creating a "small" residual. But the bound on this residual (1.43x) comes purely from empirical measurement at d=64, not from theory. There is no analytical bound on the gamma correction factor. The paper would be stronger if it acknowledged that the 1.43x is an empirical observation at micro scale, not a proven upper bound.

## Novelty Assessment

This is a follow-up validation experiment, not a novelty claim. It directly addresses the adversarial review of the parent experiment (alpha_residual_scaling_ablation). This is the correct research practice -- addressing reviewer concerns with targeted experiments.

No prior art issues. The question "does RMSNorm gamma break composition safety?" is specific to the SOLE architecture and has not been studied elsewhere.

## Experimental Design

### Strengths

1. **Comprehensive gamma sweep.** The experiment tests log-normal (8 sigma values), bimodal (4 configurations), and layer-wise (5 profiles) gamma distributions. This covers a wide range of realistic and adversarial gamma profiles.

2. **Multi-seed averaging.** Three seeds per configuration reduces noise. The baseline alpha standard deviation across seeds is visible in the raw results (0.019-0.025 range).

3. **Target-scale validation.** Test 4 at d=256, N=50 directly tests K2 with the claimed worst-case gamma. This is the right test for the stated kill criterion.

4. **Appropriate kill criteria.** K1 (<2x) and K2 (<5%) are reasonable thresholds that match the parent experiment's framework.

### Weaknesses

5. **Only 3 seeds is thin for ratio claims.** The baseline alpha at d=64 varies from 0.019 to 0.025 across seeds (a 1.3x ratio just from seed variance). The claimed 1.43x worst-case ratio is only 1.1x above the seed noise floor. More seeds (10+) would strengthen the claim that 1.43x is a real gamma effect rather than a seed-gamma interaction artifact.

   However: this is a micro-scale experiment and 3 seeds is standard practice within the project. The 51x margin on K2 absorbs this concern.

6. **The gamma profiles are synthetic, not extracted from real models.** The paper acknowledges this (Limitation 4). Log-normal with sigma=0.5 is a reasonable proxy for production gamma distributions, but extracting gamma values from a real Qwen2.5-0.5B checkpoint would take minutes and would have been definitive. This is a missed opportunity, not a fatal flaw.

7. **GELU vs SiLU.** The experiment uses GELU; production Qwen2.5 uses SiLU. The paper acknowledges this (Limitation 1) and argues similar saturation properties. This is plausible but unverified. Since the gamma correction factor comes entirely from the nonlinearity interaction (the linear case gives exact invariance), this matters more than it might seem. A single SiLU comparison run would have been cheap insurance.

8. **K2 threshold mismatch with kill criteria text.** The HYPOTHESES.yml kill criterion K2 says "gamma=5 on 25% of layers" but the experiment tests "gamma=5 on 25% of dimensions per layer" (bimodal profile) and "6 layers at gamma=5" (layer-wise profile). These are different conditions. The bimodal test is actually closer to the spirit of the concern, but the exact K2 criterion as stated is not precisely what was tested. The layer-wise test (early_high_5.0) at d=64 gives 1.31x ratio, which passes K1, but was not run at d=256 for K2.

   This is a minor issue because the bimodal test at d=256 gives 0.098% deviation (51x margin), making the specific gamma profile irrelevant at that margin.

## Hypothesis Graph Consistency

The experiment correctly depends on exp_alpha_residual_scaling_ablation and addresses that experiment's adversarial review concern. The kill criteria match the HYPOTHESES.yml entry. The status change to PROVEN is justified given both K1 and K2 pass.

One note: the HYPOTHESES.yml blocks field is empty. This experiment should arguably unblock any downstream nodes that were waiting on "macro-readiness of the alpha bound," but since no such explicit dependency exists, this is fine.

## Macro-Scale Risks (advisory)

1. **Gamma-perturbation correlation at production scale.** The one risk this experiment cannot address: if real LoRA experts trained on Qwen2.5 systematically modify high-gamma dimensions, the correction factor could exceed 1.43x. This should be tested during the macro held-out eval by extracting real gamma values and measuring the cosine between gamma magnitude and expert delta magnitude per dimension.

2. **SiLU activation.** A single comparison run with SiLU would close this gap before macro. Cost: approximately zero (swap one function).

3. **The 51x margin is the real safety net.** Even if the gamma correction factor turns out to be 5x at production scale (which would require a systematic gamma-perturbation correlation that seems unlikely), the bound would still pass K2 with a 10x margin. The architecture is robust to this concern.

## Verdict

**PROCEED**

The experiment cleanly addresses the adversarial concern from the parent experiment. The theoretical argument is sound (gamma cancels in the linear case, with bounded nonlinear residual). Both kill criteria pass with large margins (1.43x vs 2x threshold; 0.098% vs 5% threshold). The 51x margin on K2 absorbs all remaining uncertainties (seed noise, SiLU vs GELU, synthetic vs real gamma, gamma-perturbation correlation).

Minor improvements that would strengthen but are not blocking:

1. Swap GELU for SiLU in one configuration to verify the nonlinearity interaction is similar (estimated effort: 5 minutes).
2. Extract real gamma values from Qwen2.5-0.5B checkpoint and run one configuration with those values (estimated effort: 15 minutes).
3. Soften the claim "adversarial concern is fully resolved" to "adversarial concern is resolved for uncorrelated gamma; correlation risk remains for macro validation."

None of these are blocking given the 51x margin on K2.
