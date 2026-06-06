# Peer Review: LoRA Scale Ablation

## Experiment Type
Verification -- MATH.md contains Theorem 1 with proof and QED, making quantitative predictions (P1-P5) about when LoRA scale causes overwrite.

## Hack Detector
- Fix count: 0 (this is diagnostics, not a fix)
- Is MATH.md a proof or a description? Proof with QED -- Theorem 1 proves a cosine bound on the perturbation-to-base ratio. The proof is genuine (Cauchy-Schwarz based bound on output cosine similarity).
- Metric used as evidence: rho (perturbation-to-base Frobenius norm ratio), GSM8K accuracy, MMLU accuracy
- Kill criteria source: Derived from proof predictions (P1, P2)

## Self-Test Audit
1. One-sentence impossibility property: "Keeping rho < 1 ensures LoRA cannot dominate base." -- Acceptable, genuinely one property.
2. Cited theorems: Cauchy-Schwarz, isotropic energy ratio, Hu et al. 2022. All real; isotropic energy ratio is standard RMT. Conditions are stated (isotropic input). **Pass.**
3. Predicted numbers: P1-P5 with specific rho values per scale. **Pass.**
4. Falsification condition: "If scale=20 does NOT degrade benchmarks, the isotropic assumption is wrong." **Pass** -- this correctly targets the proof's central assumption.
5. Hyperparameter count: 0 (discovering correct value of existing parameter). **Pass.**
6. Hack check: "Removing a confound, not adding a fix." **Pass.**

Self-test is complete and honest. No blanks or evasions.

## Mathematical Soundness

### What Holds
- Theorem 1 is mathematically correct. The Frobenius norm energy ratio E[||Mx||^2] = ||M||_F^2 * E[||x||^2/d] for isotropic x is standard. The cosine lower bound cos(theta) >= 1/(1+rho) follows from triangle inequality applied correctly. The QED is legitimate.
- The framework (rho as diagnostic) is conceptually sound and well-grounded in Hu et al. 2022.

### What Does Not Hold
**Flaw 1: Norm estimates were wrong by an order of magnitude, but the proof structure is fine.** The predicted ||W||_F ~ 45 was actually 83.1 (1.8x), and predicted ||B^T A^T||_F ~ 5 was actually 0.6-1.2 (5-8x off). The combined error yields rho predictions ~15x too high. The paper correctly identifies this and treats the proof as falsified in its quantitative predictions. This is honest and appropriate.

**Flaw 2: The cosine bound in the proof is extremely loose.** cos(theta) >= 1/(1+rho) is a worst-case bound. At rho=0.14, this gives cos >= 0.88, which tells us almost nothing (it permits everything from near-identical to 28 degrees off). The proof cannot distinguish rho=0.01 from rho=0.14 in terms of behavioral impact. The "rho < 1 means perturbation regime" threshold is actually a *definition*, not a derived result.

**Flaw 3: The isotropic input assumption is acknowledged but not tested.** The proof's energy ratio depends on x being isotropic. Real activations in transformers are highly structured (ReLU/SiLU sparsity, layer-norm concentration, attention patterns). The ratio E[||Delta*x||^2]/E[||Wx||^2] could differ from rho^2 substantially. The paper never measures this ratio on actual activations -- only the weight-space rho.

### Subtle Issue: Training Confound in rho Measurement
MATH.md's prediction table assumes a FIXED ||B^T A^T||_F across scales, with rho scaling linearly as `scale * constant`. But the experiment trains SEPARATE adapters at each scale. The scale parameter affects gradient magnitudes during training, so B matrices evolve differently at each scale. Evidence from the data:

- Scale=1, delta_norm (mean across conditions) ~ 1.0
- Scale=20, delta_norm (mean) ~ 11.6
- If rho scaled linearly, we would expect delta_norm_20 = 20 * delta_norm_1 = 20.0

The actual ratio is 11.6, not 20.0. This means the B matrices at scale=20 converge to SMALLER values than at scale=1, likely because the larger effective learning rate causes the optimizer to settle in a different region. This is a real effect that the proof framework does not model. The proof treats scale as a post-hoc multiplier; the experiment applies it during training.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table. Honest assessment:

| Prediction | Result | Verdict |
|-----------|--------|---------|
| P1: scale<=2 degrades <=1/6 benchmarks | 50-67% of MMLU domains degraded at all scales | FALSIFIED (but see statistical note) |
| P2: scale=20 destroys base | GSM8K improves at scale=20 (0.523 vs 0.440 base) | FALSIFIED |
| P3: Composition "recovery" is dilution | Cannot test (rho << 1 everywhere) | MOOT |
| P4: Optimal scale differs single vs composed | Mild support (GSM8K peaks scale=8) | INCONCLUSIVE |
| P5: Monotonic degradation vs scale | No monotonic relationship observed | FALSIFIED |

The paper is honest about these falsifications. The core prediction (rho > 1 at scale=20) was wrong by 15x.

## Critical Statistical Concerns

**MMLU with n=20 is nearly useless as evidence.**

The 95% confidence interval for a binomial proportion at p=0.5, n=20 is +/- 0.219. This means:
- Base MMLU math = 0.550 (11/20) has 95% CI [0.331, 0.769]
- Any measurement between 0.33 and 0.77 is statistically indistinguishable from the base

The paper uses a 2% degradation threshold (MMLU drops > 0.02 from base = "degraded"). With n=20, the standard error of a proportion at p=0.5 is 0.112. A 2% threshold is 0.18 standard errors -- purely noise. **Every MMLU "degradation" claim in this paper is statistically meaningless at n=20.**

The "12/24 MMLU domains degraded at scale=1" finding? Random coin flips would produce the same pattern. The claim that "degradation is from domain specialization, not overscaling" is equally unsubstantiated -- the measurements cannot distinguish domain specialization from random variation.

**GSM8K with n=50 is borderline.** The standard error at p=0.5 is 0.071, giving 95% CI of +/- 0.14. The difference between scale=4 (0.573) and scale=20 (0.523) is 0.05, well within noise. Only the base-to-best comparison (0.44 to 0.66 at scale=4 SFT code) approaches significance (p ~ 0.02 by Fisher exact test for 22/50 vs 33/50).

## "Domain Specialization" as Explanation

The paper attributes all MMLU degradation to "domain specialization: training on medical hurts math/code MMLU." This explanation is plausible but not tested:

1. If domain specialization were real, the *trained* domain's MMLU should improve or hold while other domains drop. Looking at the data: s1.0_sft_medical has MMLU-medical = 0.550 (same as base 0.550), MMLU-math = 0.550 (same), MMLU-code = 0.500 (drop of 0.10). The medical adapter does NOT improve medical MMLU. The s1.0_ntp_medical adapter actually DROPS medical MMLU to 0.300 while RAISING math MMLU to 0.700. These patterns are incoherent with a "domain specialization" narrative -- they look like random noise at n=20.

2. No control was run: an adapter trained on random/shuffled data would establish the noise floor for MMLU degradation at n=20. Without this control, "degradation" cannot be distinguished from measurement noise.

## Data Integrity Concern

results.json explicitly lists 13 missing evaluations (all of s8.0 and s20.0, plus s4.0_ntp_code) with a status note "INCOMPLETE - log truncated." However, checkpoint.json contains complete data for all 30 conditions. PAPER.md reports numbers from checkpoint.json without noting the discrepancy. This should be disclosed: were the results.json entries never updated, or was there a data pipeline issue? The checkpoint.json data appears legitimate (includes timing data, memory measurements), but the inconsistency should be explicitly addressed.

## The GSM8K Inverted-U: Real Signal or Noise?

The most interesting pattern in this data is the GSM8K inverted-U:
- Scale=1: 0.487
- Scale=4-8: 0.573-0.593 (peak)
- Scale=20: 0.523

This is a real question: does scale=20 actually perform worse than scale=4-8? At n=50, the difference between 0.593 (scale=8) and 0.523 (scale=20) is 0.070. With SE ~ 0.071, this is about 1 standard error -- suggestive but not significant. A proper analysis would need n=200+ per condition, or a paired test on the same 50 questions across scales.

If this pattern is real, it contradicts the paper's conclusion that "scale doesn't matter much" and would suggest a genuine optimal range around scale=4-8. The paper dismisses this as "modest" but it could be the most important finding if confirmed with adequate power.

## "Prior Findings Validated" Is Too Strong

The paper concludes: "Prior findings are validated. Routing improvements, composition benefits, and adapter quality differences were real effects, not artifacts of overscaling."

This conclusion has a logical gap. The paper shows that rho < 0.15 at all scales, meaning scale=20 is not in the overwrite regime. This is a necessary condition for prior findings to be valid, but not sufficient. Prior findings could still be confounded by:
- Evaluation methodology (n=20 MMLU)
- Training data quality/quantity
- Prompt formatting effects
- Other hyperparameter choices

The correct conclusion is: "Scale=20 does not cause the base model to be overwritten, so prior findings are not invalidated by *this specific concern*." The stronger "validated" claim requires re-running the prior experiments under controlled conditions, which was not done.

## Novelty Assessment

The rho framework for diagnosing LoRA scale is not novel -- it is essentially the perturbation magnitude analysis from Hu et al. 2022 applied to a specific model. The empirical finding that Falcon-E-3B ternary weights have larger norms than estimated, and that 300-step LoRA updates are smaller than estimated, is useful calibration data but not a theoretical contribution.

The genuine value of this experiment is in providing *measured* norms for the Falcon-E ternary model, which calibrates the entire LoRA scaling question for this specific architecture. This is a service contribution, not a theoretical one.

## Macro-Scale Risks (advisory)

- The rho framework transfers directly to larger models; the key finding (measure norms, do not guess) is universally applicable
- At longer training (1000+ steps), B matrices grow more, and the rho values will be higher -- the scale=20 comfort zone may not persist
- The isotropic assumption becomes more questionable at scale as activation distributions become more structured

## Verdict

**REVISE**

The norm measurements and falsification of the overscaling hypothesis are genuine and valuable. The proof is mathematically sound. However, the paper makes claims that exceed what the evidence supports.

Required revisions:

1. **Add statistical power analysis.** Compute confidence intervals for all MMLU and GSM8K comparisons. Acknowledge that n=20 MMLU comparisons cannot distinguish signal from noise. Remove or heavily caveat all MMLU degradation claims.

2. **Add a noise-floor control.** Train one adapter on random data (or shuffled labels) and evaluate. This establishes the baseline MMLU variation from n=20 sampling alone.

3. **Downgrade "prior findings validated" to "prior findings not invalidated by overscaling."** The current claim is logically stronger than what the evidence supports.

4. **Address the training confound in rho.** Note that adapters trained at different scales produce different B matrices, so the rho prediction table (which assumes fixed ||B^T A^T||_F) is not the right model. The measured rho values at scale=20 (0.14) reflect both the scale multiplier AND the optimizer's response to that multiplier.

5. **Reconcile results.json and checkpoint.json.** Note which is the authoritative data source and why they differ.

6. **Investigate the GSM8K inverted-U.** Either run a power analysis showing n=50 is insufficient to detect the scale=8 vs scale=20 difference, or flag this as a provisional finding requiring follow-up with larger n.

**Finding status recommendation: provisional.** The norm measurement (rho < 0.15 at all scales) is the solid contribution. The behavioral claims are underpowered. The theoretical framework is correct in structure but was quantitatively falsified in its predictions, and the paper correctly identifies this. With the revisions above, this could be upgraded to supported.
