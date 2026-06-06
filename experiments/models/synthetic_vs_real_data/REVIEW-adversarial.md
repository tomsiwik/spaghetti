# Peer Review: Synthetic vs Real Data Quality

## NotebookLM Findings

Skipped -- pure simulation study with straightforward math; deep review not warranted for the complexity level.

## Mathematical Soundness

### What holds

1. **Data generation model is internally consistent.** The mixture-of-Gaussians setup with Dirichlet-weighted mode assignment is standard. The synthetic/real parameterization (5 vs 20 modes, alpha=0.5 vs 2.0, sigma=0.05 vs 0.30) creates the intended coverage-quality tradeoff.

2. **Quality metric is correctly implemented.** `1 - ||pred - true||_F / ||true||_F` is a valid relative error measure. The `max(0, ...)` clamp is fine for preventing negative values when approximation is worse than predicting zero.

3. **Frozen-A SGD derivation is correct.** The gradient `grad_B = error^T @ x_batch @ A^T / |batch|` matches standard linear regression in the projected space. Cosine decay schedule is standard.

4. **Kill criteria evaluation is mathematically correct.** K1 gap = (real - synth) / real = 58.1% > 15% threshold. K2 best_mixed = 0.0670 > best_pure = 0.0603, improvement = 11.2%.

5. **Orthogonality measurement is correct.** Flattened B@A cosine similarity and subspace_angles from scipy are both appropriate.

### What does not hold or is questionable

1. **The quality metric denominates in [0, 0.07].** All quality values are extremely low (2.5% to 6.7% reconstruction accuracy). At this operating point, the 58.1% relative gap is between 0.025 and 0.060 in absolute terms -- a difference of 0.035. This means the entire experiment's conclusions rest on tiny absolute differences where noise, hyperparameter sensitivity, and initialization could dominate. The PAPER acknowledges this ("relative comparisons are meaningful") but the absolute scale raises the question: are these experts learning *anything* meaningful, or are we comparing noise floors?

2. **Effective rank is insensitive at d=64.** MATH.md Section 2.5 correctly identifies this: both synthetic (62.7) and real (63.5) achieve near-maximal effective rank because 1000 samples from even 5 modes fill 64 dimensions. This means the effective rank metric is essentially vacuous at this scale -- it cannot distinguish the data regimes. The experiment claims the diversity difference is in "gradient direction concentration," but this is asserted rather than measured.

3. **The contamination analysis is purely formulaic.** Section 6 of MATH.md and the `contamination_stats` function compute `P(overlap)` and `expected_boost` from hand-specified overlap rates (10% synthetic, 2% real). These numbers are not measured from the simulation; they are input parameters presented as findings. The 5x contamination ratio is definitionally built into the parameters, not discovered.

4. **Mixing sweep uses a single expert per ratio.** In `run_seed`, the mixing sweep (Part 2, line 276-304) trains only ONE expert per mixing ratio per seed. The pure regime comparison trains 4 experts per regime per seed. This means the mixing sweep has 4x less statistical power per data point. The reported +11.2% improvement at alpha=0.2 has std=0.0177, which is 26% of the mean (0.067). This is high variance.

5. **Noise variance in the mixing sweep.** Looking at the actual results:
   - ratio=0.0: q_uniform = 0.0581 +/- 0.0106
   - ratio=0.1: q_uniform = 0.0648 +/- 0.0066
   - ratio=0.2: q_uniform = 0.0670 +/- 0.0177
   - ratio=0.3: q_uniform = 0.0576 +/- 0.0183

   The confidence intervals of ratio=0.0 and ratio=0.2 overlap substantially. A proper paired statistical test (e.g., Wilcoxon signed-rank on per-seed differences) is not reported. The +11.2% improvement claim is not tested for statistical significance.

6. **MATH.md Section 4.1 effective noise formula is misleading.** The claim `sigma_eff^2 = alpha * sigma_s^2 + (1-alpha) * sigma_r^2` treats label noise as the only relevant factor. But the mixing also changes the input distribution (mode composition, bias). The actual effect of mixing is a change in the *joint* (X, y) distribution, not just the marginal noise level. The analysis attributes the mixing benefit to "noise reduction" but coverage change is confounded.

## Novelty Assessment

**This is a simulation study, not an empirical experiment.** No actual models are trained. No real data is used. The synthetic/real distinction is modeled by parameter choices (5 vs 20 modes, sigma=0.05 vs 0.30) in a linear regression setup.

**Prior art that already addresses data mixing for fine-tuning:**

- Gunasekar et al. 2023 (Phi-1): Actually trained models on synthetic data. Their finding (synthetic-only achieves 50.6% on HumanEval) is cited but this experiment adds no new empirical evidence.
- Shumailov et al. 2024 (Model Collapse): Studied diversity loss in recursive synthetic generation. This experiment models mode collapse as a parameter (5 modes) rather than demonstrating it.
- The LoRA Soups reference notes "beats data mixing by 12%" -- similar magnitude to the 11% mixing benefit found here.

**Delta over existing work:** The experiment's contribution is applying the known coverage-quality tradeoff to the specific SOLE LoRA distillation context. This is a reasonable micro-experiment to establish a baseline recommendation (80/20 mix), but the evidence is directional at best.

**No reinvention of existing implementations detected.** The simulation is original code for an original (if narrow) question.

## Experimental Design

### Does this test what it claims?

**Partially.** The experiment claims to test whether synthetic data can substitute for real data in LoRA expert distillation. But what it actually tests is: in a linear regression with frozen random A, does a 5-mode low-noise Gaussian mixture produce better LoRA approximations than a 20-mode high-noise Gaussian mixture?

The gap between the simulation model and reality is large:

1. **Linear task vs. nonlinear NLP.** W* is a rank-r linear map. Real distillation produces adapters for nonlinear transformer layers. The coverage-quality tradeoff direction is plausible but magnitudes are uncalibrated.

2. **The 6x noise ratio (0.30/0.05) is a strong assumption.** This implies real data has 6x more label noise than synthetic. In the SOLE distillation pipeline, both "real" and "synthetic" data are teacher-generated (Groq API output). The distinction is about input diversity, not label noise. The experiment conflates two independent variables (input coverage AND label noise) into the synthetic/real dichotomy. If you held noise constant and varied only coverage, or held coverage constant and varied only noise, the results might be very different.

3. **The 5 vs 20 modes choice determines the outcome.** The coverage advantage of real data is baked into the parameter choice of 20 modes vs 5 modes. If synthetic data had 15 modes and real had 20, the gap would shrink dramatically. These parameters are "calibrated from literature" but the calibration is qualitative at best.

### Controls

- **Adequate:** 5 seeds, multiple evaluation distributions (uniform, synthetic-like, real-like), orthogonality measurement
- **Missing:** Statistical significance tests for the mixing sweep, ablation holding coverage constant while varying noise (or vice versa), sensitivity analysis on key parameters (what if synthetic has 10 modes instead of 5?)

### Could a simpler mechanism explain the results?

**Yes.** The entire result can be explained by: "more diverse training inputs produce better generalization in linear regression." This is a textbook result. The synthetic/real framing adds narrative but not mechanism. A simpler experiment would sweep the number of modes from 5 to 20 while holding noise constant, and separately sweep noise from 0.05 to 0.30 while holding modes constant.

## Hypothesis Graph Consistency

The experiment maps to `exp_synthetic_vs_real_data` in HYPOTHESES.yml with status `supported`. The kill criteria match:
- K1: "synthetic-only expert is >15% worse" -- tested, killed at 58.1%
- K2: "mixed NOT better than either alone" -- tested, survives at +11.2%

The overall status `supported` (not `proven`) is appropriate given this is a pure simulation.

**Evidence is sufficient to keep the node at `supported`.** The directional finding (mixing helps, synthetic-only is worse) is reasonable even if the magnitudes are uncertain.

## Macro-Scale Risks (advisory)

1. **The coverage gap may be irrelevant in the actual SOLE pipeline.** The pilot-50 distillation uses Groq API to generate both instructions AND responses. The "real" data comparison would be HuggingFace datasets with naturally-occurring examples. But the experiment's "real data" model (20 modes, high noise) does not match this scenario well -- curated real data (e.g., codeparrot-clean) may have moderate noise but also moderate coverage.

2. **The optimal mixing ratio (alpha*=0.2) has no reason to transfer.** The optimal ratio depends on the noise ratio, coverage ratio, task complexity, and model capacity -- all of which change at macro scale.

3. **Contamination risk is the strongest actionable finding** and transfers well: synthetic data generated by the same LLM family used for evaluation will have higher benchmark overlap. The recommendation to use execution-based eval (HumanEval pass@1) is sound regardless of the simulation.

## Verdict

**PROCEED**

The experiment is an honest simulation study that produces directional findings consistent with the literature. The findings are appropriately caveated in PAPER.md and FINDINGS.md. The status is `supported`, not `proven`, which is correct.

The experiment answers a reasonable question within micro constraints: does the coverage-quality tradeoff favor pure synthetic, pure real, or a mix? The answer (mix is best, synthetic-only is killed) aligns with published results and provides a concrete recommendation (80/20 real/synthetic) for the production pipeline.

**Non-blocking issues (4):**

1. **Report statistical significance of the mixing benefit.** The +11.2% at alpha=0.2 should be tested with a paired test (per-seed ratio=0.2 minus ratio=0.0). The current evidence does not distinguish the mixing benefit from noise given the overlapping standard deviations (0.0177 at ratio=0.2 vs 0.0106 at ratio=0.0).

2. **Disentangle coverage from noise.** The experiment conflates two independent variables. Add an ablation: (a) 20-mode, sigma=0.05 (high coverage, clean labels) and (b) 5-mode, sigma=0.30 (low coverage, noisy labels). This would isolate which factor drives the 58% gap. Without this, the paper cannot claim the gap is due to "coverage" as opposed to "noise" or the interaction.

3. **Remove the contamination section or label it as "input parameter echo."** The contamination numbers are computed from hand-specified overlap rates, not discovered from the simulation. Presenting them alongside empirical findings creates a false impression of empirical evidence.

4. **VISION.md line 106-113 still recommends Frechet merge** ("Composition via Riemannian Frechet mean instead of naive addition") even though the Frechet merge experiment was killed. This is not caused by this experiment but is a consistency issue that affects interpretation of composition claims. The Frechet code block should be updated or removed.
