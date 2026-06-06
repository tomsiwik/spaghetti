# Peer Review: Warmup Fraction Sensitivity (Exp 20)

## Mathematical Soundness

### Step-by-step verification of MATH.md

**Section 2.1 -- Schedule Definition**: Correct. The warmup+cosine schedule is standard. The eta(50) values in the table check out:
- f_w=0.01: S_w=32. At s=50, warmup is over (50 > 32), so cosine phase applies: eta(50) = (3e-3/2)(1 + cos(pi*(50-32)/(3200-32))) = (1.5e-3)(1 + cos(0.0178pi)) ~ 3.0e-3. Correct.
- f_w=0.02: S_w=64. At s=50, still in warmup: eta(50) = 3e-3 * 50/64 = 2.34e-3. Correct.
- f_w=0.05: S_w=160. eta(50) = 3e-3 * 50/160 = 9.375e-4. Correct.

**Section 3.2 -- Cumulative LR integral derivation**: This is the core mathematical contribution.

For S_w >= 50:
```
sum_{s=1}^{50} eta(s) = eta_peak/S_w * sum_{s=1}^{50} s = eta_peak/S_w * 1275
```
This is correct (50*51/2 = 1275). The suppression factor F = 1275/(50*S_w) = 25.5/S_w checks out.

For S_w < 50 (only applies to f_w=0.01, S_w=32):
```
sum = eta_peak/S_w * (S_w*(S_w+1)/2) + eta_peak*(50-S_w)
    = eta_peak * ((S_w+1)/2 + 50 - S_w)
    = eta_peak * (50 - (S_w-1)/2)
```
Step-by-step: warmup portion = eta_peak/32 * (32*33/2) = eta_peak * 16.5. Post-warmup portion = eta_peak * 18 = 18*eta_peak. Total = 34.5*eta_peak. The formula gives eta_peak*(50 - 31/2) = eta_peak * 34.5. Correct.

F = (50 - (S_w-1)/2)/50 = (50 - 15.5)/50 = 0.69. Matches the table.

**Numerical predictions vs. actuals (Table in Section 3.2)**:
- f_w=0.01: F=0.69, predicted = 16.5% + 0.69*(53.5%-16.5%) = 16.5% + 25.5% = 42.0%. Actual: 42.8%. Error +0.8pp.
- f_w=0.02: F=0.40, predicted = 16.5% + 0.40*37.0% = 31.3%. Actual: 31.1%. Error -0.2pp.

Wait -- the PAPER.md reports the prediction uses delta_0 and delta_spike derived from the constant-LR control. The constant baseline death@50 is 53.5% (from PAPER.md), not 51.6% (from MATH.md). The MATH.md prediction table uses 51.6% (from Exp 19), but the actual experiment measured 53.5% for constant LR at S=50. The code at line 478-488 correctly uses the actual base_rate and const_50 from the current experiment's data, not the Exp 19 numbers.

**Issue**: MATH.md Section 3.2 predictions (column "Predicted delta(50)") use 51.6% as the constant baseline (from Exp 19), but the actual experiment measured 53.5%. The PAPER.md prediction validation table shows different predicted values (42.0% vs MATH.md's ~39%), suggesting the paper correctly recalculated using actual baselines while MATH.md was written a priori. This is not an error per se -- MATH.md predictions are pre-experiment and PAPER.md validates with actual data -- but the discrepancy is confusing and should be clarified.

**Hidden Assumption -- Linear Death-vs-LR**: The model assumes P(death at step s) ~ eta(s) * C with C constant. This is a strong linearization. The 0.6pp mean error is suspiciously good for such a simple model. However, looking at the data more carefully: the model has one free parameter implicitly -- it uses the constant-LR outcome as calibration. So it is really a one-parameter model (C is fit from the constant baseline), predicting 5 data points. The goodness of fit is genuine but should be described as "one-parameter calibrated" rather than "zero-parameter prediction."

### Code vs. MATH.md Consistency

**Schedule implementation** (test_warmup_sweep.py line 64-81): The `make_warmup_cosine_schedule` function uses `optim.linear_schedule(0.0, peak_lr, steps=warmup_steps)` joined with `optim.cosine_decay`. This matches MATH.md Section 2.1 exactly.

**Minor code issue**: Line 75 uses `warmup_steps = max(1, int(total_steps * warmup_frac))` while MATH.md uses `S_w = floor(f_w * S_total)`. For f_w=0.01, int(3200*0.01) = 32, max(1,32) = 32. Same result. The `max(1, ...)` guard only matters for f_w so small that floor gives 0, which is not tested. No discrepancy for the tested fractions.

**Warmup start from 0 vs. small epsilon**: The code uses `optim.linear_schedule(0.0, peak_lr, ...)`, starting LR from exactly 0. MATH.md defines eta(0) = 0 implicitly (eta(s) = eta_peak * s/S_w at s=0). Consistent.

## Novelty Assessment

### Prior Art

The PAPER.md cites relevant work:
1. **"Why Warmup the Learning Rate?"** (NeurIPS 2019) -- warmup prevents catastrophic drift
2. **"Analyzing and Reducing the Need for Learning Rate Warmup"** (NeurIPS 2024) -- correlates no-warmup training with permanently dead ReLUs
3. **Gurbuzbalaban et al. (2024), "Maxwell's Demon"** -- revival during LR decay

The NeurIPS 2024 paper is the most relevant prior art. It directly connects warmup absence with ReLU death. What Exp 20 adds is:
- A quantitative dose-response curve (5 warmup fractions)
- The R = S_w/T_spike framework for predicting critical warmup threshold
- Validation of a simple cumulative-LR-integral model

This is a reasonable delta over the prior art -- it goes from "warmup prevents ReLU death" (binary) to "how much warmup" (continuous). The critical-ratio framework is a useful conceptual tool.

### No reinvention of existing code

The experiment correctly imports and reuses `make_lr_schedule`, `train_with_schedule`, `_compute_death_stats`, and `TOTAL_STEPS` from Exp 19. It adds only `make_warmup_cosine_schedule` for the new warmup fractions. Clean code reuse.

## Experimental Design

### Does it test the stated hypothesis?

**Yes.** The hypothesis is that warmup fraction (not just warmup vs. no warmup) matters for spike suppression, with a phase transition at R = S_w/T_spike ~ 1. The 5-point sweep with controls directly tests this.

### Controls

Two controls (constant LR, cosine-only) are carried forward from Exp 19. This is appropriate -- they establish the no-warmup baselines. The constant control serves as the denominator for all suppression calculations.

### Could a simpler mechanism explain the results?

The cumulative-LR-integral model IS the simple mechanism, and it fits. There is no need to invoke more complex explanations. This is a strength of the experiment -- the simplest possible model works.

### Statistical concerns

**3 seeds**: Standard for this project. The paper acknowledges "standard deviations are 3-10pp for some conditions." The qualitative findings (monotonic decrease, phase transition at R~1) appear robust across seeds.

**No confidence intervals on the prediction errors**: The 0.6pp MAE is reported without uncertainty. With 5 data points, a bootstrap or leave-one-out cross-validation would be informative but not strictly necessary at micro scale.

### Confound: warmup fraction and cosine phase duration

The PAPER.md Section "Micro-Scale Limitations" point 6 correctly identifies this: "Longer warmup means less time in the cosine decay phase." At f_w=0.20, the cosine phase is 2560 steps; at f_w=0.01, it is 3168 steps. This means the 20% warmup condition experiences a steeper cosine decay (faster LR decrease over fewer steps), which could independently reduce death through the revival mechanism identified in Exp 19. The paper acknowledges this confound but does not attempt to decompose it.

**Assessment**: This is a real confound but a minor one. The main finding (spike suppression at S=50) occurs before the cosine phase matters. The equilibrium finding (S=3200) is confounded, but the direction of the confound (steeper cosine helps 20% warmup) reinforces the main conclusion (more warmup = less death).

## Kill Criteria Evaluation

### Kill 1: All fractions within 5pp at S=50

Range = 42.8% - 17.2% = 25.6pp. Well above 5pp. **Correctly not triggered.**

### Kill 2: 1% warmup captures >90% of 10% benefit

Suppression at 1%: 53.5% - 42.8% = 10.7pp.
Suppression at 10%: 53.5% - 18.5% = 35.0pp.
Ratio: 10.7/35.0 = 30.6%.

**Correctly not triggered.** The 90% threshold is well-chosen -- it tests whether warmup fraction is practically irrelevant.

### Kill 3: Non-monotonic (inversion)

The code uses a 2pp tolerance for noise (line 396: `death_b > death_a + 0.02`). This is reasonable given the 3-10pp standard deviations. No inversions detected. **Correctly not triggered.**

**One issue with Kill 3**: The criterion tests monotonicity only at S=50. Looking at the full data table, at S=100 there is a potential concern: wc_20 shows 17.1% death at S=100, which is LOWER than its 17.2% at S=50 (a decrease). This is not non-monotonicity across warmup fractions (kill 3 tests across f_w, not across time), but it is noteworthy. It means 20% warmup death actually decreases slightly from S=50 to S=100, suggesting the spike for long warmup may not have peaked yet at S=50. For f_w=0.20, T_spike might effectively be longer. This does not invalidate the analysis but hints that T_spike is not truly fixed across warmup fractions.

## Hypothesis Graph Consistency

The experiment corresponds to VISION.md item 17 ("Warmup fraction sensitivity"). There is no explicit node in HYPOTHESES.yml for this experiment -- it is listed as item 17 in VISION.md "What Remains" but not in the HYPOTHESES.yml nodes. **Missing HYPOTHESES.yml entry.** This is a bookkeeping issue, not a scientific one.

## Integration Risk

### Composes with existing architecture

This experiment produces no architectural changes. It is purely an empirical characterization of a training hyperparameter's effect on a previously-measured phenomenon. It integrates cleanly as a refinement of Exp 19's macro predictions.

### No conflicts with existing components

The revised macro prediction (warmup-fraction-dependent death rates) updates the pruning yield estimates in VISION.md. No conflicts.

## Macro-Scale Risks (advisory)

1. **T_spike scaling is the critical unknown**: The entire R = S_w/T_spike framework depends on T_spike ~ 50 steps. If T_spike is fundamentally different at d=4096 (e.g., 5 steps or 500 steps), the warmup fraction prescriptions change completely. The paper correctly identifies this as the key macro uncertainty.

2. **SiLU activation eliminates hard death**: Qwen uses SiLU, not ReLU. SiLU neurons never reach exactly zero activation. The "death spike" may manifest as a "near-death spike" with magnitude-threshold behavior. The entire warmup sensitivity story may be qualitatively different for SiLU. The paper mentions this in "What Would Kill This" but could be more explicit about this being the primary macro risk.

3. **Frozen attention assumption**: All experiments freeze attention during fine-tuning. Full fine-tuning (which is what LLM pre-training does) creates larger distribution shifts. The paper notes this but the interaction with warmup fraction is unexplored.

4. **Warmup fraction and total training steps**: At macro scale, total training steps are orders of magnitude larger (300K+). The relationship between warmup fraction and absolute warmup steps changes dramatically. At S_total=300K with f_w=0.001, S_w=300 steps, which is 6x the micro T_spike. The micro finding that "1% warmup is insufficient" may not transfer because the absolute number of warmup steps is what matters (S_w vs T_spike), not the fraction.

## Overclaims

1. **"Best theory-experiment agreement in the project" (0.6pp MAE)**: This is a one-parameter model calibrated from the constant-LR control, predicting 5 data points that vary monotonically. The fact that a linear interpolation between two known endpoints fits well is less impressive than presented. However, the model does make specific quantitative predictions that could have failed (e.g., the curvature could have been wrong), so the claim is not vacuous -- just somewhat overstated.

2. **"Can predict death rates for arbitrary warmup schedules"**: This is an extrapolation claim from a linear model validated on one family of schedules (linear warmup + cosine). Non-linear warmup (e.g., exponential) or multi-phase schedules might violate the linear cumulative-LR-integral assumption. The claim should be qualified.

3. **Chinchilla prediction (40% dead)**: The paper predicts Chinchilla's 0.33% warmup yields ~40% dead neurons. This is an extrapolation beyond the tested range (minimum 1% warmup) using a model that has not been validated below 1%. The linear relationship between cumulative LR and death might not hold at very short warmup where the transition from warmup to full LR is extremely abrupt (effectively a step function). Should be flagged as speculative.

## Verdict

**PROCEED**

The experiment is well-designed, mathematically sound, and produces clear results. The 5-point warmup sweep with two controls is a clean parametric study. The cumulative-LR-integral model provides genuine predictive power. Kill criteria are well-defined and honestly evaluated. The code correctly implements the MATH.md specification. Prior art is adequately cited.

The findings are useful for the project: they refine Exp 19's macro prediction from a single point estimate to a warmup-fraction-dependent curve, and they provide a practical framework (R = S_w/T_spike) for predicting warmup sufficiency.

### Minor revisions recommended (not blocking):

1. **Clarify MATH.md prediction baseline**: The predicted death@50 values in MATH.md Section 3.2 use the Exp 19 constant baseline (51.6%), while PAPER.md validation uses the actual Exp 20 baseline (53.5%). Add a note explaining this discrepancy as "pre-experiment predictions vs. post-experiment validation."

2. **Add HYPOTHESES.yml node**: Create an entry for this experiment in the hypothesis graph.

3. **Qualify the "arbitrary schedule" prediction claim**: The cumulative-LR-integral model is validated for one schedule family. Extrapolation to non-linear warmup or schedules below f_w=1% should be flagged as speculative.

4. **Acknowledge the one-parameter nature of the prediction model**: The 0.6pp MAE is impressive but should note that the constant-LR data point effectively calibrates the model's single free parameter (C).

5. **Flag T_spike non-constancy**: The wc_20 data (death decreases from S=50 to S=100) suggests T_spike may depend on warmup fraction itself. If the spike is delayed when LR ramps more slowly, the R framework needs refinement at macro scale.
