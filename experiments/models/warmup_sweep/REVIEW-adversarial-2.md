# Peer Review (2nd Pass): Warmup Fraction Sensitivity (Exp 20)

## Scope

This review focuses on issues the 1st reviewer missed. The 1st review was thorough and well-calibrated. I will not repeat its findings. Instead I go deeper on: (a) whether the theory actually explains the mechanism or just curve-fits, (b) statistical methodology, (c) code edge cases, (d) whether macro extrapolations are justified, and (e) an assessment of the 1st reviewer's concerns.

---

## 1. The Theory Is Curve-Fitting, Not Mechanistic -- And That Matters More Than The 1st Review Acknowledged

The 1st review noted the model is "one-parameter calibrated" (point 4 of minor revisions) and that the 0.6pp MAE is "somewhat overstated." I agree but think this understates the problem.

The cumulative-LR-integral model says: delta(50) = delta(0) + F * (delta_constant(50) - delta(0)). This is equivalent to saying "death at S=50 is a linear interpolation between the baseline and the constant-LR outcome, weighted by the ratio of cumulative LR integrals." This is a **scaling law**, not a mechanistic theory. It does not explain *why* death is proportional to cumulative LR -- it just observes that it is.

Why this matters: the paper claims (Finding 3) "we can predict death rates for arbitrary warmup schedules ... as long as we know T_spike and the LR integral over [0, T_spike]." This is an extrapolation claim from a scaling law, and scaling laws break at regime boundaries. Specifically:

- **Step-function warmup** (LR=0 for S_w steps, then jump to peak): The cumulative integral over [0,50] would be zero if S_w >= 50, predicting delta(50) = delta(0) = 16.5%. But a step-function schedule creates a sudden large gradient at step S_w that could cause a *delayed* spike -- the same spike compressed into fewer steps. The cumulative-integral model cannot predict this because it ignores gradient magnitude dynamics.

- **Exponential warmup** (LR = peak * exp(-alpha*(S_w - s)) for s < S_w): This front-loads LR early in warmup. The integral could be similar to linear warmup, but the gradient dynamics are different because the model sees near-peak LR earlier.

The model works for linear warmup because linear warmup smoothly distributes the LR integral across the spike window, and the death rate happens to scale linearly with this integral. This is a happy coincidence of the linear warmup shape, not a general principle.

**Recommendation**: Downgrade "can predict arbitrary schedules" to "predicts linear warmup + cosine accurately; generalization to other warmup shapes is a testable hypothesis, not a validated result."

---

## 2. The "Phase Transition at R~1" Is a Smooth Crossover, Not a Phase Transition

The paper repeatedly uses "phase transition" (MATH.md Section 3.1, PAPER.md Finding 2). Looking at the data:

| R | % of max benefit |
|---|-----------------|
| 0.64 | 31% |
| 1.28 | 64% |
| 3.20 | 90% |
| 6.40 | 100% |
| 12.80 | 104% |

This is a smooth, monotonically increasing function with no discontinuity, inflection point, or sudden change in slope. A true phase transition would show a sharp change in behavior at a critical point. What the data shows is simply diminishing returns -- as expected from any saturating function.

The cumulative-LR-integral model itself predicts this smooth behavior: F = 25.5/S_w (for S_w >= 50) is a hyperbola, which is smooth. There is no mathematical basis for calling R=1 a "phase transition."

This is not just terminology. Calling it a "phase transition" implies that there is a sharp boundary below which warmup is ineffective and above which it works. The data shows the opposite: even R=0.64 captures 31% of the benefit. There is no binary switch.

**Recommendation**: Replace "phase transition" with "crossover region" or "characteristic scale." R=1 is where warmup duration equals the spike timescale, which is a natural reference point, but not a critical threshold.

---

## 3. Statistical Methodology Issues the 1st Review Missed

### 3.1 No Standard Deviations Reported for Key Metrics

The PAPER.md reports 3-seed means for death rates at all checkpoints but does not report standard deviations in the main tables. The 1st review noted "3-10pp standard deviations for some conditions" but did not flag that the critical kill criterion evaluations lack uncertainty quantification.

For Kill 2: suppression(0.01)/suppression(0.10) = 31%. But what is the confidence interval? With 3 seeds, the standard error of each suppression estimate could be 3-5pp. A ratio of two quantities each with ~5pp uncertainty could easily range from 20% to 45%. The 31% point estimate is well below 90%, so this particular kill criterion is robust, but the paper should acknowledge the uncertainty.

### 3.2 Seed-Correlated Conditions Share the Same Base Model

All seven conditions (constant, cosine, five warmup fractions) within one seed start from the same pretrained base model. This means the conditions are not independent -- they share pretraining randomness. The inter-condition differences (which are the main findings) are **paired comparisons**, which is actually advantageous (lower variance), but the paper never acknowledges this statistical structure. A paired t-test or signed-rank test would be more appropriate than an independent comparison.

### 3.3 The Prediction Validation Has No Degrees of Freedom Analysis

The PAPER.md claims 0.6pp MAE across 5 data points. But the model has:
- delta(0): measured (not predicted)
- delta_constant(50): measured (not predicted, used as calibration)
- The model then predicts 5 points using one formula with no free parameters beyond these calibration points

The 1st review correctly noted this is "one-parameter calibrated." But more precisely: the model uses 2 measured anchors (delta(0) and delta_constant(50)) plus the known warmup schedule to predict 5 points. With 2 anchors and 5 predictions, there are effectively 5 degrees of freedom for validation (since the 2 anchors are from different conditions, not fitted). This is actually reasonable, and the 1st review was slightly overcritical in suggesting the fit is trivially good. The suppression factor F varies from 0.04 to 0.69 across the 5 conditions -- this is a wide range, and getting all 5 within 1pp is genuinely informative.

---

## 4. Code Edge Cases

### 4.1 The optim.linear_schedule Start-from-Zero Issue

The `make_warmup_cosine_schedule` function (line 77) creates `optim.linear_schedule(0.0, peak_lr, steps=warmup_steps)`. MLX's `linear_schedule(init, end, steps)` returns `init + (end - init) * min(step / steps, 1.0)`.

At step 0, this returns LR = 0.0. At step 1, LR = peak_lr/S_w. The MATH.md formula (Section 2.1) defines eta(s) = eta_peak * s/S_w, which at s=0 gives eta(0) = 0. These match.

However, the training loop starts at `step in range(1, steps + 1)` (line 130 of lr_schedule_death). MLX's optimizer step counter is internally tracked. After the first `optimizer.update()` call, the internal step becomes 1, so the LR for the first training step is eta_peak * 1/S_w, not 0. This is correct behavior -- the first gradient update uses a non-zero LR.

But there is a subtle issue: the **MLX optimizer step counter starts at 0 and increments after each update call**, or it may start at 1. This depends on the MLX implementation. If the first update uses step=0 (LR=0), then the first training step applies zero gradient, wasting a step. If it starts at step=1, the first step uses LR = peak/S_w, matching the math.

**Assessment**: This is implementation-dependent and I cannot verify without running the code. However, the excellent theory-experiment agreement (0.6pp) suggests the code and math are consistent, regardless of which convention MLX uses. If the first step were wasted (LR=0), the effective warmup would be shifted by one step, which is negligible for S_w >= 32. Not a blocking issue.

### 4.2 Independent Runs From Deepcopy -- No Training Trajectory Confound

Each checkpoint S is trained from a fresh `copy.deepcopy(base)` (lines 143, 210), not from the previous checkpoint. This means S=50 and S=100 are independent runs, not the same trajectory sampled at two points. This is the correct design for measuring "what happens if you train for exactly S steps with this schedule" but it does mean:

- The S=50 and S=100 results for the same condition are statistically independent (within one seed)
- You cannot infer the trajectory between checkpoints -- a neuron dead at S=50 in one run may never have been dead at S=50 in the S=100 run (different random batch sequences despite same seed)

Wait -- actually, the seed IS the same. The training seed is passed to `train_with_schedule`, and `rng = random.Random(seed)` (line 123). So the first 50 steps of the S=100 run see the exact same batches as the S=50 run. The S=100 run IS an extension of the S=50 trajectory. The paper does not state this explicitly. The deepcopy ensures the same starting weights, and the same seed ensures the same batch sequence. So the S=50 results ARE a prefix of the S=100 results.

This is fine for the analysis but means the data points across step counts are NOT independent within a condition and seed. The paper treats them somewhat as independent observations when plotting curves, but they are actually a single trajectory sampled at 8 points. This does not invalidate any finding but means you cannot treat the 8 step-count points as 8 independent observations for any statistical test.

---

## 5. Macro Extrapolation Issues the 1st Review Partially Missed

### 5.1 T_spike Is Defined Circularly for Macro Prediction

The paper's macro prediction framework is: "Measure T_spike at macro scale, then compute R = S_w/T_spike to determine warmup sufficiency."

But how do you measure T_spike at macro? You run training with constant LR and profile death at S=50 (the paper suggests this). But you need to know when to profile -- the spike could peak at S=5 or S=500 at macro. You would need a fine-grained sweep over early training steps, which is expensive. The R framework is useful retrospectively (once you know T_spike) but does not provide an actionable protocol for macro without a prior on T_spike's scale.

### 5.2 The Equilibrium Predictions Conflate Two Mechanisms

The paper provides a table (Finding 4) predicting equilibrium death rates for different training regimes (e.g., "LLM pre-training: ~40-44%"). These predictions extrapolate from micro equilibrium data, but at macro scale two additional mechanisms operate:

1. **Batch normalization / layer normalization effects**: At d=64, normalization statistics are noisy. At d=4096, they are stable. This changes gradient dynamics and potentially T_spike.

2. **Training duration**: Micro trains for 3200 steps. LLM pre-training runs for 300K+ steps. The revival dynamics measured at micro (still accelerating at S=3200) would have orders of magnitude more time to operate at macro. The equilibrium death rate at S=300K could be dramatically lower than at S=3200, even with constant LR. Extrapolating the S=3200 equilibrium to macro as if it were the final equilibrium is likely wrong.

The 1st review flagged T_spike scaling (correctly) and SiLU (correctly) but did not note the training-duration extrapolation issue.

### 5.3 The Chinchilla Prediction Is Doubly Extrapolated

The paper predicts Chinchilla's 0.33% warmup yields ~40% dead. This requires:
1. Extrapolating the warmup fraction below the tested range (minimum tested: 1%)
2. Assuming T_spike ~ 50 at Chinchilla scale

Even if T_spike were 50, Chinchilla's S_w = 5000 steps (0.33% of 1.5M). R = 5000/50 = 100. This would place Chinchilla solidly in the "strong suppression" regime (R >> 3), predicting ~17% dead, not 40%. The 40% prediction appears to assume T_spike scales proportionally with total training steps, which is never stated or justified.

Let me re-read the paper... The paper says: "Chinchilla-style 0.33% warmup: ~40%". Looking at the micro data: 0.33% of 3200 steps = 10.6 steps (S_w ~ 11). R = 11/50 = 0.22. This gives F ~ 1 - 10/100 = 0.90, predicting death ~ 16.5% + 0.90*37% = 49.8%. So the paper is computing R using MICRO's S_total=3200, not Chinchilla's actual S_total=1.5M.

This is the correct approach IF T_spike ~ 50 is architecture-dependent (not dataset/duration-dependent). But it reveals that the warmup FRACTION is meaningless -- what matters is the absolute number of warmup steps relative to T_spike. At Chinchilla's actual training setup, S_w = 5000, which gives R = 100, not R = 0.22. The paper's Table in Finding 4 conflates "warmup fraction" with "absolute warmup steps" by implicitly assuming micro's S_total=3200.

This is a significant conceptual issue. The paper should state clearly: "The macro prediction depends on absolute warmup steps (S_w) relative to T_spike, NOT on warmup fraction. At macro scale with S_total >> micro, even tiny fractions can give large S_w and strong suppression."

---

## 6. Assessment of 1st Reviewer's Concerns

### Valid and well-calibrated:
- **Point 1 (MATH.md baseline discrepancy)**: Correct. The a priori predictions use Exp 19's 51.6% while actuals use Exp 20's 53.5%. Worth a clarifying note.
- **Point 3 (qualify arbitrary schedule claim)**: Correct and I reinforce this above with the step-function argument.
- **Point 4 (one-parameter model)**: Slightly overcritical. As I argue in Section 3.3, the model genuinely predicts 5 data points from 2 measured anchors across a wide range of F values. The fit quality is informative.
- **Point 5 (T_spike non-constancy from wc_20 data)**: Insightful. The wc_20 death decrease from S=50 (17.2%) to S=100 (17.1%) is within noise, but the pattern across S=50->100->200 for longer warmup fractions (wc_10: 18.5->21.1->24.7, wc_20: 17.2->17.1->19.7) shows death *increasing* after S=50, not decreasing. This is not T_spike being different -- it is the warmup itself: at S=100, wc_10 has eta(100) = 3e-3 * 100/320 = 0.94e-3 (still ramping), while wc_20 has eta(100) = 3e-3 * 100/640 = 0.47e-3. The death increase from S=50 to S=200 for longer warmup fractions reflects the continued LR ramp creating new deaths, not a shifted spike. The 1st reviewer's interpretation (T_spike depends on warmup) is plausible but the simpler explanation is that the "spike" for long warmup is replaced by a gradual accumulation during the warmup ramp.

### Missing from 1st review:
- The "phase transition" language issue
- The macro extrapolation conflation of warmup fraction with absolute warmup steps
- The correlated trajectory structure of the data
- The Chinchilla prediction arithmetic error

---

## 7. One Positive Finding the 1st Review Could Have Highlighted More

The wc_05 vs wc_10 val loss anomaly (0.4812 vs 0.4815 -- essentially identical despite 6pp difference in death rate) is noted in the paper but not discussed by either review. This is actually an important finding: it suggests that beyond ~5% warmup, the marginal alive neurons contribute negligible quality. The 6pp of neurons that are dead at wc_05 but alive at wc_10 are apparently not carrying useful information. This supports the "alive-neuron specialization" hypothesis from Exp 17 (quality comes from what alive neurons learn, not from having more alive neurons). Worth highlighting as it has implications for the pruning strategy: at certain warmup fractions, you get maximal quality with a still-significant pruning opportunity.

---

## Verdict

**PROCEED**

The experiment is well-designed, the code correctly implements the protocol, and the findings are genuine and useful. The cumulative-LR-integral model is a good empirical scaling law, and the warmup fraction sensitivity result is a real and practically important finding. The 1st review's verdict of PROCEED was correct.

### Required revisions (not blocking, but should be addressed):

1. **Replace "phase transition" with "crossover" or "characteristic scale"** throughout MATH.md and PAPER.md. The data show a smooth saturating curve, not a discontinuous transition.

2. **Clarify absolute warmup steps vs. warmup fraction in macro predictions.** The paper's Finding 4 table is misleading because it uses warmup fractions that implicitly assume micro's S_total=3200. State explicitly: "At macro scale, even 0.1% warmup can give S_w >> T_spike because S_total is orders of magnitude larger. The critical quantity is S_w/T_spike, not f_w."

3. **Downgrade the "arbitrary schedule" prediction claim** (reinforcing 1st review point 3). The model is validated for linear warmup only. Step-function, exponential, or multi-phase warmup could violate the cumulative-integral assumption due to gradient-magnitude dynamics.

4. **Report standard deviations** in the main death rate table (currently only means are shown). The code computes them (line 267) but the PAPER.md omits them.

5. **Acknowledge the correlated trajectory structure**: The 8 step-count measurements within one (condition, seed) combination are NOT independent -- they share the same batch sequence for overlapping steps. This is fine for the analysis but should be stated.
