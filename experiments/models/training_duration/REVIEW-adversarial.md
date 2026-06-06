# Peer Review: Exp 17 -- Training Duration vs Death Rate

## NotebookLM Findings

Skipped (authentication not configured). Review conducted via direct close reading of MATH.md, PAPER.md, code, tests, and full lineage context (Exp 9, 10, VISION.md, FINDINGS.md, ADVERSARIAL_REVIEW.md).

---

## Mathematical Soundness

### What holds

1. **The irreversibility argument (Section 3.2) is correctly stated and correctly falsified.** The derivation that `d L / d a_i = 0` when `f_i = 0` is valid for a single neuron in isolation. The paper correctly identifies the escape route: inter-layer coupling. When layers 0 through l-1 update their MLP weights (via their own alive capsules' gradients), the hidden states `x_l` shift, potentially moving `a_i^T x_l` from negative to positive for some data points. This does NOT require gradient flow through the dead neuron. The corrected analysis in Section 8.2 (`d(a_i^T x) / d step != 0` even when `d a_i / d step = 0`) is mathematically sound.

2. **The death rate definition (Section 2) is unambiguous.** `delta(S) = (1/P) * sum 1{f_i(S) = 0}` computed over a fixed validation set. The `f_i(S) = 0` threshold is exact (binary), not approximate. Combined with Exp 9's theorem that pruning at `tau=0` produces zero quality change, this definition is well-grounded.

3. **The profiling protocol is consistent with Exp 10.** Same `profile_activations()` function, same hyperparameters (20 batches of 32). The replication check at S=200 (52.9% vs 54.3%, delta = 1.4pp) is within expected variance from the profiling condition change (domain-only vs joint data).

4. **The Pearson correlation is correctly computed in code (lines 330-353).** The S=0 exclusion is appropriate. The paper now correctly caveats that the near-zero r is partly an artifact of different functional forms over the shared independent variable S, and attributes the "dead neurons are useless" claim to Exp 9's exact pruning result rather than to this correlation.

### What does not hold or is problematic

5. **The saturating exponential fit is acknowledged as misspecified, but still occupies significant space.** MATH.md Section 5.1 defines the model, Section 5.2 offers a power law alternative, and Section 5.3 describes the fit procedure. PAPER.md then reports fit parameters (delta_0=0.188, delta_inf=0.350, tau=25, R^2=0.914) with appropriate caveats. The problem: the experiment's key finding is that the data is NON-monotonic, yet the only model actually fit is monotonically increasing. No non-monotonic model (e.g., rising exponential + slow linear/logarithmic decay) is fit. The "Revised Trajectory Model" in MATH.md Section 8.3 proposes `delta(S) ~ delta_spike * exp(-S/tau_rise) * (1 + decay_rate * log(1 + S/S_0))` but this is never fit to data. This is a missed opportunity -- even a crude two-phase piecewise fit (exponential rise for S < 100, linear decay for S > 100) would be more informative than the misspecified monotonic model. **Severity: low-moderate.** The paper is honest about the misspecification, but dedicating MATH.md Sections 5.1-5.3 to a model known to be wrong is organizational clutter.

6. **The death rate formula in Section 3.4 is hand-wavy and never validated.** The expression `P(alive -> dead) ~ lr * ||g_i|| / ||a_i|| * phi(margin_i)` has no derivation for `phi`, no distributional assumptions, and no calibration to data. It serves as informal motivation for the saturating exponential hypothesis. Since that hypothesis was falsified, this section has no remaining value. **Severity: negligible.** Clearly labeled as motivation, not a theorem.

7. **The curve fit grid search has minimum tau = 25 at the grid boundary.** The grid is `[25, 50, 100, 150, 200, 300, 500, 800, 1200, 2000]` (line 167 in `test_training_duration.py`). The best fit tau = 25 is the minimum grid value, meaning the true optimum could be at tau = 10 or tau = 15. Since the model is misspecified anyway, this does not change conclusions, but it means the reported "time constant ~25 steps" is an upper bound on the fitted parameter, not necessarily the optimal one. **Severity: negligible.**

### Hidden assumptions examined

8. **Nested seed structure is correctly identified as a strength.** The code passes the same `seed` to `train()` for all step counts (line 100-101). Since `train()` creates `rng = random.Random(seed)`, S=50 is exactly the first 50 steps of S=3200. This means the step counts are checkpoints of the SAME training trajectory, not independent runs. This eliminates batch ordering as a confound. The paper now explicitly states this as a "Design note."

9. **The profiling seed matches the training seed.** In `run_duration_experiment()`, `profile_activations()` receives the same `seed` as `train()` (line 106). This means the profiling batches are deterministic per seed but drawn from the validation set (not the training set), so there is no train/profile leakage. However, using the same seed for profiling across all step counts means the same 640 validation positions (20 batches * 32 samples) are used for all measurements within a seed. This is correct -- it ensures the profiling set is held constant, isolating step count as the only variable.

10. **No weight decay in Adam.** The paper states this in Section 3.3 and it is confirmed by the code (no weight_decay argument in train calls). This is relevant because weight decay could push dead neurons' weights toward zero, shrinking `||a_i||` and potentially making revival harder (or easier, if it crosses the decision boundary in the right direction). The absence of weight decay simplifies the analysis.

---

## Novelty Assessment

### Prior art

The paper cites the four most relevant works:

- **Lu et al. (2019)**: Dying ReLU at initialization. This experiment extends to training dynamics -- valid extension.
- **Li et al. (2023)**: Lazy Neuron Phenomenon. Reports ~50% natural ReLU sparsity in trained transformers, consistent with the equilibrium finding here.
- **Gurbuzbalaban et al. (2024)**: Neural revival. This is the most directly relevant prior work -- it documents the revival phenomenon through inter-layer coupling, with the finding that >90% of revived neurons eventually die again. The paper cites this correctly.
- **Mirzadeh et al. (2023)**: ReLU Strikes Back. Context on ReLU sparsity as a feature for inference acceleration.

### Delta over existing work

The specific contribution is: characterizing the death rate trajectory in a multi-layer transformer with frozen attention and shared MLP fine-tuning, in the context of the composable expert composition protocol. The three-phase "spike and slow decay" characterization, the fast time constant (~25 steps), and the practical implication that training duration beyond ~100 steps barely affects the death rate are useful empirical observations.

This is a narrow but adequate contribution for a micro-experiment that builds on prior work and directly informs the macro transition of the composable experts project. No prior work characterizes death trajectories in this specific architectural context (frozen attention, all MLP layers co-trained).

---

## Experimental Design

### Strengths

1. **Clean single-variable sweep.** All step counts start from the same deepcopy of the same pretrained base, use the same frozen attention, same learning rate, same profiling protocol. The ONLY variable is S (number of fine-tuning steps). This is good experimental isolation.

2. **Geometric step spacing.** The 2x spacing uniformly samples log-time, appropriate for exponential/power-law processes. The range (50-3200) spans ~1.5 orders of magnitude.

3. **Pre-registered kill criteria.** Three specific thresholds defined before results. One triggered (death decrease > 5pp from S=200 to S=3200). This is honest, pre-registered hypothesis testing.

4. **Exp 10 replication.** The S=200 data point replicates Exp 10 within 1.4pp, validating measurement consistency.

5. **The S=0 baseline is valuable.** Profiling the pretrained base before any fine-tuning reveals that 18.8% of capsules are already dead from pretraining alone. This contextualizes the fine-tuning spike (18.8% -> 55.1% in 50 steps = 36.3pp from fine-tuning shock).

### Concerns

6. **Statistical power is marginal for the kill criterion.** The kill criterion triggers at delta(200) - delta(3200) = 5.7pp, just barely past the 5pp threshold. With only 3 seeds and std = 6.3-6.5% at those step counts, a fourth seed could push the aggregate below threshold. The paper acknowledges this ("Only 3 seeds: ... close to the 5pp threshold. More seeds would increase confidence."). At 3 seeds, the standard error of the mean death rate is approximately 6.5% / sqrt(3) = 3.75pp. A 5.7pp decrease with SE ~3.75pp gives a t-statistic of ~1.5 (p ~ 0.13, one-tailed). This is not statistically significant by conventional standards (p < 0.05). The non-monotonicity finding is directionally consistent across all 3 seeds (each shows at least one decrease > 0.5pp), which provides qualitative support, but the aggregate magnitude is not statistically distinguishable from noise at conventional significance levels.

7. **The kill criterion reference point is debatable.** Kill criterion 1 compares S=200 to S=3200, because S=200 was the Exp 10 measurement point. But the peak death rate is at S=100 (55.5%), not S=200 (52.9%). The decrease from peak to S=3200 is 8.2pp, which would be more statistically robust. The paper notes this in the Monotonicity Analysis section but uses the S=200 reference for the kill criterion. This is defensible (it answers "does the Exp 10 measurement predict macro behavior?") but understates the non-monotonicity signal.

8. **The correlation analysis adds nothing beyond Exp 9.** The Pearson r = 0.027 between death rate and val loss is meaningless as a test of whether dead neurons affect quality. Both variables are functions of S with different functional forms (non-monotonic vs monotonic). The paper now correctly attributes the "dead neurons are useless" claim to Exp 9, making the correlation analysis vestigial. It could be removed without loss. **Not blocking, but organizational dead weight.**

9. **Per-capsule identity tracking is absent.** The experiment measures aggregate death RATES but does not track whether the SAME capsules are dead at S=100 and S=3200. The "revival" interpretation (dead neurons come back to life) is inferred from the aggregate rate decrease, but the decrease could also arise from a different mechanism: some capsules that were alive at S=100 die by S=3200, while a different (larger) set of capsules that were dead at S=100 revive. Without per-capsule tracking, the paper cannot distinguish "some capsules oscillate between alive and dead" from "the alive/dead population is mostly stable with a net flow toward revival." VISION.md lists Exp 16 (capsule identity tracking) as a future direction -- this experiment would have benefited from it. **Severity: moderate.** The aggregate finding (rate decreases) is valid regardless, but the mechanistic interpretation is underconstrained.

10. **No checkpoint-based trajectory (only independent runs from base).** Each step count S starts from a fresh deepcopy of the pretrained base and trains for exactly S steps. An alternative design would train once for 3200 steps and checkpoint at S = {50, 100, 200, ...}, profiling at each checkpoint. The current design and the checkpoint design are identical due to the nested seed structure (point 8 above) -- S=50 from a fresh deepcopy with seed=42 produces the same weights as the first 50 steps of S=3200 with seed=42. But this identity is never verified in the code. A simple assertion confirming that the S=50 model weights match the first 50 steps of the S=3200 model would strengthen confidence in the nested structure. **Severity: very low.** The identity follows from deterministic RNG + deepcopy of the same base.

---

## Macro-Scale Risks (advisory)

1. **LR schedules will qualitatively change the trajectory.** The constant LR (3e-3) creates a specific dynamic: large initial gradients cause the spike, natural loss convergence softens gradients later. Warmup (standard at macro) would reduce early gradient magnitudes, potentially softening the spike. Cosine decay would reduce late-phase gradients, potentially reducing both late death and late revival. The "spike and slow decay" shape may not transfer. Gurbuzbalaban et al. specifically note that LR decay triggers brief revival episodes.

2. **SiLU/GELU at macro scale invalidates the binary death definition.** Qwen uses SiLU, not ReLU. SiLU has no hard zero, so `f_i = 0` never occurs exactly. The entire death-rate framework requires adaptation to a threshold-based definition (e.g., `f_i < epsilon` for some `epsilon`). VISION.md lists this as Exp 15 (non-ReLU pruning). The training-duration trajectory may differ qualitatively with soft activations.

3. **The logarithmic extrapolation (40-45% at 100K steps) is a 1.5-order-of-magnitude extrapolation.** The paper extrapolates ~2.5pp per 10x training from 8 data points spanning 50-3200 steps. Extrapolating to 100K steps (31x beyond max measurement) is speculative. The paper acknowledges this ("should be treated with caution") but still reports the prediction prominently.

4. **Unfrozen attention at macro scale.** Full fine-tuning (not just MLP) causes much larger distribution shifts at every layer, potentially changing both the spike magnitude and decay rate.

5. **Profiling sample size (640 positions) may be insufficient at macro scale.** At d=4096 with P=8192 capsules, 640 profiling positions might not provide enough statistical power to reliably classify each capsule as dead vs alive. The profiling budget should scale with the number of capsules.

---

## Verdict

**PROCEED**

This is a well-designed micro-experiment that honestly reports a partially falsified hypothesis. The experimental protocol cleanly isolates the independent variable (step count), the controls are adequate, the kill criteria are pre-registered, and the results are reproducible. The core finding -- death rate follows a "spike and slow decay" trajectory, not monotonic accumulation, with equilibrium around 47-55% -- is a useful calibration for macro-scale expectations.

The previous review (two passes) identified five advisories, all of which have been addressed in the current PAPER.md and MATH.md. My independent review confirms those fixes and adds the following observations:

### New findings beyond previous review

1. **The kill criterion trigger (5.7pp > 5pp) is not statistically significant.** With 3 seeds and SE ~3.75pp, the t-statistic is ~1.5 (p ~ 0.13). The qualitative finding (all 3 seeds show non-monotonicity) is directionally consistent, but the aggregate magnitude crosses the 5pp threshold by only 0.7pp. This is marginal. The experiment should note that the kill criterion is triggered directionally but not at conventional statistical significance with n=3.

2. **Per-capsule identity tracking would disambiguate revival mechanisms.** The aggregate rate decrease could reflect (a) specific dead capsules reviving, (b) population turnover with net flow toward revival, or (c) measurement noise in a plateau regime. Without per-capsule tracking, the mechanistic interpretation is underconstrained. This does not invalidate the aggregate finding but limits the theoretical insight.

3. **The curve fit tau=25 is at the grid boundary** and should be reported as "tau <= 25" or the grid should be extended downward (e.g., include tau = 5, 10, 15). This is cosmetic given the model misspecification.

4. **Code line 422-425 still contains the disavowed macro prediction** using the misspecified exponential asymptote. The branch is unreachable in the actual experiment (kill1 triggered), but it encodes a claim the paper explicitly rejects. A cleanup pass should either remove this branch or update its output.

None of these are blocking. The experiment achieves its purpose: characterizing the death rate trajectory over training duration and informing the macro transition with a revised pruning yield estimate (47-55% instead of the point estimate of 54.3%).

---

## Second Pass Verification (2026-03-04)

All five advisories from the prior review have been verified as addressed:

- Advisory 1 (exponential fit caveat): ADDRESSED in PAPER.md curve fit section.
- Advisory 2 (nested seed structure): ADDRESSED as "Design note" in PAPER.md.
- Advisory 3 (decoupled claim): ADDRESSED with Exp 9 attribution.
- Advisory 4 (Layer 0 explanation): ADDRESSED with fixed-input-distribution reasoning.
- Advisory 5 (constant LR limitation): ADDRESSED in Limitations item 5.

No new blocking issues found. Verdict stands: **PROCEED**.
