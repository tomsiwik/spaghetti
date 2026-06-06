# Peer Review: Calibration LR Scaling with N

## NotebookLM Findings

Skipped -- the experiment is a clean null result with straightforward math. Deep review not warranted for this scope.

## Mathematical Soundness

### The Adam cancellation argument (correct, with a caveat)

The core claim is that Adam's update rule cancels gradient magnitude scaling:

```
Update(N) = LR * alpha * m / (alpha * sqrt(v) + eps)
          ~ LR * m / (sqrt(v) + eps/alpha)
```

where alpha = k/N.

**Step-by-step verification:**

1. If g_N = alpha * g_2, then the first moment m_t(N) ~ alpha * m_t(2). Correct -- the exponential moving average is linear.

2. The second moment v_t(N) ~ alpha^2 * v_t(2). Correct -- v_t tracks E[g^2], and (alpha*g)^2 = alpha^2 * g^2.

3. Therefore sqrt(v_t(N)) ~ alpha * sqrt(v_t(2)), and the alpha cancels in m/sqrt(v). Correct.

4. The eps term breaks this: eps/alpha > eps when alpha < 1. At high N (alpha = 2/16 = 0.125), eps/alpha = 8*eps. For eps = 1e-8 and typical sqrt(v) ~ 1e-3 to 1e-2, eps/alpha ~ 8e-8, still negligible. **Correct -- eps is irrelevant at these scales.**

**Caveat that the paper acknowledges:** This argument requires v_t to have stabilized (i.e., accumulated enough gradient history). In the first few steps, v_t is near zero and eps dominates, meaning alpha does NOT cancel. However, with beta2=0.999 and typical gradient magnitudes, v_t stabilizes within 5-10 steps. Since the experiment evaluates at step 10+ intervals (CAL_EVAL_EVERY=10) and convergence is measured at step 100, this is fine.

**One concern not mentioned:** The argument assumes gradients are uniformly attenuated by alpha across all parameters. In practice, router parameters vs expert parameters may have different attenuation profiles. But since only router parameters are unfrozen during calibration (line 80-82 of the test script), and all router gradients scale similarly with k/N (the top-k selection mechanism is shared), uniform attenuation is a reasonable assumption.

### The scaling law fit (weak but conclusion is correct)

Log-log regression yields b=0.10, r^2=0.067. The paper concludes "no scaling needed." This is correct -- r^2=0.067 means N explains almost nothing about optimal LR. However, the fit itself is on only 4 data points, and the "optimal LR" is selected from a coarse grid (5 multipliers). The optimal LR for N=4 being 0.5x while all others are 1.0x could be noise. With 3 seeds, the confidence intervals on "optimal LR" are wide enough that all N values are plausibly optimal at 1.0x.

**Verdict: The math is sound.** The Adam cancellation argument is a well-known property (see e.g., the "scale-free" property of Adam discussed in the original Kingma & Ba paper). The experiment correctly identifies and confirms it.

## Novelty Assessment

**This is not novel.** The scale-invariance of Adam to gradient magnitude is a textbook property. The original Adam paper (Kingma & Ba, 2015) describes this as a feature, not a discovery. Any practitioner who has used Adam with varying batch sizes or gradient accumulation steps has implicitly relied on this property.

**However, novelty is not required here.** This experiment's value is as a *hypothesis killer*: the parent experiment (discriminability_n_gt_2) measured 5-7x gradient attenuation and suggested LR scaling as a fix. This experiment correctly shows the fix is unnecessary. The null result closes a branch of the hypothesis graph, which is the right outcome.

**Delta over prior art:** Zero, in terms of Adam's properties. The contribution is the empirical confirmation that this property holds for MoE router calibration specifically, and the resulting simplification of the contribution protocol.

## Experimental Design

### Strengths

1. **Full grid sweep.** 4 N values x 5 LR multipliers x 3 seeds = 60 configs. Thorough enough for the claim.

2. **Convergence tracking.** Loss curves at 10-step intervals with multiple step budgets (100, 200, 300, 600). This correctly tests both the LR and step-count hypotheses.

3. **Correct baseline.** Joint training as the reference point is the right comparison.

4. **Practical operating regime.** Testing at cos=0.0 (the actual operating point for independently trained LoRA experts) is the right choice. No need to test cos>0 since it never occurs in practice.

### Concerns

1. **Kill criteria are evaluated trivially.** KC1 asks for "monotonic relationship between N and optimal calibration steps." Since all N converge at exactly 100 steps, this is "monotonically constant." KC2 asks if "LR scaling does not compensate for gradient attenuation (quality gap persists)." Since there is no quality gap to close, it passes trivially. The experiment correctly identifies this as a null result, but the kill criteria were designed for a world where a gap exists. The criteria should have included a third condition: "quality at N=16 is within 1pp of N=2 at default LR" -- which is the actual finding.

2. **Expert generation method.** Experts for N>2 are generated via `generate_n_experts_at_cosine()`, which projects from 2 base LoRA deltas. These synthetic experts may not capture the diversity of truly independently trained experts (e.g., 8 different domain-specific LoRAs). However, the paper acknowledges this limitation, and at cos=0.0, the projection is orthogonal -- which is the worst case for gradient attenuation (maximum dilution). If anything, this biases toward finding a gap, making the null result more convincing.

3. **Only Adam tested.** The claim "no LR scaling needed" is Adam-specific. SGD, AdaGrad, or other optimizers would behave differently. The paper correctly flags this. This is not a flaw -- it is a properly scoped result.

4. **Same calibration seed as training seed.** The `seed` parameter is reused for both expert training and calibration (line 307-310). This is not ideal for independence but is unlikely to affect results given the fundamentally different optimization landscapes.

## Hypothesis Graph Consistency

The node `exp_calibration_lr_scaling_with_n` is marked "proven" with status consistent with a null result. The kill criteria in the code (lines 18-21) match those in HYPOTHESES.yml (lines 1017-1018). The evidence list accurately summarizes the findings.

**One inconsistency:** The node says "proven" but the experiment essentially *disproved* the original hypothesis (that LR should scale as N/k). "Proven" here means "the experiment ran successfully and produced a definitive answer (null result)." This usage is consistent with the project's conventions but could confuse external readers.

The experiment correctly closes the calibration branch. The note in HYPOTHESES.yml says "this closes the calibration branch of the hypothesis graph," and indeed there are no downstream nodes that depend on this experiment.

## Macro-Scale Risks (advisory)

1. **eps sensitivity at very large N.** At N=1024 with k=2, alpha = 0.002. Adam's eps/alpha = 5000*eps ~ 5e-5. If sqrt(v) is O(1e-4) for some parameters at macro scale, eps could start to matter. This is an edge case but worth monitoring.

2. **Learning rate warmup interaction.** Macro training often uses LR warmup. During warmup, the effective LR is small, and Adam's second moment is still accumulating. The interaction between warmup and gradient attenuation from large N is untested. Probably fine, but worth a sanity check.

3. **The 100-step constant may not hold at d=896+.** At micro scale (d=64), 100 steps may suffice because the router has few parameters. At macro scale with larger routers, convergence might require more steps -- but this would be independent of N, consistent with the null result.

## Verdict

**PROCEED**

This is a clean, well-executed null result that correctly closes a branch of the hypothesis graph. The math is sound (Adam's scale-invariance is a known property, and the experiment confirms it applies to MoE router calibration). The experimental design is thorough (60 configs, 3 seeds). The null result is the best possible outcome for practical deployment: the contribution protocol requires no N-dependent tuning.

Minor issues (trivial kill criteria evaluation, synthetic experts, single optimizer) are all properly acknowledged in the paper's limitations. No revisions required.
