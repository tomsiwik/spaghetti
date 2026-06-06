# Peer Review: Few-Shot Adaptation Recovery

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that a NotebookLM deep review would add latency without proportional insight. The math is simple (SVD decomposition + gradient descent), the experimental design is clear, and the kill criteria evaluation is honest. Proceeding directly to systematic attack.

## Mathematical Soundness

### SVD Warm-Start Derivation (MATH.md Section 2.2)

The SVD decomposition of the trained delta is correct. Given delta = s * A @ B where A is (in, r) and B is (r, out), the product A @ B is exactly rank-r. SVD recovery is therefore exact (no truncation error), as claimed. The sqrt(sigma) splitting between A_init and B_init is a standard choice that balances the norms of the two factors. No issues here.

**Verified in code:** `inject_lora_deltas_as_trainable()` at lines 63-115 correctly implements this: divides by scale, performs SVD, splits singular values via sqrt, assigns to A and B. The implementation matches the math.

### Gradient Expressions (MATH.md Section 2.3)

The gradients dL/dA and dL/dB are standard chain-rule results for LoRA. The key observation that "the gradient landscape depends on W_k, not W_p" is correct -- the forward pass uses the new base, so gradients are computed w.r.t. the new base's loss surface. This is trivially true but worth stating.

### Logarithmic Fit (MATH.md Section 3.3)

The claim R%(n) ~ 8.5 * ln(n) - 2.0 is fit to 5 data points (n = 1, 5, 10, 25, 50). This is a reasonable empirical description but:

1. **The fit is underdetermined.** A two-parameter model fit to 5 points with R^2 not reported. The logarithmic form is plausible (diminishing returns from fine-tuning near a minimum) but not derived from theory.
2. **Extrapolation to n=550 and n=16,000 is risky.** The logarithmic model predicts 50% at 550 steps, but at some point the adapted expert reaches the retrained quality and the model saturates. The sensitivity data (3e-3 LR) shows crossover at ~100 steps, which the logarithmic model at 1e-3 LR does not predict.
3. **No confidence interval on the fit parameters.** The 3-seed data has std=1.5pp at 50 steps, which is small, but propagated uncertainty on the extrapolation would be large.

These are minor -- the qualitative conclusion (logarithmic, not linear) is directionally correct and the extrapolation is clearly labeled as approximate.

### K2 Forgetting Model (MATH.md Section 3.4)

Linear model K2(n) ~ 1.0 + c*n with c ~ 1.3e-4 per step. This is fit to 5 data points. The linear approximation is reasonable for small n but must saturate (the expert cannot degrade indefinitely). No saturation model is proposed, which is fine for the micro-scale regime tested.

### Gap Reduction Computation

Verified in code (line 403): `gap_reduction = (zs_gap - adapted_gap) / (abs(zs_gap) + 1e-12) * 100`. This correctly computes the percentage of the zero-shot gap that was closed by adaptation. The epsilon guard (1e-12) is appropriate.

**One subtle issue:** The PAPER.md tables report "Gap Reduction" as aggregated across 4 experts per seed, then across 3 seeds. This is a mean-of-means. Since experts have different gap magnitudes, this weights all experts equally regardless of their absolute gap size. A weighted average (by gap magnitude) would be more representative. This is a minor methodological note, not a flaw -- the per-seed standard deviation is small (1.5pp), suggesting the experts are similar enough that the distinction does not matter.

### K2 Kill Criterion Implementation

There is an inconsistency between the code and the paper:

- **Code (line 494):** K2 is KILLED if `k2_worst_ratio > 1.0 + 0.001`, i.e., any degradation above 0.1%.
- **MATH.md (Section 4):** States K2 is "KILLED on strict interpretation (any degradation at all)" but "SURVIVES on practical interpretation (<5% hard-fail threshold)."
- **HYPOTHESES.yml:** States "adapted expert quality worse than zero-shot on original base (adaptation hurts)" -- no quantitative threshold.

The code uses a 0.1% threshold, while the paper discusses 0% (strict) and 5% (practical) thresholds. The actual K2 ratios are 1.014-1.018 across seeds, which is:
- Above 0% (strict kill)
- Above 0.1% (code kill)
- Below 5% (practical survival)

The paper honestly presents all three interpretations. The kill is defensible under the strict reading.

## Novelty Assessment

### Prior Art

SVD warm-starting of LoRA parameters for transfer is not novel in the general sense -- it is a straightforward application of SVD initialization. However, in the specific context of base model swapping for composable expert systems, the framing and the characterization of the adaptation-forgetting tradeoff contribute useful empirical knowledge.

The closest prior work:
- **LoRA-XS (Bałazy et al., 2024):** Uses SVD of pretrained weight differences to initialize LoRA, but for initial training, not transfer.
- **AdaLoRA (Zhang et al., 2023):** Adaptive rank allocation via SVD, but does not address base swapping.
- Standard transfer learning / domain adaptation literature addresses this at the full-model level but not specifically for LoRA expert portfolios.

The experiment's contribution is not the SVD warm-start technique itself but the empirical characterization of adaptation dynamics (logarithmic curve, forgetting tradeoff, crossover point) in the context of the SOLE architecture's base-swapping use case. This is appropriate for a micro-experiment.

### Delta Over Parent Experiment

This experiment properly extends `zero_shot_base_transfer` by asking: given the 2.8-4.2% gap established there, can it be closed cheaply? The answer is "partially, at a cost" -- a useful negative result that informs the architecture's deployment protocol.

## Experimental Design

### Does It Test the Stated Hypothesis?

**Yes.** The hypothesis is "10-50 steps reduce transfer gap by >50%." The experiment directly measures gap reduction at {1, 5, 10, 25, 50} steps across 3 seeds with 4 experts each. The kill criteria are correctly evaluated.

### Controls

1. **Zero-shot baseline (0 steps):** Present, properly computed.
2. **Retrained upper bound (300 steps):** Present, trained from scratch on the new base.
3. **K2 check (quality on original base):** Present, properly measured.
4. **3 seeds, 4 experts per seed:** Adequate for a micro-experiment. Cross-seed consistency is good (std=1.5pp).

### Could a Simpler Mechanism Explain Results?

The logarithmic adaptation curve could simply be "SGD converges slower as it approaches a minimum" -- which is the standard behavior of gradient descent, not a special property of LoRA adaptation. The paper does not claim otherwise; it correctly identifies this as a property of the optimization landscape.

The crossover effect (adapted exceeds retrained at ~100 steps) is genuinely interesting and is explained by the warm-start advantage. This is a real finding.

### Missing Controls

1. **Cold-start adaptation control.** The experiment compares SVD warm-start adaptation against (a) zero-shot and (b) full retraining. It does not compare against cold-start adaptation (standard Kaiming/zero init, same number of steps). This would directly quantify the warm-start advantage at each step count. The sensitivity analysis partially addresses this (retrained at 300 steps from cold start), but a matched-steps cold-start comparison would be more informative.

2. **Learning rate sweep.** Only two LR values tested (1e-3 primary, 3e-3 sensitivity). The paper acknowledges LR dominates adaptation speed but does not systematically explore the LR-forgetting Pareto frontier.

Neither of these is a blocking flaw for a killed experiment.

### Sensitivity Analysis Weakness

The sensitivity analysis (3e-3 LR) uses only seed 42. The paper reports borderline K1 survival (50.0% gap reduction) at this LR. Without multi-seed replication, this borderline result cannot be trusted. The paper acknowledges this: "single seed" is noted. But the HYPOTHESES.yml evidence line states "K1 borderline at lr=3e-3: 50 steps reaches exactly 50.0% gap reduction (single seed)" -- this is honest reporting.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry correctly reflects the experiment:
- **Status:** killed
- **Kill criteria match:** K1 (50 steps, >50% gap reduction) and K2 (no degradation on original base) are exactly what the code tests.
- **Evidence lines** accurately summarize the results with appropriate caveats.
- **Dependency:** Correctly depends on `exp_zero_shot_base_transfer`.
- **Blocks:** Empty, which is correct -- this is a leaf node.

## Kill Criteria Fairness

The kill criteria were pre-registered and honestly evaluated. The paper does not attempt to rescue the hypothesis by redefining thresholds post-hoc. It clearly states "KILLED" and then discusses practical implications separately. This is good scientific practice.

One could argue the 50% threshold at 50 steps was overly aggressive. The paper makes this argument (Section 4, "The threshold of 50% at 50 steps is too aggressive for this learning rate regime"). But the paper does not retroactively change the threshold -- it proposes a revised hypothesis for future work instead. This is the correct approach.

## Macro-Scale Risks (advisory)

1. **Adaptation may be faster at macro scale.** At d=4096, rank-16 SVD perturbation retains a much larger fraction of total energy than at d=64. The transfer gap itself may be smaller, and adaptation may converge faster. This could flip K1 to SURVIVES at macro.

2. **Domain data availability.** The experiment assumes the same domain data is available for adaptation. In production, base model updates may occur without access to each expert's original training data. Unsupervised or self-supervised adaptation is unexplored.

3. **LR tuning per expert.** The paper notes that optimal adaptation LR may need per-expert tuning. At N=500 experts, this becomes a hyperparameter search problem. A universal adaptation LR (or a simple schedule like "3x the original training LR") would need validation.

4. **The forgetting tax implies storage doubling.** Each adapted expert is base-specific. If you need to serve on multiple base versions simultaneously (e.g., during rolling upgrades), you need both original and adapted deltas per expert. At 6MB/expert and N=5000, this is 30GB of additional storage -- manageable but worth planning for.

5. **The deploy-then-adapt protocol is the real value.** Even though the hypothesis was killed, the practical protocol (deploy zero-shot at 3.7% gap, then adapt in background to 2.7% gap) is operationally sound. Macro should validate this workflow, not the specific 50-step/50% threshold.

## Verdict

**PROCEED**

Despite the KILLED status on both kill criteria, this experiment is well-executed and its results are correctly interpreted. The kill is honest and informative. The findings -- logarithmic adaptation dynamics, the adaptation-forgetting tradeoff, SVD warm-start advantage, and the deploy-then-adapt protocol -- are all valuable contributions to the SOLE architecture's operational design.

**Why PROCEED, not REVISE or KILL:**

1. The experiment already killed itself. There is nothing to revise -- the hypothesis was tested, falsified, and the kill was accepted cleanly.
2. The secondary findings (logarithmic dynamics, warm-start advantage, forgetting tradeoff characterization) are sound and properly documented.
3. The paper does not overreach: it proposes a revised hypothesis for future work rather than retroactively salvaging the current one.
4. The code is clean, matches the math, and produces reproducible results (std=1.5pp across seeds).

**Non-blocking suggestions for documentation:**

1. Add the cold-start adaptation control comparison to distinguish "SVD warm-start is faster" from "any adaptation is this fast" -- could be a one-paragraph note rather than a new experiment, since the retrained-from-scratch number already provides an upper bound.
2. Note the mean-of-means aggregation choice explicitly (trivial given low cross-expert variance).
3. The K2 threshold inconsistency between code (0.1%), paper (0%/5%), and HYPOTHESES.yml (qualitative) should be reconciled in a sentence. The current paper handles this well enough by presenting all three interpretations, but a canonical threshold in HYPOTHESES.yml would prevent future ambiguity.
