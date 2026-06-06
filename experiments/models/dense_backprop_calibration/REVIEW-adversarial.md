# Peer Review: Dense Backpropagation Calibration

## NotebookLM Findings

Skipped -- the experiment is already self-killed with clear results. Deep review not warranted for a cleanly killed experiment with honest reporting.

## Mathematical Soundness

### STE Implementation: Correct

The straight-through estimator is correctly implemented:

```python
weights = dense_probs + mx.stop_gradient(sparse_probs - dense_probs)
```

This evaluates to `sparse_probs` in forward (since `stop_gradient(x) = x` in forward) and has gradient `d(dense_probs)/d(theta)` in backward (since `stop_gradient` blocks the correction term). Standard STE technique, correctly applied.

### Gradient Scaling Prediction: Flawed Premise

The MATH.md predicts dense backprop should restore gradients to N/k = 4x of sparse, yielding "100% closure of the N2-N8 gap." This reasoning has a hidden assumption:

**Assumption**: Router gradient magnitude scales linearly as k/N solely due to the number of contributing experts.

**Problem**: This ignores that with N=8 experts at fixed embedding dimension d=64, each expert's output contribution is diluted by the softmax normalization. Dense backprop sends gradients through all 8 experts, but the dense softmax weights are ~1/8 each (vs ~1/2 for k=2 in the sparse case). The gradient through each expert path is proportionally smaller. The net effect is not a clean k/N restoration -- it depends on the expert output covariance structure.

The paper correctly identifies this post-hoc ("dense backprop changes gradient SHAPE, not just magnitude"), but the prediction was unsound from the start. The k/N scaling argument conflates "number of gradient paths" with "gradient magnitude per path."

### Gap Closure Metric: Problematic

The gap closure metric `1 - |grad_N2 - grad_N8_dense| / |grad_N2 - grad_N8_sparse|` is unstable when the denominator is small. At cos=0.3, the N8 sparse gradient (0.0710) is *higher* than expected, making the sparse gap tiny (0.0105). This causes the -134.8% closure -- not because dense backprop made things worse, but because the metric divides by noise.

A more robust approach would have been:
- Ratio metric: `grad_N8_dense / grad_N2` vs `grad_N8_sparse / grad_N2`
- Or using median instead of mean across cosine levels
- Or requiring the sparse gap to exceed a minimum threshold before computing closure

The mean closure of -7.0% is dominated by one noisy data point. The median closure (32.2%) is more representative but still below the 50% threshold.

### Convergence Measurement: Trivially Uninformative

All configurations converge at step 200-202 out of 300 maximum steps. The convergence target is defined as the N=8 sparse final loss -- but since both sparse and dense are evaluated every `CAL_EVAL_EVERY` steps, and both reach the target at nearly the same evaluation checkpoint, the measurement has resolution limited by the eval frequency. The 1.00x speedup may simply reflect that eval checkpoints are too coarse to distinguish a real difference.

However, this is a minor issue because the paper's real conclusion (dense backprop helps quality, not speed) does not depend on the convergence metric.

## Novelty Assessment

**Prior art acknowledged**: Default MoE (arXiv:2504.12463) is cited. That paper already demonstrated dense backprop for MoE training with EMA approximations and reported 9% tokens-to-target reduction. DenseMixer is mentioned but not cited with a full reference.

**Delta over existing work**: This experiment tests the *exact* STE variant (computing all expert outputs) as an upper bound, whereas Default MoE uses cheaper EMA approximations. The experiment's contribution is demonstrating that even with exact dense computation (the best-case scenario), the gradient magnitude gap is not the bottleneck. This is a useful negative result that saves future effort.

**No reinvention**: The experiment correctly builds on the parent chain (`discriminability_n_gt_2`) and reuses existing infrastructure.

## Experimental Design

### Does it test the stated hypothesis?

Yes. The hypothesis is: "dense backprop closes the gradient gap by >50% and achieves >=2x convergence speedup." Both kill criteria are cleanly measured and both fail. The experiment design is appropriate for the hypothesis.

### Adequate controls?

**Strengths**:
- Same base model, same deltas, same seed for dense vs sparse comparisons
- 3 seeds with mean aggregation
- N=4 intermediate check provides useful gradient amplification data point
- Both N=2 baseline and N=8 sparse baseline included

**Weaknesses**:
1. **Sparse branch computes all N expert outputs too.** In the `else` (sparse) branch of `__call__`, the loop `for e in range(self.n_experts)` iterates over all experts, but masked experts get weight ~0. The multiplication `w_e * self._run_expert_mlp(h, l_idx, e)` still computes the expert output. This means the sparse branch is NOT equivalent to true sparse routing (where non-selected experts are never computed). Gradients could still leak through the multiplication with near-zero weights. At float32 precision, `0.0 * f(x)` still propagates gradients through `f(x)` in autograd frameworks.

    Checking the code more carefully: the mask zeros out non-selected probs, but `masked_probs` could have small residuals from float precision. More importantly, the expert computation `self._run_expert_mlp(h, l_idx, e)` is always called. In MLX's autograd, if `w_e` is zero after masking but the graph includes the multiply, gradients may still flow (product rule: `d/dx [0 * f(x)] = 0 * f'(x) + f(x) * 0 = 0`, so this is actually fine numerically). The sparse baseline is valid.

2. **Same seed used for all configs within a trial.** The random seed for calibration (`seed=seed` in `calibrate_with_gradient_tracking`) is shared across all configs. This means the same batch sequence is used, which is appropriate for paired comparison but could mask seed-specific effects.

3. **Synthetic N=8 experts from 2 trained domains.** The `generate_n_experts_at_cosine` function creates N=8 experts by geometric projection from only 2 real LoRA adapters. These synthetic experts may not have the output diversity that 8 independently-trained domain experts would have. The paper acknowledges this (Limitation 2).

### Could a simpler mechanism explain the positive result?

The 0.5pp quality improvement from dense backprop could be explained by:
- **Implicit load balancing**: dense gradients push the router to distribute more evenly, which at micro scale with uniform domain mixing could help
- **Gradient smoothing**: averaging over 8 expert signals reduces per-step gradient variance, acting as implicit gradient averaging

Neither of these requires the "routing information richness" narrative the paper proposes. A simpler test: does dense backprop with a FROZEN router (no router gradients) still improve quality? If so, the benefit is from the expert weight updates, not routing.

## Hypothesis Graph Consistency

The experiment correctly maps to `exp_dense_backprop_calibration` in HYPOTHESES.yml. Kill criteria match exactly:
- KC1: gradient gap closure >50% (KILL at -7.0%)
- KC2: convergence speedup >=2x (KILL at 1.00x)

The status is correctly set to `killed`. Evidence entries accurately reflect results. The downstream node `exp_calibration_lr_scaling_with_n` is correctly identified as the practical alternative.

## Integration Risk

None, since the experiment is killed. The finding that "k/N dilution is not the bottleneck" is recorded in FINDINGS.md and correctly informs the project direction. No architecture changes needed.

## Macro-Scale Risks (advisory)

Not applicable -- experiment is killed. The advisory note in the paper ("at real scale cos~0.0002, all of this is moot") is correct and well-calibrated. Dense backprop at macro scale would cost N/k=4x training FLOPS for a mechanism that provides diminishing returns as expert discriminability increases with dimension.

## Verdict

**PROCEED** (as a killed experiment -- the kill is correct and well-documented)

The experiment is honestly and correctly killed. Both kill criteria fail clearly. The methodology is sound within micro constraints. The positive side-finding (0.5pp quality improvement via routing information) is interesting but does not justify the 4x training cost.

Minor issues that do NOT warrant revision:

1. The gap closure metric is numerically unstable at cos=0.3, but the kill would hold even with a more robust metric (median closure = 32.2%, still below 50%).
2. The convergence measurement has limited resolution, but the paper does not over-claim on this axis.
3. The "routing information richness" explanation for the quality improvement is speculative (no ablation to confirm), but is presented as interpretive rather than proven.

This is a clean negative result that correctly eliminates a hypothesis branch and redirects effort toward the more practical LR-scaling approach. No revisions needed.
