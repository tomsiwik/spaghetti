# Peer Review: Training-Time Composition Compatibility (Exp 11)

## NotebookLM Findings

Skipped (NotebookLM not authenticated). Review conducted from direct source analysis.

## Mathematical Soundness

### Orthogonality Loss (L_ortho)

The formulation is correct and well-normalized:

```
L_ortho = ||delta_A @ A_0^T||_F^2 / (||delta_A||_F^2 * ||A_0||_F^2 + eps)
```

This is a valid measure of subspace alignment. The denominator normalizes to roughly [0, 1], making the coefficient lambda_o interpretable. The Frobenius norm of the cross-product captures the projection of delta onto the base subspace.

**One concern:** The MATH.md claims "If deltas are orthogonal to base weights, they are more likely to be orthogonal to EACH OTHER." This is stated as a hypothesis but the reasoning is sound only in a specific regime. The orthogonal complement of a rank-r subspace in R^P has dimension P - r. At P=128, d=64, rank(A_0) is at most 64, so the complement is at most 64-dimensional. With N=2 domains, two random vectors in a 64-dimensional space have expected cosine similarity near zero, so the hypothesis is plausible but weakly motivated -- the deltas are ALREADY nearly orthogonal without the loss (cos = 0.170 baseline). The loss reduces this to 0.109, confirming the mechanism works directionally but the baseline is already low.

**Verdict:** Math is sound. The hypothesis connecting base-orthogonality to inter-domain-orthogonality is reasonable but the experiment correctly shows it does not translate to function-space improvement.

### Norm Matching Loss (L_norm)

```
L_norm = lambda_n * (||pool(x)||_2 / ||x||_2 - target_ratio)^2
```

Implementation uses mean-over-batch of per-position squared norms, then sqrt. This computes `sqrt(mean(||x_i||^2))` which is a valid proxy for the RMS output norm but is NOT the same as `mean(||x_i||)`. The squared-then-sqrt formulation is dominated by outlier positions. This is unlikely to matter at micro scale but is worth noting.

**Target ratio measurement (lines 232-266 of training_compat.py):** There is a bug or at least a confusing double-forward. The function computes `x_pre = layer.norm2(x + layer.attn(layer.norm1(x)))` to get the pool input, runs the pool, records the ratio -- but then ALSO runs the full forward pass again (`x = x + layer.attn(...)` followed by `x = x + layer.capsule_pool(...)`) to get x for the next layer. This means the x feeding into the next layer's measurement includes TWO attention computations and TWO pool computations for each layer. The target ratios are therefore computed on a subtly wrong forward pass. However, since this affects all conditions equally (including no_aux which ignores the ratio), it does not invalidate the comparative results.

**Verdict:** Minor implementation concern, does not affect the kill/pass determination.

### Composition by Concatenation

The claim that `pool_composed(x) = sum_n pool_n(x)` is exact due to ReLU per-neuron independence is correct. Each capsule i computes `b_i * ReLU(a_i^T x)` independently, so concatenating capsules from different pools is equivalent to summing the pools' outputs. This is a structural property of the architecture.

### The Core Insight

The paper correctly identifies the fundamental inequality:

```
ReLU(A_1 @ x) + ReLU(A_2 @ x) != ReLU((A_1 + A_2) @ x)
```

This is the key theoretical contribution: weight-space regularization CANNOT close the function-space gap for concatenation because the gap arises from the nonlinearity, not from weight properties. This is mathematically rigorous and important for the research program.

## Novelty Assessment

### Prior Art

The experiment correctly cites InfLoRA (2024) as the source of orthogonality constraints for LoRA. The delta is clear: InfLoRA enforces orthogonality for continual learning (sequential tasks, prevent forgetting), while this experiment tests orthogonality for parallel composition (independent training, reduce composition gap). The application is different even though the mechanism is borrowed.

The norm-matching loss appears to be a natural extension, not directly borrowed from any specific prior work. Output-norm regularization is common in deep learning but its application to composition compatibility is reasonable.

### References Check

`references/REFERENCES.yml` lists InfLoRA and MoE-Adapters4CL as relevant to `exp11_training_time_compat`. Both are cited. MoRAM is also listed as relevant -- the paper does not discuss self-routing as an alternative to auxiliary losses, which is a minor gap but not critical since self-routing is a different composition strategy entirely.

### Novelty Verdict

The experiment is a reasonable application of known techniques to a specific problem. The negative result (aux losses worsen concatenation) combined with the positive secondary finding (combined aux improves weight averaging) is genuinely informative.

## Experimental Design

### Does it test the hypothesis?

Yes. The hypothesis is "auxiliary loss reduces composition gap by >=50%." The experiment tests four conditions (no_aux, ortho_only, norm_only, combined) across 3 seeds, measuring zero-shot concatenation gap. The kill criterion is unambiguous and the result is clear: no condition reduces the gap.

### Controls

- **no_aux baseline:** Correct control -- same architecture, same training protocol, just aux coefficients set to zero.
- **Joint training:** Correct upper bound -- trains on all data jointly.
- **3 seeds:** Adequate for micro-scale directional evidence.
- **Diagnostic measurements:** Weight cosine and output norm variance confirm the mechanisms work at the weight level. Good practice.

### Potential Confounds

1. **Single coefficient value (0.1).** The paper acknowledges this limitation. A coefficient sweep might find a sweet spot, but given that ALL aux conditions worsen concatenation (not just marginally), it is unlikely that coefficient tuning would achieve 50% gap reduction. The direction is wrong, not just the magnitude.

2. **200 fine-tuning steps.** Short training may not give the aux losses enough time to reshape the weight landscape. However, 200 steps is the standard protocol in this project, and the ortho loss DOES measurably reduce cosine similarity (0.170 to 0.109), confirming the loss has time to take effect.

3. **N=2 only.** At N=2, the composition problem is simplest. If aux losses cannot help at N=2, they are unlikely to help at N=5. This actually strengthens the kill.

### Weight Averaging Secondary Finding

The combined condition achieving -0.2% vs joint is interesting but the paper correctly flags it needs more validation:
- 3 seeds may not be enough to confirm a 0.2% improvement (this is within typical noise range)
- N=2 only
- The mechanism is clear (norm matching helps averaging) but the magnitude needs verification

**This is appropriately reported as a secondary finding, not a primary claim.**

## Hypothesis Graph Consistency

The kill criterion in `HYPOTHESES.yml` is: "auxiliary loss reduces composition gap <50% relative to no-aux baseline." The experiment directly tests this with zero-shot concatenation. Status is set to `disproven`. This is correct.

The evidence entry accurately summarizes both the kill (concatenation) and the secondary finding (weight averaging). Consistent.

## Macro-Scale Risks (advisory)

1. **The double-forward in `measure_norm_ratios` should be fixed before any macro use.** At macro scale, this wastes compute and produces subtly wrong target ratios.

2. **The ortho loss has O(P^2 * d) cost per layer.** At macro scale with P=512+ experts, this becomes expensive. The paper acknowledges this (lines 102-103 of MATH.md). Would need low-rank approximation of the orthogonality constraint.

3. **The weight averaging finding (+combined aux) is the most promising macro direction from this experiment.** Weight averaging is O(0) at inference -- no router, no calibration. If combined aux genuinely brings it to joint-training quality, that is a significant practical result worth validating at scale.

4. **The core negative result (weight-space regularization cannot close the function-space gap for concatenation) likely holds at any scale.** The inequality `ReLU(A_1 x) + ReLU(A_2 x) != ReLU((A_1+A_2) x)` is scale-invariant.

## Verdict

**PROCEED** (as a completed, killed experiment with a valid secondary finding)

The experiment is well-designed, the math is sound, the kill criterion is unambiguous, and the result is clear. The kill is correct: no auxiliary loss reduces the zero-shot concatenation gap. All conditions worsen it.

The experiment makes a genuine contribution to the research program:
1. It establishes that the function-space gap is NOT addressable by weight-space regularization for concatenation composition. This is a principled negative result that eliminates an entire class of approaches.
2. The secondary weight-averaging finding (combined aux to -0.2% vs joint) is worth tracking but needs more seeds and N>2 to be considered robust.
3. The diagnostic measurements (cosine similarity, norm variance) confirm the losses work at the weight level, isolating the failure to the nonlinear composition step.

No revisions needed. The paper correctly self-kills on the primary hypothesis and appropriately hedges the secondary finding. The HYPOTHESES.yml entry is accurate. File to FINDINGS.md and move on.
