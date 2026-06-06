# Peer Review: SiLU Pruning (exp15_non_relu_pruning)

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that document-level analysis suffices. The math is simple (error bound from triangle inequality on a linear decomposition), and the empirical finding is clear-cut.

## Mathematical Soundness

### Error Bound (MATH.md Section 3.1): CORRECT

The per-capsule error bound is trivially correct. The MLP output decomposes as a sum over capsule contributions:

```
y(x) = sum_j b_j * SiLU(a_j^T x)
```

Removing capsule i gives `delta_y = b_i * SiLU(a_i^T x)`, and `||delta_y|| = ||b_i|| * |SiLU(a_i^T x)|`. Taking expectation gives `E[||delta_y||] <= ||b_i|| * mu_i`. This is standard and correct.

### Aggregate Bound (Section 3.2): CORRECT but loose

The triangle inequality bound `E[||delta_y||] <= sum_{i in S} ||b_i|| * mu_i` is correct but potentially very loose. If pruned capsules' contributions are correlated (likely -- they share the same input x), the actual error could be much smaller due to cancellation. The paper acknowledges this (Assumption 4). Not a flaw, just a conservative bound.

### Floor Effect Analysis (Section 4.3): PARTIALLY CORRECT

The lower bound derivation:

```
E[|SiLU(Z)|] >= 0.5 * E[Z | Z > 0] * P(Z > 0) = sigma / (2*sqrt(2*pi)) ~ 0.20*sigma
```

This holds but has a subtle issue. The step `E[Z * sigmoid(Z) * 1{Z > 0}] >= E[Z * 0.5 * 1{Z > 0}]` uses `sigmoid(z) >= 0.5` for `z > 0`, which is correct. However, the bound **ignores the negative contribution**: `|SiLU(z)| > 0` for `z < 0` too (SiLU is negative in (-inf, 0) with minimum ~-0.278). So the actual `E[|SiLU(Z)|]` is strictly *larger* than the bound states. The floor is even higher than derived, which strengthens (not weakens) the conclusion.

### Standard Deviation Computation Bug (silu_pruning.py line 114)

```python
std_arr = mx.sqrt(mx.maximum(mean_sq_arr - (sum_abs[l_idx] / total_positions) ** 2, ...))
```

This computes `sqrt(E[X^2] - (E[|X|])^2)`, which is the standard deviation of `|SiLU(a_i^T x)|`, NOT of `SiLU(a_i^T x)`. The variable name `std_activation` is misleading -- it is `std(|activation|)`. This does not affect any result since `std_arr` is computed but never used in any decision or reported number. Minor code quality issue only.

### Gaussian Assumption (Section 4.3, Assumption)

The floor analysis assumes `a_i^T x ~ N(0, sigma^2)`. After training, pre-activations are unlikely to be Gaussian -- the detector vectors `a_i` are optimized to respond to specific input patterns, creating heavy tails or multimodality. However, the floor bound is conservative (see above), so non-Gaussianity does not invalidate the conclusion.

**Mathematical Verdict: Sound.** No errors that affect conclusions.

## Novelty Assessment

### Prior Art

Magnitude-based pruning of neural networks is extensively studied (Han et al. 2015, "Learning both Weights and Connections"; Zhu & Gupta 2017). The specific application to SiLU activations in a capsule/adapter context is a reasonable micro-experiment but not novel research. The experiment is better understood as an **engineering diagnostic** (does our ReLU pruning technique transfer to SiLU?) rather than a research contribution.

### Delta Over Existing Work

The useful finding is empirical: SiLU's activation floor at ~0.05-0.09 prevents lossless pruning. This is a direct consequence of SiLU's non-zero gradient everywhere, which is well-known in the literature (Ramachandran et al. 2017 "Searching for Activation Functions"). The experiment confirms this known property in the capsule context.

### References Check

The `references/redo-dead-neurons/` entry is listed as relevant to `exp15_non_relu_pruning`. The ReDo paper profiles activation norms for dead neuron detection -- the same approach used here. The experiment correctly adapts it from binary (ReLU) to threshold-based (SiLU). No reinvention concern.

## Experimental Design

### Does it test the stated hypothesis? YES, with caveats.

The hypothesis is: "magnitude-threshold pruning on SiLU capsule MLPs can achieve meaningful compression (>10% parameter reduction) without degrading quality more than 5%."

The experiment tests this directly. The result: no threshold achieves >10% compression without crossing the activation floor.

### Kill Criterion Mismatch: MODERATE CONCERN

The HYPOTHESES.yml kill criterion is:

> "magnitude-threshold pruning on SiLU capsules degrades quality >5% vs unpruned"

The PAPER.md kill criterion adds:

> Kill criterion: magnitude-threshold pruning degrades quality >5% vs unpruned at any threshold that achieves >10% compression.

The PAPER version includes the ">10% compression" qualifier which makes the criterion harder to fail. Under the HYPOTHESES.yml criterion (no compression qualifier), the experiment trivially passes because you can always set tau=0 and prune nothing. Both formulations ultimately reach the same conclusion -- the experiment cannot be killed by the stated criterion -- but the kill criterion as written in HYPOTHESES.yml is vacuous. A pruning method that prunes nothing always "passes" a quality degradation test.

**The more useful kill criterion would have been**: "SiLU magnitude-threshold pruning achieves <10% compression at <5% quality degradation" -- i.e., kill if it FAILS to compress. The paper's actual conclusion is a negative finding (SiLU is not prunable), which the kill criterion was not designed to detect.

### Controls: ADEQUATE

- ReLU baseline under identical conditions (same seeds, architecture, data, training steps)
- Multiple thresholds swept
- Both mean_abs and max_abs methods tested
- 3 seeds with per-seed reporting

### Missing Control: MINOR

No random-pruning baseline. At tau=0.1, 32% of SiLU capsules are pruned at +1.01% degradation. How does randomly pruning 32% compare? If random pruning also gives ~1% degradation, the magnitude criterion adds nothing. If random gives much worse degradation, the magnitude criterion is doing useful work even in the "aggressive" regime. This would have been a cheap addition.

### Seed Variance at tau=0.1: WELL REPORTED

The per-seed breakdown (85.7% vs 3.7% vs 6.6% pruned) is honestly reported and correctly identified as a problem. The mean (32%) is misleading -- seed 42 is a dramatic outlier. This is excellent transparent reporting.

### Profiling Forward Pass Consistency: VERIFIED

I compared the profiling code (`silu_pruning.py` lines 79-106) against the actual model forward pass (`silu_capsule.py` lines 64-68). Both compute `nn.silu(pool.A(x_norm))` followed by `pool.B(h)`. The profiling correctly mirrors the model's computation path.

## Hypothesis Graph Consistency

The experiment matches its `HYPOTHESES.yml` node (`exp15_non_relu_pruning`). The status is `completed` and the evidence accurately summarizes the finding. However, as noted above, the kill criterion tests the wrong direction -- it asks "does pruning hurt quality?" when the real finding is "pruning cannot compress SiLU models."

The node correctly lists `blocks: [exp5_macro_match]`, since knowing SiLU is not prunable is critical information for the macro experiment.

## Integration with VISION.md

VISION.md Section "Dead neuron rate as a compression signal" states that dead capsule pruning achieves 57% compression with zero quality loss. This experiment correctly establishes that this result is **ReLU-specific** and does not transfer to SiLU. The paper's "Alternative Compression Paths for SiLU Models" section (listing SwiGLU-aware pruning, low-rank factorization, etc.) provides useful next directions.

The experiment does NOT conflict with or duplicate any existing component. It fills a necessary gap: confirming that the ReLU pruning pipeline from Exp 9 does not generalize to the activation function used in production models.

## Macro-Scale Risks (advisory)

1. **SwiGLU gating may change the picture.** Production Qwen/Llama use SwiGLU: `out = fc3(silu(fc1(x)) * fc2(x))`. The gate `fc2(x)` could push the *gated* output toward zero even when `silu(fc1(x))` is nonzero. Profiling the gated output (not raw SiLU output) might find prunable capsules. This is the paper's suggestion #2 and is the highest-priority macro follow-up.

2. **The floor may shift at scale.** At d=896, the pre-activation variance sigma could be very different from d=64. If sigma is much smaller (due to layer normalization), the floor could be lower. If sigma is larger, the floor rises. The paper acknowledges this (Assumption 2). Worth one profiling run at macro scale.

3. **Gradient-based importance may work where magnitude fails.** Fisher information or Taylor expansion of the loss change measures contribution to the loss, not just activation magnitude. A capsule with moderate activation but near-zero gradient contribution is prunable. This is orthogonal to SiLU vs ReLU and could be explored at macro scale.

## Verdict

**PROCEED**

The experiment is well-designed, the math is sound, the code matches the claimed methodology, and the conclusion is honest and correctly scoped. The finding is practically negative (SiLU is not prunable at safe thresholds) but scientifically positive (it closes a question and redirects effort). The per-seed variance reporting is exemplary.

Two minor issues that do not warrant REVISE:

1. The kill criterion tests quality degradation rather than compression failure, making it vacuous (tau=0 always passes). The paper compensates by honestly reporting the negative practical finding. Future experiments should define kill criteria that capture the failure mode being tested.

2. A random-pruning baseline at tau=0.1 would have strengthened the analysis. The cost would have been ~5 minutes of additional compute. Not blocking.

The experiment correctly establishes that ReLU pruning is strictly superior to SiLU pruning for capsule compression, and that production models using SiLU need alternative compression strategies. This is a clean, useful negative result that properly informs the macro direction.
