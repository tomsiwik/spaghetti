# Peer Review: Procrustes Expert Transfer

## NotebookLM Findings

Skipped -- the experiment is already self-killed (K2) with honest reporting. Deep review resources are better spent on active experiments.

## Mathematical Soundness

### Procrustes formulation (MATH.md Section 2.1): Correct

The SVD-based solution to the orthogonal Procrustes problem is textbook (Schonemann 1966). The implementation correctly computes M = B @ A^T, takes SVD, forms R = U @ V^T, and handles the reflection case (det < 0). No issues.

### Activation-space alignment (MATH.md Section 2.2, code lines 156-226): Correct

The hook-based activation collection, transposition convention (R @ H_A^T ~ H_B^T), and application of the same Procrustes solver are all consistent. The code correctly collects from the layer output (the residual stream after each transformer block), which is the right place to align.

### Expert delta transformation (MATH.md Section 2.3, code lines 229-348): Has a subtle issue

The MATH.md derives:
- fc1: dW_fc1_B = R_l @ dW_fc1_A (rotate input dimension)
- fc2: dW_fc2_B = dW_fc2_A @ R_l^T (rotate output dimension)

The code implements:
- fc1: `d_transformed = R_l @ d` where d has shape (d, 4d) -- this applies R_l on the left, which rotates the d-dimensional input. Correct.
- fc2: `d_transformed = d @ R_l.T` where d has shape (4d, d) -- this applies R_l^T on the right, which rotates the d-dimensional output. Correct.

However, there is a conceptual gap the paper acknowledges but does not fully address: the transformation only handles the residual-stream dimensions (d). The intermediate 4d space is model-specific and unaligned. For fc1, the output (4d) is unrotated. For fc2, the input (4d) is unrotated. This means the transformation is incomplete -- it handles only one of the two dimensions of each weight matrix. The paper honestly states this as a "fundamental limitation" (MATH.md line 77-79), which is appropriate.

### Alignment error interpretation (MATH.md Section 3): Mostly sound but one claim is imprecise

The paper states (line 107): "The expected alignment error for two random d x d matrices: E[||R @ A - B||_F / ||B||_F] ~ sqrt(2 - 2/d) ~ sqrt(2) for large d."

This formula applies when A and B are drawn from the same distribution and are independent. The actual bound for Gaussian random matrices involves the ratio of Frobenius norms and depends on the distribution. The claim is directionally correct (random matrices have high alignment error, observed values are below random) but the specific formula sqrt(2 - 2/d) needs a citation or derivation. This is non-blocking -- the qualitative conclusion (observed < random) holds regardless.

### fc2 anomaly (MATH.md Section 3.2): Imprecise claim

"fc2 weights have alignment errors ~1.1 (above sqrt(2) ~ 1.41 threshold for random)." The paper says 1.1 is "above" sqrt(2) ~ 1.41, but 1.1 < 1.41. The sentence is confusingly written -- it appears to claim fc2 is worse than random, but 1.1 < 1.41 means it is actually better than random. The qualitative point (fc2 alignment is poor because of the unaligned 4d intermediate space) is correct, but the numerical comparison is backwards. Non-blocking since the conclusion is the same either way.

## Novelty Assessment

### Prior art

Procrustes alignment for neural networks is well-studied:

1. **Ainsworth et al. (2022), Git Re-Basin**: The closest prior work. Uses weight matching (Hungarian algorithm for permutations) combined with orthogonal alignment. The paper cites this correctly.

2. **Li et al. (2015), "Convergent Learning"**: Showed that independently trained networks converge to equivalent solutions up to permutation. Not cited but implicitly addressed.

3. **Model stitching literature** (Lenc & Vedaldi 2015, Bansal et al. 2021): Activation-space alignment for representation similarity. The approach in this experiment is standard in that literature.

### Delta over existing work

The specific novelty is applying Procrustes to **LoRA expert transfer** (not full model merging). This is a reasonable and unstudied application. The experiment correctly identifies that the interesting case is not Procrustes itself (well-known) but whether LoRA deltas survive the transformation. This is a valid micro contribution.

No reinvention of existing code from the references/ folder. The experiment correctly builds on `base_free_composition` infrastructure.

## Experimental Design

### Does it test the stated hypothesis? Yes

The hypothesis is that Procrustes alignment enables cross-model expert transfer with <20% quality loss and <5% alignment error. The experiment trains two independent models, trains experts on one, transfers to the other with and without alignment, and measures both quality and alignment error. The design is clean.

### Controls are adequate

Three methods compared (naive, per-weight Procrustes, activation Procrustes). Native experts on model B serve as the gold standard. Three seeds provide consistency evidence.

### Could a simpler mechanism explain the result? Partially

The most interesting finding is that **naive transfer (no alignment) already works at 15.3% gap**. Procrustes adds only 1.7% improvement. This raises the question: at d=64, are the models similar enough that any reasonable transfer works, making Procrustes unnecessary? Or is 1.7% a genuine signal that would amplify at scale?

The paper addresses this honestly (PAPER.md Section 5 comparison table), noting that 15.3% naive gap is comparable to rank-8 SVD perturbation (16.7%). This is a good control comparison.

### Kill criteria assessment is honest and correct

The experiment correctly applies both kill criteria. K1 (quality) survives. K2 (alignment error) is killed by a large margin (26% vs 5%). The paper does not try to rescue the experiment by redefining thresholds -- it honestly reports K2 as killed while noting the threshold may be miscalibrated for independently-trained models. This is exactly the right approach.

### One weakness: the "improvement over naive" metric

The 1.7% improvement is within the standard deviation (0.9% std across seeds). The paper reports consistency (all 3 seeds positive) but does not compute a p-value or confidence interval for whether Procrustes genuinely outperforms naive. At 3 seeds, a paired t-test would give t = (0.020)/(0.009/sqrt(3)) = 3.85, p ~ 0.03 (one-tailed). This is marginally significant but should be reported. Non-blocking since the experiment is already killed on K2.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry matches:
- Kill criteria K1 (>20% worse) and K2 (>5% alignment error) match what was tested
- Status is correctly "killed"
- Evidence lines are accurate and complete
- No dependencies or blockers affected

The kill does not invalidate the parent experiment (zero_shot_base_transfer, proven), which tested same-lineage transfer. The paper correctly positions this as the harder case.

## Integration with VISION.md

VISION.md (line 116-118) includes a row:

| New model, same d (e.g. Qwen to Llama) | No (same Grassmannian) | Procrustes alignment |

This is consistent with the paper's recommendation (PAPER.md Section "Implications for SOLE Architecture"): Procrustes is the expected approach for cross-model transfer, with the caveat that at d=64 it is lossy. The kill at micro does not invalidate this VISION.md entry because the paper argues (with reasonable evidence from Git Re-Basin literature) that alignment error would drop at d=4096. This is an open question for macro validation.

## Macro-Scale Risks (advisory)

1. **Git Re-Basin alignment error at d=4096**: The paper predicts near-zero alignment error at scale based on published Git Re-Basin results. However, Git Re-Basin was tested on ResNets (same architecture, same data, different seeds) -- not on architecturally different models like Qwen vs Llama. Cross-architecture transfer is fundamentally harder because attention head structure differs. Macro should test same-architecture first (Qwen 2.5 vs Qwen 3).

2. **Attention head alignment**: This experiment uses MLP-only LoRA. At macro with all-modules LoRA, attention heads need alignment too. Multi-head attention has additional permutation symmetry (head ordering) that Procrustes alone cannot handle. May need per-head matching.

3. **The 1.7% improvement may not scale**: If naive transfer already works well at d=4096 (because large models converge to similar representations), Procrustes may provide negligible benefit over naive. The cost-benefit may not justify the implementation complexity.

4. **Intermediate space alignment**: The fundamental limitation (unaligned 4d intermediate) persists at any scale. Git Re-Basin addresses this with permutation matching, not rotation alone. A macro experiment should implement the full permutation + rotation pipeline.

## Verdict

**PROCEED** (kill is valid and correctly reported)

The experiment is already self-killed on K2, and the kill is honest and well-documented. No revision is needed because:

1. The math is sound (minor imprecisions in Section 3.2 are non-blocking).
2. The experimental design correctly tests the hypothesis.
3. The kill criteria are applied honestly without threshold manipulation.
4. The findings are valuable: (a) naive transfer works surprisingly well at 15.3% gap, (b) activation Procrustes provides consistent 1.7% improvement, (c) per-weight Procrustes is useless for MLPs, (d) the K2 threshold is miscalibrated for independent models.
5. The paper correctly identifies the path forward: same-lineage transfer (zero_shot_base_transfer) is the practical path; cross-model transfer should use Git Re-Basin (permutation + rotation) at macro.

The FINDINGS.md and HYPOTHESES.yml entries are accurate. No further action required on this experiment.
