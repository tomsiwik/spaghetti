# Learnings: exp_bitnet_lori_sparse_b

## Core Finding

LoRI-style 90% B-sparsity is redundant on BitNet-2B ternary bases because the ternary weight constraint already provides near-zero adapter interference (|cos|=0.0016), making parameter-space sparsification a solution to a problem that does not exist in this regime.

## Why This Happened (Literature-Grounded)

The LoRI mechanism (arXiv 2504.07448, COLM 2025) operates through two complementary channels: (1) frozen A matrices create approximately orthogonal random projections, and (2) 90% B-sparsity isolates parameter updates to reduce overlap. On FP16 models (Llama-3), where adapter cosine similarity is ~0.142, both channels provide meaningful interference reduction.

On BitNet-2B-4T, the ternary weight constraint {-1, 0, 1} acts as a natural decorrelator. Our prior experiment (bitnet_cosine_convergence) proved that |cos| plateaus at 0.00125 and is not a training artifact -- the ternary geometry itself produces near-random adapter directions. This is a **floor effect**: when interference is already 114x below FP16 levels, no parameter-space intervention can improve it further.

The interference decomposition from MATH.md shows why:
```
interference(i,j) = (alpha/r)^2 * ||B_i^T A_i^T A_j B_j||_F
```
When ||A_i^T A_j||_F is already near-zero (the ternary regime), modifying B structure is operating on the wrong bottleneck. The interference is dominated by the A-matrix cross-product, not B-matrix overlap.

Critically, magnitude pruning actually **increased** cosine similarity by 1.46x (0.00156 -> 0.00229). This is consistent with the signal concentration hypothesis: pruning removes low-magnitude parameters and concentrates the surviving signal into fewer dimensions, increasing the probability of cross-adapter overlap in those dimensions. The paper "What Makes a Good Prune? Maximal Unstructured Pruning for Maximal Cosine Similarity" (OpenReview, jsvvPVVzwf) confirms that magnitude pruning maximizes cosine similarity between pruned and parent networks -- the same mechanism that, when applied per-adapter, concentrates signals into overlapping high-importance positions.

## Confirming Evidence

1. **Rethinking Inter-LoRA Orthogonality** (arXiv 2510.03262): Empirical analysis shows that inter-LoRA orthogonality "does not lead to the semantic disentanglement highlighted in prior work on compositional adaptation." This EXTENDS our finding -- even if B-sparsity had improved orthogonality, it might not have improved semantic composition. Orthogonality in parameter space is necessary but not sufficient for interference-free composition.

2. **BitNet-2B Cosine Convergence** (this project, bitnet_cosine_convergence): |cos| plateaus at 0.00125 after 2000 steps (40x below 0.05 threshold), confirming the floor is architectural, not a training artifact.

3. **BitNet-2B Real Composition** (this project, bitnet_2b_real_composition): N=5, mean |cos|=0.001, composition ratio 3.59x. Dense adapters already compose near-optimally without any sparsification.

4. **BitLoRA** (ScienceDirect, Qi et al.): Confirms that ternary weights are "fundamentally incongruous" with standard LoRA architecture, requiring architectural redesign rather than parameter-space tricks.

## Contradicting Evidence

1. **LoRI's non-orthogonality benefits** (arXiv 2504.07448): LoRI identifies three benefits beyond interference reduction that could theoretically help even in low-interference regimes:
   - **Regularization**: 90% sparsity acts as a strong regularizer, preserving pretrained knowledge. However, our K1 results show individual adapter quality is nearly identical (max 1.2% degradation), meaning regularization neither helps nor hurts significantly.
   - **Catastrophic forgetting prevention**: B-sparsity isolates parameter updates across sequential tasks. This is relevant for continual learning but NOT for our simultaneous 1/N composition scenario.
   - **Positive knowledge transfer via 1% mask overlap**: LoRI finds the ~1% expected mask overlap enables beneficial cross-task transfer. On BitNet-2B, the ternary base already provides this implicitly through its constrained feature space.

2. **LoRI's capacity limitation**: LoRI's own ablation shows sparse adapters suffer from "limited capacity" requiring larger learning rates. We observed this indirectly: sparse creative writing loss actually increased during training (1.164 -> 1.580), suggesting capacity was insufficient for that domain. If we needed higher individual quality, sparsity would actively harm it.

No paper was found showing B-sparsity improving composition in a naturally low-interference regime. The mechanism is specifically designed for high-interference (FP16) contexts.

## Alternative Approaches (What We Could Try Instead)

1. **Orthogonal Gradient Projection (Ortho-LoRA)** (arXiv 2601.09684): Projects conflicting task gradients onto orthogonal complements during training. Recovers 95% of single-task performance gap. However, this solves a training-time conflict problem -- irrelevant when we train adapters independently.

2. **Orthogonal Subspaces for Model Merging** (arXiv 2505.22934, ACL 2025): Constrains LoRA adapters to orthogonal subspaces during training for better post-hoc merging. This is essentially what our Grassmannian skeleton already does (frozen A via AP packing). Our approach is more principled (geometric guarantee vs learned constraint).

3. **O-LoRA** (arXiv 2310.14152): Learns tasks in distinct orthogonal low-rank subspaces for continual learning. Relevant for sequential composition but not our parallel 1/N scaling.

4. **Runtime LoRA serving (no merge)**: Since interference is already near-zero, the most impactful next step is not further interference reduction but **runtime serving** where adapters are applied dynamically without merging. This avoids the composition question entirely at the cost of inference overhead. This is Track 3 (exp_bitnet_llamacpp_serving).

5. **Individual adapter quality improvement**: With composition essentially solved (|cos| at floor), the binding constraint shifts to individual adapter quality. The KR-Test evaluation (exp_bitnet_kr_test_evaluation) and effective delta cosine (exp_bitnet_effective_delta_cosine) address this directly.

## Implications for Next Experiments

1. **Interference reduction is a solved problem on BitNet-2B.** No further experiments targeting orthogonality improvement are needed. The ternary base provides this for free. This applies to: B-sparsity (killed), structured masks, TIES-Merging-style interventions, and any other parameter-space overlap reduction technique.

2. **The bottleneck has shifted to individual adapter quality and scaling.** With |cos| at the floor, the composition quality is bounded by individual adapter PPL. Focus should be on:
   - Better training recipes (longer training, curriculum, data quality)
   - Quality metrics beyond PPL (KR-Test, task-specific evals)
   - Scaling N (how many adapters before cumulative interference matters)

3. **Magnitude pruning is counterproductive for composition on ternary bases.** The 1.46x cosine increase from magnitude pruning is a genuine anti-pattern. If storage compression is needed, prefer random pruning (which preserves directional uniformity) over magnitude pruning.

4. **B-sparsity should be revisited ONLY if**: (a) the project returns to FP16 bases, (b) N exceeds ~1000 and cumulative interference becomes measurable, or (c) sequential/continual learning replaces parallel composition.

5. **The "orthogonality != composability" finding from arXiv 2510.03262 deserves attention.** Even with near-perfect orthogonality, there may be semantic interference that cosine similarity doesn't capture. The effective delta cosine experiment (exp_bitnet_effective_delta_cosine) should measure vec(B@A) cosine, which is closer to the functional interference than raw parameter cosine.

## New References to Add

1. **Rethinking Inter-LoRA Orthogonality in Adapter Merging** (arXiv 2510.03262) - Orthogonality does not guarantee semantic disentanglement. Directly relevant to interpreting our cosine metrics.

2. **Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging** (arXiv 2505.22934, ACL 2025) - Orthogonal subspace constraints during training. Comparison point for our Grassmannian approach.
