# Learnings: exp_bitnet_effective_delta_cosine

## Core Finding

The effective-delta cosine (measuring alignment of actual weight perturbations B@A) is 19x HIGHER than raw parameter cosine at production scale (d=2560, 210 modules), opposite to the hypothesis. Per-module A-filtering works, but concatenation across modules inflates the metric. Raw parameter cosine is the more conservative and operationally correct proxy. **KILLED.**

## Why This Happened (Literature-Grounded)

The failure has two distinct mechanisms, both well-documented in the literature:

**1. Subspace cosine does not imply high-dimensional cosine.**

When per-module effective deltas are concatenated into a mega-vector (2.08B dims vs 21.6M for raw params), the cosine of the concatenation is NOT the average of per-module cosines. It is a weighted sum of per-module inner products divided by the product of mega-vector norms. Large MLP modules (gate_proj, up_proj: 17.7M elements each) dominate the aggregate, drowning out the well-behaved attention modules. This is a known property of cosine similarity in composite spaces -- Jina AI's analysis ("Does Subspace Cosine Similarity Imply High-Dimensional Cosine Similarity?") shows that aggregate cosine can exceed any individual subspace cosine when subspace contributions have correlated signs.

**2. Concentration of measure in high dimensions.**

In ~2B dimensions, random vectors have cosine concentrated near zero with very tight variance (O(1/sqrt(d))). The effective-delta vectors, while not random, inherit this property: their cosine concentrates around ~0.019 regardless of adapter pair. Meanwhile, raw parameter vectors in ~21M dimensions have more variance, allowing some pairs to achieve near-zero cosine through cancellation. The 404x worst-case ratio (legal-creative) occurs when raw cosine is near the noise floor (~0.00006) while effective-delta cosine sits at its stable floor (~0.024). This is the "concentration trap" -- higher dimensions make cosine more stable but also raise the floor.

**3. The A-filtering is real but local.**

The per-module trace identity vec(DW_i)^T vec(DW_j) = tr(A_i^T B_i^T B_j A_j) correctly shows that A-orthogonality filters B-correlation at the module level (confirmed: per-module |cos_eff| ~ 0.001-0.005). The error was assuming this local property lifts to the global concatenation. It does not, because the concatenated inner product sums 210 independent per-module terms with uncorrelated signs, and the normalization denominator grows differently for the two metrics.

## Confirming Evidence

1. **Cao et al. (2025) "Efficient Modular Learning through Naive LoRA Summation"** (arxiv:2508.11985): Confirms that RMS cosine similarity of LoRA deltas (measured on raw parameters, not effective deltas) correlates with composition quality. Their metric choice -- raw parameter cosine -- is consistent with our finding that it is the better operational proxy. They demonstrate the Superposition Principle: independently trained LoRA modules on disjoint domains are approximately orthogonal in high-dimensional parameter space.

2. **Zhang & Zhou (2025) "Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging"** (arxiv:2505.22934, ACL 2025): Proposes OSRM to constrain LoRA subspaces to be orthogonal. Crucially, they measure interference in the *parameter* subspace (A and B matrices), not in the effective-delta space. This implicitly validates raw parameter cosine as the standard interference metric in the community.

3. **Johnson-Lindenstrauss concentration**: Classic result. In d dimensions, the cosine between random unit vectors concentrates as O(1/sqrt(d)). At d_eff=2.08B, the concentration is ~50x tighter than at d_raw=21.6M, explaining why effective-delta cosine has much lower variance across pairs (range 0.012-0.025 vs 0.00006-0.0026 for raw).

## Contradicting Evidence

1. **Zhang et al. (2025) "Rethinking Inter-LoRA Orthogonality in Adapter Merging"** (arxiv:2510.03262): Found that strict orthogonality does NOT lead to semantic disentanglement, and in some cases *reduces* quality compared to non-orthogonal merging. This challenges our assumption that lower cosine = better composition. However, their work is on diffusion models (image generation), where the relationship between parameter orthogonality and output quality may differ from language model perplexity. Our prior macro experiments (1/N scaling, LOO ranking) have validated that low cosine correlates with successful composition in the LM setting.

2. **Our own toy-scale result (d=64, single module)**: The 17x decorrelation filter at d=64 was correct for a single module but was incorrectly extrapolated to multi-module aggregation. This is a cautionary tale about scale-dependent findings -- a result at toy scale with a single module does not predict behavior at 210 modules with 96x more dimensions.

## Alternative Approaches (What We Could Try Instead)

1. **Per-module cosine monitoring (no aggregation)**: Instead of a single aggregate metric, track the distribution of per-module cosines. This preserves the A-filtering signal without the aggregation trap. Could be implemented as max/mean/p95 over modules in `tools/orthogonality.py`. Low priority since raw parameter cosine already works.

2. **AWOM metric (Adapter Weight Orthogonality Magnitude)**: Proposed in the orthogonal LoRA literature as a standardized metric. Measures Frobenius norm of the cross-correlation matrix between adapter weight matrices. Worth investigating if we ever need a more nuanced metric than raw cosine.

3. **Subspace angle (Grassmann distance)**: Instead of cosine on flattened vectors, measure the principal angles between the column spaces of A matrices. This is what our Grassmannian skeleton already uses and is immune to the aggregation trap. The existing AP packing analysis is the right tool for orthogonality measurement at the subspace level.

4. **OSRM-style constrained training** (Zhang & Zhou, ACL 2025): Constrain LoRA subspaces to be orthogonal *during training*, rather than measuring post-hoc. This guarantees zero interference by construction. However, it requires coordinated training (all adapters must be aware of each other's subspaces), which conflicts with our independent training desideratum.

## Implications for Next Experiments

1. **`tools/orthogonality.py` needs no changes.** Raw parameter cosine is validated as the correct operational metric. Do NOT implement `--effective-delta` mode.

2. **VISION.md should qualify the "17x decorrelation filter" claim** as a per-module property, not an adapter-level property. The Grassmannian skeleton provides per-module orthogonality guarantees, and this is correct and valuable -- but the headline "17x" number measured a single module at toy scale.

3. **For exp_bitnet_lori_sparse_b** (downstream dependency): The kill of effective-delta cosine does NOT invalidate the sparse-B approach. B-sparsity reduces per-module interference at the source. The learning is that interference should be measured with raw parameter cosine, not effective-delta cosine. The sparse-B experiment should use raw cosine as its primary metric.

4. **Three consecutive kills suggest the "foundation fixes" track is yielding diminishing returns.** The raw parameter cosine (mean=0.001, 50x below threshold) is already excellent. Further refinement of the orthogonality metric is academic. Priority should shift to Track 2 (base-free scaffold) and Track 3 (serving), where the unknowns are larger.

## New References to Add

| Paper | arxiv ID | Relevance |
|-------|----------|-----------|
| Efficient Modular Learning through Naive LoRA Summation | 2508.11985 | Confirms raw param cosine as standard metric; Superposition Principle for LoRA |
| Unraveling LoRA Interference (OSRM) | 2505.22934 | Orthogonal subspace constraint for LoRA merging (ACL 2025) |
| Rethinking Inter-LoRA Orthogonality | 2510.03262 | Challenges that orthogonality alone ensures compositionality |
