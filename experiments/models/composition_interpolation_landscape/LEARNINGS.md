# Learnings: exp_composition_interpolation_landscape

## Core Finding

The PPL landscape of Grassmannian LoRA adapter weight interpolation is smooth and convex. All 2-adapter pairs show high convexity (0.63-1.0), perfect monotonicity per-domain, and interior optima for mixed evaluation. The 3-adapter simplex confirms sub-additivity: edge midpoints are 3.5-5.2% below vertex chords, meaning mixing adapters is always better than pure selection. Gradient-based optimization of composition weights is viable.

## Why This Happened

### Orthogonality produces independent contributions

With Grassmannian A-matrices (|cos(A_i, A_j)| = 0.004), each adapter's contribution delta_W_i = B_i @ A_i^T operates in a near-independent subspace. The composed perturbation delta_W(w) = sum w_i * delta_W_i produces a PPL function that is approximately separable: L(w) ~ sum L_i(w_i). Separable functions are convex if each L_i is convex, which holds because PPL is convex in the logits direction and each adapter moves logits along an independent axis.

This is the mechanism the Naive LoRA Summation paper (arXiv 2508.11985) predicted: orthogonal A-matrices make adapter effects additive, which implies smooth interpolation.

### Sub-additivity from constructive transfer

The simplex midpoints being below vertex chords (3.5-5.2%) shows that mixing adapters is better than expected from linearity. This is consistent with the OSRM finding (arXiv 2505.22934) that composition works via constructive transfer, not merely non-interference. Each adapter contributes some benefit even out-of-domain (activation magnitude ratio 1.08x per exp_softmax_router_scaling), and these small out-of-domain benefits accumulate constructively.

### Slice-based adapters are functionally interchangeable

The code/engineering pair (0.12% PPL range) trained on different offsets of the same code dataset shows near-zero landscape variation. This directly confirms the softmax router scaling finding that within-cluster misrouting has 0.0% oracle gap. The landscape is flat because the adapters learn equivalent perturbations from equivalent data.

## Confirming Evidence

1. **exp_softmax_router_scaling:** Within-cluster misrouting has 0% PPL gap. Our code/engineering pair (0.12% range) confirms this is because within-cluster adapters are functionally interchangeable — the landscape is flat within clusters.

2. **exp_real_data_25_domain_adapters:** Uniform N=24 composition gives -29.1% vs base. Our finding that uniform is only 0.7% from optimal on the simplex explains why uniform works well: the landscape is smooth enough that any reasonable weighting captures most of the benefit.

3. **Model Soups (Wortsman et al., ICML 2022):** Showed weight-space averaging of fine-tuned models finds a convex basin. Our result extends this to LoRA adapter interpolation with Grassmannian orthogonality — the basin is even smoother due to enforced subspace independence.

## Contradicting Evidence

1. **The optimal is NOT uniform.** The simplex optimum (0.1, 0.5, 0.4) for medical/math/code gives math 5x the weight of medical. If the landscape were perfectly symmetric (all adapters equivalent), uniform would be optimal. The asymmetry reflects differing per-domain PPL magnitudes and adapter quality. At N=24, this asymmetry could compound.

2. **Per-domain optima are at vertices, not interior.** For single-domain evaluation, the best weight is always 1.0 on the target adapter. This means soft routing only helps for mixed/multi-domain evaluation. If the deployment scenario is single-domain, top-1 hard routing is optimal.

3. **Flatness of code/engineering may indicate adapter quality issue.** These adapters being interchangeable means training on offset slices of the same dataset produces redundant adapters. The Grassmannian skeleton provides distinct subspaces, but if the training data doesn't provide distinct signal, the B-matrices converge to equivalent perturbations. This is a data diversity problem, not an architecture problem.

## Alternative Approaches

1. **Learned routing weights via backpropagation.** Now that the landscape is confirmed convex, router output can be directly used as mixture weights and trained end-to-end with PPL loss. The smooth gradient signal means simple SGD would converge.

2. **Bayesian optimization for N>3.** For high-dimensional simplices (N=24), grid search is infeasible but Bayesian optimization works well on smooth, low-effective-dimension landscapes. The smoothness confirmed here makes BO appropriate.

3. **Per-token adaptive weights.** Instead of fixed composition weights, use the router's softmax output as per-token mixing weights. The smoothness guarantee means small weight changes produce small PPL changes, enabling stable per-token adaptation.

## Implications for Next Experiments

1. **Soft top-k routing is the next step.** Top-2 or top-3 with learned mixture weights should outperform top-1 hard routing, since mixing is always better than selection (sub-additivity result). The landscape smoothness means the mixture weights are gradient-trainable.

2. **Uniform composition is a strong baseline.** The 0.7% gap from uniform to optimal means the current 1/N composition is already near-optimal. Routing optimization is a refinement worth ~1% PPL, not a paradigm shift.

3. **Within-cluster routing precision doesn't matter.** The code/engineering flatness (0.12% range) means the router only needs to identify the correct cluster, not the exact adapter. This confirms the softmax router's design: 40% per-adapter accuracy but 0% quality gap because clusters are what matters.

4. **Data diversity is more important than adapter count.** Slice-based adapters add nothing to the landscape — they're equivalent points in weight space. Future scaling should prioritize genuinely diverse domains over N for the sake of N.
