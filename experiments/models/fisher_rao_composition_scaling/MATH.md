# Fisher-Rao Manifold Composition: Stable Adapter Merging Beyond N=5

## Type: Verification (Type 1)

**Paper:** Fisher-Rao Manifold Merging (arXiv:2603.04972, Wang, Ye, Yin 2025)
**Prior findings:** Finding #14 (1/N scaling fixes composition catastrophe), Finding #8 (L2 norm composition stable to N=25), VISION.md (composition scales to N=25 with gamma=0.982).

## A. Failure Mode Identification

**Disease:** Norm shrinkage under Euclidean averaging of adapted model parameters.

When N adapted models with parameters {theta_i} are averaged in Euclidean space as theta_merged = (1/N) sum theta_i, the resulting norm shrinks as N grows. This is because parameter vectors pointing in different directions partially cancel. The shrinkage is:

||theta_merged|| / mean(||theta_i||) = ||sum theta_i|| / (N * mean(||theta_i||))

For adapter deltas delta_i = B_i^T A_i^T with Grassmannian-orthogonal A matrices, the deltas point in nearly orthogonal directions. The cosine between random unit vectors in R^d concentrates around 0 with standard deviation 1/sqrt(d). But the *sum* of N orthogonal unit vectors has norm sqrt(N), while the scaled sum (1/N)*sum has norm sqrt(N)/N = 1/sqrt(N).

**The shrinkage ratio decays as 1/sqrt(N).** At N=10, norms shrink to ~31.6% of original. At N=25, to ~20%. This is the root cause of activation variance collapse and effective rank degradation at high N.

**This IS a stable fixed point of naive training:** once norms shrink, gradient magnitudes shrink proportionally, making recovery harder. The model settles into a low-norm, low-variance regime.

## B. The Right Question

**Wrong question:** "How do we rescale after averaging to preserve norms?"
(This is ad-hoc: 1/N, 1/sqrt(N), or learned weights are all patches.)

**Right question:** "What is the geometrically natural averaging operation on the parameter manifold that preserves norms by construction?"

**Answer:** The Frechet/Karcher mean on the unit sphere S^(d-1). By definition, the Karcher mean lies on the sphere, so its norm is 1. Combined with separate norm averaging, this decouples direction averaging from magnitude averaging, eliminating norm shrinkage by construction.

## C. Prior Mathematical Foundations

### C1. Karcher Mean on Riemannian Manifolds (Karcher 1977, Kendall 1990)

**Definition.** Given points {p_i} on a Riemannian manifold (M, g) with weights {alpha_i} summing to 1, the Karcher mean is:

  theta* = argmin_{theta in M} sum_i alpha_i * d_g(theta, p_i)^2

where d_g is the geodesic distance.

**Theorem (Karcher 1977).** On a complete Riemannian manifold with sectional curvature bounded above by kappa > 0, if all points lie within a geodesic ball of radius r < pi/(2*sqrt(kappa)), the Karcher mean exists, is unique, and the fixed-point iteration:

  theta_{t+1} = Exp_{theta_t}(eta * sum_i alpha_i * Log_{theta_t}(p_i))

converges for eta in (0, 1].

### C2. Unit Sphere S^(d-1)

The unit sphere S^(d-1) has constant sectional curvature kappa = 1. The injectivity radius is pi. By Karcher's theorem, if all points lie within a geodesic ball of radius pi/2 (i.e., all pairwise dot products > 0), the Karcher mean is unique and the iteration converges.

**Log map:** Log_p(q) = (theta/sin(theta)) * (q - cos(theta) * p), where theta = arccos(<p,q>).

**Exp map:** Exp_p(v) = cos(||v||) * p + sin(||v||) * (v/||v||).

### C3. Norm-Direction Decomposition (arXiv:2603.04972, Section 3.4)

The spherical proxy factorizes each parameter block theta as:
  theta = ||theta|| * (theta / ||theta||) = r * u

where r = ||theta|| in R+ and u in S^(d-1).

The merge computes:
- Direction: u* = KarcherMean({u_i}, {alpha_i}) on S^(d-1)
- Magnitude: r* = mean({r_i}) (arithmetic mean of source norms)
- Result: theta* = r* * u*

This decoupling is the key insight: direction averaging on the manifold cannot cause norm shrinkage because the Karcher mean stays on S^(d-1) by construction.

### C4. Norm Shrinkage of Euclidean Averaging (Jang et al. 2024, cited in 2603.04972)

**Proposition.** For N unit vectors {u_i} in R^d with pairwise cosines cos(theta_ij):

||mean(u_i)|| = sqrt((1/N) + (1/N^2) sum_{i!=j} cos(theta_ij))

When vectors are nearly orthogonal (cos ~ 0), this reduces to 1/sqrt(N). The norm shrinkage ratio is 1/sqrt(N), independent of dimension d.

## D. Proof of Guarantee

**Theorem 1 (Norm Preservation).** Let {theta_i}_{i=1}^N be N parameter vectors with norms r_i = ||theta_i|| and unit directions u_i = theta_i/r_i. Let theta_FR = r_FR * u_FR be the Fisher-Rao spherical proxy merge, and theta_E = (1/N) sum theta_i the Euclidean merge. Then:

(a) ||theta_FR|| / mean(r_i) = 1 (exact norm preservation)

(b) ||theta_E|| / mean(r_i) <= 1, with equality iff all u_i are identical

*Proof.*

(a) By construction, u_FR lies on S^(d-1), so ||u_FR|| = 1. And r_FR = mean(r_i). Therefore ||theta_FR|| = r_FR * ||u_FR|| = mean(r_i) * 1 = mean(r_i). The ratio is 1. QED for (a).

(b) theta_E = (1/N) sum r_i * u_i. By triangle inequality:
||theta_E|| = ||(1/N) sum r_i u_i|| <= (1/N) sum r_i ||u_i|| = (1/N) sum r_i = mean(r_i).
Equality holds iff all r_i u_i are non-negative multiples of a common direction, i.e., all u_i are identical (since r_i > 0). QED for (b).

**Corollary (Shrinkage Bound for Orthogonal Adapters).** When adapter deltas are orthogonal (as guaranteed by Grassmannian A-matrices with cos ~ 0.025 at d=2560), the Euclidean norm shrinkage ratio approaches 1/sqrt(N):

||theta_E|| / mean(r_i) ~ 1/sqrt(N)

At N=10, this is 0.316. At N=15, this is 0.258. Fisher-Rao stays at 1.0 for all N.

**Conjecture 2 (Activation Variance).** *(Downgraded from Theorem: the original linear response model predicted variance should DECREASE with N for Euclidean averaging. Experiment showed variance INCREASES with N for both methods. The proof predicted the wrong sign.)*

The original argument: under linear response with orthogonal adapters, Var ~ sigma^2_base + ||Delta||^2 * c. Since ||Delta_E||^2 ~ ||delta||^2/N (shrinking), variance should decrease. Since ||Delta_FR||^2 ~ ||delta||^2 (constant), variance should be stable.

**What actually happens:** Variance INCREASES with N for both methods. This indicates the linear response model is inadequate. Multi-domain composition introduces diverse hidden state trajectories that increase inter-sample variance even as weight norms shrink. The nonlinear interaction between different adapter domains creates new activation directions not present in any single adapter.

**Revised conjecture:** For norm-preserving composition (Fisher-Rao or norm-rescaled Euclidean), activation variance is stable within ~10% of N=1 (at fixed scale). For raw Euclidean averaging, activation variance increases because the norm shrinkage changes the operating point of the model nonlinearly.

**Observed:** FR act. var. ratio (N=10 vs N=1) = 1.074 (within 10%). Euclidean ratio = 1.148. Both INCREASE, contradicting the original linear response prediction.

**Conjecture 3 (Effective Rank).** *(Downgraded from Theorem: no formal proof was ever provided. The original statement was a description dressed as a theorem.)*

The original claim: effective rank should degrade with norm shrinkage because smaller singular values fall below the noise floor.

**What actually happens:** Effective rank INCREASES with N for both methods. Multi-domain composition diversifies the activation space, increasing the number of significant singular values. Single-domain (N=1, medical) has eff. rank 4.55; multi-domain (N=5) has eff. rank 5.11-5.12 (FR) or 5.48 (Euclidean).

**Revised conjecture:** Effective rank increases with domain diversity. The relevant factor is the number of distinct domains, not the composition method. Both norm-preserving methods (FR, norm-rescaled Euclidean) produce similar effective rank (~5.11), lower than raw Euclidean (~5.48), possibly because stronger adapter signal suppresses noise-driven rank inflation.

## D. Quantitative Predictions

For our setup (BitNet-2B-4T, d=2560, rank-16 Grassmannian adapters, 5 real domains):

### Predictions from Theorem 1 (proven, verified):

| Metric | N | Euclidean Prediction | FR/NRE Prediction | Source |
|--------|---|---------------------|-------------------|--------|
| Norm shrinkage ratio | 5 | 1/sqrt(5) = 0.447 | 1.0 | Theorem 1 |
| Norm shrinkage ratio | 10 | 1/sqrt(10) = 0.316* | 1.0 | Theorem 1 |
| Norm shrinkage ratio | 15 | 1/sqrt(15) = 0.258* | 1.0 | Theorem 1 |
| PPL (norm-preserved vs raw Euc) | 5+ | degraded | better | Theorem 1 corollary |

*Note: N=10,15 use synthetic adapters (noisy copies of 5 real ones). Euclidean shrinkage plateaus at 0.447 = 1/sqrt(5), confirming the effective independent dimension is 5, not N. Predictions for N>5 are only valid with truly independent adapters.

### Predictions from Conjectures 2-3 (unproven, directionally wrong in original form):

| Metric | Original Prediction | Revised Prediction | Source |
|--------|--------------------|--------------------|--------|
| Act. var. ratio (N=10/N=1) Euc | < 0.5 (WRONG) | increases (~1.15) | Conjecture 2 |
| Act. var. ratio (N=10/N=1) FR | > 0.9 | ~1.07 (correct direction, wrong mechanism) | Conjecture 2 |
| Eff. rank degradation FR | < 5% | improves by ~12% | Conjecture 3 |

**Key behavioral prediction (revised):** Norm preservation is the primary mechanism. Both Fisher-Rao (Karcher mean) and norm-rescaled Euclidean achieve it and produce equivalent PPL, activation variance, and effective rank. The advantage over raw Euclidean averaging comes entirely from preventing 1/sqrt(N) norm shrinkage, not from directional quality of the Karcher mean.

## E. Assumptions and Breaking Conditions

1. **Small adapter perturbation (linear response).** Adapter deltas must be small relative to base weights. For our rank-16 adapters at scale 20.0, the Frobenius norm of delta is O(scale * sqrt(r)) ~ 80, while base weight norms are O(sqrt(d*d)) ~ 2560. Ratio ~3%. Assumption holds.

2. **Near-orthogonal adapter deltas.** Required for the 1/sqrt(N) shrinkage analysis. Our Grassmannian A-matrices guarantee this with mean |cos| = 0.00125 (Finding #3). Assumption holds.

3. **Positive pairwise dot products of unit vectors (for Karcher convergence).** Since adapter deltas are *perturbations* to the same base model, the full parameter vectors theta_i = W + delta_i all point in nearly the same direction (cos ~ 1 - epsilon). The Karcher iteration converges trivially. Assumption holds strongly.

4. **Equal adapter norms.** The norm-preservation guarantee of Fisher-Rao is exact. But if source norms vary wildly, the "mean norm" choice may not be optimal. Our adapters have optimized per-domain scales (1.0-20.0), so norms DO vary. The direction-magnitude decoupling still works; the magnitude choice is a separate question.

**If Assumption 1 fails** (large perturbations): Linear response breaks, and the relationship between weight-space norm and activation variance becomes nonlinear. Theorem 2's prediction weakens but Theorem 1 still holds exactly.

**If Assumption 2 fails** (correlated adapters): Euclidean shrinkage is *less* severe (correlated vectors don't cancel as much), reducing the advantage of Fisher-Rao. But this means composition is redundant anyway.

## F. Worked Example (d=4)

Let d=4, N=3 adapter deltas (unit vectors):
- u1 = [1, 0, 0, 0]
- u2 = [0, 1, 0, 0]  
- u3 = [0, 0, 1, 0]

With norms r1 = r2 = r3 = 2.0.

**Euclidean average:** (1/3)(2*u1 + 2*u2 + 2*u3) = (2/3)[1, 1, 1, 0]
Norm = (2/3)*sqrt(3) = 1.155. Mean source norm = 2.0.
Shrinkage ratio = 1.155/2.0 = 0.577 = 1/sqrt(3). Matches prediction.

**Fisher-Rao:**
Direction: Karcher mean of {u1, u2, u3} on S^3.
With equal weights, this is the projection of (u1+u2+u3)/||u1+u2+u3|| = [1,1,1,0]/sqrt(3).
(For orthogonal vectors, Karcher mean = normalized Euclidean mean.)
||direction|| = 1 (on sphere).
Magnitude: mean(2.0, 2.0, 2.0) = 2.0.
Result: 2.0 * [1,1,1,0]/sqrt(3) = [1.155, 1.155, 1.155, 0].
Norm = 2.0. Shrinkage ratio = 2.0/2.0 = 1.0. Exactly preserved.

**Difference:** Euclidean output has norm 1.155; Fisher-Rao has norm 2.0. The 73% norm boost from Fisher-Rao preserves adapter contribution strength.

## G. Complexity and Architecture Connection

**Karcher mean iteration:** Per block of dimension d:
- Log map: O(d) per point, O(N*d) per iteration
- Exp map: O(d) per iteration
- Total per block: O(N*d*T) where T is iterations to convergence (typically T < 20)
- Over all blocks: O(N*D*T) where D = total parameters

For our setup: N <= 15, D ~ 2M (adapter params only), T ~ 10. Total: ~300M FLOPs. Negligible compared to model evaluation.

**Memory:** O(N*D) to hold all adapter parameters + O(D) for working state. With 5 adapters at ~50KB each, total is ~300KB. No concern.

**Integration:** The Fisher-Rao merge replaces the current `sum(delta_i / N)` composition with:
1. Normalize each delta_i to unit sphere
2. Compute Karcher mean direction
3. Rescale by mean source norm
4. Apply to base model

This is a drop-in replacement for the composition step. The existing Grassmannian skeleton, training pipeline, and routing infrastructure are unchanged.

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Norm preservation after averaging (whether via Karcher mean or simple rescaling) prevents 1/sqrt(N) shrinkage. The Karcher mean achieves this by construction; norm-rescaled Euclidean achieves it by post-hoc correction.

2. **Which existing theorem(s) does the proof build on?**
   Karcher (1977) existence/uniqueness of Frechet mean on complete Riemannian manifolds with curvature bounds. Jang et al. (2024) norm shrinkage analysis of Euclidean averaging.

3. **What specific numbers does the proof predict?**
   Norm shrinkage ratio: Euclidean 1/sqrt(N), norm-preserved methods 1.0 (Theorem 1 -- verified).
   Activation variance and effective rank predictions from original Theorems 2-3 were wrong in direction and have been downgraded to conjectures.

4. **What would FALSIFY the proof?**
   If norm-preserved composition does NOT improve PPL over raw Euclidean (would mean norm shrinkage is beneficial, not harmful). This was tested and falsified: norm preservation improves PPL by ~12% at N=5.
   If Karcher mean significantly outperforms norm-rescaled Euclidean (would indicate directional quality matters). This was tested: no significant difference found.

5. **How many hyperparameters does this approach add?**
   0 new hyperparameters. Both Karcher mean and norm-rescaled Euclidean are parameter-free.

6. **Hack check:** The experiment reveals that the Riemannian manifold machinery is overkill. A one-line norm rescaling achieves the same result. This is actually a positive finding: it identifies norm preservation as the single relevant property, simplifying the composition pipeline.
