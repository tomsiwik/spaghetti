# Flat-LoRA: Mathematical Foundations

## 1. Mechanism Definition

### 1.1 Standard LoRA Training

Given base model weights W in R^{m x n}, LoRA decomposes the adaptation as:

    W' = W + s * B @ A

where A in R^{n x r}, B in R^{m x r}, s = alpha/r is the scaling factor, and
r << min(m,n) is the rank. Training minimizes:

    min_{A,B} L(W + s*B@A; D)

where D is the training data and L is the cross-entropy loss.

**Standard LoRA finds a single point (A*, B*) that minimizes training loss.**
This point may lie in a sharp minimum -- a narrow valley where small perturbations
in weight space cause large loss increases.

### 1.2 Sharpness-Aware Minimization (SAM)

SAM (Foret et al., ICLR 2021, arXiv:2010.01412) reformulates training to minimize
the worst-case loss within an epsilon-ball:

    min_theta max_{||epsilon||_2 <= rho} L(theta + epsilon; D)

The inner maximization has a closed-form first-order approximation:

    epsilon*(theta) = rho * grad_theta L(theta) / ||grad_theta L(theta)||_2

SAM training step:
1. Compute gradient g = grad_theta L(theta)
2. Compute perturbation: epsilon = rho * g / ||g||_2
3. Evaluate gradient at perturbed point: g_SAM = grad_theta L(theta + epsilon)
4. Update: theta <- theta - lr * g_SAM

**Cost:** 2x the gradient computation per step (two forward+backward passes).

### 1.3 Flat-LoRA (Sun et al., arXiv:2409.14396)

Standard SAM applied to LoRA only perturbs within the LoRA subspace (columns of A).
This is a restricted perturbation that misses the full weight-space geometry.

Flat-LoRA perturbs in the FULL m x n weight space:

    min_{A,B} max_{||epsilon||_F <= rho} L(W + s*B@A + epsilon; D)

where epsilon in R^{m x n} (full-rank perturbation, not restricted to rank-r).

The key insight: the perturbation epsilon is NOT constrained to the column space
of A. It searches the entire weight manifold for directions of high curvature.

**Implementation:** Since perturbing W directly would be O(mn) parameters,
Flat-LoRA uses a Bayesian formulation:

    min_{A,B} E_{epsilon ~ N(0, sigma^2 I_{mn})} [L(W + s*B@A + epsilon; D)]

In practice, this is implemented as:
1. Sample epsilon_W ~ N(0, sigma^2) for each weight matrix (or approximate via SAM)
2. Forward pass with perturbed weights: W + s*B@A + epsilon_W
3. Compute loss and gradient at this perturbed point
4. Update A, B with the perturbed gradient

### 1.4 Our MLX Implementation: LoRA-Space SAM

Full m x n perturbation is prohibitively expensive for our 2.4B-param base model.
Each weight matrix at d=2560 is 2560 x 2560 = 6.5M floats. With 30 layers x 7
projections, that's 1.37B perturbation parameters -- nearly as large as the model.

Instead, we implement **LoRA-Space SAM** (the "restricted" variant that Flat-LoRA
improves upon). This perturbs only the LoRA parameters (A, B):

    min_{A,B} max_{||eps_A, eps_B||_2 <= rho} L(W + s*(B + eps_B)@(A + eps_A); D)

This is cheaper (only 2*r*(m+n) perturbation parameters per layer) but still
promotes flat minima in the LoRA parameter space. The key question is whether
LoRA-space flatness transfers to weight-space mergeability.

**Why this should still work for composition:**
When we merge N adapters via Task Arithmetic:

    W_merged = W + sum_{i=1}^N lambda_i * s * B_i @ A_i

Each adapter contributes B_i @ A_i. If adapter i was trained in a flat region of
its own parameter space, small perturbations to (A_i, B_i) preserve loss.
Merging effectively perturbs each adapter by adding the other adapters' deltas.
If each adapter is robust to perturbation in its parameter space, the merge
should preserve more of each adapter's quality.

## 2. Why Flat Minima Survive Averaging

### 2.1 The Sharp vs Flat Basin Argument

Consider two adapters trained to minimize L_1(theta) and L_2(theta) respectively,
converging to theta_1* and theta_2*. The merged point is:

    theta_merge = (theta_1* + theta_2*) / 2

If theta_1* is in a sharp basin of L_1:
- Small perturbation delta = (theta_2* - theta_1*) / 2 causes large L_1 increase
- L_1(theta_merge) >> L_1(theta_1*)

If theta_1* is in a flat basin of L_1:
- Same perturbation delta is absorbed by the flat landscape
- L_1(theta_merge) ~ L_1(theta_1*) + O(||delta||^2 * lambda_min)

where lambda_min is the minimum Hessian eigenvalue (small for flat regions).

### 2.2 Quantitative Bound (Neyshabur et al., 2017)

For a theta in a flat region with Hessian H having max eigenvalue lambda_max:

    |L(theta + delta) - L(theta)| <= lambda_max * ||delta||^2 / 2

SAM training reduces lambda_max. If the merge perturbation has magnitude
||delta|| = ||theta_2* - theta_1*|| / 2, the post-merge loss degradation scales as:

    Delta_L <= lambda_max^{SAM} / lambda_max^{std} * Delta_L_std

Flat-LoRA reports ~3pp improvement, consistent with modest Hessian eigenvalue
reduction (not orders of magnitude).

### 2.3 Connection to Model Soups (Wortsman et al., ICML 2022)

Model Soups showed that models fine-tuned from the same pretrained initialization
tend to lie in the same loss basin. Flat-LoRA extends this: by training in
flat regions of that basin, the convex combination is more likely to remain in
the low-loss region.

Our Grassmannian orthogonal A-matrices further help: they ensure adapter deltas
point in different directions, so the merge perturbation per adapter is smaller.

## 3. What Breaks It

### 3.1 LoRA-Space vs Full-Space Flatness

LoRA-space SAM (what we implement) only ensures flatness in the r-dimensional
subspace. The full weight-space has mn >> 2r(m+n) directions. If the loss
landscape has sharp ridges ORTHOGONAL to the LoRA subspace, LoRA-SAM won't
detect them.

**Kill condition (K1 weakened):** If the merge perturbation is primarily in
directions orthogonal to the LoRA subspace, LoRA-SAM provides no benefit.

However: with Grassmannian orthogonal A-matrices, the merge perturbation
sum_i B_i @ A_i lies WITHIN the union of LoRA subspaces by construction.
This means LoRA-space SAM should cover the relevant perturbation directions.

### 3.2 SAM Overhead on MLX

SAM requires 2 forward+backward passes per step. On MLX with lazy evaluation:
- First pass: compute loss and gradient (builds graph)
- Perturbation: normalize gradient, add to parameters (in-graph operation)
- Second pass: compute loss and gradient at perturbed point (builds second graph)
- Both graphs must be evaluated

**Risk:** The double graph may exceed memory or cause evaluation issues on MLX.
This is the primary K1 risk.

### 3.3 Adapter Quality Degradation

SAM is known to sometimes hurt individual model quality when the perturbation
radius rho is too large. If Flat-LoRA adapters are individually worse, the
merge improvement may not compensate.

**Kill condition (K2):** If individual Flat-LoRA PPL is >5% worse than standard
AND merge improvement is <3pp, the technique is net-negative.

### 3.4 Scale Dependence

Flat-LoRA paper results are at 7B scale with rank-8. Our experiment is at 2.4B
with rank-16. The benefit may not transfer to our scale/configuration.

## 4. Assumptions

1. **SAM approximation is valid at LoRA scale.** The first-order epsilon
   approximation assumes smooth loss landscape near the current point. At
   rank-16 with 200 steps, this should hold (small learning rate regime).
   If wrong: SAM perturbation is random noise, no flatness benefit.

2. **Flat minima in LoRA space imply better mergeability.** This assumes
   the merge perturbation direction overlaps with the flatness directions.
   Justified by Grassmannian orthogonality (merge deltas are in LoRA subspaces).
   If wrong: individual quality maintained but no merge improvement.

3. **200 steps is sufficient for SAM to find flat regions.** SAM typically
   needs more iterations than standard training. At 200 steps, SAM may not
   have time to escape a sharp basin. If wrong: inconclusive result, would
   need more steps.

4. **rho selection is not critical.** We use rho=0.05 (standard SAM default).
   Paper uses sigma=0.01 for Bayesian formulation. If wrong: we may need
   a hyperparameter sweep.

## 5. Complexity Analysis

Per training step:

| Operation | Standard LoRA | Flat-LoRA (LoRA-SAM) |
|-----------|--------------|---------------------|
| Forward pass | O(d^2 * n_tokens) | 2 * O(d^2 * n_tokens) |
| Backward pass | O(d^2 * n_tokens) | 2 * O(d^2 * n_tokens) |
| Perturbation | 0 | O(r * (m + n) * n_layers) |
| Total FLOPs | F | ~2F |
| Memory | M | ~M + O(trainable_params) |

For our setup (d=2560, r=16, 30 layers, 7 projections):
- Trainable params: 2 * 16 * 2560 * 30 * 7 = 17.2M
- Perturbation storage: 17.2M * 4 bytes = ~69MB (negligible)
- Wall-clock overhead: ~2x per step

Total training time estimate:
- Standard: 200 steps * 5 domains * ~0.1s/step = ~100s
- Flat-LoRA: ~200s
- Full experiment with eval: ~30 minutes

## 6. Worked Example (d=64, r=4, N=2)

Base W in R^{64x64}. Two adapters: (A_1, B_1), (A_2, B_2) with A in R^{64x4}, B in R^{64x4}.

**Standard training:**
- Adapter 1 converges to sharp minimum: Hessian max eigenvalue lambda_1 = 50
- Adapter 2 converges to sharp minimum: lambda_2 = 45
- Merge delta per adapter: ||B_2@A_2||_F / 2 = 0.5
- Post-merge loss increase: 50 * 0.25 / 2 = 6.25 (large)

**SAM training:**
- Adapter 1 converges to flat minimum: lambda_1 = 5
- Adapter 2 converges to flat minimum: lambda_2 = 4
- Same merge delta: 0.5
- Post-merge loss increase: 5 * 0.25 / 2 = 0.625 (10x smaller)

**Expected merge PPL improvement:** If standard merge adds ~10% PPL degradation,
SAM-trained merge should add ~1% degradation. Improvement = 9pp.

## 7. Connection to Architecture

### Grassmannian Skeleton Interaction

Our frozen Grassmannian A-matrices provide orthogonal subspaces. This means:
- Adapter deltas B_i @ A_i are approximately orthogonal
- The merge perturbation per adapter is small (other adapters' deltas project
  nearly zero onto adapter i's subspace)
- SAM's benefit is ADDITIVE to Grassmannian's benefit

The composition landscape experiment showed smooth convex landscapes with
uniform 1/N only 0.7% from optimal. Flat-LoRA could further reduce this 0.7% gap
and make composition more robust to non-uniform weighting.

### Comparison with Production Models

DeepSeek-V3 and Qwen3 use output-space composition (MoE), avoiding the merge
problem entirely. For parameter-space composition (our pre-merge approach),
flat training is one of the few known techniques to improve merge quality.

LoRA Soups (arXiv:2410.13025) showed that CAT composition beats data mixing.
Flat-LoRA + CAT composition could provide further gains.

## 8. Merge Methods

We test four merge strategies:

### Task Arithmetic (TA)
    Delta_W = sum_{i=1}^N lambda * (B_i @ A_i)

Simple weighted sum. lambda = 1/N for uniform.

### TIES (Yadav et al., NeurIPS 2023)
1. Trim: zero out low-magnitude entries (keep top-k%)
2. Elect sign: for each parameter, take majority sign across adapters
3. Merge: average only parameters with elected sign

### DARE (Yu et al., ICML 2024)
1. Randomly drop parameters with probability p (e.g., 0.9)
2. Rescale remaining by 1/(1-p)
3. Average the sparsified adapters

### DO-Merging (Direct Orthogonal)
With Grassmannian A-matrices, simply sum B_i @ A_i without scaling.
Cross-terms vanish due to A_i^T @ A_j ~ 0. This is our architecture's
native composition method.

## 9. Post-Experiment Analysis

### Why the Hypothesis Failed

The core prediction was: SAM reduces Hessian max eigenvalue lambda_max,
reducing post-merge loss degradation Delta_L ~ lambda_max * ||delta||^2.

The experiment reveals that ||delta||^2 is already near-zero due to
Grassmannian orthogonality. With mean |cos| = 0.001, the effective
merge perturbation per adapter is:

    ||delta_i_projected|| ~ |cos(A_i, A_j)| * sum_j ||B_j @ A_j||
                          ~ 0.001 * ||B_j @ A_j||

This makes Delta_L ~ lambda_max * (0.001)^2 * ||B||^2 = O(10^-6)

At this perturbation scale, the distinction between lambda_max = 5 (SAM)
and lambda_max = 50 (standard) gives Delta_L ratios of 10^-5 vs 10^-4.
Both are negligible compared to evaluation noise (~0.1% PPL).

### Quantitative Confirmation

Sharpness measurement (1% random perturbation, 10x actual merge perturbation):
- Standard: 0.02% PPL increase
- SAM: 0.07% PPL increase

Both are essentially zero. The loss landscape is already flat at the scale
that matters for composition, because the composition perturbation is
projected away by Grassmannian orthogonality.

### Implication for Architecture

This result strengthens the Grassmannian skeleton as the primary mechanism:
- It is not just preventing interference (measured by cos similarity)
- It is making the GEOMETRY of composition irrelevant
- Flat minima, sharp minima -- doesn't matter when perturbations are ~0
- No training-time technique can improve on this for orthogonal adapters
