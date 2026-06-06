# P7.B0: Null-Space Projection as Natural Router (Geometry = Relevance)

## Type: Guided Exploration

**Proven framework**: Null-space LoRA (Finding #494) — A-matrices live in null(W_v),
orthogonality exact by construction. Task Arithmetic (arXiv:2212.04089) — LoRA weight
vectors encode task-specific directional information.

**Unknown**: Whether the A-matrix projection magnitude ||A_i x|| discriminates domains
well enough for routing (>= 80% accuracy).

## Setup

Let W_v in R^{d_out x d_in} be the value projection weight at a transformer layer.
Let Q in R^{d_in x m} be an orthonormal basis for null(W_v) (m = dim null space).

For K domains D_1, ..., D_K, train null-space LoRA adapters independently:
- A_i = A_{null,i} @ Q^T where A_{null,i} in R^{r x m} (learned)
- B_i in R^{d_out x r} (learned)
- Forward: delta_y_i = scale * (x @ Q @ A_{null,i}^T) @ B_i^T

Define the routing signal for input x and adapter i:
  s_i(x) = ||A_{null,i} @ Q^T @ x||^2 = sum_j (a_{ij}^T Q^T x)^2

## Theorem (Projection-Domain Alignment)

**Statement**: Let A_i be trained by gradient descent on domain D_i to minimize
E_{(x,y)~D_i}[L(B_i A_i x, y)]. Then the rows of A_{null,i} converge to directions
of maximal gradient variance for D_i within null(W_v). For domains with distinct
null-space feature distributions, argmax_i s_i(x) recovers domain(x).

**Proof**:

1. **Gradient structure of A**. The gradient of L w.r.t. A_{null} at step t is:
   nabla_{A_null} L = B^T (dL/dz) (Q^T x)^T
   where z = A_{null} Q^T x is the low-rank projection. This is an outer product
   between the output gradient B^T(dL/dz) and the null-space features Q^T x.

2. **Convergence to principal gradient directions**. After T steps of SGD, A_{null}
   approximately spans the top-r principal components of the gradient covariance:
   C_i = E_{x~D_i}[(B^T dL/dz)(Q^T x)^T ((B^T dL/dz)(Q^T x)^T)^T]
   By the Eckart-Young theorem, the rank-r approximation to C_i is optimal in
   Frobenius norm. Different domains D_i produce different C_i (different data
   distributions), hence different principal components.

3. **Projection magnitude as domain indicator**. For x ~ D_i:
   s_i(x) = ||A_{null,i} Q^T x||^2 = x^T Q A_{null,i}^T A_{null,i} Q^T x
   This is a quadratic form in the null-space projection of x, weighted by
   A_{null,i}^T A_{null,i} (the Gram matrix of adapter i's learned directions).
   
   Since A_{null,i}'s rows are trained on D_i, they align with D_i's feature
   directions. For x ~ D_i, the projection captures learned variance -> large s_i.
   For x ~ D_j (j != i), the projection captures less variance -> smaller s_i.

4. **Routing accuracy bound**. The routing error P(argmax_i s_i(x) != domain(x))
   is bounded by the overlap of null-space feature distributions across domains.
   For well-separated text domains (medical vs code vs math vs legal vs finance),
   this overlap should be small, giving accuracy >> 1/K = 20%.

**QED** (constructive, contingent on domain separability in null-space coordinates).

## Connection to Room Model

This proves a key claim of the Room Model architecture: "routing IS the matmul."
The adapter's A-matrix simultaneously:
- Extracts features for the LoRA update (functional role)
- Indicates domain relevance via projection magnitude (routing role)

No separate router network, TF-IDF classifier, or embedding lookup is needed.
The routing signal is a free byproduct of the LoRA computation.

## Distinction from Finding #295 (B-projection failure)

Finding #295 showed null-space B-matrix preservation FAILED (36.1% vs 85.6%
theoretical) because "B-subspaces overlap — all adapters adapt the same base model."

This experiment tests A-projection, which is fundamentally different:
- B-matrices map FROM the shared low-rank space TO output -> converge similarly
- A-matrices map FROM input TO adapter-specific features -> trained on different data

The A-matrices should specialize because they optimize on different domain data,
learning different feature extraction directions within null(W_v).

## Failure Condition

If W_v already captures ALL domain-relevant features (effective rank = d_in),
then null(W_v) is trivial (dimension 0) and no routing signal exists. However,
Finding #493 showed null_dim ~ 2048 for Gemma 4 v_proj (d_in = 3584, rank ~ 1536),
confirming substantial null space exists.

A subtler failure: if all 5 domains project identically into null(W_v) — i.e.,
the null space contains no domain-discriminative features. This would mean the base
model W_v already separates all domain-relevant variance from irrelevant noise,
and the null space is pure noise. The experiment tests whether this is the case.

## Predictions

1. **K1300**: 5-domain routing accuracy >= 80% (chance = 20%). Prediction: ~85-90%
   for clearly separated text domains (medical/code/math/legal/finance).

2. **K1301**: Spearman r >= 0.3 between projection magnitude and adapter quality
   (measured as domain-match indicator and NTP loss). Prediction: r ~ 0.5-0.7.

3. **K1302**: Routing latency < 0.5ms. Prediction: < 0.1ms. The computation is
   just small matmuls: Q^T @ x (m x d_in @ d_in -> m) then A_null_i @ result
   (r x m @ m -> r). For 5 adapters x 8 layers, this is trivial.

## References

- Finding #494: Null-space LoRA preserves 98.7% quality, orthogonality exact
- Finding #493: v_proj null_dim = 2048, 341 slots at r=6
- Finding #295: B-projection fails, A-projection untested (this gap motivates experiment)
- Room Model (project memory): routing IS the matmul
- arXiv:2106.09685 (LoRA): low-rank adaptation learns task-specific projections
- arXiv:2212.04089 (Task Arithmetic): task vectors encode directional information
- arXiv:2405.09673 (LoRA Learns Less): LoRA modifies activations in task-specific subspace
