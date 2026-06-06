# SOLE vs LoRA Soups: Formal Comparison Framework

## 1. Setup

### 1.1 Shared Notation

| Symbol | Definition | Shape |
|--------|-----------|-------|
| W_s^{(l)} | Frozen base (skeleton) weight at layer l | R^{d_out x d_in} |
| dW_i^{(l)} | Expert i delta at layer l | R^{d_out x d_in} |
| k | Number of simultaneously active experts | scalar |
| N | Total experts in library | scalar |
| r | LoRA rank | scalar |
| d | Model embedding dimension | scalar |

### 1.2 Three Composition Functions

All three methods share the form:

    W_composed^{(l)} = W_s^{(l)} + sum_{i in S} c_i^{(l)} * dW_i^{(l)}

They differ only in how the coefficients c_i^{(l)} are determined.

## 2. Method Definitions

### 2.1 SOLE (Structurally Orthogonal Latent Experts)

    c_i^{(l)} = 1   for all i in S, for all l

Unit weights. No optimization. The composition is:

    W_SOLE^{(l)} = W_s^{(l)} + sum_{i in S} dW_i^{(l)}

**Computational cost of composition:** O(k * r * (d_in + d_out) * L) to
materialize the deltas. Zero optimization cost.

### 2.2 LoRA Soups CAT (Prabhakar et al., 2024)

    c_i^{(l)} = w_i^{(l)}   where w_i^{(l)} are learned per-layer scalar weights

The weights are optimized on a held-out dataset D by minimizing:

    min_{w} L(D; W_s + sum_i w_i^{(l)} * dW_i^{(l)})

**Computational cost of composition:** Same materialization cost as SOLE,
plus T optimization steps, each requiring O(k * L) forward passes through
the model. Total optimization cost: O(T * k * L * C_fwd) where C_fwd is
the cost of one forward pass.

**Number of trainable parameters:** 2 * k * L scalars (one per expert per
layer per projection direction). For k=2, L=32: 128 parameters. Minimal.

### 2.3 Uniform Averaging (Model Soups baseline)

    c_i^{(l)} = 1/k   for all i in S, for all l

Fixed weights. No optimization. The composition is:

    W_avg^{(l)} = W_s^{(l)} + (1/k) * sum_{i in S} dW_i^{(l)}

**Note:** This is NOT what Model Soups (Wortsman et al., 2022) does. Model
Soups averages full model weights. Uniform averaging of LoRA deltas is a
special case of CAT where all weights are fixed at 1/k.

## 3. When Do Methods Diverge?

### 3.1 Condition for SOLE = CAT

If the optimal CAT weights satisfy w_i^{(l)} approx 1 for all i, l, then
CAT recovers SOLE. This happens when:

1. **Experts are near-orthogonal**: cos(vec(dW_i), vec(dW_j)) ~ 0. Cross-terms
   in the loss are negligible, so the optimal weight for each expert is
   independent of the others, and equals the single-expert optimal weight.

2. **Expert magnitudes are calibrated**: Each expert's delta has the right
   scale relative to the skeleton. If training uses standard LoRA scaling
   (alpha/r), this is automatically satisfied.

3. **No destructive interference**: The SiLU nonlinearity does not create
   significant cross-terms between experts. Bounded by:

       |interference| <= C * k^2 * (r/sqrt(d))^2 * max ||dW_i||^2

   For orthogonal experts at high d, this is negligible.

**Implication:** At d >= 896 (where cos ~ 0.0002), SOLE and optimally-trained
CAT should produce nearly identical results. CAT's optimization would converge
to w_i ~ 1.0, recovering SOLE at higher computational cost.

### 3.2 When CAT Wins

CAT can outperform SOLE when:

1. **Experts have significant overlap**: High cos(dW_i, dW_j) means cross-terms
   matter, and optimal c_i != 1. CAT can learn to down-weight overlapping experts.

2. **Expert magnitudes are miscalibrated**: Different training regimes produce
   deltas with inconsistent norms. CAT can rescale.

3. **Task distribution differs from expert training**: The target task weights
   experts differently than uniform. CAT learns task-specific reweighting.

### 3.3 When SOLE Wins (Operational)

Even if CAT achieves marginally better loss, SOLE wins on:

1. **Setup cost**: SOLE requires zero optimization. CAT requires T steps on
   held-out data from the composed task.

2. **Expert addition/removal**: Adding expert k+1 to an SOLE composition is
   instant. CAT requires retraining all weights (the new expert changes the
   optimal weights for existing experts through cross-terms).

3. **Evolution compatibility**: Clone-and-compete requires evaluating new
   expert versions without recalibrating the composition. SOLE supports this
   natively; CAT would need retraining after each clone resolution.

4. **Scaling**: SOLE composition is O(1) per expert addition. CAT weight
   optimization is O(T * k * L) and grows with k.

## 4. Convergence Analysis

### 4.1 CAT Weight Dynamics

The gradient of loss w.r.t. CAT weight w_i^{(l)} is:

    dL/dw_i^{(l)} = dL/dW_composed^{(l)} : dW_i^{(l)}

where : denotes the Frobenius inner product.

At convergence (gradient = 0), the optimal weight satisfies:

    w_i^{(l)} = -[sum_{j} w_j * <dW_j, dW_i>]^{-1} * <dW_i, gradient_of_loss_wrt_W_s>

When experts are orthogonal (<dW_i, dW_j> ~ 0 for i != j), this simplifies to:

    w_i^{(l)} ~ -<dW_i, dL/dW_s> / ||dW_i||^2

If the expert was well-trained (i.e., dW_i already points in the negative
gradient direction for its domain data), this ratio is approximately 1.

### 4.2 Expected CAT-SOLE Gap

    |L_CAT - L_SOLE| <= O(k^2 * cos_max^2 * ||dW||^4 / d^2)

where cos_max = max_{i!=j} |cos(dW_i, dW_j)|.

At d=896, cos_max ~ 0.0002:

    Gap <= O(k^2 * 4e-8 * ||dW||^4 / d^2) ~ negligible

At d=64 (micro), cos_max ~ 0.01:

    Gap <= O(k^2 * 1e-4 * ||dW||^4 / d^2) ~ small but measurable if ||dW|| is large

## 5. Micro-Scale Empirical Validation

### 5.1 Configuration

d=64, d_ff=256, r=8, L=4, N=6 (3 clusters x 2 domains), k=2 or 6.

### 5.2 Results

All three methods produce identical NTP loss to 4 decimal places across all
compositions and all 3 seeds. This is consistent with the theory:

- Mean |cos| = 0.0023 (near-orthogonal)
- Expert deltas have very small magnitude (specialist improvement < 0.01 nats)
- CAT weights converge to approximately 1.0

The quality comparison is vacuous at micro scale because experts do not
specialize sufficiently. The timing comparison is meaningful:

| Method | Composition Time (N=2) | Composition Time (N=6) |
|--------|----------------------|----------------------|
| SOLE    | 0.17s                | 0.45s                |
| Avg    | 0.17s                | 0.44s                |
| CAT    | 8.6s (52x)           | 33.5s (75x)          |

CAT overhead grows superlinearly with N because the number of scalar weights
to optimize is 2*k*L.

## 6. Assumptions

1. Experts are trained independently with standard LoRA (alpha/r scaling).
2. Expert deltas are low-rank (rank r << min(d_in, d_out)).
3. Experts occupy near-orthogonal subspaces (cos ~ O(r/sqrt(d_in * d_out))).
4. The composition target distribution is a mixture of the expert training
   distributions (not an entirely new distribution).
