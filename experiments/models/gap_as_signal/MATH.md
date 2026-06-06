# Gap-as-Signal: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Definition |
|--------|-----------|------------|
| W_base | (d, d') | Frozen base model weight matrix |
| Delta_i | (d, d') | LoRA delta for expert i: (alpha/r) * A_i @ B_i |
| A_i | (d, r) | LoRA down-projection for expert i |
| B_i | (r, d') | LoRA up-projection for expert i |
| r | scalar | LoRA rank |
| alpha | scalar | LoRA scaling factor |
| N | scalar | Number of experts |
| x | (B, T, d) | Input hidden states |
| f_joint(x) | (B, T, V) | Output of jointly-trained model |
| f_comp(x) | (B, T, V) | Output of composed (averaged) model |
| g(x) | (B, T, V) | Output of routed model |
| cos(i,j) | scalar | Cosine similarity between flattened Delta_i, Delta_j |
| G | scalar | Function-space gap magnitude |
| S | scalar | Calibration speed (inverse of steps or final quality gap) |

## Setup

Expert i applies a linear correction to the base MLP:

```
MLP_i(x) = ReLU((W_base + Delta_i^{fc1}) @ x) @ (W_base + Delta_i^{fc2})^T
```

The delta is pure linear in weight space: `Delta_i = (alpha/r) * A_i @ B_i`.
This is a rank-r matrix living in R^{d x d'}.

## The Gap

### Definition 1: Function-Space Gap

For a composed model (task arithmetic, simple average of deltas) and a
jointly-trained model:

```
G_CE = |CE(f_comp, targets) - CE(f_joint, targets)|
G_KL = KL(f_joint || f_comp)
```

where CE is cross-entropy loss and KL is KL divergence over the output
probability distributions.

### Definition 2: Cosine Similarity of Deltas

Flatten all delta matrices for expert i into a single vector delta_i in R^D
where D = n_layer * (d*4d + 4d*d) = n_layer * 2 * 4 * d^2.

```
cos(i, j) = <delta_i, delta_j> / (||delta_i|| * ||delta_j||)
```

At micro scale (d=64, r=8, n_layer=4): D = 4 * 2 * 4 * 64^2 = 131,072.

## The Hypothesis

### Claim (Gap-as-Signal)

The function-space gap G is a monotonically increasing function of expert
correlation cos(i,j), and calibration quality Q (closeness to joint after
calibration) is a monotonically decreasing function of cos(i,j):

```
cos(i,j) -> G_CE: monotonically increasing
cos(i,j) -> Q = |final_loss - joint_loss| / joint_loss: monotonically increasing
```

Equivalently: orthogonal experts (cos~0) produce the conditions for
fast/high-quality calibration, while correlated experts (cos~1) produce
conditions for slow/poor calibration.

### Mechanism

Why does this hold? Consider the router's learning problem. The router
must learn to assign each token x to the expert whose delta minimizes
prediction error for that token.

For two experts with deltas Delta_A and Delta_B:

1. **Orthogonal case (cos=0):** The deltas modify different subspaces of
   the weight matrix. Token x activates one subspace strongly and the
   other weakly. The router's gradient signal is:
   ```
   grad_router = (loss_A(x) - loss_B(x)) * d(routing)/d(router_params)
   ```
   When deltas are orthogonal, `loss_A(x) != loss_B(x)` for most tokens,
   giving a strong gradient signal.

2. **Correlated case (cos->1):** The deltas modify the same subspace.
   Both experts produce similar outputs for all tokens:
   ```
   loss_A(x) ~ loss_B(x) for all x
   ```
   The router gradient vanishes: there's nothing to learn because the
   experts are interchangeable.

3. **The gap as gradient magnitude:** The function-space gap
   G = ||f_comp - f_joint|| is proportional to the discriminability of
   the experts. When G is large, the router has a strong supervision
   signal for learning token-to-expert assignment.

### Formal Prediction

```
Q(cos) proportional to cos(i,j)       (quality gap grows with correlation)
G_CE(cos) proportional to cos(i,j)    (function-space gap grows with correlation)
Q(cos) proportional to G_CE(cos)      (gap predicts quality, transitively)
```

Kill criterion: r^2(G_CE, Q) < 0.3.

## Projection Method

To test the hypothesis cleanly, we need controlled orthogonality. We
train two experts naturally (cos ~ 0.01, naturally orthogonal at d=64),
then GEOMETRICALLY PROJECT expert B to achieve target cosine:

Given flattened deltas a, b in R^D:
```
a_hat = a / ||a||
b_perp = b - <b, a_hat> * a_hat
b_perp_hat = b_perp / ||b_perp||

b_proj(c) = c * ||b|| * a_hat + sqrt(1 - c^2) * ||b|| * b_perp_hat
```

Properties:
- `cos(a, b_proj(c)) = c` (exact, by construction)
- `||b_proj(c)|| = ||b||` (preserves expert magnitude)
- `b_proj(0) is orthogonal to a` (Gram-Schmidt)
- `b_proj(1) = ||b|| * a_hat` (parallel to a)

This isolates the orthogonality variable while keeping expert norm constant.

## Computational Cost

The gap measurement requires:
- Forward pass through composed model: O(n_layer * d^2 * T) per token
- Forward pass through joint model: O(n_layer * d^2 * T) per token
- KL divergence computation: O(V * B * T)
- Total: O(2 * n_layer * d^2 * T + V * B * T)

At micro scale (d=64, T=32, B=32, V=28, n_layer=4):
- ~2M FLOPs per gap measurement (negligible)

The projection is O(D) = O(n_layer * d^2), also negligible.

## Worked Example (Micro Scale)

d=64, n_head=4, n_layer=4, r=8, 2 experts:

1. Delta dimension: D = 4 * 2 * (64*256 + 256*64) = 4 * 2 * 32768 = 262,144
2. Natural cosine: cos ~ 0.01 (near-orthogonal, consistent with r/sqrt(D) ~ 0.016)
3. At target cos = 0.0:
   - CE gap ~ 0.007 (small gap)
   - Final quality: +2.1% vs joint
4. At target cos = 0.9:
   - CE gap ~ 0.035 (5x larger gap)
   - Final quality: +12.1% vs joint (5.8x worse)
5. Correlation r^2(CE_gap, quality) = 0.74

## Assumptions

1. **Linearity of LoRA deltas:** The projection modifies weight-space cosine.
   The relationship to function-space gap is mediated by the nonlinear ReLU
   activation. The claim is that weight-space orthogonality is a sufficient
   (though not necessary) predictor of function-space discriminability.

2. **Fixed calibration budget:** We measure quality after a fixed number of
   calibration steps (300). The hypothesis that orthogonal experts calibrate
   FASTER (fewer steps needed) is harder to test at micro scale because the
   model is small enough that even poor experts converge quickly. The quality
   metric (final loss relative to joint) is a proxy.

3. **Two-expert case:** We test N=2. At N>2, the gap structure becomes a
   matrix of pairwise cosines. The prediction generalizes: the minimum
   pairwise cosine across all expert pairs determines calibration quality.

4. **Micro-scale limitations:** At d=64, the model has limited capacity.
   The absolute quality gap between cos=0 and cos=0.9 is ~10% at micro
   scale. At macro scale (d=896+), we expect larger gaps because the experts
   can specialize more strongly.
