# Training-Time Composition Compatibility: Mathematical Foundations

## Notation

| Symbol | Shape | Definition |
|--------|-------|------------|
| x | (B, T, d) | Input hidden states |
| A_l | (P, d) | Layer l capsule detector matrix |
| B_l | (d, P) | Layer l capsule expansion matrix |
| A_l^0 | (P, d) | Base (pre-fine-tuning) snapshot of A_l |
| delta_A_l | (P, d) | Weight delta: A_l - A_l^0 |
| d | scalar | Model dimension (64 at micro) |
| P | scalar | Capsules per pool (128 at micro) |
| L | scalar | Number of layers (4 at micro) |
| N | scalar | Number of domains (2 at micro) |

## Capsule Pool Forward Pass

For layer l:

```
pool_l(x) = B_l @ ReLU(A_l @ x)
```

Where A_l is (P, d) and B_l is (d, P). Each capsule i computes:

```
capsule_i(x) = b_i * ReLU(a_i^T x)
```

## Composition by Concatenation

For N domains, composed pool:

```
pool_composed(x) = [B_1, ..., B_N] @ ReLU([A_1; ...; A_N] @ x)
                  = sum_{n=1}^{N} B_n @ ReLU(A_n @ x)
                  = sum_{n=1}^{N} pool_n(x)
```

This is exact (not approximate) due to ReLU's per-neuron independence.

## The Composition Gap

Define the composition gap G as:

```
G = E_x[ L(sum_n pool_n(x)) - L(pool_joint(x)) ]
```

where L is the downstream loss. G measures the function-space difference
between independently-trained capsules summed vs jointly-trained.

Previous experiments: G ~ +5-7% for zero-shot, ~+1.3% for weight averaging.

## Auxiliary Loss 1: Weight Orthogonality (L_ortho)

Inspired by InfLoRA's orthogonality constraints. During fine-tuning of domain
n, penalize the component of delta that lies in the span of the base weights:

```
L_ortho = lambda_o * sum_l [ ||delta_A_l @ (A_l^0)^T||_F^2 / (||delta_A_l||_F^2 * ||A_l^0||_F^2 + eps)
                            + ||delta_B_l @ (B_l^0)^T||_F^2 / (||delta_B_l||_F^2 * ||B_l^0||_F^2 + eps) ]
```

**Interpretation:** The numerator ||delta_A @ A_0^T||_F^2 measures the squared
Frobenius norm of the matrix delta_A @ A_0^T, which is large when delta and A_0
share principal directions. The denominator normalizes to [0, 1].

**Computational cost:** O(P^2 * d) per layer (matrix multiply). Negligible vs
the forward pass O(B * T * P * d).

**Hypothesis:** If deltas are orthogonal to base weights, they are more likely
to be orthogonal to EACH OTHER (since they're all constrained to the same
orthogonal complement). This should reduce inter-domain interference at
composition time.

## Auxiliary Loss 2: Output-Norm Matching (L_norm)

Penalize deviation of pool output norms from the base model's output norms:

```
r_l = ||pool_l(x)||_2 / ||x||_2       (output-to-input norm ratio)
r_l^0 = E_x[ ||pool_l^0(x)||_2 / ||x||_2 ]   (target ratio from base)

L_norm = lambda_n * sum_l (r_l - r_l^0)^2
```

**Interpretation:** When composing by concatenation, the composed output is
pool_1(x) + pool_2(x). If both pools have similar output norms, they
contribute roughly equally. If one pool has 5x the norm of the other, it
dominates and the smaller pool's contribution is noise.

**Computational cost:** O(B * T * d) per layer (norm computation). Negligible.

## Worked Example (d=64, P=128, N=2)

**Orthogonality loss:**
- A^0 is (128, 64), delta_A is (128, 64)
- delta_A @ (A^0)^T is (128, 128) -- 128^2 * 64 = 1,048,576 MADs
- Forward pass: B * T * P * d = 32 * 32 * 128 * 64 = 8,388,608 MADs
- Overhead: ~12.5% per layer = 4 * 12.5% = ~50% total
- At micro scale this is acceptable; at macro scale would need approximation

**Norm loss:**
- Two norms of (B, T, d) tensors = 2 * B * T * d MADs
- Negligible overhead (<0.1%)

## Assumptions

1. **Orthogonality to base implies orthogonality between domains.** This is the
   key assumption. If the orthogonal complement of the base subspace is high-
   dimensional (d >> rank(A^0)), different domain deltas will naturally spread
   in different directions. At d=64, P=128, the base subspace is at most rank 64,
   leaving a 64-dimensional complement in P-space.

2. **Output norm predicts composition contribution.** Assumes the downstream
   loss is sensitive to the relative magnitudes of pool outputs, not just their
   directions. Supported by the "loudness" experiments (IDEA-RELU-ROUTER.md).

3. **Auxiliary losses don't degrade domain-specific quality.** The regularization
   constrains the solution space, which could reduce specialization. This is the
   composability-specialization tradeoff.
