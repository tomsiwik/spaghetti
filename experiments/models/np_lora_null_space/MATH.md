# NP-LoRA: Null Space Projection for Interference-Free Adapter Composition

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| W_0 | (d_out, d_in) | Frozen base weight |
| A_i | (r, d_in) | Frozen LoRA down-projection for adapter i |
| B_i | (d_out, r) | Trainable LoRA up-projection for adapter i |
| Delta_i | (d_out, d_in) | Effective delta = B_i @ A_i |
| N | scalar | Number of adapters to compose |
| r | scalar | LoRA rank |
| d | scalar | Model hidden dimension (d_in or d_out) |
| P_i | (d_in, d_in) | Null space projector for adapter i |

## Setup

Each adapter contributes an additive weight update:

    W_composed = W_0 + (1/N) * sum_{i=1}^{N} Delta_i

where Delta_i = B_i @ A_i has rank at most r.

## The Cross-Term Interference Problem

When composing N adapters by summation, the composed output for input x is:

    y = (W_0 + sum Delta_i / N) @ x
      = W_0 @ x + (1/N) * sum (B_i @ A_i @ x)

The interference between adapters i and j is captured by the cross-term
in the squared Frobenius norm of the composed delta:

    ||sum Delta_i||_F^2 = sum ||Delta_i||_F^2 + 2 * sum_{i<j} <Delta_i, Delta_j>_F

where <Delta_i, Delta_j>_F = tr(Delta_i^T Delta_j).

## Grassmannian Approach (Our Current Method)

We pre-compute A_i on the Grassmannian Gr(r, d) via QR factorization such that:

    A_i @ A_j^T = 0 for all i != j

This guarantees:

    <Delta_i, Delta_j>_F = tr(A_i^T B_i^T B_j A_j) = tr((B_i^T B_j)(A_j A_i^T))

Since A_j A_i^T = 0 (subspaces are orthogonal), the cross-term vanishes
regardless of B matrix correlation. Empirically: mean |cos| = 2.54e-7.

Capacity: N_max = d/r adapters (256/8 = 32 at our micro scale).

## NP-LoRA Approach (This Experiment)

Instead of constraining A matrices a priori, NP-LoRA post-hoc projects each
adapter's effective delta into the null space of all other adapters.

### Step 1: Collect Effective Deltas

For each adapter i, compute the vectorized effective delta:

    delta_i = vec(Delta_i) = vec(B_i @ A_i), where delta_i in R^{d_out * d_in}

### Step 2: Build the Interference Subspace

For adapter i, stack all OTHER adapters' deltas into a matrix:

    S_i = [delta_1, ..., delta_{i-1}, delta_{i+1}, ..., delta_N]^T
    S_i has shape (N-1, d_out * d_in)

### Step 3: Compute Null Space Projector via SVD

Compute the SVD of S_i:

    S_i = U Sigma V^T

The null space of S_i is spanned by the columns of V corresponding to
zero singular values. The projector onto the null space is:

    P_i = I - V_r V_r^T

where V_r contains the columns of V corresponding to nonzero singular values
(i.e., V_r has shape (d_out*d_in, rank(S_i))).

### Step 4: Project and Reshape

    delta_i_proj = P_i @ delta_i
    Delta_i_proj = unvec(delta_i_proj)  -- reshape to (d_out, d_in)

### Step 5: Compose

    W_composed = W_0 + (1/N) * sum Delta_i_proj

By construction: <Delta_i_proj, Delta_j>_F = 0 for all j != i, since
delta_i_proj lies in the null space of S_i which contains delta_j.

## Per-Layer Variant (Practical)

The vectorized approach requires SVD of a (N-1, d_out*d_in) matrix, which
is prohibitively large for real models. The practical variant operates
per-layer:

For each weight matrix W_l, collect the per-layer deltas:

    delta_{i,l} = vec(B_{i,l} @ A_{i,l}), shape (d_out_l * d_in_l,)

Build per-layer interference matrix S_{i,l} and project. This is much
cheaper: SVD of (N-1, d_out*d_in) per layer rather than one huge SVD.

## Computational Cost Analysis

### Per-layer SVD cost

For a single layer with dimensions d_out x d_in:
- Matrix S_i shape: (N-1, d_out * d_in)
- SVD cost: O(min(N-1, d_out*d_in) * (N-1) * d_out * d_in)
- For N << d_out*d_in (typical): O(N^2 * d_out * d_in)
- The projection P_i @ delta_i: O(N * d_out * d_in)

### Total for all N adapters, all L layers

    Cost = O(N * L * N^2 * d_out * d_in) = O(N^3 * L * d^2)

### At our micro scale (d=256, r=8, L=6*6+1=37 weight matrices, N=5)

- Per matrix: S_i shape (4, 65536), SVD of thin matrix = O(4^2 * 65536) ~ 1M ops
- Per adapter: 37 layers * 1M ~ 37M ops
- Total: 5 * 37M ~ 185M ops
- Expected time: well under 1 second

### At N=50 (K2 test)

- Per matrix: S_i shape (49, 65536), SVD ~ O(49^2 * 65536) ~ 157M ops
- Total: 50 * 37 * 157M ~ 290B ops
- Expected time: potentially several seconds
- Key insight: we can use thin SVD since N << d^2

## Worked Example (d_in=4, d_out=4, r=2, N=2)

Adapter 1: A_1 = [[1,0,0,0],[0,1,0,0]], B_1 = [[1,0],[0,1],[0,0],[0,0]]
  Delta_1 = B_1 @ A_1 = [[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]
  delta_1 = [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]

Adapter 2: A_2 = [[0,0,1,0],[0,0,0,1]], B_2 = [[0,0],[0,0],[1,0],[0,1]]
  Delta_2 = B_2 @ A_2 = [[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]]
  delta_2 = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1]

These are already orthogonal (dot product = 0), so NP-LoRA projection
is a no-op. This is exactly what we expect with Grassmannian A matrices.

Now with NON-orthogonal A matrices:
A_1 = [[1,1,0,0]/sqrt(2),[0,0,1,0]], B_1 = [[1,0],[0,1],[0,0],[0,0]]
A_2 = [[1,0,0,0],[0,1,0,0]], B_2 = [[0,0],[0,0],[1,0],[0,1]]

Delta_1 = [[.707,.707,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]
Delta_2 = [[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]]

dot(delta_1, delta_2) = 0 (still orthogonal because B subspaces don't overlap)

This shows: with separated output subspaces (B), NP-LoRA is unnecessary.
The Grassmannian approach separates INPUT subspaces (A), which combined
with typically-decorrelated B matrices, achieves near-zero interference.

## Assumptions

1. Adapter deltas have rank at most r (by construction)
2. Per-layer projection is sufficient (no cross-layer interference)
3. Projection preserves adapter quality (projecting away interference
   components does not destroy useful signal)
4. SVD is numerically stable for our matrix sizes
5. Thin SVD is sufficiently accurate (vs. full SVD)

## Prediction

With Grassmannian A matrices (|cos| ~ 1e-7), NP-LoRA projection will be
near-identity, producing negligible improvement. The composition ratio will
be essentially identical (within noise) to the Grassmannian baseline.

With random (non-Grassmannian) A matrices, NP-LoRA may improve composition
quality significantly, validating that it solves a real problem -- just one
that Grassmannian init already solves more elegantly.
