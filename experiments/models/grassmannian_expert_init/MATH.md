# Grassmannian Expert Init: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | {64, 128, 256} (micro); 4096 (production) |
| r | LoRA rank | 8 (micro), 16 (production) |
| N | Number of expert subspace slots | {12, 20, 40} (micro); 500+ (production) |
| Gr(r, d) | Grassmannian manifold | Space of r-dim subspaces of R^d |
| U_i | Orthonormal frame for slot i | (d, r), U_i^T U_i = I_r |
| G | Block Gram matrix | (Nr, Nr); G_{ij} = U_i^T U_j |
| mu | Coherence: max off-diagonal block norm | ||U_i^T U_j||_F for i != j |
| mu_W | Welch bound (theoretical floor for mu) | function of N, r, d |
| d_chord | Chordal distance on Gr(r, d) | sqrt(r - ||U^T V||_F^2) |

## 2. The Grassmannian Manifold and Expert Subspaces

### 2.1 LoRA Adapters as Points on Gr(r, d)

A LoRA adapter with frozen A: (d, r) and trained B: (r, d_out) produces a
weight delta dW = (alpha/r) * A @ B. The column space of A is an r-dimensional
subspace of R^d, i.e., a point on Gr(r, d).

Two adapters with column spaces span(A_i) and span(A_j) interfere when these
subspaces overlap. The interference is measured by:

    coherence(i, j) = ||A_i^T A_j||_F

where A_i, A_j have orthonormal columns. This equals the sum of squared cosines
of principal angles between the subspaces:

    ||U_i^T U_j||_F^2 = sum_{k=1}^r cos^2(theta_k)

Perfect orthogonality: coherence = 0 (all principal angles = pi/2).
Maximum interference: coherence = sqrt(r) (subspaces coincide).

### 2.2 The Packing Problem

**Problem:** Given N, r, d, find N points on Gr(r, d) that maximize the
minimum pairwise chordal distance:

    max_{U_1,...,U_N} min_{i != j} d_chord(U_i, U_j)

Equivalently, minimize the maximum coherence:

    min_{U_1,...,U_N} max_{i != j} ||U_i^T U_j||_F

This is the Grassmannian packing problem (Dhillon et al., 2008).

## 3. The Welch Bound

### 3.1 Statement

For N subspaces of dimension r in R^d, the maximum coherence is bounded below:

    mu_max >= mu_W = sqrt(r * (Nr - d) / (d * (Nr - r)))

**Derivation.** Consider the (Nr x Nr) block Gram matrix G with diagonal
blocks I_r. We have:

    tr(G^2) = sum_{i,j} ||G_{ij}||_F^2 = N*r + sum_{i!=j} ||U_i^T U_j||_F^2

Since G is PSD with eigenvalues summing to Nr and at most d positive eigenvalues:

    tr(G^2) >= (Nr)^2 / d    (Cauchy-Schwarz on eigenvalues)

Therefore:

    sum_{i!=j} ||U_i^T U_j||_F^2 >= (Nr)^2/d - Nr = Nr(Nr - d)/d

There are N(N-1) off-diagonal pairs, so:

    max_{i!=j} ||U_i^T U_j||_F^2 >= Nr(Nr - d) / (d * N(N-1))
                                   = r(Nr - d) / (d(N-1))

Taking the square root gives the Welch bound.

### 3.2 Achievability

When Nr <= d: mu_W = 0 (perfect orthogonality is possible). This is the
trivial regime -- there is enough room in R^d for N mutually orthogonal
r-dimensional subspaces.

When Nr > d: mu_W > 0 (some interference is unavoidable). This is the
interesting regime where AP provides value.

| d   | r | N  | Nr  | Nr/d  | mu_W   | Regime |
|-----|---|-----|-----|-------|--------|--------|
| 64  | 8 | 12  | 96  | 1.50  | 0.213  | packing needed |
| 128 | 8 | 20  | 160 | 1.25  | 0.115  | packing needed |
| 256 | 8 | 40  | 320 | 1.25  | 0.080  | packing needed |
| 4096| 16| 500 |8000 | 1.95  | 0.044  | packing needed |

## 4. Alternating Projection Algorithm

### 4.1 Algorithm

The AP algorithm (Tropp, Dhillon, Heath, Strohmer 2005) alternates between
two constraint sets to find a valid Gram matrix with low coherence.

**Input:** N, r, d, target coherence mu_target >= mu_W

**Initialize:** Random N points on Gr(r, d); form Gram matrix G.

**Repeat for T iterations:**

1. **Structural projection S(G):**
   For each off-diagonal block G_{ij} with ||G_{ij}||_F > mu_target:

       G_{ij} <- G_{ij} * (mu_target / ||G_{ij}||_F)

   Diagonal blocks: G_{ii} <- I_r

2. **Spectral projection P(G):**
   Eigendecompose: G = V Lambda V^T
   Keep top-d eigenvalues, clamp negatives:

       Lambda'_k = max(lambda_k, 0) for k <= d, 0 for k > d

   Rescale to tr(G') = Nr:

       Lambda' <- Lambda' * (Nr / sum Lambda')

   Reconstruct: G <- V Lambda' V^T

**Output:** Extract frames from final G via eigendecomposition + QR.

### 4.2 Complexity

Per iteration:
- Structural projection: O(N^2 * r^2) (scan all blocks, clip)
- Spectral projection: O((Nr)^3) (eigendecomposition of Nr x Nr matrix)
- Total per iteration: O((Nr)^3)

Total: O(T * (Nr)^3)

| Scale | Nr | O((Nr)^3) | T | Total |
|-------|-----|-----------|---|-------|
| Micro (d=64, N=12, r=8) | 96 | 885K | 500 | ~442M ops |
| Micro (d=256, N=40, r=8) | 320 | 32.8M | 500 | ~16.4B ops |
| Prod (d=4096, N=500, r=16) | 8000 | 512B | 5000 | ~2.56P ops |

At production scale, GPU-accelerated eigh (torch.linalg.eigh) handles the
8000x8000 eigendecomposition in milliseconds. TAAP (Targeted coherence with
Accelerated AP, Meszaros et al.) adds momentum for 10x faster convergence.

### 4.3 Convergence

The algorithm converges to a fixed point (Bauschke & Borwein, 1993) because
both constraint sets are convex (the structural set is a box constraint on
block norms; the spectral set is an intersection of spectral constraints).

In practice, 500 iterations suffice for micro scale. The max coherence
decreases monotonically and plateaus within 100-200 iterations.

## 5. Expert Initialization from Skeleton

### 5.1 Frame-to-LoRA Mapping

Given a skeleton of N frames {U_1, ..., U_N} on Gr(r, d):

    Expert i: A_i = U_i    (d, r) orthonormal frame
              B_i = 0      (r, d_out) zero-initialized (standard LoRA)

For multi-layer models, the frame is applied to the first layer's A matrix.
Subsequent layers use rotated versions of the same frame (within the same
r-dimensional subspace):

    A_i^{(l)} = U_i @ R_l

where R_l: (r, r) is a deterministic rotation matrix. This ensures all
layers of expert i operate within the same subspace slot.

### 5.2 Zero Drift is a Design Property (Not an Empirical Finding)

During B-only training (A frozen), the gradient updates to B do not change
the column space of A. The LoRA delta dW = (alpha/r) * A @ B has column
space span(A) = span(U_i), which is exactly the assigned slot.

**This is a mathematical tautology, not an experimental result.** If A is
frozen, span(A) cannot change. The measured "drift" of 0.02-0.03% in the
experiment is float32 arithmetic noise (the QR decomposition in the distance
measurement introduces rounding error; the actual subspace is unchanged).

This zero-drift property is a design choice (use frozen-A LoRA), not a
discovery. It should not be counted as a "survived kill criterion" because
the experiment could never fail K3 under this design.

With full A+B training (e.g., DoRA, GaLore), gradient updates to A would
move the subspace away from the assigned slot, and the skeleton guarantee
would degrade. The SOLE architecture assumes frozen-A training.

### 5.3 Interference Bound

For two experts initialized into slots U_i, U_j of the skeleton:

    ||delta_W_i^T delta_W_j||_F = (alpha/r)^2 * ||B_i^T A_i^T A_j B_j||_F
                                <= (alpha/r)^2 * ||B_i|| * ||A_i^T A_j||_F * ||B_j||
                                <= (alpha/r)^2 * ||B_i|| * mu * ||B_j||

where mu is the skeleton's coherence (capped by AP at mu_target ~ 1.2 * mu_W).

The interference scales linearly with the skeleton coherence mu. AP minimizes
mu, directly minimizing the worst-case interference bound.

## 6. Three-Condition Comparison: Isolating the Packing Effect

### 6.1 The Confound in Two-Condition Comparison

The original experiment compared AP-orthonormal against random-Gaussian init.
This conflates two effects:
1. **Packing effect:** AP optimal placement vs. random placement on Gr(r, d)
2. **Orthonormality effect:** Orthonormal A vs. scaled Gaussian A

To isolate the packing effect, we add a third condition: **Haar-random
orthonormal** init -- A matrices drawn uniformly from the Stiefel manifold
St(r, d), without any AP packing optimization.

### 6.2 Pre-Training Coherence

Random-Gaussian init coherence: For A_i ~ N(0, 2/d * I), the columns are not
orthonormal, so ||A_i^T A_j||_F includes both the subspace overlap and the
non-orthonormality of individual frames. This inflates the measured coherence.

Haar-random orthonormal init coherence: E[||U_i^T U_j||_F^2] = r^2/d for
Haar-random subspaces (from E[tr(P_U P_V)] = r^2/d, where P_U = U U^T is
the projection matrix). Thus E[||U_i^T U_j||_F] ~ r/sqrt(d) (with a
Jensen's inequality correction factor that is close to 1 for large d).

AP init coherence: mu_target ~ 1.2 * mu_W, where mu_W = sqrt(r(Nr-d)/(d(Nr-r))).

### 6.3 Post-Training: Three-Condition Results

With frozen A, the subspace is locked. The flattened delta vector cosine
measures interference in the FULL parameter space (dimension D = 2 * n_layers * d * d_ff).

**Empirical decomposition of improvement:**

    Total improvement (AP vs Gaussian) = Packing effect * Orthonormality effect

At d=128: Total = 2.0x, Packing = 1.52x, Ortho = 1.32x (1.52 * 1.32 ~ 2.0)
At d=256: Total = 1.32x, Packing = 1.33x, Ortho = 0.99x (packing dominates)
At d=64:  Total = 1.27x, Packing = 1.23x, Ortho = 1.03x (neither significant)

The packing effect is the dominant factor at d >= 128. The orthonormality
effect is inconsistent and often not statistically significant.

### 6.4 Statistical Tests

Wilcoxon signed-rank test on paired samples (same domain pairs across conditions):
- H0: AP cosines and random-orthonormal cosines have the same distribution
- H1: AP cosines are stochastically smaller (one-sided, alternative='less')
- 56 paired observations per dimension (28 pairs * 2 seeds)

| d   | AP vs Ortho p | AP vs Gauss p | Ortho vs Gauss p |
|-----|---------------|---------------|------------------|
| 64  | 0.096 n.s.    | 0.038 *       | 0.372 n.s.       |
| 128 | 0.009 **      | 0.002 **      | 0.055 borderline |
| 256 | 0.012 *       | 0.007 **      | 0.628 n.s.       |

The packing effect (AP vs Ortho) is significant at d >= 128 but not at d=64.
The orthonormality effect (Ortho vs Gauss) is never clearly significant.

## 7. Worked Numerical Example

d=128, r=8, N=20, layers=2, d_ff=512:

    Nr = 160 > d = 128 (packing regime)
    Welch bound: mu_W = sqrt(8 * (160 - 128) / (128 * (160 - 8)))
                      = sqrt(8 * 32 / (128 * 152))
                      = sqrt(256 / 19456)
                      = sqrt(0.01316)
                      = 0.1147

    AP target: mu = 1.2 * 0.1147 = 0.1376

    After 500 iterations of AP:
    - Max coherence: 0.324 (not yet at target -- more iterations needed)
    - Mean coherence: 0.324
    - Random baseline coherence: 0.706

    Pre-training improvement: 0.706 / 0.324 = 2.18x

    Train 8 experts on distinct Markov domains (B-only, A frozen),
    three conditions:
    - AP-orthonormal delta-vector |cos|: 0.0032 (mean), 0.010 (max)
    - Random-orthonormal delta-vector |cos|: 0.0049 (mean), 0.014 (max)
    - Random-Gaussian delta-vector |cos|: 0.0064 (mean), 0.015 (max)

    Packing effect (AP vs random-ortho): 0.0049 / 0.0032 = 1.52x (p=0.009)
    Orthonormality effect (ortho vs Gauss): 0.0064 / 0.0049 = 1.32x (p=0.055)
    Total improvement (AP vs Gauss): 0.0064 / 0.0032 = 2.0x

    Slot drift: 0.0007 / 2.828 = 0.025% (zero by construction -- frozen A)

    Delta vector dimension: D = 2 * 2 * 128 * 512 = 262,144
    Both AP and random cos << sqrt(r/d) = 0.25 because D >> d.

## 8. Capacity and Scaling

### 8.1 Maximum N for Given Coherence Target

From the Welch bound, rearranging for N:

    mu^2 >= r(Nr - d) / (d(Nr - r))
    mu^2 * d * (Nr - r) >= r * (Nr - d)
    mu^2 * d * Nr - mu^2 * d * r >= r * Nr - r * d
    Nr * (mu^2 * d - r) >= mu^2 * d * r - r * d
    Nr >= r * d * (mu^2 - 1) / (mu^2 * d - r)

For mu << 1 (low interference target):

    N_max ~ d / r    (at mu ~ sqrt(r/d))

For the SOLE capacity claim N_max ~ d^2/r^2, we need:

    N_max ~ d^2/r^2 at mu ~ sqrt(r/d) * sqrt(r/d) = r/d

This is consistent: with the flattened delta vector analysis (dimension D ~ d^2),
the effective capacity is D/r^2 ~ d^2/r^2.

### 8.2 Production Estimates

| Config | Nr | mu_W | AP time (est.) | Regime |
|--------|-----|------|----------------|--------|
| d=4096, r=16, N=500 | 8000 | 0.044 | ~5-10 min (GPU) | moderate packing |
| d=4096, r=16, N=5000 | 80000 | 0.014 | ~1 hr (GPU, TAAP) | dense packing |
| d=4096, r=16, N=65000 | 1.04M | -- | infeasible directly | need hierarchical |

For N > 5000, hierarchical packing (group experts into meta-clusters, pack
within each cluster) or iterative slot allocation (pack first K, then extend)
is needed. The skeleton does not need to be computed all at once.

## 9. Assumptions and Limitations

1. **B-only training with frozen A.** The zero-drift result is a consequence of
   this standard LoRA practice. Full A+B training (e.g., GaLore, DoRA) would
   allow subspace drift, weakening the slot guarantee. However, B-only training
   is the dominant practice and is assumed throughout SOLE.

2. **Per-layer subspace assignment.** We assign the frame to layer 0's A matrix
   and use rotations for other layers. A more sophisticated approach would assign
   independent frames per layer (multiplying the packing problem by L). This is
   unnecessary if inter-layer coherence is not the bottleneck.

3. **Micro-scale toy data.** Losses near random (~3.466) throughout training.
   The AP advantage is measured in delta-vector cosine, which is a geometric
   property independent of learning signal strength.

4. **AP convergence at micro scale.** With 500 iterations, AP achieves coherence
   2-3x above the Welch bound. More iterations or accelerated methods (TAAP)
   would close this gap.

5. **The benefit is marginal at micro scale.** Post-training ratio is 1.3-2x,
   meaning AP gives a modest improvement over random. At production scale with
   higher N/d ratio, the benefit may be larger.
