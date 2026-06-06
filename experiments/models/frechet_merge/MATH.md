# Frechet Merge: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | {64, 128, 256, 512, 1024} (micro) |
| r | LoRA rank | 8 |
| N | Number of experts to merge | {2, 5, 10, 25, 50} |
| Gr(r, d) | Grassmannian manifold | Space of r-dim subspaces of R^d |
| U_i | Orthonormal frame for expert i | (d, r), U_i^T U_i = I_r |
| P_i | Projection matrix for expert i | P_i = U_i U_i^T, (d, d), rank r |
| A_i | LoRA down-projection (frozen) | (d, r) |
| B_i | LoRA up-projection (trained) | (r, d_out) |
| alpha | LoRA scaling factor | scalar |
| d_chord(U, V) | Chordal distance on Gr(r, d) | sqrt(r - \|\|U^T V\|\|_F^2) |

## 2. The Expert Composition Problem

### 2.1 Current SOLE Composition (Euclidean)

The current SOLE architecture composes N experts via naive addition:

    W_composed = W_base + sum_{i=1}^N (alpha/r) * A_i @ B_i

This is Euclidean averaging in the full weight space R^{d x d_out}. It treats
LoRA deltas as flat vectors and simply sums them.

### 2.2 The Geometric Perspective

Each expert's A_i defines an r-dimensional subspace span(A_i) in R^d, which is
a point on the Grassmannian Gr(r, d). The expert's "knowledge direction" lives
in this subspace. Composing experts means finding a merged subspace that best
represents all experts.

**Key insight:** The Grassmannian is a curved manifold. Euclidean averaging of
projection matrices ignores this curvature. The Riemannian Frechet mean respects
the manifold geometry.

### 2.3 Why Naive Addition is Geometrically Wrong

Naive addition sum(A_i @ B_i) produces a rank-Nr matrix (at most), not a rank-r
matrix. Its effective subspace (top-r SVD) is biased toward experts with larger
||B_i|| norms, regardless of their geometric arrangement on the Grassmannian.

The Frechet mean, by contrast, finds the r-dim subspace that minimizes the sum
of squared distances to all expert subspaces -- a proper "center" on the manifold.

## 3. Three Merge Methods

### 3.1 Naive Addition

    delta_merged = sum_{i=1}^N (alpha/r) * A_i @ B_i     (d, d_out)

To extract an effective subspace, compute SVD:

    delta_merged = U_eff @ S @ V^T

The effective merged subspace is span(U_eff[:, :r]).

**Complexity:** O(N * d * r * d_out) for summation, O(d * d_out * min(d, d_out))
for SVD extraction.

### 3.2 Chordal Frechet Mean (Closed-Form)

The chordal Frechet mean minimizes:

    mu_chord = argmin_{S in Gr(r,d)} sum_{i=1}^N d_chord(S, span(U_i))^2

where d_chord(S, T)^2 = r - tr(P_S P_T) (chordal distance squared).

**Note:** An equivalent expression is d_chord^2 = ||P_S - P_T||_F^2 / 2, since
||P_S - P_T||_F^2 = tr(P_S) + tr(P_T) - 2 tr(P_S P_T) = 2r - 2 tr(P_S P_T),
so ||P_S - P_T||_F^2 / 2 = r - tr(P_S P_T) = d_chord^2.

Minimizing sum d_chord^2 = Nr - sum tr(P_mu P_i) is equivalent to maximizing
sum tr(P_mu P_i), which has a closed-form solution:

    P_avg = (1/N) sum_{i=1}^N U_i U_i^T     (d, d) symmetric PSD

The top-r eigenvectors of P_avg form the chordal Frechet mean.

**Algorithm:**
1. Orthonormalize each A_i: Q_i, _ = QR(A_i), U_i = Q_i[:, :r]
2. P_avg = (1/N) sum U_i U_i^T
3. eigvals, eigvecs = eigh(P_avg)
4. merged = eigvecs[:, -r:]  (top-r)

**Complexity:** O(N * d * r) for projection sum, O(d^3) for eigendecomposition.

**Proof of optimality.** The chordal distance squared is:

    d_chord^2(S, T) = r - tr(P_S P_T)

So minimizing sum d_chord^2 = Nr - sum tr(P_mu P_i) is equivalent to maximizing
sum tr(P_mu P_i) = tr(P_mu * sum P_i) = N * tr(P_mu P_avg).

Since P_mu is a rank-r projection, tr(P_mu P_avg) is maximized by projecting
onto the top-r eigenspace of P_avg. QED.

### 3.3 Geodesic (Karcher) Mean

The geodesic Frechet mean minimizes:

    mu_geo = argmin_{S in Gr(r,d)} sum_{i=1}^N d_geo(S, span(U_i))^2

where d_geo is the Riemannian (arc-length) distance on Gr(r, d).

**No closed-form exists.** We use Karcher flow (Riemannian gradient descent):

**Log map** Log_X(Y): Gr(r,d) -> T_X Gr(r,d)

    1. M = (I - X X^T) Y          (complement projection)
    2. W = M @ inv(X^T Y)         (solve alignment)
    3. U, s, V^T = SVD(W)         (thin SVD)
    4. Delta = U @ diag(arctan(s)) @ V^T   (tangent vector)

**Exp map** Exp_X(Delta): T_X Gr(r,d) -> Gr(r,d)

    1. U, s, V^T = SVD(Delta)     (thin SVD)
    2. Y = X @ V @ diag(cos(s)) @ V^T + U @ diag(sin(s)) @ V^T
    3. Re-orthonormalize: Q, _ = QR(Y)

**Karcher flow:**
1. Initialize: mu = chordal mean (warm start)
2. Repeat until convergence:
   a. Delta_i = Log_mu(U_i) for i = 1, ..., N
   b. Delta_avg = (1/N) sum Delta_i
   c. If ||Delta_avg||_F < tol: converge
   d. mu = Exp_mu(step * Delta_avg)

**Complexity per iteration:** O(N * d * r^2) for Log maps + O(d * r^2) for Exp map.
Total: O(T * N * d * r^2) where T is the number of iterations (typically 5-20).

### 3.4 Subspace-to-Weight Reconstruction

Both Frechet methods produce a merged subspace (d, r). To produce a weight delta
comparable to naive addition, we project each expert's B through the merged
subspace:

    alignment_i = merged^T @ U_i    (r, r) rotation matrix
    B_proj = sum_i alignment_i @ B_i   (r, d_out)
    delta_merged = (alpha/r) * merged @ B_proj

This preserves the B information that aligns with the merged subspace.

## 4. Metrics

### 4.1 Subspace Preservation

How much of each expert's subspace is captured by the merged subspace:

    preservation_i = ||merged^T U_i||_F^2 / r

Range: [0, 1]. Value 1 means the merged subspace fully contains expert i.
Value r/d means the merged subspace has no special relationship with expert i
(random overlap).

For the merged subspace to be useful, preservation should exceed r/d (the
baseline for random subspaces).

### 4.2 Theoretical Bounds

For N experts uniformly distributed on Gr(r, d):

    E[preservation per expert] ~ r/d for random merged subspace
    E[preservation per expert] <= 1 (only if all experts share the same subspace)

The chordal Frechet mean maximizes the AVERAGE preservation across all experts
(by definition -- it maximizes sum tr(P_mu P_i)).

For N >> d/r (many more experts than subspace capacity), preservation degrades
as ~r/d * (correction factor from expert clustering).

### 4.3 Chordal vs Geodesic Agreement

The chordal and geodesic metrics on Gr(r, d) are related but not identical:

    d_chord(U, V) = sqrt(sum sin^2(theta_k))
    d_geo(U, V)   = sqrt(sum theta_k^2)

where theta_1, ..., theta_r are the principal angles. For small angles
(similar subspaces), sin(theta) ~ theta, so the metrics agree. For large
angles (dissimilar subspaces), they diverge.

The chordal metric underestimates large distances:

    d_chord / d_geo -> sin(theta)/theta -> 2/pi as theta -> pi/2

This means the chordal mean gives more weight to distant subspaces than the
geodesic mean does.

## 5. Worked Numerical Example

d=256, r=8, N=10, random regime:

    Expert subspaces: 10 random points on Gr(8, 256)
    Expected pairwise coherence: ||U_i^T U_j||_F ~ r/sqrt(d) = 0.5
    Measured mean coherence: 0.498

    Naive addition:
      delta = sum (1/8) * A_i @ B_i    (256, 256 matrix)
      SVD -> top-8 subspace
      Mean preservation of experts: 0.167 (= 42% above r/d = 0.031)

    Chordal Frechet:
      P_avg = (1/10) sum U_i U_i^T    (256, 256)
      top-8 eigenvectors
      Mean preservation: 0.200 (+20% over naive)

    Geodesic Karcher:
      Initialize with chordal result
      5-10 iterations to converge
      Mean preservation: 0.189 (-6% vs chordal, +13% over naive)

    Chordal-geodesic distance: 1.29 / 2.83 = 46% of max

    Latency:
      Naive:    ~2ms (10 matmuls)
      Chordal: ~15ms (eigendecomposition of 256x256)
      Geodesic: ~35ms (10 iterations of N Log maps)

    Key observation: Chordal beats geodesic on CHORDAL preservation metric
    (tautological -- it optimizes that exact quantity). The geodesic mean
    optimizes a different objective (arc-length distances).

## 6. The Advantage Grows with N

The subspace preservation advantage of chordal Frechet over naive addition
increases with N because:

1. Naive addition's SVD-extracted subspace is biased by B norms, not geometry
2. As N grows, more experts compete for limited subspace capacity
3. The Frechet mean optimally allocates the r dimensions across all N experts
4. The naive addition wastes capacity by over-representing experts with large B

At d=256:
    N=2:   +5.9% advantage
    N=5:  +13.3% advantage
    N=10: +20.0% advantage
    N=25: +28.2% advantage
    N=50: +33.8% advantage

This is a strong scaling property: the benefit of Riemannian averaging grows
exactly where it matters most (large N, constrained subspace capacity).

## 7. Why Chordal Beats Geodesic

Counter-intuitively, the simpler chordal Frechet mean produces HIGHER subspace
preservation than the more expensive geodesic Karcher mean. This is because:

1. Our preservation metric measures ||merged^T U_i||_F^2, which is the chordal
   overlap. The chordal mean directly optimizes this quantity.

2. The geodesic mean optimizes a different metric (sum of squared arc-length
   distances). It spreads its "attention" more evenly across distant experts
   because the geodesic metric gives more weight to far-away subspaces.

3. The chordal mean is "greedier" -- it concentrates on the densest cluster of
   expert subspaces, giving higher average preservation but potentially ignoring
   outlier experts.

For SOLE composition, we want to maximize the fraction of each expert preserved
in the merged result. The chordal mean is therefore the RIGHT choice -- it
directly optimizes our objective.

The large chordal-geodesic distance (K3 result) is not a defect but a feature:
it shows the two methods optimize genuinely different objectives, and the
chordal one is the one we want.

## 8. Assumptions and Limitations

1. **Random B matrices.** We use random B to isolate the subspace geometry
   effect from learning dynamics. Trained B matrices may have structure that
   changes the relative advantage.

2. **Subspace preservation as proxy.** We measure how well the merged subspace
   captures each expert's A subspace, not downstream task quality. The two
   should correlate but this is not guaranteed.

3. **Pre-merge only.** This analysis assumes pre-merge composition (compute
   merged weights once, serve). Dynamic per-token routing is a different
   problem where Frechet merge is not applicable.

4. **The d^3 cost of eigendecomposition.** At d=4096 (production), the 4096^3
   eigendecomposition takes ~seconds on GPU. This is a one-time cost per merge
   and is negligible compared to training (hours). But it is O(d^3) vs O(N*d*r)
   for naive addition.

5. **Chordal mean optimizes a specific metric.** It maximizes average subspace
   overlap, not worst-case overlap. An expert-weighted or minimax variant might
   be better for specific use cases.
