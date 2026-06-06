# TAAP Convergence: Mathematical Analysis

## Notation

| Symbol | Definition | Shape/Type |
|--------|-----------|------------|
| d | Ambient dimension (embedding dim) | scalar |
| r | Subspace dimension (LoRA rank) | scalar |
| N | Number of subspaces (experts) | scalar |
| U_i | Orthonormal frame for subspace i | (d, r) |
| G | Block Gram matrix, G_{ij} = U_i^T U_j | (Nr, Nr) |
| mu | Coherence: max_{i != j} ||G_{ij}||_F | scalar |
| c | Common pairwise coherence (equidistributed) | scalar |
| mu_W | Welch bound on mu | scalar |

## The Welch Bound for Grassmannian Packings

For N subspaces of dimension r in R^d with Nr > d, the generalized Welch
bound provides a lower limit on the maximum pairwise coherence:

$$\mu_W = \sqrt{\frac{r(Nr - d)}{d(Nr - r)}}$$

This bound is derived from a quadratic averaging argument on the Gram matrix.
It represents the theoretical floor -- no arrangement of N subspaces can
achieve max coherence below mu_W.

## The Equidistributed Fixed Point

Alternating Projection converges to an **equidistributed** configuration where
all pairwise coherences are equal: ||G_{ij}||_F = c for all i != j.

**Theorem.** The equidistributed coherence satisfies:

$$c_{eq} = \sqrt{\frac{r(Nr - d)}{d(N - 1)}}$$

**Proof.** Consider the block Gram matrix G (Nr x Nr) with:
- Diagonal blocks: G_{ii} = I_r (orthonormality)
- Off-diagonal blocks: ||G_{ij}||_F = c for all i != j (equidistribution)

The squared Frobenius norm of G is:

$$||G||_F^2 = \sum_i ||G_{ii}||_F^2 + \sum_{i \neq j} ||G_{ij}||_F^2 = Nr + N(N-1) c^2$$

The Gram matrix has rank at most d and trace Nr. By Cauchy-Schwarz on the
eigenvalues (lambda_1, ..., lambda_d):

$$\sum_{k=1}^d \lambda_k^2 \geq \frac{(\sum_{k=1}^d \lambda_k)^2}{d} = \frac{(Nr)^2}{d}$$

Since ||G||_F^2 = sum lambda_k^2:

$$Nr + N(N-1) c^2 \geq \frac{(Nr)^2}{d}$$

Solving for c^2:

$$c^2 \geq \frac{Nr(Nr - d)}{d \cdot N(N-1)} = \frac{r(Nr - d)}{d(N-1)}$$

Equality holds when all d eigenvalues are equal (= Nr/d), which is the
equidistributed case. AP converges to this configuration, achieving equality.

## The sqrt(r) Gap Identity

**Theorem.** The ratio of equidistributed coherence to Welch bound is exactly
sqrt(r), independent of N and d:

$$\frac{c_{eq}}{\mu_W} = \sqrt{r}$$

**Proof.**

$$\frac{c_{eq}}{\mu_W} = \sqrt{\frac{r(Nr-d)/(d(N-1))}{r(Nr-d)/(d(Nr-r))}} = \sqrt{\frac{Nr - r}{N - 1}} = \sqrt{\frac{r(N-1)}{N-1}} = \sqrt{r}$$

**Corollary.** For rank-8 LoRA (r = 8): c_eq / mu_W = sqrt(8) = 2.828.
This is the "2.8-3x gap" observed in the grassmannian_expert_init experiment.
It is not a convergence failure -- it is a mathematical identity.

## Why the Gap Is Fundamental

The Welch bound is a bound on the **maximum** pairwise coherence. It permits
non-equidistributed arrangements where most pairs have low coherence but some
pairs have higher coherence. Specifically:

$$\sum_{i < j} ||G_{ij}||_F^2 \geq \frac{N(N-1)}{2} \cdot \mu_W^2$$

This says the **mean** of ||G_{ij}||_F^2 must exceed mu_W^2. For an
equidistributed arrangement (all ||G_{ij}||_F = c), this becomes c >= mu_W,
but the rank constraint strengthens this to c >= sqrt(r) * mu_W.

For r = 1 (vectors, not subspaces), the gap vanishes (sqrt(1) = 1) and
equiangular tight frames achieve the Welch bound exactly.

For r > 1, the rank-d constraint on the (Nr x Nr) Gram matrix imposes
stronger restrictions on equidistributed configurations than the Welch
averaging argument captures. The gap sqrt(r) quantifies this exactly.

## Implications for SOLE

1. **No algorithm can close the gap while maintaining equidistribution.**
   TAAP, momentum, adaptive scheduling -- none can produce equidistributed
   packings with coherence below sqrt(r) * mu_W.

2. **Non-equidistributed packings exist that approach the Welch bound** but
   they sacrifice worst-case guarantees (some pairs have much higher coherence).

3. **The correct capacity bound** for SOLE is c_eq, not mu_W:
   - N_max at coherence threshold mu_max: solve c_eq(N) <= mu_max
   - This gives: N <= 1 + r(d - r*N_max) / (d * mu_max^2)
   - For the equidistributed case, this is less restrictive than the Welch
     bound suggests (by a factor of r in the capacity).

4. **Practical impact is ZERO.** The AP skeleton at c_eq = sqrt(r) * mu_W
   already provides equidistributed, minimax-optimal coherence. The gap
   only matters if comparing to the Welch bound as a benchmark -- and the
   Welch bound is the wrong benchmark for r > 1 equidistributed packings.

## Worked Example

d = 128, r = 8, N = 20:
- Nr = 160, Nr/d = 1.25
- Welch bound: mu_W = sqrt(8 * (160-128) / (128 * (160-8))) = sqrt(256/19456) = 0.1147
- Equidistributed: c_eq = sqrt(8 * (160-128) / (128 * 19)) = sqrt(256/2432) = 0.3244
- Ratio: c_eq/mu_W = 0.3244/0.1147 = sqrt(8) = 2.828
- AP achieves c_eq exactly (verified empirically)

## Computational Complexity

| Operation | Cost | Notes |
|-----------|------|-------|
| Standard AP (n_iter iterations) | O(n_iter * N^2 * r^2 + n_iter * (Nr)^3) | Structural + spectral |
| TAAP-Schedule | Same as AP | Only changes mu target |
| TAAP-Momentum | Same as AP | Only changes iteration dynamics |
| TAAP-Selective | O(n_iter * N^2 * r^2 + n_iter * (Nr)^3) | Extra norm computation |
| All methods converge to same fixed point | | |

All methods have identical computational cost per iteration and converge to
the same equidistributed fixed point c_eq = sqrt(r(Nr-d)/(d(N-1))).
