# Mixed-Rank Grassmannian Capacity: Mathematical Foundations

## Setup and Notation

| Symbol | Meaning | Shape/Range |
|--------|---------|-------------|
| d | ambient dimension | scalar, e.g. 64, 4096 |
| N | number of experts | scalar |
| r_i | rank of expert i | scalar, r_i << d |
| U_i | orthonormal frame for expert i | (d, r_i) |
| Gr(r, d) | Grassmannian manifold of r-planes in R^d | manifold |
| G | block Gram matrix | (sum(r_i), sum(r_i)) |
| G_{ij} | block of Gram matrix | (r_i, r_j) |
| sigma_k(G_{ij}) | k-th singular value of G_{ij} | scalar in [0, 1] |

## Cross-Rank Coherence

For two subspaces U_i in Gr(r_i, d) and U_j in Gr(r_j, d), the
cross-coherence is defined via the singular values of U_i^T U_j:

```
G_{ij} = U_i^T U_j     shape: (r_i, r_j)
```

The number of principal angles is min(r_i, r_j). The singular values
sigma_1 >= ... >= sigma_{min(r_i, r_j)} of G_{ij} are the cosines of
the principal angles.

**Normalized coherence** (in [0, 1] regardless of rank pair):

```
coh(i, j) = ||G_{ij}||_F / sqrt(min(r_i, r_j))
           = sqrt(sum_k sigma_k^2) / sqrt(min(r_i, r_j))
```

**Chordal distance** (generalized to mixed ranks):

```
d_c^2(i, j) = min(r_i, r_j) - ||G_{ij}||_F^2
            = min(r_i, r_j) - sum_k sigma_k^2
```

**Normalized chordal distance** (in [0, 1]):

```
d_c_norm(i, j) = d_c(i, j) / sqrt(min(r_i, r_j))
```

When d_c_norm = 1, the subspaces are fully orthogonal (zero overlap in
the smaller subspace). When d_c_norm = 0, one subspace is contained in
the other.

## Block Gram Matrix Structure

The full Gram matrix G has size (sum(r_i), sum(r_i)):

```
G = [ I_{r_1}    G_{12}    ...  G_{1N}  ]
    [ G_{21}     I_{r_2}   ...  G_{2N}  ]
    [ ...        ...       ...  ...     ]
    [ G_{N1}     G_{N2}    ...  I_{r_N} ]
```

**Valid Gram matrix constraints:**
1. Positive semidefinite: G >= 0
2. Rank at most d: rank(G) <= d
3. Correct trace: tr(G) = sum(r_i) (each diagonal block is identity)

## Mixed-Rank Alternating Projection

### Structural Projection (modified for mixed ranks)

For each off-diagonal block G_{ij}:

```
threshold(i, j) = mu_target * sqrt(min(r_i, r_j))
```

If ||G_{ij}||_F > threshold:
```
G_{ij} <- G_{ij} * threshold / ||G_{ij}||_F
```

The rank-dependent threshold ensures normalized coherence is capped at
mu_target for all rank pairs, preventing large-rank experts from
dominating the coherence budget.

### Spectral Projection (unchanged)

Eigendecompose G, keep top-d non-negative eigenvalues, rescale trace to
sum(r_i). This is identical to uniform-rank AP because the spectral
constraint operates on the full Gram matrix, not individual blocks.

### Welch Bound Estimate

No closed-form Welch bound exists for mixed-rank systems. We use a
conservative estimate based on the maximum rank:

```
mu_welch_est = sqrt(r_max * (N * r_max - d) / (d * (N * r_max - r_max)))
```

This overestimates the true bound (because smaller-rank experts use less
space), making the AP target mu_target = 1.5 * mu_welch_est conservative.

## Capacity Analysis

### Uniform-Rank Capacity

For N experts all at rank r:
```
N_max = d^2 / r^2
```

This is the number of r-dimensional subspaces that can be packed in R^d
with bounded coherence (derived from counting dimensions: each subspace
uses r(d - r) degrees of freedom on the Grassmannian).

### Conservative Mixed-Rank Bound

The bottleneck is the largest-rank expert:
```
N_max_conservative = min_i(d^2 / r_i^2) = d^2 / r_max^2
```

This is pessimistic because it assumes ALL experts have the maximum rank.

### Additive Capacity Bound (conjectured)

If experts are distributed across ranks with counts n_r at each rank r,
the available subspace dimensions partition:

```
sum_r n_r * r <= d   (orthogonality constraint)
N_max_additive = sum_r (d^2 / r^2) * (n_r / N)   (weighted average)
```

For the equal distribution {4, 8, 16} with N/3 each:
```
N_max_additive = (256 + 64 + 16) / 3 = 112
```

This is 7x the conservative bound of 16.

## Crowding Effect

When a large-rank expert (r_L) coexists with small-rank experts (r_S),
the large expert's projection onto each small expert's subspace has
Frobenius norm bounded by:

```
||U_L^T U_S||_F <= sqrt(r_S)   (trivial bound)
```

The coherence coh(L, S) = ||U_L^T U_S||_F / sqrt(r_S) can reach 1.0,
meaning the large-rank subspace can fully contain the small-rank subspace.

The empirical finding is that large-rank experts increase the minimum
coherence floor (from ~0.17 for all-r4 to ~0.57 for 19r4+1r16), but
this still leaves normalized chordal distance > 0.8, which is well above
the interference threshold.

## Worked Example (d=64, ranks={4,8,16})

Configuration: 20 experts, mixed_equal distribution (7 at r=4, 7 at r=8, 6 at r=16).

**Gram matrix size:** sum(r_i) = 7*4 + 7*8 + 6*16 = 28 + 56 + 96 = 180.
So G is 180 x 180.

**Theoretical bounds:**
- Conservative N_max = 64^2 / 16^2 = 16
- Per-rank: N_max(r=4) = 256, N_max(r=8) = 64, N_max(r=16) = 16
- N=20 already exceeds the conservative bound by 1.25x

**Empirical result at N=20:**
- min normalized chordal distance = 0.819 (well above 0 cliff)
- max normalized coherence = 0.573

**Comparison to uniform baselines at N=20:**
- r=4: min_dist = 0.993, max_coh = 0.115
- r=8: min_dist = 0.960, max_coh = 0.281
- r=16: min_dist = 0.889, max_coh = 0.459

The mixed-rank packing is dominated by the r=16 component -- the
min_dist (0.819) is lower than the r=16 baseline (0.889), but only by
~8%, far below the 50% degradation threshold.

## Computational Cost

**AP iteration cost:**
- Block Gram computation: O(sum_i sum_j r_i * r_j * d) = O(N^2 * r_avg^2 * d)
- Structural projection: O(N^2 * r_max^2) (iterate blocks)
- Spectral projection: O(T^3) where T = sum(r_i) (eigendecomposition)

For d=64, N=80, mixed ranks: T = sum(r_i) ~ 80 * 9.3 = 744.
Spectral projection is O(744^3) ~ 4.1 * 10^8, dominating the cost.

At production d=4096, N=500: T ~ 500 * 9.3 = 4650. Cost is O(4650^3)
~ 10^11, which would take ~minutes on modern hardware. For N>1000 with
mixed ranks, hierarchical packing would be needed.

## Assumptions

1. **Random initialization.** Frames start as random Grassmannian points.
   With pre-assigned slots (e.g., from domain clustering), quality may
   differ.

2. **Fixed rank per expert.** Each expert has a single rank across all
   layers. Per-layer rank variation (AdaLoRA) would create a more complex
   mixed-Grassmannian problem.

3. **No training interaction.** The capacity analysis is geometric (A-matrix
   only). Post-training B-matrix overlap may reduce effective capacity
   (as shown in the minimax packing experiment).

4. **Coherence normalization.** We normalize by sqrt(min(r_i, r_j)), which
   treats all principal angles equally. Alternative normalizations
   (e.g., by max singular value) would give different capacity estimates.
