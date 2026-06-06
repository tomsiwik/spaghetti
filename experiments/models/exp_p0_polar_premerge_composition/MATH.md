# MATH.md: PoLAR Pre-Merge Composition

## Context

Finding #526 (KILLED): Pre-merge composition fails because adapter perturbations share
non-orthogonal directions, not because of magnitude. Standard LoRA at 737x different norms
produces identical 0% GSM8K. The disease is direction overlap; only structural orthogonality
can cure it.

Finding #442 (SUPPORTED): PoLAR with joint Stiefel retraction guarantees sr(DW) = r exactly
(verified to 7 decimal places). Standard LoRA collapses to sr ~ 1.77.

**Question:** Does PoLAR's spectral regularity (sr = r) enable safe pre-merge composition?

## Theorem 1: Spectral Regularity Bounds Inter-Adapter Interference

**Setup.** Let DW_i = A_i B_i be a PoLAR adapter with A_i in St(r, d_in) (columns
orthonormal) and B_i in St(r, d_out)^T (rows orthonormal). Let {DW_i}_{i=1}^N be N
independently initialized adapters.

**Definition.** The cross-term interference when applying merged weights to a domain-j
input x is:

  interference_j(x) = sum_{i != j} DW_i @ x = sum_{i != j} A_i (B_i x)

**Claim.** For random Stiefel initialization, the expected squared Frobenius norm of
inter-adapter overlap satisfies:

  E[||A_i^T A_j||_F^2] = r^2 / d_in    (i != j)

**Proof.** A_i and A_j are independent random elements of St(r, d_in). Each column of A_i
is uniformly distributed on S^{d_in - 1}. For independent uniform vectors u, v on S^{d-1}:

  E[<u, v>^2] = 1/d

The matrix A_i^T A_j has r^2 entries, each with E[(A_i^T A_j)_{kl}^2] = 1/d_in.
Since the entries are not independent (columns within each matrix are orthonormal), the
cross-terms reduce variance slightly, but the expectation is exact:

  E[||A_i^T A_j||_F^2] = sum_{k,l} E[(a_{ik}^T a_{jl})^2] = r^2 / d_in

For r = 6, d_in = 2816: E[overlap] = 36/2816 = 0.0128.   QED.

## Theorem 2: Spectral Spread Reduces Per-Direction Interference

**Setup.** Compare two adapters with same ||DW||_F but different spectral ratios:
- Rank-collapsed (sr ~ 1): energy concentrated in sigma_1, remaining sigmas ~ 0
- Spectrally regular (sr = r): all sigmas equal: sigma_k = ||DW||_F / sqrt(r)

**Claim.** The maximum interference per overlapping direction is:

  Standard LoRA (sr ~ 1):  max_interference ~ ||DW||_F^2 * (r / d_in)
  PoLAR (sr = r):          max_interference ~ ||DW||_F^2 * (1 / d_in)

giving a factor-r reduction in worst-case per-direction interference.

**Proof.** For rank-collapsed LoRA, sigma_1 ~ ||DW||_F, so the energy in the top
direction is ||DW||_F^2. A random r-dimensional subspace overlaps this direction
with probability r/d_in, giving interference ~ ||DW||_F^2 * r/d_in.

For PoLAR, each direction carries ||DW||_F^2 / r energy. The same overlap gives
interference per direction ~ (||DW||_F^2 / r) * (r/d_in) = ||DW||_F^2 / d_in.

The ratio is r = 6.   QED.

## Theorem 3: Pre-Merge Error Bound

**Setup.** W_merged = W_base + sum_i DW_i. For domain-j query x, the ideal output is
(W_base + DW_j)x. The pre-merge error is:

  error_j = sum_{i != j} DW_i x = sum_{i != j} A_i (B_i x)

**Bound.** For PoLAR adapters with B_i on Stiefel:

  E[||error_j||^2] <= (N-1) * r / d_in * ||x||^2

(Since each B_i preserves norms: ||B_i x||^2 = ||P_i x||^2 where P_i projects
onto B_i's row space, and A_i maps this through an r-dimensional subspace
overlapping other adapters' subspaces with measure r/d_in.)

For N=3, r=6, d_in=2816:

  E[||error_j||^2 / ||x||^2] <= 2 * 6/2816 = 0.0043

This is 0.43% of the input energy — negligible interference.

**Contrast with standard LoRA:** Rank collapse (sr ~ 1.77) concentrates energy,
making the effective interference (N-1) * 1/d_in * ||DW||_F^2, but ||DW||_F^2
is amplified by the rank-1 dominant direction, and trained adapters' top directions
may correlate (all trying to correct the same base model deficiency).

## Predictions

### Quantitative
1. PoLAR sr(DW_i) >= 5.5 for all 3 adapters (from Finding #442: sr = r exactly)
2. Inter-adapter cosine similarity < 0.05 (from Theorem 1: overlap = r^2/d_in = 0.013)
3. Pre-merge error ratio < 1% of input energy (from Theorem 3)

### Behavioral
4. PoLAR solo GSM8K >= 50% (adapters must be useful before testing pre-merge)
5. PoLAR pre-merge GSM8K >= 40% (0.43% interference should not destroy performance)
6. Standard LoRA pre-merge GSM8K = 0-1% (confirmed in Findings #510, #526)

### Kill Criteria Mapping
- K1451 (pre-merge GSM8K >= 50%): Tests Theorem 3 prediction that 0.43% interference is survivable
- K1452 (sr >= 5.0): Tests PoLAR Stiefel guarantee (Theorem 1/2 prerequisite)
- K1453 (inter-adapter cos < 0.1): Tests Theorem 1 orthogonality prediction
- K1454 (solo GSM8K >= 50%): Validates adapter quality (necessary condition for meaningful pre-merge test)

## What Would Kill This

1. **PoLAR training destroys adapter quality** (sr = r but poor task performance):
   Implies Stiefel constraint over-regularizes, preventing task-specific learning.
   Impossibility: r dimensions insufficient for task representation under norm constraint.

2. **Inter-adapter cosine >> 0.05** (adapters converge to shared directions):
   Implies gradient dynamics drive independently-trained adapters to correlated
   subspaces (all correct the same base deficiency). Random initialization not
   sufficient; explicit Grassmannian allocation needed.

3. **Pre-merge fails despite low cosine** (interference in functional space, not weight space):
   Implies the error bound in Theorem 3 is loose — weight-space orthogonality
   does not guarantee functional-space orthogonality. The same input x activates
   different features through DW_j and DW_i even when A_i^T A_j ~ 0.

## Type
Verification (if K1 passes) / Guided exploration (if K1 fails — learn which
kill mechanism operates)
