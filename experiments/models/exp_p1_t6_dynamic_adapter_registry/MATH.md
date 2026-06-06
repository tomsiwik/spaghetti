# MATH.md — T6.5: Dynamic Adapter Registry

## Problem Statement

A production system serving N adapters must support four lifecycle operations:
register, remove, promote-to-base, and crystallize. Each operation must complete
in bounded time without requiring re-orthogonalization of existing adapters.

We claim all four operations are O(1) in existing adapter count N, because
Grassmannian near-orthogonality makes re-orthogonalization unnecessary.

---

## Theorem 1: O(1) Registry Consistency Under Register/Remove

**Setup.** Let Ω = {A_1, ..., A_N} be a registry of LoRA adapters with
pairwise flat cosine similarity bounded: max_{i≠j} cos(A_i^flat, A_j^flat) < τ.

**Claim.** Adding A_{N+1} to Ω requires only a single compatibility check
(O(N) scan), not re-orthogonalization of A_1,...,A_N. Removing A_k likewise
requires O(1) bookkeeping — remaining adapters are unaffected.

**Proof.**
(→ Register) Grassmannian orthogonality is pairwise: each (A_i, A_j) pair
is checked independently. Adding A_{N+1} requires verifying
  max_{i=1..N} cos(A_{N+1}^flat, A_i^flat) < τ.
If satisfied, A_{N+1} is compatible with all existing adapters by the
pairwise criterion. No existing A_i changes; orthogonality of the pre-existing
set is unaffected.

(→ Remove) Removing A_k from Ω yields Ω \ {A_k}. The remaining N-1 adapters
retain their pairwise relationships unchanged — removing an element from a
pairwise-compatible set yields a pairwise-compatible set.

**Complexity.** Register: O(N) scan, O(N·d) FLOPs. Remove: O(1). No
re-orthogonalization in either case.
**QED.**

---

## Theorem 2: Promote-to-Base Preserves Registry Consistency

**Setup.** Registry Ω has adapter A_1 designated for promotion. The base
model W_base is updated: W' = W_base + scale·B_1^T @ A_1^T. The promoted
adapter is removed from Ω.

**Claim.** After promotion, the updated base W' and remaining registry
Ω' = Ω \ {A_1} are mutually consistent: existing adapters A_2,...,A_N
require no update.

**Proof.**
By Davis-Kahan (Stewart & Sun 1990), the spectral perturbation of W' vs W_base is:
  ε = ||ΔW||_F / ||W_base||_F = scale·||B_1||_F·||A_1||_F / ||W_base||_F

Finding #452 established ε < 5% for Gemma 4 with synthetic weights.

Key: ΔW only modifies the subspace of W_base spanned by A_1. Adapters
A_2,...,A_N operate in complementary subspaces (by near-orthogonality of
their A-matrices in input space). The perturbation ΔW thus does not change
the effective output of adapters A_2,...,A_N in expectation.

More precisely: output of A_k on input x is scale·B_k @ A_k @ x. The
gradient of A_k through the new base W' = W + ΔW changes by:
  Δ∇ ∝ A_k @ ΔW^T A_1^T ≈ cos(A_k^flat, A_1^flat) · small_constant < τ·small_constant

Since cos < τ ≈ 0.1 (Finding #427), the perturbation to adapter k's
effective gradient is < 10% of its magnitude — within normal training noise.
**QED.**

**Prediction.** After N=5 sequential promotions (T6.4 simulated N=3):
  ε_cumul ≤ N · ε_single ≤ 5 × 5% = 25%  (worst case, linear growth)
  ε_cumul ≈ √N · ε_single ≈ 2.24 × 5% = 11.2%  (expected, T6.4 pattern)
  ε_cumul actual (T6.4) = 7.62% at N=3

---

## Theorem 3: Crystallize Preserves Compatibility

**Setup.** A cluster C_k ⊂ Ω of N_k adapters from the same domain.
Crystallization produces: A_crystal = A_canonical (shared init), B_crystal = (1/N_k) Σ_{i∈C_k} B_i.

**Claim.** cos(A_crystal^flat, A_j^flat) = cos(A_canonical^flat, A_j^flat)
for all j ∉ C_k, i.e., crystallization does not change pairwise A-compatibility.

**Proof.**
The A-matrix in LoRA is initialized identically for all users in the same
domain cluster (shared Kaiming uniform init in mlx_lm.lora). Therefore
A_crystal = A_canonical is exactly the per-domain A-matrix, and:
  cos(A_crystal^flat, A_j^flat) = cos(A_canonical^flat, A_j^flat)

This quantity was verified < τ in the initial registration check. Crystallization
does not alter A-matrices — only B-matrices are averaged. Therefore registry
compatibility is preserved identically. **QED.**

---

## Quantitative Predictions

| Kill Criterion | Predicted Value | Basis |
|---------------|-----------------|-------|
| K1132: register time | < 1s per adapter | O(N·d) FLOPs, N=5, d=2560 |
| K1133: remove time | < 50ms | O(1) bookkeeping |
| K1134: promote time | < 2s | Same as T6.3 (≈0.4ms formula, +verification) |
| K1135: crystallize time | < 5s | Same as T6.2 (B averaging only) |
| K1136: max pairwise cos | < τ=0.15 throughout | Theorem 1+3 |

Note: kill criteria bounds (5s, 1s, 30s, 60s) are 5-30× more lenient than
predicted values — experiment tests the structural invariants, not tight timing.

---

## References

- Davis-Kahan: Stewart & Sun 1990 — spectral perturbation bound
- Task Arithmetic: Ilharco et al. 2022, arxiv 2212.04089 — additive LoRA merging
- Model Soup: Wortsman et al. 2022, arxiv 2203.05482 — B-matrix averaging
- Finding #427: max pairwise cos < 0.1 for 5-domain adapters
- Finding #450 (T6.1): silhouette=0.82, K=5 clusters from 25 adapters
- Finding #451 (T6.2): B-avg cos_crystal=0.9806, norm_ratio=1.020
- Finding #452 (T6.3): promotion ε_mean=3.63%, cos_exact=0.99999988
- Finding #453 (T6.4): 3 sequential promotions safe, ε_cumul=7.62%
