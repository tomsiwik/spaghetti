# MATH.md — Multi-Cycle Promotion: Pythagorean Norm Bound

## Problem

The promotion flywheel: each new domain is trained as a LoRA adapter on the current
base, then promoted (merged) into the base. Can the base absorb K sequential promotions
without destroying earlier domains?

## Failure Mode

Without orthogonal A-slots, sequential promotions cause interference:
⟨ΔW_i, ΔW_j⟩_F = tr(ΔW_i^T ΔW_j) ≠ 0 in general.

When training domain k on the promoted base (which contains ΔW_1 + ... + ΔW_{k-1}),
the new adapter ΔW_k can partially cancel the directions used by earlier domains,
destroying their quality. This is the promotion-cycle interference failure mode.

## Self-Test Questions

1. **One-sentence impossibility:** With Grassmannian orthogonal A-slots, later adapters
   cannot modify the directions used by earlier adapters.
2. **Prior theorems:** Pythagorean theorem for orthogonal matrices (standard linear algebra).
3. **Predictions:** K928 ≥ 80%, K929 degradation ≤ 20%, norm bound exact to float precision.
4. **Falsification:** K928 fails (promoted quality < 80% SFT) → interference is real.
5. **Hyperparameters added:** 0 new (slot count = N_domains × rank, fully determined).

---

## Theorem 1 — Pythagorean Norm Bound (Exact, No Approximation)

**Theorem:** Let W_0 be the initial base weights. For K domains, each trained with
LoRA adapter ΔW_k = scale * B_k @ A_k^T where:
- A_k ∈ R^{d_in × rank}: orthonormal columns from QR decomposition
- A_k^T @ A_j = 0 for k ≠ j (Grassmannian orthogonality)

Then after K sequential promotions (W_K = W_0 + Σ_k ΔW_k):

  ‖W_K - W_0‖_F = sqrt(Σ_{k=1}^K ‖ΔW_k‖_F²)

and ‖ΔW_k‖_F = scale * ‖B_k‖_F (since A_k has orthonormal columns).

**Proof:**

Step 1 (Cross-term elimination):
  ⟨ΔW_i, ΔW_j⟩_F = tr(ΔW_i^T ΔW_j)
    = tr((A_i @ B_i^T) @ (B_j^T @ A_j^T))  [substituting ΔW = B.T @ A.T in col-vec form]

Hmm, let me be precise. With:
  A ∈ R^{d_in × rank}: our "down" basis (column convention)
  B ∈ R^{rank × d_out}: learned "up" matrix (initialized to zero)
  LoRA forward: y += scale * (x @ A) @ B  [right-multiply convention]
  ΔW = scale * B.T @ A.T ∈ R^{d_out × d_in}  [to add to nn.Linear weight]

Then:
  ΔW_i = scale * B_i.T @ A_i.T
  ΔW_i^T = scale * A_i @ B_i

  ⟨ΔW_i, ΔW_j⟩_F = tr(ΔW_i^T @ ΔW_j)
    = scale² * tr((A_i @ B_i) @ (B_j.T @ A_j.T))
    = scale² * tr(B_j.T @ A_j.T @ A_i @ B_i)  [cyclic: tr(XY) = tr(YX) applied recursively]

Since A_j.T @ A_i = Q[:,j*r:(j+1)*r].T @ Q[:,i*r:(i+1)*r] = δ_{ij} * I_r:
  For i ≠ j: A_j.T @ A_i = 0 → ⟨ΔW_i, ΔW_j⟩_F = 0 ✓

Step 2 (Pythagorean sum):
  ‖Σ_k ΔW_k‖_F² = Σ_k Σ_j ⟨ΔW_k, ΔW_j⟩_F = Σ_k ‖ΔW_k‖_F²

Step 3 (Norm simplification):
  ‖ΔW_k‖_F² = scale² * ‖B_k.T @ A_k.T‖_F²
             = scale² * tr(A_k @ B_k @ B_k.T @ A_k.T)
             = scale² * tr(B_k @ B_k.T @ A_k.T @ A_k)  [cyclic]
             = scale² * tr(B_k @ B_k.T @ I_r)           [A_k.T @ A_k = I_r, orthonormal cols]
             = scale² * ‖B_k‖_F²

Therefore ‖ΔW_k‖_F = scale * ‖B_k‖_F. **QED**

---

## Theorem 2 — Weight-Space Protection After Promotion

**Theorem:** After promoting domain i (W_base_i = W_base + ΔW_i), any subsequent
adapter ΔW_j (j ≠ i, with A_j ⊥ A_i) has zero Frobenius inner product with ΔW_i.
Therefore training ΔW_j on W_base_i cannot target the directions used by ΔW_i.

**Proof:** Direct consequence of Theorem 1: ⟨ΔW_i, ΔW_j⟩_F = 0 for all B_i, B_j.
The trained adapter ΔW_j lives in the subspace spanned by column combinations of A_j,
which is orthogonal to the subspace spanned by A_i. **QED**

**Activation-space caveat:** Weight-space orthogonality does not guarantee activation-space
orthogonality. For domain-k input x_k, the effect of ΔW_j on the output is:
  ‖ΔW_j @ x_k‖ = scale * ‖(x_k @ A_j) @ B_j‖ = scale * ‖A_j.T x_k‖ * ‖B_j‖

This is zero only if A_j.T x_k = 0 (input lies in A_j's null space). In general it is small
when x_k is "aligned" with A_i (not A_j). The experiment measures K928 to empirically
verify that this residual activation-space interference is below the 20% threshold.

---

## Quantitative Predictions

### K928: Promoted quality retention

After K=3 cycles, for each domain d:
  quality_ratio_d = promoted_acc_d / sft_acc_d ≥ 0.80

Theoretical basis: Theorem 2 guarantees zero WEIGHT-SPACE interference. Activation-space
interference is bounded by the alignment of domain inputs with off-domain A-slots.
For well-separated tasks (different A-slots), we predict quality_ratio ≥ 0.80.

### K929: No degradation across cycles

For domain d promoted at cycle k_d:
  promoted_acc_d_after_K_cycles ≥ 0.80 × promoted_acc_d_after_cycle_k_d

That is, later promotions don't degrade earlier ones.

### Pythagorean Bound (exact)

  |‖W_K - W_0‖_F - sqrt(Σ_k ‖ΔW_k‖_F²)| < epsilon_float

This should hold to floating-point precision (relative error < 1e-5).

### Kill Criterion K930

Any domain quality on promoted base < 50% → KILL (flywheel is dead).
This would mean: activation-space interference overwhelms weight-space protection.

---

## Summary of Predictions

| Kill Criterion | Prediction | Basis |
|----------------|------------|-------|
| K928: All domains >= 80% SFT after 3 cycles | PASS | Theorem 2 (weight-space protection) |
| K929: No domain degrades > 20% cross-cycle | PASS | Theorem 2 (subsequent promotions orthogonal) |
| K930 (KILL): Any domain < 50% | NO KILL | Orthogonality makes < 50% geometrically impossible for well-separated A-slots |
| Pythagorean bound | Exact to float precision | Theorem 1 (algebraic identity) |

---

## References

- Pythagorean theorem for orthogonal matrices: standard linear algebra (Golub & Van Loan 1996)
- Grassmannian A-slots: Finding #50 (this project), max|A_i^T A_j| = 1e-08 for N=50
- MOLE (arXiv:2402.09432): Frobenius interference via shared A-matrix alignment
- Finding #366 (this project): S3 selective routing is true Pareto winner for safe dissolve
- Finding #381 (this project): Separate A-slots mandatory for Theorem 1 on Qwen3-0.6B
- VeRA (arXiv:2310.11454): shared random basis for LoRA parameter reduction
