# MATH.md — exp_p3_b2_fullw_orthogonal_compose

## Background: Why B-Matrix GS Was Insufficient (P3.B1 Finding #462)

P3.B1 applied Gram-Schmidt orthogonalization to the **B-matrices** only:
- B_P' ⊥ B_D (algebraically exact, cos = 2.5e-7)
- Result: style compliance 76% → 60%, Δ = 16pp (K1174 FAIL, threshold ≤ 10pp)

**Root cause**: LoRA computes ΔW = A × B (using mlx_lm's storage convention: x @ lora_a @ lora_b).
The full interference term is:

  ⟨ΔW_P, ΔW_D⟩_F = tr(ΔW_P^T ΔW_D) = tr((la_P @ lb_P)^T (la_D @ lb_D))
                   = tr(lb_P^T (la_P^T la_D) lb_D)

After B-only GS (lb_P' ⊥ lb_D in row space):

  ⟨ΔW_P', ΔW_D⟩_F = tr(lb_P'^T (la_P^T la_D) lb_D)

This is zero ONLY IF la_P^T la_D = 0. But la_P and la_D are both initialized with
random Gaussian in ℝ^{d_in × r} and trained from the same base model — their columns
are NOT orthogonal (typical cosine ≈ 0.02–0.08 for random matrices of rank 4–6 in ℝ^{2048}).
The A-matrix cross-term explains the residual 16pp in P3.B1.

---

## Theorem 1: Full ΔW Null-Space Orthogonality

**Setup**: For each overlap layer ℓ (layers 26–41 where both adapters are active):
- ΔW_D = la_D @ lb_D ∈ ℝ^{d_in × d_out}, rank r_D = 6
- ΔW_P = la_P @ lb_P ∈ ℝ^{d_in × d_out}, rank r_P = 4

**Theorem 1**: Define the column-space projector of ΔW_D:

  SVD(ΔW_D) = U_D Σ_D V_D^T,   U_D ∈ ℝ^{d_in × r_D}  (orthonormal columns)

  Projection:  ΔW_P' = ΔW_P - U_D (U_D^T ΔW_P)

Then:

  ⟨ΔW_P', ΔW_D⟩_F = tr(ΔW_P'^T ΔW_D) = 0   (EXACTLY)

**Proof**:

  ΔW_D = U_D Σ_D V_D^T

  ΔW_P'^T ΔW_D = (ΔW_P - U_D U_D^T ΔW_P)^T ΔW_D
               = ΔW_P^T ΔW_D - ΔW_P^T U_D U_D^T ΔW_D
               = ΔW_P^T ΔW_D - ΔW_P^T U_D U_D^T U_D Σ_D V_D^T
               = ΔW_P^T ΔW_D - ΔW_P^T U_D Σ_D V_D^T  [since U_D^T U_D = I_{r_D}]
               = ΔW_P^T (U_D Σ_D V_D^T) - ΔW_P^T (U_D Σ_D V_D^T)
               = 0   □

Note: U_D is the left singular vectors of ΔW_D (column space basis), NOT the QR
basis of the B-matrix as in P3.B1. This is strictly more general:
- P3.B1 projected onto the complement of span(rows of lb_D) = column space of lb_D^T
- P3.B2 projects onto the complement of span(cols of ΔW_D) = column space of la_D @ lb_D

**The A-matrix cross-term is eliminated because**:

  tr(ΔW_P'^T ΔW_D) = tr(lb_P'^T (la_P^T la_D) lb_D) + ...  [all terms vanish by Theorem 1]

The full ΔW orthogonality subsumes B-matrix orthogonality: if ΔW_P' ⊥ ΔW_D, then
trivially lb_P'^T (la_P^T la_D) lb_D contributes zero to the Frobenius inner product.

---

## Theorem 2: SVD Re-factorization Preserves Rank

**Claim**: ΔW_P' can be exactly re-factorized to rank ≤ r_P.

**Proof**:
  rank(ΔW_P) = rank(la_P @ lb_P) ≤ min(r_P, d_in, d_out) = r_P
  rank(ΔW_P') = rank((I - U_D U_D^T) ΔW_P) ≤ rank(ΔW_P) ≤ r_P

Since the projection only removes components (cannot increase rank), ΔW_P' has
numerical rank ≤ r_P. Truncated SVD at rank r_P recovers ΔW_P' exactly (within float32
numerical precision, error ≈ 1e-7 relative).

Re-factorization:
  SVD(ΔW_P') → U_P', Σ_P', V_P'^T  (keeping top r_P singular values/vectors)
  la_P' = U_P'[:, :r_P] × sqrt(Σ_P'[:r_P])       ∈ ℝ^{d_in × r_P}
  lb_P' = sqrt(Σ_P'[:r_P])[:, None] × V_P'^T[:r_P, :]  ∈ ℝ^{r_P × d_out}

Then: la_P' @ lb_P' ≡ ΔW_P' exactly.   □

---

## Theorem 3: Power Equalization (same structure as P3.B1 Theorem 2)

After SVD re-factorization, apply power equalization over overlap layers (26–41):

  S_D = Σ_{ℓ=26}^{41} ||MATH_SCALE × lb_D_ℓ||_F   (math power, B-norm proxy)
  S_P = Σ_{ℓ=26}^{41} ||la_P'_ℓ @ lb_P'_ℓ||_F    (personal power, full ΔW norm)
  α = S_D / S_P

Equalizing: la_P'_ℓ ← la_P'_ℓ × sqrt(α),  lb_P'_ℓ ← lb_P'_ℓ × sqrt(α)

Result: power_ratio_after = 1.0 exactly.   □

**Note**: We use the Frobenius norm of ΔW_P' (not just lb_P') for equalization
because after SVD re-factorization, la and lb share the scale symmetrically.

---

## Behavioral Prediction: Why K1182 Should Pass

P3.B1 measured 16pp residual after B-only GS. Our hypothesis: this residual is the
A-matrix cross-term contribution.

**Evidence**:
1. B-matrix orthogonality was exact in P3.B1 (cosine = 2.5e-7)
2. The residual cannot come from B-matrix interference (it was zeroed)
3. By elimination: A-matrix cross-term is the only remaining interference source

**Prediction**: Full ΔW orthogonality removes the A-matrix cross-term entirely.
If the hypothesis is correct: style compliance delta ≤ 10pp (K1182 PASS).

**Kill condition**: If K1182 still fails (>10pp) despite exact ΔW orthogonality,
then the interference is NOT linear-algebraic — it arises from non-linear
interactions in the transformer (attention, normalization, or next-layer effects).
In that case, no linear projection of LoRA weights can fix simultaneous T2+T3 composition,
and the correct approach is sequential application (T2 → T3 as residual, not additive).

---

## Kill Criteria Predictions

| Kill Criterion | Prediction | Source |
|----------------|-----------|--------|
| K1180: max_cos(ΔW_P', ΔW_D)_F < 1e-6 | ≈ 1e-7 (float32 precision) | Theorem 1 |
| K1181: power_ratio == 1.0 | = 1.0 exactly | Theorem 3 |
| K1182: style compliance Δ ≤ 10pp | ≤ 10pp | Hypothesis: A-matrix cross-term |
| K1183: math MCQ Δ ≤ 5pp | ≤ 5pp | la_D, lb_D unchanged by projection |

K1180, K1181: algebraic guarantees — cannot fail modulo floating point.

K1182: behavioral prediction with ~65% confidence.
- PASS case: 16pp residual was A-matrix cross-term (finding from P3.B1 failure analysis)
- FAIL case: Non-linear interference from attention/norm layers (would require sequential composition)

K1183: domain adapter is unchanged (la_D, lb_D not modified); math MCQ unaffected.

---

## Connection to arxiv 2402.03513 (Null-Space LoRA)

The original Null-Space LoRA paper projects NEW adapter weights during training into the
null space of existing adapters to prevent catastrophic forgetting. Our approach applies
the same geometric principle POST-HOC to compose two already-trained adapters.

Difference: we project ΔW_P onto the null space of ΔW_D^T (complement of column space
of ΔW_D), then re-factorize. This is equivalent to the paper's projection but applied
offline rather than during training gradient steps.

The paper proves that null-space projection preserves the existing adapter's outputs for
any input, which is precisely the K1183 guarantee (domain adapter outputs unchanged).
