# MATH.md — exp_p3_b1_ortho_t2t3_compose

## Theorem 1: Gram-Schmidt Orthogonality (Exact)

**Problem**: Finding #460 showed that naive T2+T3 composition fails because:
- max B-matrix cosine ε_B = 0.1607 > threshold 0.1
- Power imbalance S_D/S_P = 2.96×
- Result: personal style catastrophically suppressed (100pp loss)

**Claim**: Gram-Schmidt projection of the personal B-matrices onto the complement of the
math B-matrix row space produces EXACT zero cosine between all domain-personal B-matrix
direction pairs.

**Setup**: Let r_D = 6, r_P = 4, d = d_out.
- B_D ∈ ℝ^{r_D × d}: domain (math) B-matrix, scaled → lb_math_s = scale_D × B_D (each row: output direction)
- B_P ∈ ℝ^{r_P × d}: personal B-matrix, scaled → lb_pers_s = scale_P × B_P

**GS Projection**:
1. Compute reduced QR decomposition of lb_math_s^T:
   Q_D, R = QR(lb_math_s^T)  where Q_D ∈ ℝ^{d × r_D} (orthonormal columns = ONB for row space of lb_math_s)
2. Project lb_pers_s onto the complement:
   lb_pers_s' = lb_pers_s - lb_pers_s @ Q_D @ Q_D^T     ∈ ℝ^{r_P × d}

**Theorem 1 (GS Orthogonality)**: After the projection above:
```
∀ i ∈ {0,...,r_D-1}, j ∈ {0,...,r_P-1}:
    lb_math_s[i,:] · lb_pers_s'[j,:] = 0
```

Equivalently, max_{i,j} |cos(lb_math_s[i,:], lb_pers_s'[j,:])| = 0 exactly.

**Proof**:
Let b_d = lb_math_s[i,:] and b_p = lb_pers_s[j,:].
Since b_d^T is in the column space of Q_D, write b_d = (Q_D c)^T = c^T Q_D^T for some c ∈ ℝ^{r_D}.

The projected row: b_p' = b_p - b_p Q_D Q_D^T

Inner product:
  b_d · b_p' = c^T Q_D^T (b_p - b_p Q_D Q_D^T)^T
             = c^T Q_D^T b_p^T - c^T Q_D^T (Q_D Q_D^T b_p^T)
             = c^T Q_D^T b_p^T - c^T (Q_D^T Q_D) Q_D^T b_p^T
             = c^T Q_D^T b_p^T - c^T I Q_D^T b_p^T    [Q_D^T Q_D = I since Q_D is ONB]
             = c^T Q_D^T b_p^T - c^T Q_D^T b_p^T = 0  □

Applied per-layer across all 16 overlap layers → max cosine = 0 EXACTLY for every layer.

---

## Theorem 2: Power Equalization

**Claim**: Explicit scale adjustment equalizes effective power between domain and personal
adapters after GS projection.

**Definitions** (overlap-only, layers 26-41 where both adapters are active):
- Effective power per adapter (aggregated across overlap layers): S = Σ_{l=26}^{41} ||lb_scaled_l||_F
  (B-matrix Frobenius norm, scales baked in; A-matrix norms treated as approximately equal)
- S_D_overlap = Σ_{l=26}^{41} ||scale_D × B_D_l||_F
- S_P_raw = Σ_{l=26}^{41} ||lb_pers_s'_l||_F  (after GS, before equalization)

**Power equalization** (overlap-only):
  α = S_D_overlap / S_P_raw  (equalization factor)
  lb_pers_final_l = α × lb_pers_s'_l

**Theorem 2 (Power Equalization)**:
  S_P_final = Σ_{l=26}^{41} ||α × lb_pers_s'_l||_F = α × S_P_raw = S_D_overlap
  → power_ratio = S_D_overlap / S_P_final = 1.0 exactly  □

**Rationale for overlap-only**: Power competition only occurs in layers where both
adapters are simultaneously active (layers 26-41). For layers 0-25 (math-only), there
is no competition and no power ratio issue.

**Note on norm reduction**: GS projection reduces ||lb_pers_s'|| ≤ ||lb_pers_s|| since
projection onto a complement only removes components, never adds. So α ≥ 1: we amplify
the personal adapter to restore its original effective contribution PLUS equalize with domain.

---

## Kill Criteria Predictions

| Kill Criterion | Prediction | Derivation |
|----------------|-----------|------------|
| K1172: max cosine < 0.05 | = 0.0 exactly | Theorem 1: GS projection is exact |
| K1173: power ratio < 1.5 | = 1.0 exactly | Theorem 2: equalization is exact |
| K1174: style compliance loss < 10pp | ≤ 10pp | From Theorem 1: no suppression of personal directions |
| K1175: math MCQ within 5pp | ≤ 5pp | GS preserves B_D unchanged; domain signal intact |

K1172 and K1173 are algebraic guarantees — they cannot fail (modulo float32 rounding to ~1e-7).

K1174 and K1175 are behavioral predictions. The reasoning:
- K1175: Math adapter B_D unchanged by GS (only B_P is projected). Math MCQ should be unaffected.
- K1174: After GS, personal directions are orthogonal to domain. Power equalization ensures
  personal adapter is "heard" (not drowned out). Expected: ≤ 10pp degradation.

If K1174 or K1175 fail despite algebraic guarantees: the failure would indicate that behavioral
quality depends on interaction effects BEYOND B-matrix direction alignment (e.g., rank collapse,
A-matrix interference). This would require extending the orthogonality criterion to full ΔW space.

---

## Connection to Finding #460

Finding #460 impossibility structure:
  ε_B × (S_D/S_P) = 0.1607 × 2.96 = 0.476 >> threshold 0.132 (3.6× violation)

After this fix:
  ε_B_fixed × (S_D/S_P_fixed) = 0.0 × 1.0 = 0.0  [exactly 0]

The structural impossible (ε_B × power_ratio >> 0.132) is eliminated by design.

**Citation**: Finding #460 (impossibility), Finding #429 (hot-add T3.6), Gram-Schmidt theorem
(standard linear algebra), arxiv 2106.09685 (LoRA: "adapters in different orthogonal subspaces
do not interfere" — applied here explicitly).
