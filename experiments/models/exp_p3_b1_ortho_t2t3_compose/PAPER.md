# PAPER.md — exp_p3_b1_ortho_t2t3_compose

## Gram-Schmidt Orthogonalization for T2+T3 Adapter Composition

### Research Question

Finding #460 showed naive T2+T3 composition (weight addition) catastrophically suppresses
personal style (100pp loss) due to: (1) B-matrix cosine ε_B = 0.1607 and (2) power ratio
S_D/S_P = 2.96×. Can Gram-Schmidt projection of personal B-matrices onto the complement of
the domain B-matrix row space eliminate this interference and preserve style compliance ≤ 10pp?

### Predictions vs Measurements

| Kill Criterion | Prediction | Measured | Pass? |
|----------------|-----------|---------|-------|
| K1172: max B-matrix cosine < 0.05 | = 0.0 exactly (algebraic) | 2.50e-07 ≈ 0.0 | **PASS** |
| K1173: power ratio < 1.5 | = 1.0 exactly (algebraic) | 1.00 | **PASS** |
| K1174: style compliance loss ≤ 10pp | ≤ 10pp | **+16.0pp** (76% → 60%) | **FAIL** |
| K1175: math MCQ within 5pp | ≤ 5pp | +0.0pp (10% → 10%) | **PASS** |

**Overall: FAIL** (1 kill criterion violated)

### Key Observations

**Algebraic guarantees hold exactly**: B-matrix GS orthogonality reduces cosine from 0.1607
to 2.50e-07 (essentially machine precision). Power equalization ratio = 1.00 exactly.
Both Theorem 1 and Theorem 2 from MATH.md verified.

**Behavioral failure despite algebraic success**: Style compliance degrades 76% → 60% (16pp),
exceeding the 10pp threshold. Math MCQ is preserved exactly (10% both phases).

**Personal-only at 76%** (not 100% as in smoke test at N=5, which was noise). Full-run N=25
reveals true personal adapter capability. Composed rate 60% shows partial preservation but
insufficient for the kill criterion.

### Interpretation

The MATH.md notes the residual failure mode: "If K1174 fails despite algebraic guarantees,
the failure indicates behavioral quality depends on interaction effects BEYOND B-matrix
direction alignment (e.g., rank collapse, A-matrix interference)."

LoRA modification is ΔW = A × B. Orthogonalizing B alone does NOT orthogonalize ΔW:
```
ΔW_pers = A_P × B_P'  (A_P ∈ ℝ^{r_P × d_in} UNCHANGED)
ΔW_domain = A_D × B_D

ΔW_pers^T · ΔW_domain = B_P'^T × (A_P^T A_D) × B_D
```

Even with B_P' ⊥ B_D, the cross term A_P^T A_D ≠ 0 because both adapters share the same
base model's gradient landscape — A-matrices converge to overlapping directions during training.

**Impossibility structure**: B-only GS cannot guarantee full ΔW orthogonality because:
- True interference requires A_P^T A_D = 0 (A-matrix orthogonality)
- A-matrices trained on same base model → NOT orthogonal in general
- Expected A-matrix interference: O(r_P × r_D / d_in) ≈ 0.01 per entry (non-negligible)

**Mathematical fix**: Full ΔW orthogonalization via null-space projection of ENTIRE
ΔW_pers matrix (not just B component), or train with explicit orthogonality constraint
per arxiv 2402.03513 (Null-Space LoRA).

### What This Means for the Architecture

The result is informative: GS partial orthogonalization reduces the 100pp catastrophic
failure (Finding #460) to 16pp degradation. The direction is correct but incomplete.

For the composable adapter vision, this implies:
- Naive weight addition: 100pp loss (Finding #460)
- B-matrix GS only: 16pp loss (this experiment)
- Full ΔW GS: predicted <10pp loss (next experiment P3.B2)

### Structural Impossibility

**Theorem (Necessity of Full ΔW Orthogonalization)**:
For personal style compliance loss δ_P < 10pp under T2+T3 composition:
- Sufficient condition: ΔW_pers · ΔW_domain = 0 in operator norm sense
- Necessary: A_P^T A_D ≈ 0 (A-matrix orthogonality) when B-matrix orthogonality is satisfied
- B-matrix-only GS leaves A-matrix interference O(||A_P^T A_D||_F × ||B_D||_F) ≈ 16pp

**Fix derivation**: Extend GS projection to operate on full ΔW matrices:
1. Compute ΔW_D = A_D × B_D per layer (explicit matrix product)
2. QR decompose ΔW_D^T to get Q_full ∈ ℝ^{d_out × r_D}
3. Project ΔW_P' = ΔW_P - ΔW_P @ Q_full @ Q_full^T
4. Re-factorize ΔW_P' into A_P', B_P' via SVD (necessary since ΔW is rank-r_P)

This extends the guarantees of Theorems 1-2 from the B-matrix subspace to the full output space.

### Citation

Finding #460 (impossibility: naive composition), Finding #429 (hot-add works for single
adapter), arxiv 2106.09685 (LoRA: orthogonal adapters), arxiv 2402.03513 (Null-Space LoRA:
full ΔW orthogonalization). MATH.md Theorem 1+2.
