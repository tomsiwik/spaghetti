# MATH — exp_p3_b3_fullw_ortho_alpha1

## Problem Statement

P3.B2 (Finding #463) showed full ΔW null-space orthogonalization achieves exact algebraic
independence (cos=9.66e-18) but style compliance degraded 36pp (76%→40%) — WORSE than P3.B1
B-GS which degraded 16pp (76%→60%). Both experiments achieved algebraic orthogonality; the
behavioral divergence must trace to a non-algebraic variable.

**Confound identified**: Power equalization factor.
- P3.B1 B-GS: α = S_D_B / S_P_B = 1.369 (B-matrix Frobenius norms)
- P3.B2 Full-ΔW: α = S_D_ΔW / S_P_ΔW = 4.349 (full ΔW Frobenius norms)

The personal adapter was amplified 4.349× in P3.B2 vs 1.369× in P3.B1. The hypothesis:
over-amplification disrupts the personal style signal because LayerNorm and attention
mechanisms respond non-linearly to adapter output magnitude.

## Theorem 1 (Natural-Scale Null-Space Composition)

**Setup**: Let W_D (rank r_D=6, scale 6.0) be the domain adapter and W_P (rank r_P=4,
scale 4.0) the personal adapter, both trained on Gemma 4 (gemma-4-e4b-it-4bit).

Let ΔW_D = la_D @ lb_D (where lb_D absorbs MATH_SCALE=6.0 baked in),
    ΔW_P = la_P @ lb_P (natural la/lb without additional scaling).

Define:
- U_D ∈ ℝ^{d_in × r_D}: left singular vectors of ΔW_D (column-space basis)
- ΔW_P' = ΔW_P - U_D(U_D^T ΔW_P)   [null-space projection]
- W_composed = W_base + ΔW_D + ΔW_P'   [additive composition, α=1.0]

**Theorem**: Under this construction,
1. ⟨ΔW_P', ΔW_D⟩_F = 0 exactly (algebraic)
2. ||ΔW_P'||_F ≤ ||ΔW_P||_F (projection is a contraction, Pythagoras)
3. The natural scale ||ΔW_P'||_F / ||ΔW_D||_F ≪ 1 (personal adapter is naturally small)

**Proof**:
1. ΔW_P' ⊥ U_D by construction. Since col(ΔW_D) ⊆ col(U_D), we have:
   ⟨ΔW_P', ΔW_D⟩_F = tr(ΔW_P'^T ΔW_D) = tr(ΔW_P'^T U_D Σ V^T) = 0  □

2. ΔW_P' = (I - U_D U_D^T) ΔW_P. The operator (I - U_D U_D^T) is an orthogonal projector,
   so ||ΔW_P'||_F = ||(I - U_D U_D^T) ΔW_P||_F ≤ ||(I - U_D U_D^T)||_op × ||ΔW_P||_F = ||ΔW_P||_F  □

3. From P3.B2 measurements: total_norm_math_overlap = 47.86, total_norm_pers_raw = 11.01.
   Without equalization: effective ratio = ||ΔW_P'||_F / ||ΔW_D||_F = 11.01/47.86 = 0.23.
   With P3.B2 equalization: ratio becomes 1.0 (4.349× amplification of personal adapter).  □

**QED**

## Mechanism Analysis (LayerNorm Interaction)

Gemma 4 uses RMSNorm with learned scale γ. For input h and residual δh = ΔW_P' x:
   h' = RMSNorm(h + δh) × γ

When ||δh|| is small (α=1.0), the perturbation is absorbed gracefully.
When ||δh|| is large (α=4.349×), the denominator √(|h+δh|²/d) shifts significantly,
altering all downstream activations proportionally.

The prediction: α=1.0 keeps ||ΔW_P' x|| within the "safe operating range" for
Gemma 4's LayerNorm, bounding behavioral degradation to ≤ 10pp.

## Quantitative Predictions

From P3.B2 Finding #463:
- equalization_factor α: 4.349 (P3.B2) → 1.0 (P3.B3, no rescaling)
- total_norm_pers_raw: 11.01 (P3.B2 ΔW_P')
- total_norm_math_overlap: 47.86 (domain adapter)
- Natural personal/domain norm ratio: 11.01/47.86 = 0.23 (vs 1.0 with equalization)

**Kill criteria predictions**:
- K1188 (algebraic cosine): = 0.0 exactly (same math as P3.B2, α doesn't affect orthogonality)
  Probability: 99% PASS (algebraic guarantee)
- K1189 (style Δ ≤ 10pp): personal_only=76%, composed target ≥ 66%
  - If α was the confound: composed ≈ 70-76%, Δ = 0-6pp → PASS
  - If non-linear interference: composed ≈ 60%, Δ ≈ 16pp → FAIL
  Probability: 60% PASS (isolates the confound)
- K1190 (math Δ ≤ 5pp): same domain adapter, math should hold
  Probability: 80% PASS

## Connection to Prior Results

| Experiment | α | Algebraic cos | Style Δ | Status |
|---|---|---|---|---|
| P3.B1 (B-GS) | 1.369 | 2.5e-7 | +16pp | KILLED (>10pp) |
| P3.B2 (Full-ΔW) | 4.349 | 9.66e-18 | +36pp | KILLED (worse) |
| P3.B3 (α=1.0) | 1.0 | ~9.66e-18 | ? | This experiment |

If P3.B3 style Δ < 16pp (P3.B1 bound): full-ΔW orthogonality is strictly better than
B-GS, and the α confound explanation is confirmed.
If P3.B3 style Δ ≥ 16pp: non-linear interference is the dominant mechanism regardless
of algebraic orthogonality; sequential composition (P3.B4) becomes necessary.

## Failure Mode (if KILLED)

If K1189 FAIL despite α=1.0: non-linear interference confirmed.
Impossibility structure for additive composition:
  For two adapters A and B with ΔW_A ⊥ ΔW_B, the composed model:
    f(x) = transformer(x + ΔW_A x + ΔW_B x)
  is NOT equivalent to f_A(x) + f_B(x) - f_base(x) because transformer
  non-linearities (LayerNorm, SiLU in SwiGLU, softmax attention) break superposition.
  This makes additive composition fundamentally limited for behavioral preservation.

Fix: P3.B4 — sequential composition (run domain adapter first, then personal).
Cite: arxiv 2402.03513 notes this as a limitation of the additive approach.
