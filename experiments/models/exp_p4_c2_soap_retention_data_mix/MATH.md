# MATH.md — P4.C2: SOAP Retention Fix via General-Knowledge Data Mix

## Problem Statement

P4.C1 (Finding #480) revealed: SOAP adapter (v_proj+o_proj, rank-16, N=100 SOAP examples)
achieves SOAP format compliance +70pp but retention = 0.80 (failed K1236 threshold 0.90).
Legal and LaTeX adapters retained at 1.00. SOAP is the only failure case.

**Disease (not symptom):** The training distribution D_SOAP (clinical notes) semantically
overlaps with D_general (general knowledge). Clinical training updates v_proj value vectors
that encode general-world facts (anatomy, physiology, medications are also general knowledge),
overwriting the general-knowledge value representations.

The question is NOT "how to prevent retention loss?" but rather: **"what training distribution
makes value vector overwrite geometrically impossible for general knowledge directions?"**

---

## Theorem 1: Data Mixing Prevents Value Vector Overwrite

**Setup:**
- Let V_SOAP ⊂ R^{d_model} = column space of general knowledge value vectors that SOAP
  training will overwrite (the "endangered" subspace)
- Let V_general = the same subspace, which general-knowledge training reinforces
- LoRA update: ΔW_v = B · A, where B ∈ R^{d_model × r} and A ∈ R^{r × d_k}

**Claim:** If training batch = [x_SOAP : x_general] at ratio α:(1-α), with α < 1, then
the optimal LoRA update cannot simultaneously maximize SOAP format improvement AND destroy
general knowledge retention.

**Proof:**

Let L_total = α · L_SOAP + (1-α) · L_general be the combined loss.

The gradient of B is:
  ∇_B L_total = α · ∇_B L_SOAP + (1-α) · ∇_B L_general

For general knowledge to be destroyed, we need:
  ∇_B L_SOAP to point in a direction that zeroes out V_general

But we also have the constraint that:
  ∇_B L_general points in a direction that REINFORCES V_general

At α=0.5, the net gradient is the average of these two conflicting signals.

Formally: the optimal B* that minimizes L_total must satisfy:
  P_V · (α · ∇_B L_SOAP + (1-α) · ∇_B L_general) = 0
  where P_V = projection onto the direction that destroys V_general

At α=0.5, if ∇_B L_SOAP and ∇_B L_general have opposite projections onto the
"destruction direction," the net gradient in that direction → 0, preventing overwrite.

**Caveat:** This is a gradient-level argument, not a convergence guarantee. The LoRA
rank-16 subspace may not be rich enough to satisfy both objectives simultaneously,
potentially degrading SOAP format improvement. We measure this trade-off empirically.

**QED (conditional on rank sufficient for both tasks)**

---

## Theorem 2: Domain Specificity Requires Data Orthogonality

**Background:** Why did Legal (retention=1.00) succeed while SOAP failed?

Legal training data (contracts, NDA boilerplate) is semantically ORTHOGONAL to the
general knowledge retention questions (geography, biology, physics, literature).
Legal format vocabulary ("WHEREAS", "hereinafter") does not overlap with the
feature space of general knowledge questions.

SOAP clinical data is NOT orthogonal: anatomy, physiology, and medication knowledge
appear in BOTH SOAP training examples AND general knowledge questions.

**Prediction (verifiable):** If we compute cos(δ_general, δ_SOAP) for the v_proj update
direction, it should be significantly higher than cos(δ_general, δ_legal).

The data-mixing solution forces the optimizer to find a B that satisfies:
  cos(B_SOAP, B_legal) ≈ cos(B_SOAP_P4C1, B_legal_P4C1)   [no change in Legal]
  cos(B_SOAP, B_general) → 0                              [new constraint]

This is the Grassmannian isolation condition applied to the v_proj parameter space.

---

## Quantitative Predictions

| Metric | P4.C1 Baseline | P4.C2 Prediction | Kill Threshold |
|--------|---------------|-------------------|----------------|
| SOAP format improvement | +70pp | +50pp to +70pp | K1237: ≥50pp |
| SOAP retention ratio | 0.80 | 0.90 to 0.95 | K1238: ≥0.90 |
| Legal retention (sanity) | 1.00 | ~1.00 (unaffected) | K1239: ≥0.95 |
| Training data | 100 SOAP | 50 SOAP + 50 general | — |

**Expected trade-off:** Up to 20pp loss in SOAP format improvement is acceptable if
retention improves to ≥0.90. The model must allocate LoRA capacity to both objectives.

---

## Failure Mode (Kill Condition)

If K1237 FAILS: SOAP format improvement < 50pp despite mixed training →
**Impossibility structure:** Rank-16 LoRA has insufficient capacity to simultaneously:
1. Override RLHF behavioral format prior (requires large update in v_proj)
2. Preserve general knowledge representations (requires small/orthogonal update)

In this case, the minimum viable solution is either:
a) Higher rank (rank-32 or rank-64) — more capacity
b) Domain-specific adapter (SOAP trained on purely clinical-vocabulary text,
   explicitly avoiding anatomy/physiology overlap with general knowledge)
c) Dual-path adapter: q_proj (no format effect, good retention) as regularizer
   + v_proj+o_proj at lower scale for format

---

## References

- Finding #480: P4.C1 — SOAP retention=0.80, Legal=1.00, LaTeX=1.00
- Finding #440: T3.4 Grassmannian isolation at N=100 (parameter-space orthogonality)
- Geva et al. (2012.14913): attention value vectors as key-value memories
- Kirkpatrick et al. (1612.00796): EWC — weight regularization for retention (data mixing
  is simpler: replays general-knowledge examples instead of Fisher-weighted regularization)
