# MATH: C1.3 — PoLAR Scale Invariance

## Motivation

C1.2 showed standard LoRA on Gemma 4 has 13.3pp accuracy variance across scale={5,10,20}.
C1.1 showed PoLAR achieves sr(ΔW) = r exactly via joint Stiefel constraint — but KC08
(behavioral comparison) was vacuous (0% vs 0%, format mismatch). C1.3 fixes the evaluation
and tests whether joint Stiefel provides genuine scale invariance.

---

## Theorem 1: PoLAR Adapters Are Scale-Predictable

**Given:**
- LoRA adapter ΔW = s · BA, where s is the scale parameter
- B ∈ ℝ^{d_out × r}, A ∈ ℝ^{r × d_in}
- PoLAR training constraint: **both** B rows and A columns are unit-norm
  (equivalently: polar retraction is applied to both A and B)

**Standard LoRA scaling behavior:**
Let μ_B = mean row norm of B (learned, not constrained). Then:
  effective_scale = s × μ_B

If training was done at s₀ with learned μ_B^train, inference at s₁ produces:
  effective_scale_infer = s₁ × μ_B^train

The output magnitude scales as (s₁ / s₀) × (s₀ × μ_B^train) = s₁ × μ_B^train.
Crucially, **μ_B^train is itself training-scale-dependent** — AdamW adapts gradient
scale to the loss landscape at s₀, baking s₀ into the learned weight magnitudes.

**Claim:** For standard LoRA, changing s → αs changes effective magnitude
by α but also shifts the implicit regularization regime, leading to non-linear
accuracy degradation at α > 2.

**PoLAR scaling behavior:**
Since polar retraction enforces ||B_i||₂ = 1 for all rows i during training:
  effective_scale_infer = s₁ × 1.0 = s₁

The weight magnitudes are decoupled from s₀. Changing s → αs changes output
by exactly α, **without** any implicit coupling to the training regime.

**Proof:**
Let B have orthonormal rows: B B^T = I_r (enforced by retraction every τ steps).
At inference with scale s₁:
  ΔW(s₁) = s₁ · BA
  ||ΔW(s₁)||_F² = s₁² · tr(A^T B^T B A) = s₁² · tr(A^T A)

Since A also has orthonormal columns (A^T A = I_r after joint Stiefel):
  ||ΔW(s₁)||_F² = s₁² · r

The Frobenius norm scales exactly linearly with s₁. There is no s₀ dependence.

**QED**

**Corollary (Scale Invariance):**
Accuracy variance across s₁ ∈ {αs₀ : α ∈ {0.5, 1, 2, 4}} satisfies:
  Var_PoLAR < Var_LoRA

provided that the accuracy function is locally Lipschitz in the effective scale
and PoLAR's effective scale range (3 to 24) stays within the Lipschitz region.

---

## Theorem 2: Matched Train/Eval Fixes KC08 Vacuousness

C1.1's KC08 trained on synthetic multi-domain data (5 math+code+medical+legal+finance
sentences) and evaluated on GSM8K (multi-step arithmetic word problems). The 0%/0%
result occurred because neither adapter learned multi-step reasoning.

**Fix:** Train both adapters (PoLAR and LoRA) on:
- GSM8K-style multi-step arithmetic (math problems with 2-4 steps)
- 100 training examples, 30 eval examples
- Evaluation: exact-match on final numeric answer

This gives both adapters a chance to learn something, so behavioral comparison is meaningful.

---

## Quantitative Predictions

### From Theorem 1 (Scale Invariance)

| Adapter | Training scale | Eval scales | Predicted variance |
|---------|---------------|-------------|-------------------|
| PoLAR r=6 | 6.0 | {3, 6, 12, 24} | < 5pp |
| LoRA r=6 | 6.0 | {3, 6, 12, 24} | > 10pp (from C1.2: 13.3pp at narrower range) |

### From Theorem 2 (Matched eval)

| Config | Predicted GSM8K accuracy |
|--------|--------------------------|
| PoLAR r=6 at training scale | 30-50% (500 training steps, limited data) |
| LoRA r=6 at training scale | 30-50% (same data and steps) |
| PoLAR ≥ 80% of LoRA | YES (Theorem 2 predicts no regression) |

### Kill Criteria Mapping

- **KC13 PASS:** PoLAR accuracy variance < 5pp across scale={3,6,12,24}
  KILL CONDITION: variance ≥ 5pp → Theorem 1 fails
  
- **KC14 PASS:** Var_PoLAR < Var_LoRA
  KILL CONDITION: PoLAR variance ≥ LoRA variance → no structural benefit
  
- **KC15 PASS:** PoLAR at scale=6 ≥ 80% of LoRA at scale=6
  KILL CONDITION: PoLAR < 80% of LoRA → Stiefel constraint hurts quality

---

## Failure Modes

1. **Retraction interval too coarse (τ too large):** Rows drift from unit norm
   between retractions, weakening Theorem 1's guarantee. Fix: retract every 10 steps.

2. **Training data too easy:** If all questions resolve to single-digit answers,
   both adapters may hit 100% at all scales (ceiling effect). Fix: use multi-step
   problems requiring 2-4 arithmetic operations.

3. **QK-norm dominates:** Gemma 4's QK-norm may normalize out scale effects
   regardless of adapter type. If Var_LoRA < 10pp, this confirms QK-norm
   provides scale protection (but PoLAR should still be ≤ LoRA).

---

## Connection to Architecture

Gemma 4 E4B uses:
- RMSNorm on Q and K before attention (QK-norm)
- This bounds attention logit scale regardless of Q/K magnitude
- LoRA adapters on q_proj/k_proj: normalized AFTER the LoRA delta

The QK-norm path means: adapter output → add to W·x → apply QK-norm.
Scale only affects the pre-norm activation, which is then normalized.
This partially explains C1.2's surprising KC10 result (0pp degradation at 2-3× scale).

PoLAR's advantage: even with QK-norm, adapters on v_proj/o_proj/mlp_proj are
NOT normalized. Our adapters target q_proj/k_proj (as per T2.1), so scale
effects propagate through QK-norm. Theorem 1 predicts PoLAR + QK-norm gives
the strongest scale invariance.
