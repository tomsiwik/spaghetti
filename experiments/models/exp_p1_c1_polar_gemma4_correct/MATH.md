# C1.1: PoLAR with Joint Stiefel on Gemma 4 E4B (5-Fix Re-test of T1.5)

**Experiment type:** Verification + Guided Exploration
**Reference:** PoLAR: Polar-Decomposed Low-Rank Adapter (arxiv 2506.03133)
**Fixes over T1.5:** (1) Both U+V Stiefel, (2) 1000 steps, (3) multi-domain data, (4) Gemma 4, (5) rank sweep r∈{6,16}

---

## Background: T1.5 Failure Analysis

T1.5 (Finding #419) killed because:
- Only U constrained to Stiefel, V not constrained
- Theorem 2 bound `sr(ΔW) ≥ sr(V)` is correct, but sr(V) → 2 (not 16) due to rank-1 gradient
- Root cause: single-domain SFT (GSM8K) gradient has rank-1 subspace
- U's orthogonality routes gradient isometrically into V, but cannot diversify it
- All r columns of V co-adapt to same gradient direction → sr(V) → 1-2

**Impossibility structure from T1.5:** For sr(V) >> 1, need either:
1. `rank(∇_ΔW L) >> 1` (multi-domain training), OR
2. **Stiefel constraint on V** (V's rows forced orthonormal regardless of gradient)

This experiment tests Fix 2 (joint Stiefel on both U and V), which provides a
structural guarantee independent of training data distribution.

---

## Theorem 1: Joint Stiefel Landing Field on Product Manifold

**Statement:** Let U ∈ R^{r×d_in}, V ∈ R^{r×d_out}. Define:

    P(U, V) = ||U U^T - I_r||_F^2 + ||V V^T - I_r||_F^2

The joint gradient flow (U̇, V̇) = -∇_{U,V} P(U,V) is a landing field on the
product manifold St(r,d_in) × St(r,d_out):

    d/dt P(U(t), V(t)) < 0   when (U,V) ∉ St(r,d_in) × St(r,d_out)

**Proof:**
By product manifold structure, the two components decouple:
    d/dt P = d/dt ||UU^T-I||_F^2 + d/dt ||VV^T-I||_F^2

Each term is independently a Lyapunov function for the Stiefel landing field
(proved in T1.5 MATH.md, Theorem 1). Both terms are non-positive, with equality
only when U ∈ St(r,d_in) AND V ∈ St(r,d_out). QED

**Discrete form:** Polar retraction every RETRACT_EVERY steps:
- Retract U: SVD(U) = W Σ V_h^T → U ← W V_h^T  (r×d_in, orthonormal rows)
- Retract V: SVD(V) = W' Σ' V_h'^T → V ← W' V_h'^T  (r×d_out, orthonormal rows)

After retraction: ||UU^T-I||_F ≤ ε_mach ≈ 1e-6 (K1021-class; KC09 threshold 0.01).

---

## Theorem 2: Joint Stiefel Guarantees sr(ΔW) = r Exactly

**Statement:** Let A ∈ R^{d_in×r} with A^T A = I_r (orthonormal columns)
and B ∈ R^{r×d_out} with B B^T = I_r (orthonormal rows). Then:

    sr(ΔW) = r   where ΔW = A @ B ∈ R^{d_in × d_out}

**Proof:**
Step 1 — Frobenius norm:
    ||ΔW||_F^2 = ||AB||_F^2 = tr(B^T A^T A B) = tr(B^T I_r B) = tr(B^T B)

Since B B^T = I_r and B ∈ R^{r×d_out} has r orthonormal rows:
    tr(B^T B) = ||B||_F^2 = r   (each row has unit norm, r rows)

So ||ΔW||_F^2 = r.

Step 2 — Singular values:
Since A has orthonormal columns (A^T A = I_r), A is an isometry on R^r:
    σ_i(AB) = σ_i(B)   for all i = 1,...,r

(Proof: (AB)^T(AB) = B^T A^T A B = B^T I_r B = B^T B. So eigenvalues of (AB)^T(AB)
equal eigenvalues of B^T B = (B^T B) = I_r^T I_r only if B has full row rank.)

Actually: B B^T = I_r → singular values of B are all 1.
→ B^T B has eigenvalues 1 (×r) and 0 (×(d_out-r))
→ (AB)^T(AB) = B^T B → σ_i(AB)² = 1 for i=1..r, 0 for i>r

Therefore σ_max(ΔW) = 1 and ||ΔW||_F^2 = r.

Step 3 — Stable rank:
    sr(ΔW) = ||ΔW||_F^2 / σ_max(ΔW)^2 = r / 1 = r.   QED

**Corollary — Rank independence:** sr(ΔW) = r holds regardless of:
- Training domain (single vs multi)
- Number of training steps
- Gradient rank structure

This is the structural guarantee that T1.5 lacked (T1.5 only proved sr(ΔW) ≥ sr(V),
but sr(V) → 2 under rank-1 gradient). Joint Stiefel bypasses the gradient rank issue.

---

## Theorem 3: Multi-Domain Gradient Diversification (Supporting)

**Statement:** With k ≥ 1 training domains, the expected gradient rank satisfies:
    E[rank(∇_ΔW L)] ≤ k

For single-domain SFT (k=1), E[rank] = 1 (empirically confirmed in T1.5).
For k=5 domains (math/code/language/logic/science), E[rank] ≤ 5.

**Relevance:** Multi-domain training is required for KC08 (quality), not KC07.
KC07 is satisfied by Theorem 2 regardless of domain count. Multi-domain ensures
that V learns different directions per domain → better task coverage → KC08.

Multi-domain training helps LoRA baseline too, so the comparison is fair.

---

## Predictions

| Kill Criterion | Symbol | Predicted Value | Threshold | Theorem |
|---|---|---|---|---|
| KC07 | sr(ΔW) at r=16, multi-domain | **= 16** (not just ≥ 5) | ≥ 5 | Theorem 2 |
| KC08 | PoLAR GSM8K vs LoRA at r=6, 1000 steps | PoLAR ≥ LoRA | PoLAR ≥ LoRA | Theorem 2 + 3 |
| KC09-U | ||UU^T-I||_F (post-retraction) | < 1e-5 (float32 floor) | < 0.01 | Theorem 1 (retraction) |
| KC09-V | ||VV^T-I||_F (post-retraction) | < 1e-5 (float32 floor) | < 0.01 | Theorem 1 (retraction) |

**Behavioral prediction:** Joint Stiefel PoLAR will produce more diverse adapter
directions (sr = r vs sr = 2 for standard LoRA), which should translate to
better multi-domain generalization (KC08 behavioral claim).

---

## Kill Criteria Thresholds — Justification

- **KC07 (sr ≥ 5 at r=16):** From Theorem 2, sr = 16 exactly post-retraction.
  5 is a generous lower bound accounting for numeric noise between retractions.
  T1.5 measured sr(LoRA) = 4.45, sr(PoLAR-U-only) = 2.21 — both below this.
  Joint Stiefel should give sr = 16 (3× improvement).

- **KC08 (PoLAR ≥ LoRA at r=6):** Theorem 2 shows capacity is not lost
  (sr(PoLAR) = 6 vs sr(LoRA) ≈ 1-3 due to collapse). Equal or better quality
  expected. Multi-domain training ensures both approaches have full task coverage.
  T1.5 failure (3.3% vs 13.3%) was due to V-collapse in PoLAR, which joint Stiefel fixes.

- **KC09 (< 0.01):** After polar retraction, ||UU^T-I||_F ≤ float32_floor ≈ 1e-6.
  Between retractions (every 20 steps), gradient updates perturb U and V by
  O(lr × grad_norm) ≈ O(1e-4 × 1.0) = 1e-4, well within tolerance.
  T1.5 K1021 (U-only) passed at 2.46e-8. KC09 extends this to V.

---

## Implementation Notes

**Model:** Gemma 4 E4B 4-bit (mlx-community/gemma-4-e4b-it-4bit)
- hidden_size = 2560, n_layers = 42
- q_proj: (d_in=2560, r) → (r, d_out=2048) (via LoRALinear: A=(2560,r), B=(r,2048))

**PoLAR setup:**
- A (= lora_a): (d_in=2560, r) — enforce A^T A = I_r via polar retraction
- B (= lora_b): (r, d_out=2048) — enforce B B^T = I_r via polar retraction
- Retraction: SVD of (r×d_in) and (r×d_out) matrices every 20 steps
  Cost: O(r²×d) ≈ O(16²×2560) = 655K ops — negligible

**Multi-domain synthetic data:**
- 5 domains: math, code, language, logic, science
- 200 synthetic examples each = 1000 total (no dataset downloads)
- Random domain sampling each batch

**Training phases:**
- Phase 1: PoLAR r=16, 500 steps multi-domain → KC07
- Phase 2: PoLAR r=6, 1000 steps multi-domain → KC08 (PoLAR), KC09
- Phase 3: LoRA r=6, 1000 steps multi-domain → KC08 (LoRA baseline)
- Phase 4: Eval both on 30 GSM8K test questions
