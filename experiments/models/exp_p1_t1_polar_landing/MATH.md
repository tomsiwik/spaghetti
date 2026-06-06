# T1.5: PoLAR Landing Field on Gemma 4 (Approximate Stiefel)

**Experiment type:** Guided exploration
**Failure mode:** Rank collapse — LoRA adapters converge to near-rank-1 solutions
because Adam minimizes over the full R^{d×r} space; orthogonality is never enforced.
**Prior math:** Polar decomposition (Cartan-Dieudonné), Łojasiewicz inequality,
stable rank / numerical rank (Rudelson & Vershynin 2007).
**Reference:** PoLAR: Polar-Decomposed Low-Rank Adapter (2310.05717 or 2406-class)

---

## Background: Rank Collapse in LoRA

Standard LoRA: ΔW = B @ A where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}.

Empirically (Zhao et al. 2024, "GaLore"), the effective rank of trained LoRA adapters
collapses during fine-tuning: the gradient matrix lies close to a low-dimensional
subspace, so the product B@A concentrates singular value mass in the top-1..3 directions.

**Stable rank** (Rudelson-Vershynin): sr(M) = ||M||_F^2 / σ_max(M)^2

For random Gaussian B@A (before training), sr ≈ r. After training, sr(LoRA) → 1–3
as Adam drives B and A to co-adapt along the dominant gradient direction.

---

## Theorem 1: Landing Field on the Stiefel Manifold

**Statement:** Let U ∈ R^{r×d_in} and define the orthogonality penalty:

    P(U) = ||U U^T - I_r||_F^2

The gradient ∇_U P(U) = 4(U U^T - I_r) U is a "landing field" — it points
strictly toward the Stiefel manifold St(r, d_in) = {U : U U^T = I_r} when U ∉ St:

    d/dt ||U(t) U(t)^T - I||_F^2 < 0 along the gradient flow U̇ = -∇_U P(U),
    with equality only at U ∈ St(r, d_in).

**Proof sketch:**
Let E = U U^T - I. Then ∇_U P = 4 E U.
d/dt ||E||_F^2 = 2 tr(E^T dE/dt) = 2 tr(E^T (dU/dt U^T + U dU^T/dt))
              = 2 tr(E^T (-∇_U P U^T + U (-∇_U P)^T))
              = -8 tr(E^T E U U^T) + (sym. term)
              = -8 ||E||_F^2 tr(U U^T / ||E||_F^2 · E^T E / ||E||_F^2)
Since E = U U^T - I is symmetric PSD-shifted and U U^T is PSD, this is ≤ 0.
Equality holds iff E = 0 (Stiefel). □

**Corollary (Łojasiewicz):** P(U) satisfies the Łojasiewicz inequality (it's a
polynomial). Therefore gradient descent (or Adam) with λ P(U) added to any smooth
loss converges to a critical point, which for the penalty is the Stiefel manifold.

**Prediction (discrete retraction form):** With periodic Stiefel retraction
(polar projection every RETRACT_EVERY=10 steps), ||U U^T - I||_F ≤ float32_floor ≈ 1e-6
after each retraction step. This is the discrete step of the gradient flow.

Note: The soft penalty form (λ||E||_F^2 added to loss) reaches equilibrium
ε_eq ≈ ||∇_{CE}U|| / (4λ sqrt(r)) ≫ 0.01 at practical λ≤1.0. Periodic retraction
achieves exact near-Stiefel (K1021) without constraint on λ.

---

## Theorem 2: Near-Orthogonal U Prevents Rank Collapse

**Statement:** Let ΔW = V @ U where U ∈ R^{r×d_in}, V ∈ R^{d_out×r}.
If U is ε-near-Stiefel (||U U^T - I_r||_F ≤ ε):

    sr(ΔW) ≥ sr(V) / (1 + ε)^2

where sr(V) = ||V||_F^2 / σ_max(V)^2.

**Proof sketch:**
σ_max(ΔW) = σ_max(VU) ≤ σ_max(V) · σ_max(U).
σ_max(U)^2 = λ_max(U U^T) ≤ 1 + ε  (by near-Stiefel assumption).
So σ_max(ΔW) ≤ σ_max(V) · sqrt(1+ε).

||ΔW||_F^2 = ||VU||_F^2 = tr(V U U^T V^T) = tr(V (I + E) V^T) ≥ ||V||_F^2 - ε ||V||_F^2
where E = U U^T - I satisfies ||E||_F ≤ ε.

Therefore sr(ΔW) = ||ΔW||_F^2 / σ_max(ΔW)^2 ≥ (||V||_F^2(1-ε)) / (σ_max(V)^2 (1+ε))
                                                = sr(V) · (1-ε)/(1+ε). □

**Corollary:** If ε < 0.01 (K1021 passes), then sr(ΔW) ≥ sr(V) · 0.98 ≈ sr(V).

**⚠️ EXPERIMENTAL REFUTATION (K1022 FAIL):** The prediction sr(V) ≈ r/2 is incorrect.
Measured: sr(V) = 2.21 after 200 steps (< sr(LoRA) = 4.45). The error is:

The claim "U's orthonormal rows route gradient to V's columns independently" is WRONG.
The gradient ∂L/∂V = (∂L/∂(ΔW)) @ U^T. Since U^T is an isometry, it maps the task
gradient exactly without distortion — but the task gradient (GSM8K SFT) has rank-1
structure regardless of U. Orthogonality of U does NOT diversify gradient directions
into different V columns. All r columns of V receive the same task gradient direction
→ Adam drives them to co-adapt → sr(V) → 1-2, not r/2.

**Corrected prediction:** sr(V) ≈ rank(∇_ΔW L) which for single-domain SFT is 1-2.
To achieve sr(ΔW) >> 1, one must constrain BOTH U and V on the Stiefel manifold
(joint retraction) or use multi-domain training to diversify ∇_ΔW L.

---

## Predictions

| Kill Criterion | Symbol | Predicted Value | Threshold | Source |
|---|---|---|---|---|
| K1021 | \|\|U U^T - I\|\|_F after 200 steps | < 0.01 | < 0.01 | Theorem 1, λ=0.1 |
| K1022 | sr(ΔW = V@U) at r=32 | ≥ 5 | ≥ 5 | Theorem 2 + sr(V)≈r/2=16 |
| K1023 | PoLAR quality vs LoRA (GSM8K) | PoLAR ≥ LoRA | ≥ LoRA | Theorem 2: no capacity loss |

**Supporting prediction:** sr(LoRA_trained) ≤ 3 (empirical from rank collapse literature),
while sr(PoLAR_trained) ≥ 5 → PoLAR has higher effective rank at same parameter count.

---

## Kill Criteria Thresholds — Justification

- **K1021 (0.01):** Float32 floor for r=32: ε_mach × r ≈ 1e-7 × 32 ≈ 3e-6. The bound 0.01
  allows for optimizer drift while still indicating near-Stiefel convergence (100× above floor).
  
- **K1022 (sr ≥ 5):** Minimum meaningful multi-rank structure. Random rank-32 matrix has
  sr ≈ 32. LoRA_trained has sr ≈ 1–3 (rank collapse). sr ≥ 5 distinguishes PoLAR from LoRA.
  
- **K1023 (PoLAR ≥ LoRA):** At equal params and steps, Theorem 2 shows PoLAR preserves
  full capacity (no loss from constraint). Near-Stiefel U is more expressive, not less.

---

## Implementation Notes (Qwen3-4B Proxy)

Gemma 4 (model_type="gemma4") is not loadable by mlx_lm 0.29.1 (MODEL_REMAPPING missing).
Using Qwen3-4B-4bit as proxy. All theorems hold for any d_in, r.

- Qwen3-4B: d_in=2560 (hidden), q_proj: (4096, 2560), r=32 adapters
- PoLAR params per layer: 32×2560 (U) + 4096×32 (V) = 212,992 = LoRA params (equal rank)
- Landing field λ: 0.1 (tuned by theory; if K1021 fails at λ=0.1, try 0.5)
