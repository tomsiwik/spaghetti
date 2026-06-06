# MATH.md — Cayley-Retracted Riemannian Adam for HRA on Gemma 4 E4B

**Experiment:** `exp_followup_cayley_riemannian_adam`
**Type:** Frontier extension (proven: Stiefel retraction theory; gap: empirical LLM fit with HRA)
**Parent killed experiment:** `exp_p1_t1_householder_vs_lora` (Finding #416, K1013 FAIL)
**Platform:** `mlx-community/gemma-4-e4b-it-4bit` via `mlx_lm 0.31.2` on M5 Pro 48GB

---

## Setup

**Base:** Gemma 4 E4B 4-bit. `hidden_size = d_in = 2560`, `n_layers = 42`.
**Adapter target:** `v_proj` (weight shape `(out=512, in=2560)` per F#627, F#668, F#673 — proven Gemma 4 target).
**Adapters under test:**
- `HRA_riem`: V ∈ R^{r × d_in}, rows start orthonormal via partitioned-QR Grassmannian init (T1.1 Finding #415), evolved with **Riemannian Adam via Cayley retraction** (Wen & Yin 2013, arxiv:1208.4298; Bécigneul & Ganea 2019, arxiv:1810.00760).
- `LoRA`: `A ∈ R^{r × d_in}`, `B ∈ R^{d_out × r}`, standard Euclidean AdamW (baseline).
- `HRA_euc`: same V parameterization optimized with Euclidean AdamW (parent regime — reference point for convergence delta).

**Rank:** r = 16 (match parent). Scope is convergence/optimization, not params budget.
**References:**
- HRA: arxiv:2405.17484 (Yuan et al. 2024)
- Stiefel Cayley retraction: arxiv:1208.4298 §2.4 (Wen & Yin)
- Riemannian Adam: arxiv:1810.00760 (Bécigneul & Ganea)

---

## Theorem 1 — Cayley retraction keeps V on Stiefel manifold exactly

**Claim.** Let V ∈ St(r, d_in) = {V : VVᵀ = I_r}, Ξ ∈ T_V St (tangent, i.e. ΞVᵀ + VΞᵀ = 0). Define skew `A = Ξᵀ V − Vᵀ Ξ ∈ R^{d_in × d_in}` and, for step size τ, the Cayley transform
```
V⁺ = Vᵀ · Q(τ) where Q(τ) = (I − (τ/2)A)⁻¹(I + (τ/2)A), transposed back.
```
Equivalently, using the Wen–Yin low-rank form with U = [Ξᵀ, V ᵀ] ∈ R^{d_in × 2r} and `Y = [V ᵀ, −Ξᵀ]`:
```
V⁺ᵀ = Vᵀ − τ · U · (I_{2r} + (τ/2) Yᵀ U)⁻¹ · Yᵀ Vᵀ.
```
Then **V⁺V⁺ᵀ = I_r exactly, independent of τ**.

**Proof (sketch).** `Q(τ)` is the Cayley transform of a skew matrix, hence orthogonal (standard result, Horn & Johnson §2.5). Therefore `V⁺V⁺ᵀ = V Q(τ)ᵀ Q(τ) Vᵀ = VVᵀ = I_r`. The Wen–Yin rewrite is algebraically equivalent by the Sherman–Morrison–Woodbury identity and requires only a `2r × 2r = 32 × 32` inverse per step per layer. **QED.**

**Failure-mode this theorem prevents:** with Euclidean Adam (parent regime), `V_{t+1} = V_t − η G` drifts off Stiefel — rows lose unit-norm and mutual orthogonality — so `H⁽r⁾ = ∏ᵢ(I − 2vᵢvᵢᵀ / ‖vᵢ‖²)` becomes an ill-conditioned product of near-reflections, not a proper orthogonal transform. Finding #416 documented the symptom: HRA loss plateau, never reaching <0.5.

---

## Theorem 2 — Riemannian gradient of the causal-LM loss equals projected Euclidean gradient

**Claim.** Let L : St(r, d_in) → R and G = ∂L/∂V the unconstrained Euclidean gradient. The Riemannian gradient w.r.t. the canonical metric on St(r, d_in) is
```
grad L(V) = G − (1/2) V (Vᵀ G + Gᵀ V).
```
It lies in T_V St (verify: `(grad L) Vᵀ + V (grad L)ᵀ = (GVᵀ − (1/2)V(VᵀG + GᵀV)Vᵀ) + (VGᵀ − (1/2)V(VᵀG+GᵀV)Vᵀ)ᵀ`; the second and fourth terms cancel because VVᵀ = I_r, collapsing the expression to the skew-symmetric part of GVᵀ, which satisfies the tangent constraint).

**QED** (standard Stiefel projection, Edelman–Arias–Smith 1998 §2.2).

---

## Theorem 3 — Riemannian Adam with Cayley retraction converges on L(V) at rate matching (or beating) Euclidean Adam on LoRA for same rank

**Claim (informal, Guided Exploration).** For the Stiefel-parameterized HRA layer, let m_t, v_t be first- and second-moment accumulators updated in the tangent space, and
```
V_{t+1} = R_{V_t}(−η · m̂_t / (√v̂_t + ε))    # Cayley retraction
m_{t+1} = Π_{T_{V_{t+1}}}(m_t)                    # tangent transport via re-projection
```
where `Π_V` is the canonical Stiefel projection from Theorem 2. Bécigneul & Ganea (2019) Theorem 4.1 proves **O(1/√T) convergence** on geodesically-convex objectives; LLM loss isn't geodesically-convex, but their experiments (§5, Poincaré-GloVe and Hyperbolic NN) confirm the same ordering holds empirically.

**Predicted ordering for our setup:**
```
steps_HRA_riem (loss<0.5)  ≤  steps_LoRA (loss<0.5)  <  steps_HRA_euc (loss<0.5)
                                                           └─ = 300+ (never) per parent F#416
```

The `≤` (not `<`) is because HRA has **more effective directions per parameter** (sr = r vs sr ≈ 1 for LoRA at random init, Theorem 1 of T1.1 Finding #415). At equal rank, HRA's adapter-update subspace is r-dimensional from step 1; LoRA's is ~1-dimensional and only expands as training progresses. With a correct Riemannian optimizer that respects the Stiefel constraint, HRA's larger effective rank should produce ≤ LoRA convergence steps.

**QED (guided exploration: prediction tied to Wen–Yin + Bécigneul–Ganea guarantees; empirical transfer to Gemma 4 E4B is the unknown being tested).**

---

## Quantitative Predictions & Kill Criteria

| # | Metric | Prediction | KC Type | Threshold |
|---|--------|-----------|---------|-----------|
| K1559 | `steps_to_loss<0.5 (HRA_riem) / steps_to_loss<0.5 (LoRA)` | ≤ 1.0 | **proxy** (convergence) | ratio ≤ 1.0 ⇒ PASS |
| K_target | `MMLU_acc(HRA_riem) − MMLU_acc(LoRA)` | ≥ −3 pp | **target** (downstream task) | delta ≥ −3 pp ⇒ PASS |
| K_stiefel | `max_layer ‖V_final V_finalᵀ − I_r‖_F` | ≤ 1e-4 | **structural** (retraction fidelity) | Frobenius ≤ 1e-4 ⇒ PASS |
| K_euc_control | `steps_to_loss<0.5 (HRA_euc)` | > 1.5 × LoRA-steps | **control** (parent reproduction) | HRA_euc must NOT converge within 1.5× LoRA; this confirms parent's F#416 failure persists on Gemma 4, isolating the Riemannian-Adam fix as the causal variable |

**Target-gated kill rule (Finding #666):** `K1559` is a proxy (training-step-based). It is paired with `K_target` (MMLU accuracy). Verdict table:

| K1559 (proxy) | K_target (MMLU) | Verdict |
|--------------|-----------------|---------|
| PASS | PASS | **SUPPORTED** — Riemannian Adam fixes parent's K1013 failure without degrading task quality |
| PASS | FAIL | partial: Riemannian optimization converges but representations drift from base — file proxy-vs-target follow-up |
| FAIL | PASS | proxy mis-calibrated: training loss isn't the right convergence proxy; re-examine metric |
| FAIL | FAIL | **KILLED** — Cayley/Riemannian hypothesis does not rescue HRA at r=16 on Gemma 4 |

`K_stiefel` and `K_euc_control` are structural / control checks; they gate experimental validity, not the headline claim.

---

## Implementation plan (micro scale)

- **Training:** 150 SFT steps on GSM8K train split, `batch=1`, `seq_len=256`, `lr=5e-5`, `grad_clip=1.0`.
- **Adapter injection:** replace `v_proj` in all 42 layers of Gemma 4 E4B. Freeze base. Trainable params only on adapter.
- **HRA init:** partitioned-QR orthonormal rows (Grassmannian A init per F#415/#562) so V starts on Stiefel.
- **Riemannian Adam core loop per step:**
  1. `G = ∂L/∂V` (Euclidean gradient from autodiff).
  2. `Ξ = G − (1/2) V (Vᵀ G + Gᵀ V)` (Theorem 2 projection).
  3. `m ← β₁ m + (1−β₁) Ξ`, `v ← β₂ v + (1−β₂) Ξ²`.
  4. `d = m̂ / (√v̂ + ε)`.
  5. Re-project `d` onto T_V St (tangent transport via projection, Bécigneul–Ganea §3.2 "simple-transport").
  6. Cayley retraction via Wen–Yin low-rank form (`2r × 2r` solve).
  7. After step: `m ← Π_{T_{V_new}}(m)` (transport momentum to new tangent).
- **LoRA baseline:** `mlx_lm.tuner.lora.LoRALinear` on same `v_proj`, AdamW, same LR and steps.
- **Euclidean-Adam HRA control:** HRA parameterization + vanilla AdamW on `V` (no projection, no Cayley) — replicates parent regime on Gemma 4 so we can factor out model-change effects.
- **Evaluation:** MMLU zero-shot n = 30 (micro), GSM8K pass@1 n = 30 (micro). Both adapters and base evaluated from same seeded data split.

---

## Assumptions (researcher-hat pick-and-proceed)

1. **Simple tangent transport** (re-projection) rather than full parallel transport. Bécigneul–Ganea §3.2 show this suffices for Adam-class methods on Stiefel in practice. If K1559 fails, parallel transport is the first thing to try before declaring the hypothesis killed.
2. **Rows-orthonormal convention** VVᵀ = I_r with `V ∈ R^{r × d_in}`; equivalent to Vᵀ ∈ St(d_in, r). Partitioned-QR init gives this directly.
3. **Element-wise second moment** (like standard Adam) rather than scalar (geometric) second moment. Works in practice, minor constant-factor variance.
4. **Micro-scale n=30 MMLU / n=30 GSM8K** — ±9pp noise at 95% CI. `K_target`'s −3pp threshold is intentionally loose for the micro scale; a macro rerun would tighten this.

---

## Antipattern pre-flight (PLAN.md §1, auto-injected fixes)

- [x] **composition math:** N = 1 here (single adapter per run). Not applicable.
- [x] **LORA_SCALE:** baseline uses default `scale=float(r)=16` (standard LoRA). HRA has no analogous scale (multiplicative parameterization). Not using unsafe 20.
- [x] **tautological routing:** no routing; single-adapter run.
- [x] **shutil.copy adapter:** adapter weights saved via `mx.savez`, not copied.
- [x] **hardcoded pass=True:** all KCs derived from measurements.
- [x] **eval truncation / thinking mode:** GSM8K and MMLU use plain prompts; Gemma 4 thinking not invoked (training data has no `<thought>` traces).
- [x] **proxy model:** target is Gemma 4 E4B, which now loads in `mlx_lm 0.31.2` (verified). Not proxying to Qwen3 this time — parent's proxy was because of a prior `mlx_lm` bug, resolved.
- [x] **code-bug convergence sentinel:** `conv_step = TRAIN_STEPS + 1` only when threshold never crossed; K1559 logic branches on `converged` flag explicitly (propagated fix from parent v2 rerun).
- [x] **smoke-as-full:** `SMOKE_TEST=1` runs 5 steps / n=3 eval and records `is_smoke=true`; those results complete only as `provisional`, never `supported`.

---

## Architectural implication

If **SUPPORTED** (K1559 PASS + K_target PASS): Cayley-retracted Riemannian Adam is the correct optimizer for Stiefel-parameterized adapters, unblocking the T1.6 bakeoff and making HRA / Givens / Cayley a viable adapter family for Pierre. Promote to LEARNINGS.md + finding, route unblock for `exp_p1_t1_algorithm_bakeoff`.

If **KILLED** (both FAIL): the Stiefel-parameterization hypothesis for LLM adapters is structurally wrong at this model scale — either the base-model geometry doesn't have enough "orthogonal slack" at 4-bit quantization, or the Cayley retraction introduces a systematic bias at r ≪ d_in. LoRA + Grassmannian init remains the P1 path (Finding #415 already supports this).

If **PARTIAL** (proxy mismatch): file a follow-up with finer-grained convergence metrics (gradient norm, parameter-change norm, task probe every 20 steps) to find the actual convergence signal.
