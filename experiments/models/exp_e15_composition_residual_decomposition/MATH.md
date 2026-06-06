# MATH.md — exp_e15_composition_residual_decomposition

**Claim:** SVD decomposition of trained LoRA B matrices reveals a rank threshold separating compositional signal from cross-adapter interference, enabling filtered composition with tau < 0.3 (vs current tau ≈ 0.48, F#752).

---

## 0. Scope & skill invocation

**Platform skills invoked:** `/mlx-dev`, `/fast-mlx` — both invoked before code.
**Model:** `mlx-community/gemma-4-e4b-it-4bit` (PLAN.md Part 2 dev target).
**Adapter config:** LoRA r=6 on `v_proj + o_proj` per F#627. Grassmannian A-matrices. LORA_SCALE=6.
**Dataset:** GSM8K train split (domain differentiation via prompt templates — math, code, general).

---

## 1. Failure mode

Primary degenerate: "B matrices of independently trained adapters have overlapping column spaces (high U_i^T U_j), causing per-sample output coupling that Grassmannian A-orthogonality cannot prevent. SVD filtering removes capacity uniformly (all singular values carry signal proportionally), so no rank threshold exists — tau degrades monotonically with adapter quality."

This makes E15 testable: if the failure mode holds, K2045 kills.

## 2. Cited prior math / findings

- **F#752 (supported):** tau ≈ 0.48 at Gemma 4 E4B. Composition residual is 48% of sum of individual effects.
- **F#815 (provisional, E14):** Grassmannian decorrelates activations ~33% (mean|cos| 0.034 vs 0.051) but σ_max(B^T B) ≈ 45 makes theoretical bound vacuous. B-matrix coupling dominates.
- **E2 kill (F#803):** Null-space is rank deficiency, not Grassmannian property. Cross-layer nonlinear coupling is real.
- **Aghajanyan et al. arxiv:2012.13255 (Intrinsic Dimensionality):** Fine-tuned models occupy low-dimensional subspace. Effective intrinsic dimension d_eff ≪ d_model.
- **Task Singular Vectors (arxiv:2412.00081):** Task-specific information concentrates in top singular vectors of weight perturbations.

## 3. Mathematical framework

### Setup
For layer ℓ, adapter i has LoRA decomposition:
  ΔW_i = s · B_i A_i, where A_i ∈ ℝ^{r × d_in}, B_i ∈ ℝ^{d_out × r}, s = lora_scale

Grassmannian construction ensures A_i^T A_j = 0 for i ≠ j (verified in E14 to machine precision).

### SVD of B matrices
SVD(B_i) = U_i Σ_i V_i^T, where U_i ∈ ℝ^{d_out × r}, Σ_i = diag(σ_1^(i), ..., σ_r^(i)), V_i ∈ ℝ^{r × r}.

The cross-adapter output coupling for input x:
  ⟨δh_i, δh_j⟩ = s² · (A_i x)^T B_i^T B_j (A_j x)
                 = s² · (A_i x)^T V_i Σ_i U_i^T U_j Σ_j V_j^T (A_j x)

Since A_i ⊥ A_j (Grassmannian), E_x[⟨δh_i, δh_j⟩] = 0 over isotropic inputs.
But per-sample coupling depends on U_i^T U_j (the output-space alignment of B matrices).

### Rank-k filtering
Define B_i^{(k)} = U_i^{(k)} Σ_i^{(k)} V_i^{(k)T} using only the top-k singular vectors.
Then ΔW_i^{(k)} = s · B_i^{(k)} A_i.

**Lemma (coupling reduction).** If we truncate to rank k < r:
  ||B_i^{(k)T} B_j^{(k)}||_F ≤ σ_k^(i) σ_k^(j) ||U_i^{(k)T} U_j^{(k)}||_F

This bounds the coupling by the k-th singular values. If task signal concentrates in top-k and cross-adapter coupling concentrates in bottom-(r-k), filtering improves composition.

### Composition residual
Given N adapters composed (merged weights ΔW = Σ ΔW_i):
  h_composed = W x + (Σ B_i A_i) x
  h_individual_sum = Σ (W x + B_i A_i x) - (N-1) W x = W x + (Σ B_i A_i) x

Wait — for linear composition, h_composed = h_individual_sum exactly! The residual tau ≈ 0.48 from F#752 must come from nonlinear effects (activation functions between layers, not within-layer linear algebra).

**Revised theorem:** The composition residual is NOT a per-layer phenomenon. It emerges from cross-layer nonlinear propagation: adapter i's output at layer ℓ changes the INPUT to layer ℓ+1, which then passes through nonlinearities (GELU in FFN, softmax in attention) before reaching layer ℓ+1's adapter j. The per-layer linear decomposition is exact; the multi-layer nonlinear propagation is not.

**This changes what SVD filtering can do:** We cannot filter tau to zero because the residual is inherently nonlinear. But we CAN reduce the magnitude of per-layer adapter outputs (by rank-reducing B), which reduces the nonlinear coupling amplitude.

### Predictions

1. **Per-layer SVD spectrum:** Top 2-3 SVs of B_i carry 70%+ of Frobenius norm (consistent with intrinsic dimensionality result).
2. **Cross-adapter U alignment:** U_i^T U_j has structure — top SVs align more than bottom SVs (because top SVs point toward the output space's "important" directions).
3. **Filtered tau:** Keeping only top-k=4 of r=6 SVs reduces tau from 0.48 to 0.25-0.35 (50-70% of full-rank) because we remove the noisiest SVs while preserving 80%+ of adapter signal.
4. **Quality retention:** Rank-4 filtered adapters retain >85% of full-rank quality (GSM8K accuracy within 3pp of full-rank).

## 4. Kill-criterion map (pre-registered)

| KC | Measured quantity | Threshold | Type |
|---|---|---|---|
| K2045 | SVD spectrum structure: fraction of ||B||_F in top-k SVs | ≥ 60% in top-3 of r=6 | proxy (structural) |
| K2045_target | Exists rank k where tau(k)/tau(full) < 0.7 AND quality(k) > 0.8*quality(full) | At least one k satisfies | target (paired with K2045) |
| K2046 | Best achievable tau with filtered composition | tau < 0.3 at any k | target (behavioral) |
| K2046_quality | GSM8K accuracy at the tau-minimizing k | Within 5pp of full-rank baseline | target (paired with K2046) |

## 5. Predicted measurements

- K2045: top-3 SVs carry 75% of ||B||_F → PASS
- K2045_target: k=4 gives tau ratio 0.65 with quality retention 0.88 → PASS
- K2046: tau(k=4) ≈ 0.31 → PASS (marginal)
- K2046_quality: GSM8K accuracy within 2pp → PASS
