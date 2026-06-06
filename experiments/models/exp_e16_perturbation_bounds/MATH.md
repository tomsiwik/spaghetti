# E16: Tight Bounds for NRE Composition Error

## Type
Verification — perturbation theory framework (F#172) applied to derive measurable bounds; experiment confirms predictions.

## Prior work
- **F#172** (supported): N² interference bound predicts N_max ~ 50 for γ>0.95. Unverified at scale.
- **F#815** (E14): B₁ᵀB₂ coupling dominates interference; Grassmannian constrains A, not B.
- **F#817** (E15): Per-layer linear composition is exact; tau≈0.48 comes from cross-layer GELU/softmax propagation.
- **arxiv:2012.13255**: Intrinsic dimensionality — fine-tuned adapters occupy low-rank subspace.

## Failure mode prevented
Vacuous bounds that predict >100% error while actual NRE is <5%, giving no useful engineering guidance for choosing N_max.

## Mathematical framework

### Setup
Transformer with L layers, each: h^(l+1) = GELU(W^(l) h^(l) + b^(l)).
N LoRA adapters of rank r on v_proj: ΔW_i^(l) = B_i^(l) A_i^(l) / r.
Per-layer perturbation: δ_i^(l) = B_i^(l) A_i^(l) h^(l) / r.

### Theorem 1 (Per-layer composition error)
**Statement.** For adapters {1,...,N} at layer l, the per-layer composition error is:

ε^(l) = GELU(z + Σᵢ δᵢ) − Σᵢ[GELU(z + δᵢ) − GELU(z)] − GELU(z)

where z = W^(l) h^(l) + b^(l).

By Taylor expansion (element-wise, using GELU ∈ C²):

ε^(l) = GELU″(z) ⊙ Σ_{i<j} (δᵢ ⊙ δⱼ) + O(‖Σδᵢ‖³)

**Proof.** GELU(z + Σδ) = GELU(z) + GELU′(z)⊙Σδ + ½GELU″(z)⊙(Σδ)² + O(‖Σδ‖³).
Each individual: GELU(z + δᵢ) − GELU(z) = GELU′(z)⊙δᵢ + ½GELU″(z)⊙δᵢ² + O(‖δᵢ‖³).
Subtracting: ε = ½GELU″(z)⊙[(Σδᵢ)² − Σδᵢ²] + H.O.T. = GELU″(z) ⊙ Σ_{i<j} δᵢ⊙δⱼ + H.O.T. ∎

### Lemma 1 (GELU″ bound)
For GELU(z) = z·Φ(z): GELU″(z) = φ(z)(2 − z²) where φ = N(0,1) PDF.
Maximum: |GELU″(z)| ≤ 2φ(0) ≈ 0.798 at z=0.
For |z| > 2: |GELU″(z)| < 0.054 (rapid decay). ∎

### Theorem 2 (Output-level NRE bound)
**Statement.** The Normalized Reconstruction Error at the output layer is bounded by:

NRE(N) ≤ Σ_l (Π_{l'=l+1}^{L} κ_{l'}) · ‖GELU″(z^(l))‖_∞ · Σ_{i<j} |⟨δᵢ^(l), δⱼ^(l)⟩|^{1/2} · (1/‖h_base‖)

where κ_{l'} is the local Lipschitz constant of layer l'.

**Simplified bound (uniform Lipschitz):**

NRE(N) ≤ 0.798 · (N choose 2) · κ^{L} · max_l [max_{i<j} ‖δᵢ^(l) ⊙ δⱼ^(l)‖₁] / ‖h_base‖

**Proof sketch.** Per-layer error ε^(l) propagates through subsequent layers. Each layer amplifies by at most κ_{l'} (Lipschitz of GELU∘Linear). Sum over all layers that contribute cross-terms. ∎

### Theorem 3 (N² scaling)
**Statement.** For uniform adapter norms (‖δᵢ‖ ≈ δ for all i), NRE(N) scales as:

NRE(N) ∝ N(N−1)/2 · δ² ∝ N²

This follows directly from Theorem 2: the cross-term count is (N choose 2) = N(N−1)/2.

### Practical bound (computable from single forward pass)
Given base model hidden states {h^(l)} and adapter matrices {Aᵢ^(l), Bᵢ^(l)}:

**Per-layer predicted error:**
ε̂(l) = Σ_{i<j} Σ_d |GELU″(z_d^(l))| · |δᵢ,d^(l)| · |δⱼ,d^(l)|

**Accumulated NRE prediction:**
NRE_pred(N) = Σ_l ε̂(l) / ‖h_base_output‖

(Assuming κ ≈ 1 for pre-trained networks near equilibrium — Lipschitz constant near 1 is standard for converged transformers.)

## Predictions

| N | Predicted NRE scaling | Expected measured range |
|---|---|---|
| 2 | 1× (baseline) | 0.01–0.05 |
| 5 | 10× | 0.1–0.5 |
| 10 | 45× | 0.5–2.0 |
| 25 | 300× | 3.0–15.0 |

If κ > 1 for even a few layers, the accumulated error will be larger. The experiment tests whether κ ≈ 1 holds or whether certain layers amplify errors.

## Kill criteria (pre-registered)

**K2047 (non-vacuous bound):** The predicted NRE bound must be within 20× of measured NRE for N≤10. If predicted > 20× measured, the bound is vacuous and useless for engineering. Specifically: for N=2, if measured NRE < 0.05 but predicted > 1.0, FAIL.

**K2048 (scaling holds):** The N² scaling relationship NRE(N)/NRE(2) ≈ (N choose 2) must hold within 3× for N∈{5,10,25}. If the ratio deviates by >3× at N≥25, the theory doesn't scale.

**Target KC (behavioral, paired per F#666):**
**K2047_target:** For N≤10 composed adapters, GSM8K accuracy must remain >50% of base (composition doesn't destroy reasoning). This ensures the bound region is behaviorally relevant.

## Platform
- Model: `mlx-community/gemma-4-e4b-it-4bit`
- MLX on M5 Pro 48GB
- mlx-lm for model loading
- LoRA rank=6, scale=6, v_proj only (per F#627 target module)
- Grassmannian A matrices via QR partitioning
