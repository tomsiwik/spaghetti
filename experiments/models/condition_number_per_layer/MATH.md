# MATH.md — Condition Number κ(W) per Layer for Qwen3-0.6B

## Problem Statement

Before using the **epsilon-map** framework for ternary promotion safety, we need to
know the condition numbers κ(W) of each transformer layer's weight matrices. The
condition number determines how quantization errors compound across K promotion cycles.

**Epsilon-map context:** In the M2P system, "promotion" means reusing the ternary base
model's frozen weights across multiple adapter compositions. Each use incurs a small
quantization error ε. Across K cycles, the error compounds multiplicatively, bounded
by κ(W). If κ is small (< 20), K>5 cycles are safe. If κ > 200, promotion is
fundamentally unsafe and needs scale reduction.

---

## Theorem 1: Condition Number Bounds Quantization Error Amplification

**Setup.** Let W ∈ ℝ^(m×n) be a weight matrix with condition number κ(W) = σ_max/σ_min.
Let W_q = W + ΔW be the ternary-quantized version where ‖ΔW‖_F ≤ ε·‖W‖_F (relative
quantization error). Let y = Wx be the true output for input x with ‖x‖₂ = 1.

**Theorem (Stewart, 1973 — Matrix Perturbation Theory).**
The relative output error satisfies:

    ‖(W_q - W)x‖₂ / ‖Wx‖₂ ≤ κ(W) · (‖ΔW‖_F / ‖W‖_F) = κ(W) · ε

For K sequential applications (K promotion cycles) of independently quantized W_q:

    ‖Δy_total‖ / ‖y_true‖ ≤ K · κ(W) · ε          [first-order, independent errors]

**Key result:** The per-layer condition number κ(W) is the *amplification factor* for
quantization errors. Small κ → errors stay bounded over many cycles.

**Reference:** Stewart (1973), "Error and Perturbation Bounds for Subspaces Associated
with Certain Eigenvalue Problems". See also: Higham (2002), "Accuracy and Stability of
Numerical Algorithms", §7.4 (condition numbers and backward error).

For BitNet-style ternary quantization (Ma et al., arXiv 2402.17764):
- Typical relative quantization error ε ≈ 0.01–0.03 (1–3%) for 1.58-bit models
- With κ(W) = 10 and K=5: total error ≤ 5 × 10 × 0.02 = 1.0 (borderline)
- With κ(W) = 20 and K=5: total error ≤ 5 × 20 × 0.02 = 2.0 (unsafe)
- With κ(W) = 200 and K=1: total error ≤ 1 × 200 × 0.02 = 4.0 (fundamentally unsafe)

---

## Theorem 2: Random Matrix Baseline (Marchenko-Pastur)

For i.i.d. Gaussian W ~ N(0, 1/n) with aspect ratio r = m/n ≥ 1:

    κ_MP(W) = ((1 + √r) / (1 - √r))² for r < 1; undefined (κ → ∞) as r → 1

**Implication:** Random matrices with r = 2 (e.g., q_proj: 2048×1024) have theoretical
κ_MP ≈ ((1 + √2)/(1 - √2))² → ill-conditioned (negative denominator → κ → ∞).

However, trained transformer weights are **NOT random**. They develop low-rank structure
(Aghajanyan et al., arXiv 2012.13255) with relatively flat singular value spectra,
which typically REDUCES condition numbers relative to random matrices.

**Prediction for trained Qwen3-0.6B weights:**
- Singular values are more uniformly distributed than Marchenko-Pastur
- Expected κ range: 10–200 (much better than unbounded MP limit)
- Near-zero singular values (degenerate layers) are not expected in trained models

---

## Predictions

| Kill Criterion | Prediction | Threshold |
|----------------|-----------|-----------|
| K942: Finite κ for all 28 layers | PASS — trained models are full rank | All κ < 10^6 |
| K943: Mean κ > 200 (KILL) | PASS (NOT killed) — trained weights have bounded κ | mean κ < 200 |

**Calibration targets (from experiment notes):**
- κ < 20: promotion safe for K > 5 cycles
- 20 ≤ κ < 100: promotion safe for K ≤ 5 cycles (reduce scale for more)
- 100 ≤ κ < 200: promotion requires scale reduction
- κ ≥ 200: KILL — promotion fundamentally unsafe

---

## Measurement Protocol

For each of 28 transformer layers, measure κ for:
1. `self_attn.q_proj` — W ∈ ℝ^(2048×1024) [q heads × head_dim, d_model]
2. `self_attn.k_proj` — W ∈ ℝ^(1024×1024) [kv heads × head_dim, d_model]
3. `self_attn.v_proj` — W ∈ ℝ^(1024×1024)
4. `self_attn.o_proj` — W ∈ ℝ^(1024×2048)
5. `mlp.gate_proj` — W ∈ ℝ^(intermediate×1024)
6. `mlp.up_proj` — W ∈ ℝ^(intermediate×1024)
7. `mlp.down_proj` — W ∈ ℝ^(1024×intermediate)

**Method:** Gram matrix eigendecomposition (numerically stable for large matrices):
- For W ∈ ℝ^(m×n): compute G = Wᵀ W ∈ ℝ^(n×n) if m≥n, else W Wᵀ ∈ ℝ^(m×m)
- Eigenvalues λ of G are σ²
- κ(W) = √(λ_max / λ_min)

**Note on 4-bit quantization:** We measure κ of the *dequantized* float weights (the
values stored as 4-bit integers × scale). This is the actual W_q used in forward passes.
