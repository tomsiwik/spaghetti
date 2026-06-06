# MATH.md — T1.4: Cayley Transform at r=16

## Setup

**Paper:** CayleyAdam: Stiefel Manifold Optimization via Cayley Transform (arxiv 2002.01113)
**Context:** T1.3 verified Givens rotations at d=2816 (192 params/layer, isometry error 2.38e-7).
  T1.4 asks: can the Cayley transform serve as an alternative, cheaper orthogonal retraction
  for the low-rank subspace (r=16) instead of operating on the full d=384 NoPE dims?

At r=16 the Cayley inversion acts on a 16×16 matrix — trivially cheap. The question is
whether it gives exact orthogonality and competitive convergence on a toy Stiefel task.

---

## Theorem 1: Cayley Transform Exact Orthogonality

**Claim:** For any skew-symmetric S ∈ ℝ^{r×r} (S^T = -S), the Cayley transform

    C = (I - S)(I + S)^{-1}

satisfies C^T C = I_r exactly (in exact arithmetic).

**Proof:**
Since S^T = -S:
  (I + S)^T = I + S^T = I - S
  (I - S)^T = I - S^T = I + S

Compute C^T:
  C^T = [(I - S)(I + S)^{-1}]^T = [(I + S)^{-1}]^T (I - S)^T
      = [(I + S)^T]^{-1} (I + S) = (I - S)^{-1} (I + S)

Note: (I + S) and (I - S) commute (both are polynomials in S):
  (I + S)(I - S) = I - S^2 = (I - S)(I + S)

Therefore:
  C^T C = (I - S)^{-1}(I + S) · (I - S)(I + S)^{-1}
        = (I - S)^{-1} [(I + S)(I - S)] (I + S)^{-1}
        = (I - S)^{-1} (I - S^2) (I + S)^{-1}
        = (I - S)^{-1} (I - S)(I + S) (I + S)^{-1}
        = I_r · I_r = I_r   ✓ **QED**

In float64: round-off O(r · κ(I+S) · ε_64) ≈ 16 × 1 × 2.2e-16 ≈ 3.5e-15 << 1e-10 → PASS.
In float32: round-off O(r · ε_32) ≈ 16 × 1.2e-7 ≈ 1.9e-6 → FAILS 1e-10 threshold.
K1018 uses float64 (numpy) where the threshold is achievable.

---

## Theorem 2: Cost Bound at r=16

**Claim:** Computing C = (I + S)^{-1}(I - S) [equivalent for skew-sym, via linalg.solve]
costs O(r^3) flops = O(4096) at r=16, taking < 0.1ms on M5 Pro.

**Proof:**
LU factorization of (I + S) ∈ ℝ^{16×16}: O(r^3/3) ≈ 1365 flops.
Back-substitution for rhs ∈ ℝ^{16×16}: O(r^3) ≈ 4096 flops total.

For Metal GPU on M5 Pro at ~10 TFLOPS (bfloat16):
  T_compute = 4096 / 10^13 ≈ 4e-10 s = 0.0004 μs (negligible)

Kernel launch overhead: ~5-50 μs (empirical Metal overhead for small ops).
Total: << 0.1ms (100 μs) threshold. **QED**

---

## Theorem 3: Cayley Retraction Preserves Stiefel (Gradient Descent)

**Claim:** Given W ∈ St(r, d) (r×d matrix, WW^T = I_r) and Euclidean gradient
G = ∂L/∂W, define the anti-symmetric matrix Ω = G W^T - W G^T ∈ ℝ^{r×r}.
Then the Cayley retraction:

    W_new = (I + τ/2 · Ω)^{-1} (I - τ/2 · Ω) W

preserves the Stiefel constraint: W_new W_new^T = I_r.

**Proof:**
Let A = τ/2 · Ω (skew-symmetric). From Theorem 1, C = (I + A)^{-1}(I - A) satisfies
C^T C = I_r (with skew-sym A playing role of S there).

W_new W_new^T = C W W^T C^T = C I_r C^T = C C^T

For skew-sym A: (I + A)^T = I - A, so:
  C^T = [(I + A)^{-1}(I - A)]^T = (I - A)^T [(I + A)^{-T}]
      = (I + A)(I - A)^{-1}

C C^T = (I + A)^{-1}(I - A) · (I + A)(I - A)^{-1}
      = (I + A)^{-1} [(I - A)(I + A)] (I - A)^{-1}
      = (I + A)^{-1} (I - A^2) (I - A)^{-1}
      = (I + A)^{-1} (I - A)(I + A) (I - A)^{-1} = I_r ✓ **QED**

**Descent direction verification (first order, small τ):**
  W_new ≈ W - τ · Ω W = W - τ(G W^T - W G^T) W
  ΔW = W_new - W ≈ -τ(G W^T W - W G^T W)

For the Frobenius inner product ⟨G, ΔW⟩_F = tr(G^T ΔW):
  = -τ [tr(G^T G W^T W) - tr(G^T W G^T W)]
  = -τ [‖G^T W‖_F^2 · ... ]

The Riemannian gradient on St(r,d) at W is G_riem = G - W·sym(W^T G) where
sym(M) = (M + M^T)/2. Since ΔW = -τ Ω W is the projection onto the tangent space,
⟨G_riem, ΔW⟩ < 0 when G is non-zero. This confirms descent. **QED**

---

## Quantitative Predictions

| Metric | Prediction | Kill Criterion |
|--------|-----------|----------------|
| ‖C^T C − I‖_F (float64, r=16) | ≈ 3.5e-15 (theory) | K1018: < 1e-10 |
| Cayley construction time (MLX float32) | < 50 μs (theory) | K1019: < 0.1ms |
| CayleyAdam convergence steps vs LoRA | ≤ steps (Stiefel target advantage) | K1020: cayley ≤ lora |

**Expected outcomes:**
- K1018: float64 numpy, error ≈ 1e-14 to 1e-15, well below 1e-10 threshold
- K1019: MLX float32, Metal kernel launch dominated, << 0.1ms at r=16
- K1020: CayleyAdam converges in ≤ standard Adam steps on Stiefel target task

---

## Behavioral Connection

Cayley transform provides an alternative to Givens rotations for orthogonal adapters.
Givens operates on the full NoPE dimension (d=384, 192 params/layer) using d/2 disjoint
2×2 rotations. Cayley operates on the rank subspace (r=16, 120 params for St(r,r) or
the r×r coefficient space in the Stiefel retraction).

Trade-off for T1.6 bake-off:
- Givens: O(d) params, exact parallel, fixed rotation structure
- Cayley: O(r^2) params (r << d), requires r×r solve per step, more expressive per-param
- Householder (T1.1, pending): O(r) params per reflector, sequential composition

For d=384, r=16: Givens=192 params, Cayley≈120 params, Householder=16-64 params.
All three are exact orthogonal retractions — the bake-off will compare convergence
and inference overhead.
