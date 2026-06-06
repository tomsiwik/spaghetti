# MATH.md: Activation-Space Interference Scaling

## TYPE: guided-exploration
## PROVEN FRAMEWORK: Grassmannian parameter-space orthogonality (#3, #341)
## UNKNOWN: How activation-space interference scales with N adapters

---

## Problem

Parameter-space orthogonality is proven: ⟨Δ_i, Δ_j⟩_F = 0 for Grassmannian A-slots.
But activation-space interference is NOT guaranteed:

    h_out = W·x + Σ_i B_i(A_i·x)

Even with A_i⊥A_j, the terms B_i(A_i·x) and B_j(A_j·x) live in the same output
space and can interfere. Finding #353 measured max|cos| = 0.29 at N=5.

The adversarial review (critique #6) requires us to measure how this scales with N.

## Metric Definition

For each pair (i,j) of composed adapters at each token position t:

    activation_cos(i,j,t) = |cos(B_i A_i x_t, B_j A_j x_t)|

where x_t ∈ R^d is the (d,) hidden-state vector at position t (a single token), and
B_i A_i x_t ∈ R^{d_out} is the adapter output for adapter i at that token.

We report:
- max over all (pair, token) combinations (worst case)
- mean over all (pair, token) combinations (typical case)
- fit power law: max_cos ~ c · N^α

**Measurement scope:** Layer 0, two representative modules:
  - wq (mi=0): d_out = D_MODEL = 256; input = norm1(x)
  - fc1 (mi=4): d_out = 4·D_MODEL = 1024; input = norm2(x + attn(x))

The worst-case max_cos is taken across both modules. Results are reported per-module
to allow comparison of a narrow module (wq) vs. a wide module (fc1).

**Implementation note:** Activations are materialized to numpy after mx.eval, then
per-token cosines are computed in numpy. This matches the definition above exactly —
each cos is computed from a (d_out,) vector, not a flattened trajectory.

## Theoretical Bounds

**Upper bound (trivial):** max|cos| ≤ 1 (unit vectors).

**Lower bound (if A orthogonal, B random):** For random B_i with ||B_i||_F ~ σ,
the expected activation cos is:

    E[|cos(B_i A_i x, B_j A_j x)|] = O(1/sqrt(d_out))

At d_out=256: E ≈ 1/16 ≈ 0.06. This is the "random B" baseline.
If measured cos >> 0.06, the B-matrices have learned correlated structure.
If measured cos ≈ 0.06, interference is random and doesn't grow with N.

**Scaling prediction:**
- If α ≈ 0: interference is constant (best case — each adapter is independent in activation space)
- If α ≈ 0.5: interference grows as sqrt(N) (central limit — random walk in activation space)
- If α ≈ 1.0: interference grows linearly (worst case — correlated B-matrices)

## Experiment Design

At d=256, L=2:
- Train N ∈ {2, 3, 5, 8, 10} per-domain M2P adapters
- Need 10 total domains. Use: arithmetic, sort, reverse, repeat, parity,
  cipher (new), counting, dedup, mapping, interleave
- For each N, compose all N adapters and measure activation cos on held-out data
- Also measure: per-domain quality degradation vs single-adapter (composition tax)

## Kill Criteria

**K_activation:** per-token max |cos| at N=10 < 0.5
  — evaluated as the worst case across both wq and fc1 modules.
  — metric is the per-token max (not the global trajectory cosine).
**K_scaling:** fitted α < 0.5 (sub-linear growth)
**K_quality_N10:** composition quality at N=10 ≥ 80% of best-single

## Predictions

| N | Predicted max|cos| | Predicted quality |
|---|:---:|:---:|
| 2 | 0.10-0.20 | 95%+ |
| 5 | 0.20-0.35 | 90%+ |
| 10 | 0.30-0.50 | 80%+ |
| α | 0.3-0.5 | — |
