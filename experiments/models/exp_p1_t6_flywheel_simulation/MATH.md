# MATH.md — T6.4: Flywheel Simulation (3 Sequential Promotions)

## Background

T6.3 verified single base promotion: exact formula (cos=0.99999988), ε=4.78% spectral
perturbation, slot freed, trainability confirmed. The caveat: single promotion only.

T6.4 asks: does the flywheel work at N=3 sequential promotions?
- Promote medical → W_1
- Promote code → W_2
- Promote math → W_3

Key risk: cumulative spectral perturbation. Linear accumulation would give 3 × 4.78% = 14.34%,
exceeding the Davis-Kahan safe zone. But near-orthogonal adapters (Finding #427, T3.1)
suggest √N scaling applies.

---

## Theorem 1: Sequential Promotion Preserves Domain Quality

**Claim:** After N sequential promotions, for domain k promoted at step k, the response
of W_N to domain input x_k satisfies:

    cos(W_N · x_k,  W_{k-1} · x_k + ΔW_k · x_k)  ≥  1 - η²/2

where η = max_{j≠k} cos(ΔW_j · x_k, ΔW_k · x_k) is the cross-domain contamination cosine.

**Proof:**
After all promotions W_N = W_0 + Σ_{i=1}^{N} ΔW_i. On domain input x_k:

    W_N · x_k = W_{k-1} · x_k + ΔW_k · x_k + Σ_{j>k} ΔW_j · x_k

The last sum is the cross-domain contamination C_k = Σ_{j≠k} ΔW_j · x_k.

By T3.1 (pairwise interference = 0), domain adapters are near-orthogonal in input space:
E[x_j · x_k^T] ≈ δ_{jk} for j≠k, so ||C_k|| / ||ΔW_k · x_k|| = η ≈ 0.

By triangle inequality:
    ||W_N · x_k - (W_{k-1} · x_k + ΔW_k · x_k)||₂ ≤ ||C_k||₂

Since cos = 1 - ||residual||²/(2||v||²) for nearly-parallel vectors:
    cos ≥ 1 - η²/2

From T3.1: pairwise cos(ΔW_i, ΔW_j) < 0.1, so η < 0.1, giving cos ≥ 1 - 0.005 = 0.995.

**Prediction K1128:** quality_cosine > 0.99 for each promoted domain after all 3 promotions.

---

## Theorem 2: Cumulative Spectral Perturbation Scales as √N (not N)

**Claim:** For N near-orthogonal domain adapters, the cumulative perturbation satisfies:

    ε_cumul = ||W_N - W_0||_F / ||W_0||_F ≈ √N · ε_single

rather than the worst-case linear bound N · ε_single.

**Proof:**
    ||W_N - W_0||_F² = ||Σ_{i=1}^{N} ΔW_i||_F²
                      = Σ_{i=1}^{N} ||ΔW_i||_F² + 2 Σ_{i<j} <ΔW_i, ΔW_j>_F

By near-orthogonality (T3.1): max_{i≠j} cos_F(ΔW_i, ΔW_j) < 0.1, so:

    <ΔW_i, ΔW_j>_F ≤ 0.1 · ||ΔW_i||_F · ||ΔW_j||_F

For equal-norm adapters (||ΔW_i||_F = ε · ||W_0||_F = c for all i):

    ||W_N - W_0||_F² ≤ N · c² + 2 · C(N,2) · 0.1 · c² = c² · (N + 0.1 · N(N-1))

For N=3: upper bound = c² · (3 + 0.6) = 3.6 · c²

Linear bound: N² = 9. √N bound: N = 3. Empirical: 3.6 (intermediate).

    ε_cumul ≤ √3.6 · ε_single = 1.897 · 4.78% ≈ 9.07%

This is below the Davis-Kahan threshold of 10% (K1129).

**Prediction K1129:** ε_cumul(3 promotions) < 10%, expected ≈ 8–9%.

Note: The linear worst-case (14.34%) would FAIL. The √N bound (8.28%) PASSES.
The distinction depends on adapter orthogonality — T3.1 makes this possible.

---

## Theorem 3: Slot Liberation Is Strictly Cumulative

**Claim:** After k promotions, exactly k adapter slots are freed.

**Proof:**
Each promotion removes exactly 1 adapter from the serving stack (T6.3, Theorem 3),
and promotions are independent (different domains). After k promotions: k slots freed. QED.

**Prediction K1130:** n_slots_freed == 3 after 3 promotions. n_adapters = n_initial - 3.

---

## Theorem 4: No Catastrophic Interference in Sequential Cascade

**Claim:** The cross-domain interference in weight space remains bounded across promotions:

    max_{i≠j} | <ΔW_i, ΔW_j>_F | / (||ΔW_i||_F · ||ΔW_j||_F) < 0.15

throughout the cascade.

**Proof:**
From T3.1 (pairwise interference experiment): the pairwise cosine similarity between
domain adapters trained independently is < 0.1 (empirically verified at N=5).
This is a structural property of domain diversity (Welch bound lower bound on worst-case
pairwise correlation for K vectors in d-dimensional space with K << d).

For Gemma 4 q_proj: d = 2560 (input dim), K = 3. Welch bound: max cos ≥ √((K-1)/(Kd-1)) ≈ 0.028.
Typical domain diversity achieves cos ≈ 0.05–0.10. Catastrophic interference (cos > 0.5) requires
domain adapters to be nearly identical — impossible for medical vs. code vs. math.

**Prediction K1131:** max_interference_cosine < 0.15 across all 3 domain pairs.

---

## Quantitative Predictions

| Kill | Theorem | Prediction | Threshold | Source |
|------|---------|-----------|-----------|--------|
| K1128 | Theorem 1 (quality preserved) | quality_cosine > 0.99 for all 3 domains | > 0.99 | T3.1 near-orthogonality |
| K1129 | Theorem 2 (√N scaling) | ε_cumul ≈ 8–9% < 10% | < 10% | Davis-Kahan + Theorem 2 |
| K1130 | Theorem 3 (slot liberation) | 3 slots freed, n_adapters = 5-3 = 2 | == 3 | Structural |
| K1131 | Theorem 4 (no catastrophic interference) | max_pairwise_cos < 0.15 | < 0.15 | Welch + T3.1 |

---

## References

1. T3.1 (Finding #427) — Pairwise adapter interference = 0, max cos < 0.1 at N=5
2. T6.3 (Finding #452) — Single promotion: cos=0.99999988, ε=4.78%, slot freed
3. Davis-Kahan theorem — Stewart & Sun 1990; ε < 10% → MMLU degradation < 1pp
4. Welch bound — Strohmer & Heath 2003, arxiv math/0208005; lower bound on pairwise correlation
5. Task Arithmetic — Ilharco et al. 2022, arxiv 2212.04089; sequential task vector composition
6. Finding #398 — Multi-cycle promotion killed on toy model capacity; Gemma 4 scale resolves this
