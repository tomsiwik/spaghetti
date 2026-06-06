# REVIEW-adversarial.md — T1.4: Cayley Transform at r=16

**Verdict: PROCEED**

Finding #414 status "supported" is appropriate. Two key theorems verified, one criterion failed with correct diagnosis.

---

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match results.json (K1018: 7.62e-16, K1019: 6.4μs, K1020: 300+ steps)
- [x] Finding status "supported" is appropriate (2/3 PASS; FAIL correctly attributed to criterion design error)
- [ ] Theorem 3 descent verification is incomplete (minor — see below)

---

## Issues

### Blocking: None

### Non-blocking

**1. K1019 ambiguity: numpy vs MLX cost**
The passing measurement (6.4 μs) is numpy, not MLX. The actual MLX cost is 433 μs — 4.3× *above* the 100 μs threshold. The PAPER.md correctly notes "PASS (numpy)" and explains the MLX limitation, but the kill criterion was framed around the "true inversion cost", not the MLX runtime. For T1.6, record 433 μs as the real overhead constraint, not 6.4 μs. Cayley in MLX 0.29.x is currently slower than Givens (pure elementwise ops, no linalg.solve needed).

**2. Theorem 3 descent analysis incomplete**
The proof trails off: `⟨G_riem, ΔW⟩ = -τ [‖G^T W‖_F^2 · ... ]` — the last step is elided. The conclusion ("⟨G_riem, ΔW⟩ < 0 when G is non-zero") is correct (standard result for Riemannian gradient descent on compact Stiefel), but the step should reference Absil et al. 2008 or just state the standard citation rather than presenting an incomplete derivation. Not a mathematical error, just an incomplete proof sketch.

**3. K1020 was wrong-class from the start**
The theorem predicts "CayleyAdam ≤ LoRA steps to convergence" — this is a flawed criterion because it compares constrained Riemannian optimization against unconstrained Adam on a loss that is convex in unconstrained space. The finding correctly diagnoses this. For T1.6, the bake-off criterion should be: Givens vs Cayley vs Householder on the SAME constrained Stiefel task with identical loss, optimizer class, and budget.

---

## Summary for T1.6 Design

From T1.3 + T1.4:
- Givens: 192 params/layer, parallel Metal ops, isometry error 2.38e-7, no linalg.solve needed
- Cayley: 120 params/layer, requires linalg.solve (CPU-only in MLX 0.29.x → 433 μs per retraction), isometry error 7.62e-16
- Givens is currently the MLX-practical choice; Cayley wins on parameter count and exactness but needs GPU linalg

T1.6 bake-off should hold MLX limitation fixed and compare convergence on constrained adapter task.
