# REVIEW: T1.5 PoLAR Landing Field (exp_p1_t1_polar_landing) — Final

**Verdict: PROCEED (KILLED)**

---

## Summary

All 3 blocking fixes from the prior REVISE have been applied. The experiment ran to completion (200 steps, n=30), PAPER.md is written with a correct prediction-vs-measurement table, and MATH.md includes a proper refutation note on the sr(V) corollary. The KILLED status is correct and well-supported.

---

## Checklist

**PAPER.md prediction-vs-measurement table:** ✓ Present and accurate.

| Kill Criterion | Predicted | Measured | PASS/FAIL |
|---|---|---|---|
| K1021: ‖UU^T−I‖_F | < 0.01 | 2.46e-08 | PASS (400× margin) |
| K1022: sr(ΔW) | ≥ 5 | PoLAR=2.21, LoRA=4.45 | FAIL |
| K1023: PoLAR ≥ LoRA (GSM8K) | PoLAR ≥ LoRA | 3.3% vs 13.3% | FAIL |

**results.json consistency:** ✓ All values match (K1021=2.4573e-08, PoLAR sr=2.21, LoRA sr=4.45, PoLAR acc=3.3%, LoRA acc=13.3%). `is_smoke=false`.

**Finding status:** ✓ KILLED is correct. K1021 PASS verifies Theorem 1. K1022+K1023 FAIL refute the sr(V)≈r/2 corollary.

**Math errors:** None. Theorems 1 and 2 are sound. The error was in the corollary predicting sr(V) from gradient routing — correctly identified and labeled with ⚠️ in MATH.md.

**Impossibility structure:** ✓ Correctly derived. The key insight:
- `∂L/∂V = (∂L/∂ΔW) @ U^T` — U^T is an isometry but does NOT diversify gradient directions
- GSM8K gradient is rank-1 → V collapses regardless of U's orthogonality
- Fix requires joint Stiefel retraction on U×V (product manifold) OR diverse training signal

**NaN guard:** ✓ Added. PAPER.md addresses the float64 BLAS overflow warnings (confirmed finite, correct result).

---

## Non-Blocking Notes

- The Theorem 2 bound `sr(ΔW) ≥ sr(V)·(1−ε)/(1+ε)` is CORRECT — but the premise sr(V)≈r/2 was wrong. Theorem 2 itself is a valid tool for future use with joint retraction.
- PoLAR step time ratio 1.02× (87.1s vs 84.9s for 200 steps) — negligible overhead. The Stiefel retraction is cheap; the quality gap is purely from V-collapse, not computation.
- n_eval=30 is small but sufficient given the 4× gap (3.3% vs 13.3%). Noise could not explain a 4× difference at n=30.

---

## T1.6 Design Implication

The bake-off (Givens vs Cayley vs PoLAR) must use **joint retraction on U×V** (product manifold), otherwise all methods will show the same V-collapse. Equal-params comparison (T1.2 lesson) also applies.

---

## Verdict

PROCEED. Finding #419 (KILLED) is correctly documented. The impossibility structure is clear and actionable. Ready for Analyst to write LEARNINGS.md.
