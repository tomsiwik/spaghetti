# REVIEW-adversarial.md — T1.3: Givens Rotation Orthogonality at d=2816

**Verdict: PROCEED**
**Date:** 2026-04-09
**Reviewer:** Adversarial

---

## Review Summary

All three kill criteria pass with strong margins. The math is correct. The PAPER.md prediction-vs-measurement table is complete and values match results.json exactly.

---

## Kill Criteria Verification

| Kill | Evidence in results.json | Verdict |
|------|--------------------------|---------|
| K1015 | `k1015_isometry_err: 2.384e-07`, `k1015_pass: true` — margin 420× over threshold 1e-4 | ✅ PASS |
| K1016 | `structural_parallel: true`, `single_kernel: true`, `n_pairs: 1408`, `t_parallel_ms: 3.14ms` | ✅ PASS |
| K1017 | `k1017_params_per_layer: 1408`, `k1017_pass: true` | ✅ PASS |

All values in PAPER.md match results.json exactly. No fabrication risk.

---

## Math Review

**Theorem 1 (Single-layer orthogonality):** Block-diagonal structure from disjoint pairs → O^T O = I_d. Algebraically sound. Float32 bound √d × ε_mach ≈ 6.4e-6 is conservative; measured isometry error (2.384e-07) is well below even this bound.

**Theorem 2 (Parallel execution):** Disjoint index sets {2k, 2k+1} ∩ {2l, 2l+1} = ∅ → zero data dependency → structurally parallel. Correct and confirmed by implementation.

**Theorem 3 (O(d) params):** L × d/2 counts correctly. 1408 < 2816 = O(d) ✅.

**Measurement artifact insight:** The claim that explicit O^T O test fails at large d due to O(d^{3/2} × ε_mach) accumulation is numerically correct. Theory predicts ≈ 0.018; measured 0.034 (2× theory, consistent with random sign accumulation). The isometry test (O(ε_mach) independent of d) is the right tool. This is a genuine contribution to experimental method.

---

## Non-blocking Notes

1. **PAPER.md says "Finding #TBD"** — should be updated to #413. (Analyst pass can fix this.)

2. **Multi-layer isometry constant at 2.384e-07 across L=1..8:** The constant value suggests this is the float32 precision floor of the test itself (dominated by cos/sin representation error), not accumulated rotation error. This is actually positive evidence — composition doesn't add error. Not a flaw, but worth noting in LEARNINGS.md.

3. **Behavioral link is future work:** The experiment correctly verifies mathematical properties but does not test that a Givens adapter improves quality on a real task. Finding status "supported" (not "conclusive") correctly captures this — behavioral validation is T1.6 (bake-off) or later.

---

## Conclusion

T1.3 is a clean verification experiment. The math is sound, predictions match measurements with large margins, and the measurement artifact insight is methodologically valuable. 

**No revisions needed. Proceed to Analyst.**
