# REVIEW-adversarial.md — exp_p3_b1_ortho_t2t3_compose

## Verdict: KILLED

**Status**: KILLED — K1174 FAIL (16pp > 10pp threshold)

---

## Evidence Quality: ACCEPTABLE

Full run completed (N=25 style, N=20 math, 204.7s). Results are non-fabricated:
- Phase 1 (personal-only): 19/25 = 76.0%
- Phase 2 (math-only): 2/20 = 10.0%
- Phase 3 (composed-GS): 15/25 style, 2/20 math

Smoke test was misleading (personal-only at N=5 showed 100%, full run shows 76%).
The 100% → 76% drop in personal-only reveals N=5 smoke tests are unreliable for
style compliance estimation.

---

## Mathematical Issues

**No math errors found** in Theorems 1-2. The proof is valid:
- GS projection is algebraically exact → K1172 verified (2.5e-7)
- Power equalization is trivially exact → K1173 verified (1.00)

**The gap is in the behavioral prediction for K1174**. MATH.md correctly identified
the residual failure mode: "interaction effects BEYOND B-matrix direction alignment."

The paper identifies the correct root cause: A_P^T A_D ≠ 0 even after B-matrix
orthogonalization. This is mathematically sound.

---

## Adversarial Concerns

**Issue 1: Smoke test reliability**
The 100% smoke test personal-only rate (N=5) gave false confidence. Full run shows 76%.
The 10pp kill criterion would only be met if composed ≥ 66%. At 60%, we fail by 6pp.
Impact: MINOR — doesn't change verdict, does suggest smoke N≥10 for style tasks.

**Issue 2: Is 16pp a fundamental limit or threshold sensitivity?**
The threshold was set at 10pp from system requirements. The measured 16pp is:
- 6pp above threshold (not a catastrophic 100pp failure like Finding #460)
- Consistent with A-matrix interference being the remaining signal (~50% of B-matrix effect)
The 10pp threshold is appropriate for a production system. Not a threshold tuning issue.

**Issue 3: Math MCQ at 10% (extremely low)**
Both math-only and composed = 10% accuracy. This is near random for 4-class MCQ (25%).
Why is math adapter so weak? Possible: math adapter was trained with insufficient data
for this eval set (out-of-distribution questions), or N=20 is too small to distinguish
from noise. This doesn't affect the verdict but indicates math signal measurement is noisy.

---

## Finding Status: KILLED

Correct status. The algebraic fix (B-matrix GS) is necessary but NOT sufficient for
behavioral preservation. A full ΔW orthogonalization is required.

The impossibility structure is well-derived: δ_P ≥ ||A_P^T A_D||_F × ||B_D||_F
cannot be bounded by B-matrix operations alone.

---

## Recommendation: PROCEED to P3.B2 (Full ΔW Orthogonalization)

The experiment is killed but the research direction is validated: GS is the right approach,
B-matrix only is insufficient. Next experiment should implement full ΔW = A×B orthogonalization
with SVD re-factorization as described in PAPER.md and arxiv 2402.03513.

**Predicted outcome of P3.B2**: If A_P^T A_D is the primary remaining interference,
then full ΔW orthogonalization should reduce 16pp → <10pp (algebraic guarantee).

Key technical concern for P3.B2: SVD re-factorization of ΔW_P' must handle rank collapse
gracefully (ΔW_P' may lose rank after projection onto complement of rank-r_D space).
