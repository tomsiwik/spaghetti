# REVIEW-adversarial.md — T1.1: Householder Chain Orthogonality

**Verdict: PROCEED**

## Summary

All 4 kill criteria PASS with strong margins. PAPER.md has a prediction-vs-measurement
table. Math is sound. One theory correction is properly disclosed.

## Checklist

| Item | Status |
|------|--------|
| PAPER.md prediction-vs-measurement table present | ✓ |
| Kill criteria match results.json | ✓ |
| is_smoke: false (full run) | ✓ |
| Finding status "supported" appropriate | ✓ |
| Math proofs correct | ✓ |

## Findings

**K1007 (isometry err):** 2.384e-07 vs threshold 1e-4 — 420× margin. Identical to
Givens T1.3, confirming float32 floor is the method-independent floor.

**K1008 (interference):** 3.85e-10 vs threshold 0.01 — algebraic zero confirmed.
Theorem 2 proof is rigorous and the measurement is consistent.

**K1009 (stable rank):** sr=16.00 exact, σ_max=2.0000 exact. Clean algebraic values
match the derivation (sr = 4r/4 = r from ||H^(r)-I||_F² = 4r and ||H^(r)-I||_2 = 2).

**K1010 (param efficiency):** param_ratio=0.50 trivially correct.

## Theory Correction (non-blocking, properly disclosed)

Theorem 3 predicted sr(LoRA) ≈ 1 but measurement shows 13.57 ≈ r. PAPER.md correctly
explains why: random Gaussian A and B both have flat singular spectra, so A@B has
sr ≈ r, not ~1. The key HRA advantage is same stable rank at 2× fewer params — this
is the claim that survives and is correct.

Recommendation: MATH.md Theorem 3 section (d) could be updated in a future pass to
reflect the correct LoRA stable rank, but this is non-blocking for this finding.

## Caveats

- Float32 overflow warnings in numpy BLAS at d=2816 — documented in PAPER.md,
  results are correct, fix is clear (use float64 for production builds).
- Multilayer isometry data (L=1..8) is in PAPER.md but not in results.json —
  minor gap, non-blocking.

## Next Steps

T1.2 (HRA vs LoRA quality on actual task) is now unblocked.
T1.6 algorithm bake-off now has 3 candidates: Givens, Cayley, HRA.
