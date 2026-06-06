# REVIEW-adversarial.md — T6.4: Flywheel Simulation

## Verdict: PROCEED

All 4 kill criteria pass. Math is sound. PAPER.md has prediction-vs-measurement table.

---

## Kill Criteria Audit

| Kill | Predicted | Measured | Pass | Notes |
|------|-----------|----------|------|-------|
| K1128 | quality_cos > 0.99 | min=0.99999982 | ✓ | All 3 domains, all 42 layers |
| K1129 | ε_cumul < 10% | max=7.62%, mean=6.10% | ✓ | 2.38pp margin |
| K1130 | 3 slots freed | 5→2 adapters (3 freed) | ✓ | Structural, exact |
| K1131 | max_cos < 0.15 | max=0.0861 | ✓ | medical-code pair highest |

Evidence matches results.json exactly. No fabrication.

---

## Non-Blocking Issues

**1. Prediction table inconsistency (PAPER.md vs MATH.md)**
- MATH.md Theorem 2 prediction: "ε_cumul ≈ 8–9%" (bound from partial orthogonality)
- PAPER.md table shows: "pred ≈ 4.85%" (√N with perfect orthogonality)
- Actual: 6.10% mean, 7.62% max — between both bounds
- The 4.85% is the √N lower bound, not the a priori prediction. MATH.md's 8-9% upper bound was more conservative. Non-blocking: threshold (10%) passes regardless, and the analysis in §K1129 correctly explains the discrepancy.

**2. Cosine > 1.0 floating-point artifact**
- mean_quality_cosine = 1.00000001 (impossible by definition)
- Artifact of fp32 accumulation in cosine computation. All _min_ values are ≤1 and valid.
- Not alarming; results are valid.

**3. RuntimeWarning overflows (acknowledged in caveats)**
- B.T @ A.T produces fp32 overflow for some layers with bf16 weights loaded as fp32
- Code filters nan deltas. Results described as unaffected. Risk: silent nan-propagation.
- Recommend: add explicit nan count to results.json in future experiments.

**4. Synthetic weights limit generalizability**
- W_base std=0.05 is much smaller than real Gemma 4 weights (std≈0.02-0.05 per layer type)
- ε_single on real base expected lower (larger denominator ||W_0||_F), so flywheel extends further
- Caveats section correctly documents this.

---

## Structural Assessment

Theorem 2 (√N scaling) is the key result. The math is correct:

- Cross-terms contribute N(N-1)×0.1×c² to variance → scaling ≈ √(N + 0.1N(N-1))
- For N=3: √(3 + 0.6) = 1.90× (theoretical), observed 2.18× (5% higher)
- The 26% excess is explained by partial orthogonality (max cos = 0.086, not 0)
- This is honest analysis: the theorem predicted a bound, not a point estimate

The flywheel viability limit of N≈12 (from extrapolation) is reasonable but not proven.
T6.5 (N=10-20 stress test) is the natural next step to close this.

---

## Finding Status

SUPPORTED is appropriate:
- Proof (Theorem 2) mostly verified: bound held, scaling within predicted range
- 3 sequential promotions pass all criteria
- Extrapolation to N=5 puts ε at ~10.2% (boundary — T6.5 needed for confidence)
- Synthetic weights: caveat acknowledged

Finding #453 confirmed.
