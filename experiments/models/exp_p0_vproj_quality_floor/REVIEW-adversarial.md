# REVIEW — exp_p0_vproj_quality_floor

**Verdict: PROCEED (kill justified)**

## Evidence Consistency

Kill criteria results in PAPER.md match results.json exactly:
- K1320: math=50%, code=10%, med=15%, legal=25%, fin=15% — all below 60%. **FAIL confirmed.**
- K1321: mean=23% vs threshold 50%. **FAIL confirmed.**
- K1322: legal=25% vs threshold 40%. **FAIL confirmed.**
- K1323: max 12.6 min vs threshold 30 min. **PASS confirmed.**

No fabricated data. Per-query results in results.json support the aggregated rates.

## Prediction-vs-Measurement Table

Present and correctly formatted. Predictions were dramatically wrong (67% predicted mean
vs 23% measured), which is actually a strong signal — the theoretical framework was
incomplete, not the execution.

## Root Cause Quality

The disease identification is the strongest part of this experiment:
- **Symptom:** Low adapter quality with HF data
- **Assumed disease:** Insufficient training data (wrong)
- **Actual disease:** Training-evaluation distribution mismatch

The impossibility structure is geometrically sound: v_proj+o_proj directly modifies token
distribution (Finding #504). Training on code tokens → model outputs code tokens, not
explanatory prose. This is not a hyperparameter issue — it's structural.

## Finding #506 Status

Finding status is "supported" (the insight that distribution > quantity IS supported by evidence).
Experiment status is "killed" (predictions refuted). Both are correct.

## Minor Notes (non-blocking)

1. MATH.md cites Finding #149 (data saturation at N=200-500) to justify N=500, but the failure
   wasn't about quantity at all. The citation is technically correct but misleading — it led to
   the wrong frame (more data = better). Future experiments should distinguish between data
   quantity claims and data distribution claims when citing saturation findings.

2. The two paths forward (explanatory data vs matching benchmarks) are well-formulated and
   actionable. The project should decide which path serves the VISION before the next experiment.
