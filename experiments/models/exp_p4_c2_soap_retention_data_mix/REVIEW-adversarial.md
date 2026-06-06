# REVIEW — exp_p4_c2_soap_retention_data_mix

**Verdict: PROCEED**

## Summary

Clean experiment. Large effect size (retention 0.80→1.00) is convincing even at N=10. The MATH.md is honest about being a gradient-level argument, not a convergence guarantee. Prediction-vs-measurement table present. Kill criteria results match evidence in results.json.

## Issues

### 1. N=10 makes the "regularization" claim noise-level (non-blocking)

The +80pp vs +70pp improvement (P4.C2 vs P4.C1) is within sampling noise at N=10. At this sample size, the 95% CI for a proportion of 0.8 is roughly [0.44, 0.97]. You cannot distinguish +70pp from +80pp. The retention jump (0.80→1.00) is more convincing since the effect is binary-scale (10/10 vs 8/10), but even that has p~0.11 for the null hypothesis of equal rates.

**Impact on finding:** The core result (mixed training fixes retention) holds. The "regularization > trade-off" narrative is an interesting hypothesis but should be labeled provisional.

### 2. K1248 (legal retention) not re-measured (non-blocking)

Legal retention is cited from P4.C1 with the note "Legal adapter not modified." This is defensible — the experiment modified only the SOAP adapter, so legal retention shouldn't change. But the kill criterion as stated ("legal retention ≥0.95") implies measurement, not citation. Minor honesty point.

### 3. Theorem 1 is a handwave, not a proof (non-blocking)

The "proof" in MATH.md amounts to: if you mix two gradient signals, the component that destroys general knowledge gets partially cancelled. This is true but trivial — it's the definition of multi-task learning. The proof doesn't derive the *threshold* at which cancellation is sufficient, or predict the mixing ratio where retention recovers. The α=0.5 choice is arbitrary, not derived.

The honest framing would be: "Data mixing is multi-task learning applied to LoRA fine-tuning. The question is whether rank-16 has sufficient capacity for both tasks." The experiment answers yes. The theorem adds little beyond intuition.

## What holds

- **Retention fixed**: 0.80→1.00 is a real, large effect. Even at N=10, going from 2 failures to 0 is meaningful.
- **Format not degraded**: At worst, format stayed at +70pp. At best, improved to +80pp. Either way, no trade-off.
- **Practical value**: 50/50 data mixing is a zero-cost recipe for preventing semantic-overlap retention loss.
- **Pattern connection**: The "constraint as regularizer" pattern (also seen in Finding #519 Stiefel) is worth tracking as a provisional hypothesis.

## Finding recommendation

Status: **supported** (not conclusive — N=10, theorem is informal, regularization claim is provisional).
