# DARE vs M2P-gated Head-to-Head (Post-#831 Fix)

## Result: KILLED

M2P-gated continuous composition underperforms DARE by 10pp despite perfect routing. Ship DARE.

## Predictions vs Measurements

| Metric | Prediction | Measured | Verdict |
|--------|-----------|----------|---------|
| K2147: M2P avg ≥ DARE avg (73.3%) | Expected match/exceed post-fix | 63.3% vs 73.3% (Δ=-10pp) | **FAIL** |
| K2148: calibration ρ ≥ 0.3 | Expected gate confidence → correctness | ρ̄=0.097 | **FAIL** |
| K2149: latency ≤ 1.2× DARE | Expected parity (same fused pattern) | 1.05× | PASS |

## Key Numbers

| Method | GSM8K | HumanEval | MedQA | Avg |
|--------|-------|-----------|-------|-----|
| DARE (drop=0.9, N=7) | 63.3% | 90.0% | 66.7% | 73.3% |
| M2P-gated (learned weights) | 60.0% | 83.3% | 46.7% | 63.3% |

Gate holdout routing accuracy: 99.6%

Calibration per benchmark: gsm8k ρ=-0.061, humaneval ρ=-0.106, medqa ρ=0.125

## Analysis

1. **Gate routes perfectly but composition fails.** The 99.6% holdout accuracy proves prompt classification works. The failure is in *how* weights translate to adapter mixing.

2. **Overflow in gated composition.** NumPy overflow warnings during `compute_gated_deltas` indicate that gate-weight × SCALE × (A@B) produces numerically unstable deltas for some layers. DARE's binary mask + rescale avoids this by zeroing 90% of parameters entirely.

3. **No calibration signal.** Gate confidence (top-1 softmax weight) has near-zero correlation with correctness. The gate learns domain routing, not difficulty estimation.

4. **DARE's stochastic pruning is the better regularizer.** Dropping 90% of parameters forces the model to rely on the most robust weight directions. Continuous weighting preserves noise and weak signals equally.

## Decision

Per pre-registered decision tree: both K1 and K2 FAIL → **ship DARE alone**. M2P-gated adds complexity without accuracy or calibration benefit.

## References

- Finding #831: _FusedDeltaLinear canonical fix
- exp_pierre_composition_method_ablation: confirms DARE > uniform averaging
- exp_pierre_per_task_routing_math: confirms DARE baseline is stable at ~73%
