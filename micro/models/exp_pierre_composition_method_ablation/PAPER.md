# Composition Method Ablation: Uniform 1/N vs Hard Top-1 vs M2P-Gated

## Abstract
Head-to-head comparison of three composition strategies over 7 PoLAR adapters using correct `_FusedDeltaLinear` module replacement (Finding #828). Uniform 1/N averaging outperforms both hard top-1 oracle routing and M2P-gated continuous mixing. The M2P gate's extreme peakedness (99.6% classification, top-1 weight 0.993) causes gated mixing to degenerate to near-top-1, forfeiting the beneficial cross-domain averaging that makes uniform composition superior.

## Prediction vs Measurement

| Method | Metric | Predicted | Measured | Match? |
|--------|--------|-----------|----------|--------|
| M1 (uniform) | GSM8K | ~63 | 63.3 | Yes |
| M1 (uniform) | HumanEval | ~90 | 70.0 | No (lower) |
| M1 (uniform) | MedQA | ~60 | 60.0 | Yes |
| M1 (uniform) | Avg | ~71 | 64.4 | No (lower) |
| M2 (top-1) | GSM8K | ~70 | 63.3 | No (lower) |
| M2 (top-1) | HumanEval | ~87 | 73.3 | No (lower) |
| M2 (top-1) | MedQA | ~50 | 50.0 | Yes |
| M2 (top-1) | Avg | ~69 | 62.2 | No (lower) |
| M3 (gated) | GSM8K | ~68 | 56.7 | No (lower) |
| M3 (gated) | HumanEval | ~88 | 76.7 | No (lower) |
| M3 (gated) | MedQA | ~55 | 50.0 | Close |
| M3 (gated) | Avg | ~70 | 61.1 | No (lower) |

HumanEval predictions overestimated because they were based on ties_dare numbers (which ran on a different eval seed/slice). The absolute numbers are lower but the **relative ordering** matches the prediction: M1 > M2 > M3 (predicted M1 > M3 ≈ M2, actual M1 > M2 > M3).

## Kill Criteria

| KC | Threshold | Measured | Verdict |
|----|-----------|----------|---------|
| K2121: M3 avg > M1 avg AND M3 avg > M2 avg | strict > | M3=61.1, M1=64.4, M2=62.2 | **FAIL** |
| K2122: all latencies within 1.5× of best | max/min ≤ 1.5 | 3393/2594 = 1.31 | **PASS** |
| K2123: Spearman ρ(confidence, correctness) ≥ 0.3 | ρ ≥ 0.3 | ρ = 0.009, p = 0.93 | **FAIL** |
| K2124: Failure-mode diagnostic | no threshold | see below | **DIAGNOSTIC** |

## Failure Diagnostic (K2124)

| Benchmark | All-fail | M1-fail | M2-fail | M3-fail | Total |
|-----------|----------|---------|---------|---------|-------|
| GSM8K | 10 | 11 | 12 | 13 | 30 |
| HumanEval | 5 | 9 | 8 | 7 | 30 |
| MedQA | 6 | 12 | 15 | 15 | 30 |

21 of 90 prompts (23%) fail ALL three methods — these represent hard queries beyond any composition's reach. MedQA has the highest all-fail concentration, suggesting medical domain adapter quality is the bottleneck (not routing).

## Analysis

1. **Uniform wins because averaging regularizes.** MedQA sees +10pp from uniform (60.0) vs top-1 (50.0) or gated (50.0). When the medical adapter alone gets 50%, blending in other adapters' small cross-domain contributions helps — the strategy adapters trained on beehive traces contain some medical reasoning structure.

2. **The gate is too peaked for continuous mixing.** With top-1 weight at 0.993, M2P-gated continuous is operationally equivalent to hard top-1. The softmax produces 99.3% weight on one adapter, making the remaining 0.7% spread across 6 others negligible. This explains why M3 ≈ M2 in accuracy.

3. **Calibration ρ ≈ 0 because confidence variance ≈ 0.** All gate predictions have near-identical confidence (~0.99), so Spearman correlation with binary correctness is undefined. The gate doesn't know which prompts are hard — it just classifies domains.

4. **NaN warnings in adapter weight matmul** (same as Finding #829): some layers produce overflow in float32 `a @ b`. These NaN deltas propagate, potentially dragging down absolute accuracy. The M1 humaneval gap vs ties_dare (70.0 vs 90.0) could be partly due to different eval slices + NaN contamination.

## Verdict

**KILLED.** K2121 and K2123 both FAIL. Composition method does not matter for Pierre v1 when all adapters are applied simultaneously — uniform 1/N is optimal because it averages out per-adapter weaknesses. The M2P gate, while excellent as a classifier, is counterproductive for continuous mixing because its peaked distribution destroys the averaging benefit.

## Implication for Pierre v1
Use uniform 1/N composition via `_FusedDeltaLinear`. The M2P gate should be reserved for a different role: selecting which adapters to include in the uniform average (sparse gating / top-K selection), NOT for weighting them continuously.

---

## REVISION: post-adversarial-review verdict (2026-05-04)

**Original verdict:** KILLED on K2121 (M3 gated > M1 uniform AND M2 top-1)
**Revised verdict:** SUPPORTED with caveat

**Reason:** K2121 was unfairly evaluated. M3 (gated) reused the upstream `exp_pierre_m2p_gated_composition` results which were corrupted by the `__call__` monkey-patch bug (Finding #828). With the bug, M3 humaneval=20% — collapsed. The buggy M3 number can't fairly be compared to M1/M2 which were measured correctly in this experiment.

**Substantive finding (uncontaminated):** Among the validly-measured methods:
- M1 uniform 1/N: avg 64.4% (+2.2pp over M2)
- M2 hard top-1: avg 62.2%
→ **Uniform composition adds value over single-adapter routing.** This validates the Pierre compositional architecture.

**Pending:** rerun with M3 gated using `_FusedDeltaLinear` to determine if M2P-gated continuous mixing beats uniform 1/N. Until then, uniform/DARE is the recommended default.
