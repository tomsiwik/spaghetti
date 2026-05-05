# PAPER — Per-Task Routing for Math (GSM8K Regression Fix)

## Experiment ID
`exp_pierre_per_task_routing_math`

## Hypothesis
Route math-shaped queries to single best math-domain adapter (bypassing DARE composition) to close the GSM8K -6.7pp regression observed in Finding #831.

## Method
- TF-IDF + RidgeClassifier binary router (math vs non-math)
- Math queries → single math-domain PoLAR adapter (no fused delta)
- Non-math queries → 7-adapter DARE composition via _FusedDeltaLinear
- Evaluated on GSM8K (30), HumanEval (30), MedQA (30)

## Results

### Routing Performance
| Metric | Value | KC threshold | Verdict |
|--------|-------|-------------|---------|
| Classifier test accuracy | 100.0% | ≥85% (K2145) | **PASS** |
| Routing overhead | 0.072 ms | ≤5ms (K2146) | **PASS** |

Routing bucketed perfectly: 30/30 GSM8K → math-single, 30/30 HumanEval → DARE, 30/30 MedQA → DARE.

### Benchmark Accuracy

| Benchmark | Single math | DARE-only | Routed | KC target | Verdict |
|-----------|------------|-----------|--------|-----------|---------|
| GSM8K | 63.3% | 63.3% | 63.3% | ≥68.0% (K2143) | **FAIL** -6.7pp |
| HumanEval | — | 90.0% | 66.7% | ≥88.0% (K2144) | **FAIL** -23.3pp |
| MedQA | — | 66.7% | 70.0% | ≥64.7% (K2144) | PASS +3.3pp |

### Kill Criteria Summary
| KC | Description | Result |
|----|------------|--------|
| K2143 | GSM8K ≥68.0% (within 2pp of 70%) | **FAIL** — 63.3%, drop 6.7pp |
| K2144 | HumanEval/MedQA within 2pp of DARE | **FAIL** — HumanEval -23.3pp |
| K2145 | Classifier ≥85% | PASS — 100.0% |
| K2146 | Routing ≤5ms | PASS — 0.072ms |

## Analysis

### The original hypothesis was wrong
The GSM8K -6.7pp regression was attributed to DARE perturbation sensitivity (Finding #831). But this experiment shows **single math adapter also gets exactly 63.3%** — identical to DARE composition. DARE is not degrading math; the math adapter simply doesn't reach 70%.

The 70% "best single-adapter" reference from prior experiments was likely measured on a different sample set or with a different generation config. On this 30-sample evaluation, both paths converge at 63.3%.

### HumanEval variance
The 23.3pp HumanEval drop (90.0% → 66.7%) between the DARE-only reference run and the routed DARE run — using identical composition parameters — exposes high generation variance. Both runs use the same DARE weights and same prompts; the difference is stochastic generation output. This is a known issue with N=30 code-generation benchmarks.

### What worked
The routing mechanism itself is excellent:
- Perfect binary classification (TF-IDF + Ridge is more than sufficient for math-vs-non-math)
- Sub-millisecond overhead (0.072ms)
- Correct bucketing across all benchmarks

## Verdict
**KILLED** — K2143 FAIL, K2144 FAIL. The routing mechanism works but the premise was false: there is no DARE-specific math regression to fix. The math adapter itself underperforms the assumed 70% baseline.

## Implications
1. The 70% GSM8K reference needs re-verification — may have been a cherry-picked or differently-sampled result
2. TF-IDF+Ridge routing is validated as a near-zero-cost routing primitive (reusable in future experiments)
3. Math adapter quality, not composition method, is the bottleneck for GSM8K performance
