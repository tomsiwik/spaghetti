# M2P-Gated Continuous-Weight Composition — Results

## Summary

Learned softmax gate achieves near-perfect domain classification (99.6%) but gated composition **catastrophically degrades** behavioral accuracy compared to single-adapter baselines. The composition application mechanism is broken.

## Prediction vs Measurement

| Metric | Predicted | Measured | Status |
|--------|-----------|----------|--------|
| Gated ≥ best-single (GSM8K) | ≥ 70.0% | 53.3% (-16.7pp) | FAIL |
| Gated ≥ best-single (HumanEval) | ≥ 86.7% | 20.0% (-66.7pp) | FAIL |
| Gated ≥ best-single (MedQA) | ≥ 50.0% | 6.7% (-43.3pp) | FAIL |
| Avg gate entropy ≤ 1.5 nats | ≤ 1.5 | 0.039 | PASS |
| Top-1 weight ≥ 0.5 | ≥ 0.5 | 0.993 | PASS |
| P95 latency ≤ 250ms | ≤ 250ms | 357.1ms | FAIL |
| Calibration ≥ 3pp gap | ≥ 3pp | Mixed (0, 40, -13.3) | FAIL |

## Kill Criteria Verdict

| KC | Result |
|----|--------|
| K2116 (TARGET): Gated ≥ best single on all 3 benchmarks | **FAIL** — drops of 16.7, 66.7, 43.3pp |
| K2117 (PROXY): Avg entropy ≤ 1.5 | PASS — 0.039 |
| K2118 (PROXY): Top-1 weight ≥ 0.5 | PASS — 0.993 |
| K2119 (PROXY): P95 latency ≤ 250ms | FAIL — 357.1ms |
| K2120 (PROXY): Calibration ≥ 3pp | FAIL — inconsistent |

## Verdict: KILLED

K2116 (target metric) fails catastrophically. The gate itself works perfectly — it routes to the correct adapter with near-binary confidence. The failure is entirely in the **composition application path**: monkey-patching `__call__` on PoLARLinear modules via `apply_gated_composition` destroys the model's forward pass.

## Root Cause Analysis

The gate outputs weights ≈ [0, 0, 0, 0, 1.05, 0, 0] (peaked at the correct adapter). At w≈1.05 for a single adapter, this should reproduce single-adapter behavior exactly. The 53.3% GSM8K vs 63.3% single-adapter (domain_math) shows that even when routing correctly, the patched forward produces worse outputs. This points to:

1. **`__call__` override via `__get__` doesn't persist correctly in MLX module graph** — the model may fall back to the base linear during generation
2. **The `base(x)` reference in the patched forward may not match what PoLARLinear actually computes** — the original forward might have additional logic (normalization, scaling conventions)
3. **Bucket-averaging destroys per-prompt routing precision** — though with w≈1.0 this shouldn't matter much

## Best Single-Adapter Baselines (for reference)

| Adapter | GSM8K | HumanEval | MedQA |
|---------|-------|-----------|-------|
| strategy_full | 63.3 | 83.3 | 0.0 |
| strategy_prepare | 66.7 | 76.7 | 16.7 |
| strategy_act | 63.3 | 70.0 | 26.7 |
| strategy_integrate | 63.3 | 63.3 | 26.7 |
| domain_math | 63.3 | 46.7 | 33.3 |
| domain_code | 60.0 | 86.7 | 33.3 |
| domain_medical | 70.0 | 53.3 | 50.0 |
| **Best single** | **70.0** | **86.7** | **50.0** |

## Key Finding

The M2P gate architecture (2-layer MLP on mean-pooled embeddings) learns domain routing to near-perfect accuracy with minimal entropy — the routing problem is solved. The unsolved problem is **applying** the routed composition without degrading the model's forward pass. This is an implementation/framework issue, not a theoretical one.

## Implications for Follow-ups

- The gate design is validated — reuse in future composition experiments
- Fix composition application: either (a) proper PoLAR weight injection instead of __call__ override, or (b) pre-merge weights before model load
- Latency budget needs 100ms trimmed — suggests gate should be cached or the composition pre-computed for repeated queries
