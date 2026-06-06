# K=2 Strategy x Domain Composition — Pierre's Headline Product Config

## Abstract

Tests Pierre's product story: one strategy adapter (how to think) + one domain adapter (what to know) composed via Fisher-Rao at K=2. Measures 3 pairs against K=7 Fisher-Rao baseline (64.7%). Result: **KILLED**. K=2 avg=54.0%, 10.7pp below K=7. Strategy adapter destructively interferes with code specialization (-12pp humaneval).

## Method

Three K=2 pairs tested sequentially:
- `strategy_full + domain_math` (emphasis: GSM8K)
- `strategy_full + domain_code` (emphasis: HumanEval)
- `strategy_full + domain_medical` (emphasis: MedQA)

Each pair runs the 4-method matrix: single_best (domain alone), Fisher-Rao K=2, method-under-test (=Fisher-Rao, trivially identical), DARE full-delta K=2. N=50 per benchmark, seed=42.

## Results

### Per-pair breakdown

| Pair | Method | GSM8K | HumanEval | MedQA | Avg |
|------|--------|-------|-----------|-------|-----|
| strategy+math | single_best | 66% | - | - | 66.0 |
| strategy+math | FR K=2 | 70% | 70% | 12% | 50.7 |
| strategy+math | DARE K=2 | 64% | 56% | 28% | 49.3 |
| strategy+code | single_best | - | 78% | - | 78.0 |
| strategy+code | FR K=2 | 62% | 66% | 44% | 57.3 |
| strategy+code | DARE K=2 | 48% | 56% | 24% | 42.7 |
| strategy+med | single_best | - | - | 42% | 42.0 |
| strategy+med | FR K=2 | 64% | 58% | 40% | 54.0 |
| strategy+med | DARE K=2 | 74% | 62% | 50% | 62.0 |

### Matched-benchmark comparison

| Pair | Domain alone | K=2 FR | Delta | K=7 FR ref |
|------|-------------|--------|-------|------------|
| math → GSM8K | 66% | 70% | +4pp | 68% |
| code → HumanEval | 78% | 66% | **-12pp** | 68% |
| medical → MedQA | 42% | 40% | -2pp | 58% |
| **Average** | **62.0%** | **58.7%** | **-3.3pp** | **64.7%** |

### Aggregate

K=2 Fisher-Rao avg (all benchmarks, all pairs): **54.0%**
K=7 Fisher-Rao reference: **64.7%**
Gap: **-10.7pp**

## Kill Criteria Evaluation

| KC | Criterion | Result | Verdict |
|----|-----------|--------|---------|
| K1 | K=2 avg ≥ 64.7% (K=7 ref) | 54.0% | **FAIL** |
| K2 | K=2 ≥ best-single + 2pp on matched bench | code: -12pp | **FAIL** |
| K3 | No pair drops >2pp below domain-alone | code: -12pp | **FAIL** |
| K4 | Preprocessing ≤ 5s | 0.01s | PASS |

**Verdict: KILLED** (K1, K2, K3 all FAIL)

## Key Findings

1. **Strategy adapter destructively interferes with code**: domain_code alone=78% HumanEval, composed with strategy_full=66% (-12pp). The strategy adapter overwrites code-specific features.

2. **K=7 > K=2**: The additional adapters in K=7 compensate for strategy interference. K=7 FR=64.7% vs K=2 FR=54.0% avg. The "less interference at K=2" hypothesis is wrong — more adapters help, not hurt.

3. **DARE worse than Fisher-Rao at K=2**: Stochastic pruning (drop_rate=0.9) is too aggressive with only 2 adapters. Exception: strategy+medical DARE=62% beats FR=54%, suggesting DARE's pruning helps remove strategy interference for medical.

4. **Strategy helps math but hurts code**: +4pp GSM8K, -12pp HumanEval. Strategy adapter's "think step by step" behavior helps structured reasoning but disrupts code generation fluency.

5. **Product story invalidated**: "One strategy + one domain" is not Pierre's shipping config. K=7 Fisher-Rao remains the correct default. Domain-alone routing may be better for code tasks.
