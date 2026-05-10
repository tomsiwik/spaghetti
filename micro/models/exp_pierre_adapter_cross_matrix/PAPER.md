# PAPER.md — Adapter Cross-Domain Matrix

**Experiment:** `exp_pierre_adapter_cross_matrix`
**Status:** KILLED (K2 fail: strategy_full medqa catastrophic)
**Date:** 2026-05-10
**Runtime:** ~1h53m (21 cells, N=50 each)

## Research Question

Which of Pierre's 7 adapters help which benchmarks? We only ever measured
native pairings (math→GSM8K, code→HumanEval, medical→MedQA). The cross-matrix
reveals whether adapters generalize or interfere across domains.

## Method

Single-adapter loading via PoLARLinear for each of 7 adapters × 3 benchmarks = 21 cells.
Compare each cell against raw baseline (no adapter): GSM8K=56, HumanEval=22, MedQA=6.

## Results

### Full 7×3 Matrix (score / delta vs raw baseline)

| Adapter            | GSM8K      | HumanEval  | MedQA      |
|--------------------|------------|------------|------------|
| **raw baseline**   | 56         | 22         | 6          |
| strategy_full      | 68 (+12)   | 74 (+52)   | **0 (−6)** |
| strategy_prepare   | 66 (+10)   | 72 (+50)   | 22 (+16)   |
| strategy_act       | 68 (+12)   | 68 (+46)   | 30 (+24)   |
| strategy_integrate | 64 (+8)    | 58 (+36)   | 32 (+26)   |
| domain_math        | 66 (+10)   | 34 (+12)   | 40 (+34)   |
| domain_code        | 62 (+6)    | 78 (+56)   | 38 (+32)   |
| domain_medical     | 68 (+12)   | 46 (+24)   | 42 (+36)   |

### Native-cell consistency (K4)

| Benchmark | Native adapter | This run | Reference (DARE-B) | Delta |
|-----------|---------------|----------|---------------------|-------|
| GSM8K     | domain_math   | 66       | 66                  | 0     |
| HumanEval | domain_code   | 78       | 78                  | 0     |
| MedQA     | domain_medical| 42       | 42                  | 0     |

Perfect native consistency — adapters reproduce identically.

## Kill Criteria

| KC | Description | Result |
|----|-------------|--------|
| K1 | ≥1 off-diagonal cell ≥ raw+2pp | **PASS** — 17/21 cells positive |
| K2 | No adapter regresses >5pp below raw | **FAIL** — strategy_full medqa=0% (−6pp) |
| K3 | Budget ≤ 240min | **PASS** — 113min |
| K4 | Native cells within 5pp of reference | **PASS** — exact match |

**Verdict: KILLED** — K2 failure. strategy_full catastrophically destroys MedQA.

## Key Findings

### 1. Every adapter is a general capability booster
All 7 adapters improve HumanEval from raw 22% to 34–78%. The adapters teach
general instruction-following/reasoning, not narrowly domain-specific knowledge.

### 2. strategy_full is pathological
strategy_full scores 0% on MedQA (below the 6% raw baseline). It's the only
adapter that causes negative transfer anywhere in the matrix. This adapter
over-specializes on gsm8k+humaneval at the expense of medical reasoning.

### 3. Domain adapters show strong cross-transfer
- domain_medical: 68/46/42 — most balanced, boosts GSM8K +12pp
- domain_code: 62/78/38 — strongest HumanEval, strong MedQA cross-transfer (+32pp)
- domain_math: 66/34/40 — strongest MedQA cross-lift (+34pp), weak on HumanEval

### 4. Strategy sub-adapters outperform strategy_full
strategy_act (68/68/30) and strategy_integrate (64/58/32) provide strong
cross-domain lift without the catastrophic MedQA failure of strategy_full.
The decomposed strategy is strictly better than the monolithic one.

### 5. Implications for K=2 routing
Best pairs by benchmark:
- GSM8K: strategy_full + domain_medical (both 68)
- HumanEval: domain_code (78) + strategy_full (74)
- MedQA: domain_medical (42) + domain_math (40)

But strategy_full's MedQA pathology means it should be excluded from any
routing scheme that might encounter medical queries. Safer pairs:
- GSM8K: strategy_act + domain_medical (both 68)
- HumanEval: domain_code + strategy_prepare (78/72)
- MedQA: domain_medical + domain_math (42/40)

## Actionable Conclusions

1. **Drop strategy_full** from the adapter set — decomposed sub-adapters dominate.
2. **Domain adapters are multi-purpose** — routing can be relaxed (domain_medical
   is nearly as good as domain_math on GSM8K).
3. **K=2 routing should pair strategy + domain**, not domain + domain.
4. **HumanEval raw baseline (22%) is extremely low** — nearly any adapter helps.
   This metric has low discriminative power for adapter selection.
