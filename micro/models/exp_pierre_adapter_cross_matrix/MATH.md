# MATH.md — Adapter cross-domain matrix (which adapter helps which benchmark?)

## Hypothesis

Pierre has 7 adapters but we've only ever measured each on its "native"
benchmark (math adapter on GSM8K, code on HumanEval, medical on MedQA).

What about cross-effects? Does:
- The medical adapter help GSM8K (medical reasoning trains chain-of-thought)?
- The code adapter help MedQA (structured logic)?
- A strategy adapter help any benchmark in particular?

This is a 7×3 matrix of single-adapter effects. Filling it cleanly tells
us:
1. Which adapter pairings would be most synergistic for K=2 routing.
2. Whether any "domain" adapter is actually multi-purpose.
3. Where strategy adapters add value vs raw base.

> **Build the 7-adapter × 3-benchmark cross-matrix at N=50. Identify
> useful cross-effects.**

## Method

For each (adapter × benchmark) cell:
- Load that single adapter into PoLARLinear (its native A, native B).
- Run the corresponding benchmark eval.
- Record cell value.

Compare against:
- Raw base (no adapter): the floor.
- Best single adapter on that benchmark: the per-bench oracle.

## Pre-registered Kill Criteria

- **K1 (DISCOVERY)** At least one off-diagonal cell (e.g., medical on GSM8K) ≥ raw + 2pp. Discovery: that adapter generalizes beyond its training domain.
- **K2 (NEGATIVE TRANSFER)** No adapter regresses below raw by >5pp on any benchmark (no destructive cross-domain effects).
- **K3 (BUDGET)** Full 7×3 = 21 evals at N=50 ≈ 3-4h. Budget cap: 240 min total.
- **K4 (CONSISTENCY)** Native-cell scores (math on GSM8K etc.) within 5pp of measurements in `exp_pierre_dare_b_vs_fisher_rao` single_best (66/78/42 reference).

## Why this matters

Direct input for **K=2 routing strategy**. Currently we route math→domain_math etc., but if domain_medical also helps math, the routing could differ. Also informs **adapter pruning** — if any adapter is universally useless, drop it from the mix.

## Output

A 7×3 matrix (7 adapters × 3 benchmarks) plus diagonal-vs-off-diagonal
analysis. Format:
```
                    GSM8K  HumanEval  MedQA
strategy_full       __     __         __
strategy_prepare    __     __         __
strategy_act        __     __         __
strategy_integrate  __     __         __
domain_math         __ *   __         __
domain_code         __     __ *       __
domain_medical      __     __         __ *
                    (* = native cell)
```

## References

- Sibling: `exp_pierre_compose_k2_strategy_x_domain` (uses cross-matrix to inform pair choice)
- Prior: `exp_pierre_dare_b_vs_fisher_rao` (single_best diagonal: 66 / 78 / 42)
