# MATH.md — K=2 strategy×domain composition (Pierre's headline product config)

## Hypothesis

Pierre's product story is **strategy × domain** orthogonal-axis composition:
one strategy adapter (how to think) layered with one domain adapter
(what to know). This is the K=2 case — fundamentally different from our
K=7 default which mixes 4 strategies + 3 domains and may have
strategy-vs-strategy interference dominating.

> **Does K=2 strategy×domain composition (`strategy_full + domain_math`)
> preserve the per-axis specialization of each adapter?**

The DARE / Pico / TIES / ACE / OrthoMerge methods were measured at K=7.
K=2 may compose much better (less interference), letting Fisher-Rao win
where it loses at K=7.

## Adapter set

K=2 = `strategy_full + domain_math` (or `domain_code` / `domain_medical`
in further variants). For this experiment we test all three pairs:
- `strategy_full + domain_math` → measure GSM8K
- `strategy_full + domain_code` → measure HumanEval
- `strategy_full + domain_medical` → measure MedQA

Reports per-pair AND across-pair average.

## Pre-registered Kill Criteria

- **K1** Fisher-Rao K=2 avg ≥ K=7 Fisher-Rao avg (64.7% from prior). PASS = K=2 is at least as good.
- **K2** Fisher-Rao K=2 avg ≥ best-single-adapter avg + 2pp on the matching benchmark.
- **K3** No single per-pair score drops below the corresponding domain-alone score by >2pp (composition not destructive).
- **K4** Preprocessing budget ≤ 5s.

## Why this experiment matters

This is **the actual product config**. If K=2 strategy×domain composition
preserves per-axis specialization, Pierre can ship today on Fisher-Rao
without the K=7 composition gap mattering. The 4 false-killed experiments
(particularly `polar_mild_adapters_compose`) targeted exactly this case.

## References

- False-killed precedent: `exp_polar_mild_adapters_compose` (re-run as `_v2`)
- Sibling re-run: `exp_pierre_polar_mild_compose_v2`
- Prior K=7 measurement: `exp_pierre_dare_b_vs_fisher_rao`
