# MATH.md — K=3 strategy-only composition (skill stack interference test)

## Hypothesis

Pierre has 4 strategy adapters: `strategy_full`, `strategy_prepare`,
`strategy_act`, `strategy_integrate`. These are conceptually
**complementary skills** (not competing tasks) — preparing, acting,
integrating, and full are different reasoning modes.

> **Does composing 3 strategies (without any domain adapter) preserve
> generic reasoning quality? Or do strategy adapters interfere with
> each other?**

If strategies interfere, that's evidence the K=7 default is overloaded
on the strategy axis — and Pierre's product story should ship K=2
(strategy + domain) not K=7 default.

## Adapter set

K=3 = `strategy_full + strategy_prepare + strategy_act` (drop `strategy_integrate` — keeps 3 to limit interference space).

## Pre-registered Kill Criteria

- **K1** Fisher-Rao K=3-strategy avg ≥ Fisher-Rao K=7 avg (64.7%). Strategies should at minimum match K=7 mixed.
- **K2** No benchmark drops below `strategy_full` single adapter by >3pp (no destructive interference within strategy axis).
- **K3** Preprocessing ≤ 5s.
- **K4** Cross-comparison: K=3-strategy avg vs K=3-domain avg (sibling experiment) — tells us whether strategy-axis or domain-axis is the lower-interference composition.

## References

- Sibling: `exp_pierre_compose_k3_domain_only`
- Prior K=7: `exp_pierre_dare_b_vs_fisher_rao`
