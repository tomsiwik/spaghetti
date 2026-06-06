# MATH.md — K=3 domain-only composition (knowledge stack interference test)

## Hypothesis

Pierre's 3 domain adapters (`domain_math`, `domain_code`, `domain_medical`)
cover **different knowledge bases** and should have minimal overlap. If
K=3-domain composition preserves per-domain accuracy, it's the cleanest
multi-task composition test we can run.

> **Does K=3 domain-only composition preserve per-domain accuracy?**

This is the simplest possible composition test: orthogonal knowledge
domains, no strategy mixing, K small enough that interference should
be minimal.

## Adapter set

K=3 = `domain_math + domain_code + domain_medical`.

## Pre-registered Kill Criteria

- **K1** Each per-domain benchmark score ≥ corresponding single-adapter score − 2pp:
  - GSM8K ≥ domain_math single − 2pp
  - HumanEval ≥ domain_code single − 2pp
  - MedQA ≥ domain_medical single − 2pp
- **K2** Aggregate avg ≥ best-single-adapter-per-bench avg (62.0% from `exp_pierre_dare_b_vs_fisher_rao`).
- **K3** Preprocessing ≤ 5s.
- **K4** K=3-domain avg vs K=3-strategy avg (sibling) — comparison tells us which axis is more compose-friendly.

## Why this matters

If domain composition works cleanly at K=3 but strategy composition
fragments at K=3-strategy, that's strong evidence Pierre's adapter
training should focus on **domain expansion** rather than **strategy
expansion** — fewer strategies, more domains.

## References

- Sibling: `exp_pierre_compose_k3_strategy_only`
- Prior single-adapter measurements: `exp_pierre_dare_b_vs_fisher_rao` single_best (math 66, code 78, medical 42)
