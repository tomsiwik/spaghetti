# MATH.md — Phase 1 E2E viability re-run (false-killed in original)

## Hypothesis

The original `exp_pierre_phase1_e2e_viability` was a false-kill per Finding
#831 audit (research agent `afdac038d5737221a`). Original numbers
(53.3/20.0/6.7) match the exact bug-fingerprint — instance-level
`m.__call__ = ...` pattern that MLX silently bypasses.

This experiment **re-runs the Phase 1 product configuration** with the
canonical `_FusedDeltaLinear` pattern via our shared `eval_runner.py`.

> **Does Pierre's actual Phase 1 product config (3 strategy + 3 domain
> adapters, K=6) preserve per-adapter capability under composition with
> the corrected infrastructure?**

## Adapter set

K=6 = 3 strategy + 3 domain (matching original Phase 1 spec):
- `strategy_full`, `strategy_prepare`, `strategy_act`
- `domain_math`, `domain_code`, `domain_medical`

(Drops `strategy_integrate` — original Phase 1 was 3+3.)

## Pre-registered Kill Criteria

- **K1** Fisher-Rao K=6 avg ≥ Fisher-Rao K=7 avg − 1pp (close to K=7 default).
- **K2** Each per-domain benchmark ≥ corresponding single-adapter score − 3pp.
- **K3** Preprocessing ≤ 5s.
- **K4** Compares to original killed numbers (53.3/20.0/6.7) — re-run should **dramatically exceed** these. PASS = no benchmark drops below 40% (the original collapsed to <20% on 2 of 3).

## Why this matters

Direct test: does Pierre's documented product config — the one users would
deploy — actually work? The original kill called this an "E2E viability"
gate. With Finding #831 fix, we revisit the gate cleanly.

## References

- False-killed original: `exp_pierre_phase1_e2e_viability`
- Audit: research agent `afdac038d5737221a`
- Sibling: `exp_pierre_compose_k3_strategy_only`, `exp_pierre_compose_k3_domain_only`
