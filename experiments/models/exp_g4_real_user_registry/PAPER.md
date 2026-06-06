# PAPER — exp_g4_real_user_registry

## Verdict: KILLED_PREEMPTIVE (5-theorem stack, defense-in-depth)

Registry re-run claim on "real heterogeneous Gemma 4 adapters" is
unreachable a priori: inventory lacks user adapters, SC gate missing,
KC unpinned, and F#454 scope caveat makes K1615 phase-ambiguous.

## Kill criteria outcome (from results.json)

| KC    | Threshold         | Outcome                                     |
|-------|-------------------|---------------------------------------------|
| K1613 | register < 10ms   | fail (unreachable: T1 ∧ T3 ∧ T4)            |
| K1614 | crystallize < 5ms | fail (unreachable: T1 ∧ T3 ∧ T4)            |
| K1615 | max_cos < 0.15    | fail (unreachable: T5 phase-ambiguous)      |

## 5-theorem evidence

- **T1 Inventory**: T2.1 adapters `{code, math, medical}` — 0 user
  adapters on Gemma 4; shortfall = 2 (heterogeneous ⇒ ≥2 users).
- **T2 Budget**: 2 users × 20.92 min = 41.84 min > 30 min iter budget.
- **T3 SC gate**: DB literal `Success Criteria: NONE` — SUPPORTED path
  ungated per PLAN.md §1.
- **T4 KC pinning**: 0/5 required keywords {hardware, rank, phase,
  epsilon, heterogeneity}. K1615 especially under-specified —
  pre- vs post-crystallization phase unspecified.
- **T5 Non-transfer**: F#454 caveat LITERAL: "Kill thresholds
  1000-30M× above measured values (non-discriminating). K1136
  'throughout' tested on final state only; intermediate
  max_cos=0.9580 when user variants coexist before crystallization."
  K1615's 0.15 threshold is trivially met post-crystallization but
  violated by 0.96 during coexistence. Phase-ambiguous KC ⇒ non-
  falsifiable.

Defense-in-depth: T1 ∨ T3 ∨ T4 ∨ T5 each alone blocks SUPPORTED; T2
reinforces. Five independent orthogonal blocks.

## Comparison with prior art

F#454 (SUPPORTED, 2026-04-11, exp_p1_t6_dynamic_adapter_registry):
register 1.20ms, crystallize 1.85ms, max_cos_post=0.1221 — all pass
by 3-4 orders of magnitude. The prior finding itself flagged
non-discriminating thresholds and phase ambiguity — this re-run
does nothing to address either caveat; it only shifts the substrate
from a prior BitNet/P1-era run to Gemma 4 while leaving the same
semantic gaps in K1613/K1614/K1615.

## Cohort context

21st preemptive-kill in the audit-2026-04-17/g4-gemma4 drain (20th
composition-bug branch, 1st scale-safety branch). Reinforces:

- ap-017 (composition-bug): count 20 → 21
- ap-framework-incomplete (SC=[])
- ap-scale-misclassified (micro scope on real Gemma 4 claim)

Reusable preempts (a)-(g) from ap-017 registry not directly used —
F#454 SUPPORTED caveat provides its own reusable preempt (h) for
future registry-operation claims.

## Runner notes (ap-027 N/A)

Runner is pure-stdlib (pathlib + subprocess + json + re). No MLX/torch,
no GPU, 1s wall. `experiment run` via pueue (mandatory per reference
card); no bare `uv run python`. DB mutation only via
`experiment complete`.

## Non-blocking runner gap (cohort-wide)

T4 keyword check still does raw substring. Cohort patch owed:
require enumerated-domain regex `\{[A-Za-z_]+(,\s*[A-Za-z_]+){2,}\}`
or numeric epsilon pattern `epsilon\s*=\s*[0-9.e-]+`, rather than
raw word scan. Not this experiment's blocker — T1 ∧ T3 ∧ T5 already
each alone block SUPPORTED.
