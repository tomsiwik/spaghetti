# LEARNINGS — exp_g4_vproj_compose_n25_clean

## Core Finding
KILLED_PREEMPTIVE — 20th audit-2026-04-17 cohort preempt, 19th composition-
bug branch. K1612 ("4/5 domains ≥ 100% quality vs solo") unreachable: T1
shortfall=22, T3 SC=[], T4 MATH-level 0/5 pinned, T5 F#505 N=5→N=25
non-transfer. Defense-in-depth: T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED.

## Why
- T1: T2.1 V3 cascade delivered 3 adapters (math/code/medical); N=25 needs 22 more.
- T3: DB `Success Criteria: NONE ⚠ INCOMPLETE`; no SUPPORTED predicate.
- T4: K1612 pins 0 of 5 fields (ε, baseline, pooled, delta-sum formula,
  domain-list). Runner cosmetic pass on `domain` in `domains`; MATH holds.
- T5: F#505 scope N=5 v_proj+o_proj, self-caveat "Solo baseline 3x lower...
  n=20 underpowered... KC miscalibrated" — magnifies at N=25.

## Implications for Next Experiment
- Pivot OFF audit-2026-04-17 cohort (6 prior handoffs, claims keep returning
  cohort members). Non-cohort P≤2: exp_g4_polar_scale_invariance,
  exp_g4_single_domain_vproj_think, exp_g4_activation_bounds_vproj. P0
  macro candidates may need operator handoff.
- Preempts (a)-(g) registered under ap-017: cohort re-claim drains via same
  5-theorem stack. F#505 novelty: first SUPPORTED-source preempt —
  source-verdict is not a guardrail, scope-caveat-literal is.
- Operator unblock (SC add + 2 new domain datasets + K pinning OR N≤3
  re-scope) remains sole cohort accelerator.
- Runner patch owed cohort-wide: T4 enumerated-domain regex or numeric ε.
