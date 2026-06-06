# LEARNINGS — exp_g4_real_user_registry

## Core Finding
KILLED_PREEMPTIVE (5-theorem defense-in-depth). 21st cohort preempt
(20th composition-bug branch) in audit-2026-04-17/g4-gemma4. T1 ∨ T3 ∨
T4 ∨ T5 each alone blocks; T2 reinforces.

## Why
- T1: 0 user adapters on Gemma 4 ({code,math,medical} domain-not-user);
  shortfall=2.
- T3: DB literal `Success Criteria: NONE` — SUPPORTED ungated.
- T4: 0/5 keywords {hardware, rank, phase, epsilon, heterogeneity};
  K1615 phase-ambiguous (pre- vs post-crystallization unpinned).
- T5: F#454 LITERAL caveat "intermediate max_cos=0.9580 when user
  variants coexist before crystallization" — K1615 (0.15) trivially
  met post-crystallization but violated (0.96) during coexistence.
  Phase-ambiguous KC ⇒ non-falsifiable a priori.

## Implications for Next Experiment
- F#454 registered as reusable preempt (h) under ap-017 (2nd
  SUPPORTED-source preempt after F#505 g): source verdict is not
  the gate; scope-caveat-literal is. Re-runs need phase-pinned KC.
- Pivot OFF cohort. P≤2 non-cohort: exp_g4_polar_scale_invariance,
  exp_g4_single_domain_vproj_think, exp_g4_activation_bounds_vproj,
  P0 macro. If cohort claim returns, drain via 5-theorem stack
  (preempts a-h registered). Else → RESEARCH_BACKLOG_DRAINED.
- Operator unblock only cohort accelerator (SC + user adapters +
  phase-pinned K1615 + N≤3 re-scope).
- Runner patch owed cohort-wide: T4 enumerated-domain regex or ε.
