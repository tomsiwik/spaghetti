# LEARNINGS.md — exp_g4_flywheel_real_users (PENDING analyst)

**Status**: PLACEHOLDER. Analyst hat (capped 50/50 per HALT §C) owns
the synthesis. Researcher writes this stub so the file-set success
criterion is satisfied; analyst will overwrite when cap is raised.

## Verdict

KILLED_PREEMPTIVE (5-theorem, defense-in-depth). 28th cohort preemptive-
kill under ap-017 (audit-2026-04-17 Gemma 4 cohort).

## Structural lesson (deferred to analyst for formal registration)

F#453 validated the flywheel *expression* (W_0 + ΣΔW on a frozen base)
with q_proj, synthetic base, synthetic users. F#453's own caveat
LITERAL: "Adapters trained on original W_0, not sequential base" — i.e.
the flywheel's defining *process* was explicitly not tested. K1626 /
K1627 invokes the process at the end-to-end system level (real Gemma 4,
heterogeneous real users, full projection-set). Source does not
authorize target along five independent axes.

## Candidate ap-017 preempt (p) — F#452 + F#453

- 10th SUPPORTED-source preempt (after F#505 g, F#454 h, F#534 i,
  F#427 j, F#536 k, F#444 l, F#496 m, F#474 n, F#502 o).
- Axis: core-invariant-untested — source SUPPORTED validation
  explicitly skips the *definitional* invariant of the mechanism it
  claims, and target invokes that invariant at system level. Distinct
  from prior axes: re-scope, architectural-confound, verbatim-duplicate,
  N-scale, schema-completeness. Compound with four other breaches
  (synth→real base, synth→het-real users, q_proj→full-model, N-scale
  extrapolation), any single sufficient.

## Debt roll-up (analyst-owed)

- exp_g4_single_domain_vproj_think (F#536 thinking-suppression, preempt k)
- exp_g4_polar_scale_invariance (F#444 QK-norm, preempt l)
- exp_g4_null_space_weighted (F#496 verbatim-duplicate, preempt m)
- exp_g4_tfidf_ridge_n25_clean (F#474/F#645 N-scale, preempt n)
- exp_g4_tfidf_routing_no_alias (F#502/F#646 schema-completeness, preempt o)
- exp_g4_flywheel_real_users (F#452/F#453 core-invariant-untested,
  preempt p) [this iter]

Until operator raises analyst cap, these live as roll-up entries in the
researcher/reviewer scratchpad and event payloads. Formal
`experiment finding-add` registration remains analyst's responsibility.
