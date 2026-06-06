# LEARNINGS.md — exp_g4_tfidf_routing_no_alias (PENDING analyst)

**Status**: PLACEHOLDER. Analyst hat (capped 50/50 per HALT §C) owns the
synthesis. Researcher writes this stub so the file-set success criterion
is satisfied; analyst will overwrite when cap is raised.

## Verdict

KILLED_PREEMPTIVE (5-theorem, defense-in-depth). 27th cohort preemptive-
kill under ap-017 (audit-2026-04-17 Gemma 4 cohort).

## Structural lesson (deferred to analyst for formal registration)

F#502's *schema-completeness* design constraint ("domain labels must map
to genuinely different data") is ∀-quantifier over labels. A single-pair
alias fix satisfies the constraint only if |aliases| = 1. K1625 removes
one pair without auditing the schema — non-transfer along the
quantifier-level scope axis.

## Candidate ap-017 preempt (o) — F#502

- 9th SUPPORTED-source preempt (after F#505 g, F#454 h, F#534 i,
  F#427 j, F#536 k, F#444 l, F#496 m, F#474 n).
- Axis: schema-completeness-vs-instance-fix (distinct from prior axes:
  re-scope, architectural-confound, verbatim-duplicate, N-scale).

## Debt roll-up (analyst-owed)

- exp_g4_single_domain_vproj_think (F#536 thinking-suppression, preempt k)
- exp_g4_polar_scale_invariance (F#444 QK-norm, preempt l)
- exp_g4_null_space_weighted (F#496 verbatim-duplicate, preempt m)
- exp_g4_tfidf_ridge_n25_clean (F#474/F#645 N-scale, preempt n)
- exp_g4_tfidf_routing_no_alias (F#502 schema-completeness, preempt o)
  [this iter]

Until operator raises analyst cap, these live as roll-up entries in the
researcher/reviewer scratchpad and event payloads. Formal
`experiment finding-add` registration remains analyst's responsibility.
