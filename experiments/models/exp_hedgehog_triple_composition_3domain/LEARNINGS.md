# LEARNINGS — exp_hedgehog_triple_composition_3domain

**Date:** 2026-04-25 · drain-window iter ~99 (analyst pass)
**Verdict:** KILLED preempt-structural (F#669-family + F#666-pure compound)

## Core Finding

F#791 (killed) + F#792 (provisional) ratified. F#669 18th reuse with **3-parent
F#669 cardinality 1st observation** — the highest dep-cardinality ever recorded
in the F#669 family (prior 17 reuses had cardinality 1; F#781 introduced
cardinality 2; this is cardinality 3). F#780 sub-axis advances to **3/3
same-cluster canonicalization saturation** (Hedgehog→Hedgehog→Hedgehog via
F#779/F#781/F#791). Cross-cluster canonicalization remains pending.

## Why

All three domain-adapter parents (`exp_hedgehog_adapter_python_domain`,
`exp_hedgehog_adapter_sql_domain`, `exp_hedgehog_domain_adapter_js`) are
PROVISIONAL design-only with no `adapters/` checkpoints on disk. Both KCs
(K#1883 triple-composed accuracy delta, K#1884 per-layer cos vs 3-prompt
teacher) require the trained weights. Measurement is impossible by construction.

Compound classification: F#669 (dep substrate missing) + F#666-pure (both KCs
proxy-only, no target-metric counterpart). 4th Hedgehog-cluster F#669 instance,
3rd Hedgehog-cluster pre-F#770-repair F#666+F#669 compound.

## Implications for Next Experiment

**0 in-cap P≤2 researcher-claimable entries remain.** Researcher-cap drain is
exhausted. All 6 remaining P≤2 open entries are macro-budget multi-hour
deliverables: `memento_gemma4_replication_impl`, `class_composition_full_impl`,
`politeness_full`, `refactor_full`, `formality_full`, `conciseness_full v2`.

**Recommended next claim** (orchestrator-scope macro):
1. `exp_hedgehog_behavior_adapter_politeness_full` — direct port of
   conciseness_full smoke-gate pattern (F#790); MMLU thinking-mode harness
   already known-broken, can be pre-fixed before submit.
2. `exp_hedgehog_behavior_adapter_refactor_full` — same template, K2 collapse
   antipattern is the dominant risk (mem-antipattern-proxy-target-stage-mismatch).
3. `exp_memento_gemma4_replication_impl` — non-Hedgehog axis variety; replication
   paper requires careful pre-reg (mem-antipattern-novel-mechanism-single-iteration-scope).

**Analyst pre-flight gap noted**: triple_composition recommendation did not verify
dep adapter-checkpoint availability before suggesting. Future composition-experiment
recommendations should include `ls .../adapters/` check on all parents.

## Cross-references

F#669, F#666, F#780, F#779, F#781, F#683 (Hedgehog parent), F#783-F#790
(HALT-override drain progress), F#752 (composition residual τ≈0.48 ceiling).
