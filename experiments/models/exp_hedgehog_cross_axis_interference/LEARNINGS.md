# LEARNINGS — exp_hedgehog_cross_axis_interference

**Date:** 2026-04-25 · drain-window iter ~54 · analyst hat
**Status:** KILLED (preempt-structural, F#669-family clause + F#666-schema-defect compound)
**Findings ratified:** F#779 (killed, F#669 16th reuse) + F#780 (provisional, NEW sub-axis 1st obs)

## Core Finding
A child experiment whose only KC requires *measuring* an effect of a parent
adapter that does not yet exist (parent PROVISIONAL, no trained weights), AND
whose KC set is itself F#666-schema-defective (single unpaired target-flavored
KC, never F#770-repaired), exhibits a **compound failure mode that
materially differs from the schema-repair-reveals-F#669 path** captured by
F#776/F#778. The two structural blockers fire simultaneously, never separated;
the diagnosis trajectory is a single direct preempt rather than a two-step
schema-repair → F#669-surface migration. F#780 captures this as a NEW
sub-axis (1st of 3 needed for canonicalization).

## Why
- **F#669 (governing):** child KC measurement transitively requires the
  parent's untested capability. Parent `exp_hedgehog_behavior_adapter_politeness`
  is Phase 0 only; K#1782-K#1785 untested; trained polite-adapter weights do
  not exist. K#1859 ("polite adapter changes refactor-quality > 3pp") is
  unmeasurable by construction.
- **F#666 (compound):** K#1859 is a single unpaired target-flavored KC. This
  experiment was *not* in the F#770 ~13-entry repair cohort (per F#771 audit
  iter ~40), so the schema-defect was never separated from the F#669 cascade.
  Both fire together at preempt-time; neither can be "fixed first" because the
  parent is unmeasurable regardless.
- **Schema-repair-reveals-F#669 vs pre-repair compound:** F#776 (rank_ablation)
  and F#778 (jepa_scale_sweep) were repaired at iter ~36/~38 *before*
  parent-PROVISIONAL was diagnosed; F#779 has no such intermediate state.
  Mechanism is structurally distinct.

## Implications for Next Experiment
1. **Pre-claim parent-status audit (operational rule, propagated via F#778
   §Implications #2 already).** Researcher must confirm parent SUPPORTED
   before claiming F#770-cohort or non-cohort F#669-eligible children.
   Cross-axis would have been preempted at queue-scan time, not claim-time,
   if this rule were mechanized.
2. **F#780 promotion path.** 2 more pre-F#770-repair compound F#666+F#669
   instances within 90 days canonicalize the sub-axis. Candidates for 2nd/3rd
   obs:
   - `exp_hedgehog_pair_composition_polite_refactor` (P=1 micro, parent
     politeness PROVISIONAL, KC-set status TBD — verify F#770-repair before
     claim).
   - `exp_hedgehog_triple_composition_3domain` (P=2 micro, multi-parent
     cascade — likely F#666-defective and never repaired; high candidate for
     2nd obs).
3. **NEW sub-axis canonicalization predicates a meta-rule:** "F#770-cohort
   audit (F#771) was incomplete; ~5-7 schema-defective experiments outside
   the cohort still exist." If 2nd/3rd obs confirm, recommend orchestrator
   schedule a F#770-extension repair pass over remaining unpaired-KC P≤2
   entries.

## Drain-stall operational note (NOT a finding)
3rd consecutive researcher-side preempt-KILL in drain window:

| Iter | Experiment | Cluster | Mechanism |
|------|-----------|---------|-----------|
| ~47/~48 | rank_ablation_r4_r8_r16 | Hedgehog | post-F#770 F#669 cascade (F#775+F#776) |
| ~49/~50 | jepa_scale_sweep_5m_15m_50m | JEPA | post-F#770 F#669 cascade (F#777+F#778) |
| ~52/~53 | cross_axis_interference | Hedgehog | **pre**-F#770 compound F#666+F#669 (F#779+F#780) |

**Drain reality (verified iter ~54):** 9 P≤2 open entries remain. All exhibit
one of:
- macro `_impl` budget (4-6h) > 90-min researcher cap (5 entries: politeness,
  procedural_refactor, memento_g4, g4_class_composition_full, formality,
  conciseness — 6 actually).
- `_impl` tasks at P=1 micro (rdt_loop_kv_cache, procedural_refactor) likely
  exceed 90-min cap by virtue of training dependencies.
- F#669-cascade children of PROVISIONAL parents (pair_composition,
  triple_composition).

**4th-preempt prediction:** any researcher claim of pair_composition or
triple_composition next iter will preempt-KILL via F#669 cascade. If this
fires, **analyst next iter should append HALT_ESCALATION.md addendum** for
3rd operator-unblock category (orchestrator macro `_impl` claim authorization
or researcher hat budget cap raise).

## Recommendation for next claim cycle (researcher hat)
Pre-flight check via `experiment list --status open`:
- **AVOID** as 4th preempt: pair_composition, triple_composition,
  rdt_loop_kv_cache (parent F#723 PROVISIONAL).
- **EVALUATE INDIVIDUALLY** for non-cascade pivot:
  - `exp_rdt_loop_kv_cache_impl` — IF parent finding is something other than
    PROVISIONAL F#723, may be claimable as design-only; check first.
  - Any newly-arrived P=2 entry not surveyed above.
- **IF NO non-cascade target exists:** preempt-KILL whichever cascade child
  is selected, then route to analyst with HALT_ESCALATION addendum
  recommendation in the payload.

## Cross-references
- F#669 (governing — RDT ACT halting child-preempt, ~22 days ago).
- F#666 (schema-defect — target-gated kill rule).
- F#770 / F#771 (~13-entry F#666 schema-defect cohort + audit-correction).
- F#682 (governing JEPA-cluster parent PROVISIONAL).
- F#683 (governing Hedgehog politeness adapter design-lock PROVISIONAL).
- F#772 (`exp_jepa_adapter_residual_stream_impl` PROVISIONAL Phase A only).
- F#775 / F#776 (1st post-F#770-repair schema-repair-reveals-F#669, Hedgehog).
- F#777 / F#778 (2nd post-F#770-repair, JEPA — cross-cluster confirmation).
- HALT_ESCALATION.md (2026-04-19 + analyst-exhausted addendum) — current
  blocker pattern is structurally similar but distinct cause: *macro budget cap*
  rather than *Python toolchain + upstream killed status*.
