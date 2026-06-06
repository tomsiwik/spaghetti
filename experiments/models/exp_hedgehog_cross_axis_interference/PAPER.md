# PAPER.md — exp_hedgehog_cross_axis_interference

**Verdict:** KILLED (preempt-structural, F#669-family clause — 16th F#669 reuse; compound F#666+F#669)
**Date:** 2026-04-25 · drain-window iter ~52 · researcher hat
**Disposition:** No measurement performed; preempt-KILL with diagnostic results.json.

## §1 Hypothesis (verbatim)

Tests axis independence: if the polite adapter affects refactoring quality,
it's encoding information (not routing knowledge). Important for composition
thesis.

## §2 Prediction-vs-measurement table

| KC    | Pre-registered prediction                                             | Measurement | Outcome |
|-------|-----------------------------------------------------------------------|-------------|---------|
| K#1859 | Polite adapter changes refactor-quality score > 3pp on refactor-only prompts | UNTESTED-PREEMPT | fail (untested-preempt) |

**No measurement performed.** Preempt-KILL is filed before any model load.

## §3 Why the experiment was preempt-KILLed (executive summary)

Two structural blockers fire simultaneously (compound):

1. **F#669 cascade (primary).** `depends_on:
   exp_hedgehog_behavior_adapter_politeness` is PROVISIONAL (Phase 0
   neutral-prompts only; K#1782-K#1785 untested; trained polite-adapter
   weights do not exist). K#1859 requires applying the trained polite adapter
   to refactor-only prompts and measuring refactor-quality delta. Without
   trained weights, K#1859 is unmeasurable by construction.

2. **F#666 schema-defect (compound).** K#1859 is the only KC, target-flavored
   but unpaired (no structural-counterpart proxy KC for triangulation). This
   entry was eligible for the F#770 ~13-entry schema-repair cohort but was
   **not** included (per F#771 audit-correction iter ~40). It remains
   schema-defective.

The compound character is what makes this a NEW sub-mechanism. F#776 / F#778
captured the **post**-F#770-repair diagnosis migration (schema repair surfaces
the underlying F#669). This entry exhibits the **pre**-F#770-repair direct
path: parent-PROVISIONAL → cascade, with the schema defect sitting alongside
unaddressed. Different diagnosis trajectory; structurally distinct from F#775
and F#777.

## §4 Findings expected from this preempt (reviewer to file)

- **F#NEW1:** F#669 16th reuse, 2nd Hedgehog-cluster F#669, 1st
  Hedgehog-cluster pre-F#770-repair compound F#666+F#669.
- **F#NEW2:** Compound F#666+F#669-pre-repair sub-form NEW sub-axis (1st
  instance). 2 more pre-repair compound instances within 90 days canonicalize
  the sub-axis.

## §5 Antipattern scan (per reviewer checklist)

- composition math: N/A
- LORA_SCALE: N/A
- shutil.copy as adapter: N/A
- hardcoded `pass: True`: carved out (kill_results = fail)
- eval template truncation: N/A
- proxy-model substitution: N/A
- KC modification post-hoc: **DID NOT OCCUR** — K#1859 byte-for-byte identical to
  2026-04-23 DB record. No relaxation, no rewording, no threshold change.

## §6 Drain diagnostic (NOT a finding — operational note)

This is the **3rd consecutive researcher-side preempt-KILL** in the drain
window:

| Iter | Experiment | Cluster | Mechanism | Findings |
|------|-----------|---------|-----------|----------|
| ~47/~48 | rank_ablation_r4_r8_r16 | Hedgehog | post-F#770 F#669 cascade | F#775+F#776 |
| ~49/~50 | jepa_scale_sweep_5m_15m_50m | JEPA | post-F#770 F#669 cascade | F#777+F#778 |
| ~52 (this) | cross_axis_interference | Hedgehog | **pre**-F#770 compound F#666+F#669 | F#NEW1+F#NEW2 |

Doom-loop self-check (researcher hat workflow §0):
- `doom_loop.py` exit=0 in this iter.
- Different cluster (alternates Hedgehog↔JEPA).
- **Different mechanism** (post-F#770 → post-F#770 → pre-F#770 compound — 3rd
  iter is structurally distinct sub-mechanism).
- Different parent (F#683 → F#682 → F#683).
- Different finding-index pair.

Per `mem-pattern-triple-fire`, the repetition is substantively distinct (each
iter advances a different sub-mechanism). NOT a doom-loop.

**Drain reality:** all remaining P≤2 open entries either (a) cascade off
PROVISIONAL parents (F#669) or (b) require >90-min macro budgets exceeding the
researcher cap. Preempt-KILL of cascade children is the **only** in-cap
progress path until the orchestrator unlocks parent _impl macro budgets.

**Recommended unblock path** (NOT performable in this iter):
- Promote `exp_hedgehog_behavior_adapter_politeness_impl` (P=1 macro, 4-6h
  budget). When it lands SUPPORTED, F#683-cluster children (cross_axis
  retroactively, pair_composition, triple_composition) become measurable.
- Same pattern for `exp_jepa_adapter_residual_stream_impl` (PROVISIONAL F#772,
  parent of jepa_scale_sweep) and `exp_rdt_loop_kv_cache_impl` (P=1 micro
  3-4h, parent of rdt_jepa_loop_adapter).

Operator action: raise researcher hat budget cap OR explicitly green-light
macro _impl claims one-at-a-time outside drain-window cap accounting.

## §7 Drain accounting (anticipated post-reviewer-formalization)

- P≤2 open queue: 11 → 10 (cross_axis killed).
- Active queue: 1 → 0 (cross_axis off active list when reviewer runs `experiment complete`).
- Finding-ledger: 39 → 41 (F#NEW1 + F#NEW2).

## §8 References

- F#669 (governing): RDT ACT halting child-preempt, ~22 days ago.
- F#666 (schema-defect): target-gated kill rule.
- F#770 / F#771: ~13-entry F#666 schema-defect cohort + audit.
- F#775 / F#776: rank_ablation kill + 1st schema-repair-reveals-F#669 obs.
- F#777 / F#778: jepa_scale_sweep kill + 2nd cross-cluster obs (Hedgehog→JEPA).
- Parent: `exp_hedgehog_behavior_adapter_politeness` (F#683-cluster, PROVISIONAL).

## §9 Hand-off

- Reviewer: write `REVIEW-adversarial.md` (preempt-KILL no-measurement template
  per F#775/F#777 pattern), file F#NEW1 (F#669 16th reuse) + F#NEW2 (compound
  F#666+F#669-pre-repair sub-form NEW sub-axis), set status killed via
  `experiment complete --status killed --k 1859:fail`, route `review.killed`.
- Files written by researcher: MATH.md, run_experiment.py, results.json,
  PAPER.md. Reviewer adds REVIEW-adversarial.md only.
- Analyst next iter: write LEARNINGS.md ratifying F#NEW1+F#NEW2, capture the
  drain-diagnostic operational note (not as a finding) as a routing
  recommendation for next claim cycle (avoid 4th consecutive preempt-KILL —
  break to a non-cascade mechanism if any P≤2 entry permits, OR escalate to
  HALT_ESCALATION addendum if all P≤2 cascade).
