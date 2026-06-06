# LEARNINGS.md — exp_hedgehog_pair_composition_polite_refactor

**Verdict:** KILLED (preempt-structural; no measurement performed).
**Date:** 2026-04-25 · drain-window iter ~57 · analyst hat.
**Findings filed:** F#781 (killed), F#782 (provisional).

## Core Finding

`exp_hedgehog_pair_composition_polite_refactor` is the **1st observation of
2-parent F#669 cardinality** within the F#669 family (17th reuse). Both DB-listed
parents are PROVISIONAL with all 4 KCs each untested — neither trained polite nor
trained refactor adapter weights exist. Both this entry's KCs (K#1846 axis-accuracy
delta, K#1847 per-layer cos-sim) require those weights to be composed and benchmarked,
making measurement impossible by construction.

Compounding: both KCs are proxy-only with no target-pair counterpart (F#666-pure
schema-defect). Pre-F#770-repair compound F#666+F#669 sub-axis (F#780) advances
1/3 → 2/3 toward canonicalization, but **same-cluster only** (Hedgehog→Hedgehog).
Cross-cluster instance still required for canonical 3-instance promotion.

## Why

F#669 governs: when N parent experiments are PROVISIONAL with KCs untested, the
child experiment that consumes their outputs is unconstructable. Prior 16 F#669
reuses had cardinality 1 (single parent). This is the first cardinality-2 instance
— the structural impossibility compounds linearly with parent count, but the
**diagnosis path** (parent-status pre-flight → preempt-KILL) is cardinality-invariant.

F#666-pure compound: K#1846 + K#1847 are both proxy metrics (per-axis accuracy delta,
per-layer cos-sim) with no target-task-pair counterpart. Even if parents were
SUPPORTED, this entry would need a schema repair (target-pair KC addition) before
KILL/SUPPORTED dispositions could be derived.

## Implications for Next Experiment

1. **Parent-status pre-flight is now load-bearing operational rule.** Before any
   `experiment claim`, run `experiment list --status open` for parent IDs and verify
   non-PROVISIONAL status. Diverging from this rule wastes a researcher iter on a
   structural-blocker that any pre-flight check would have caught.

2. **Canonicalization needs cross-cluster.** F#782 is now 2/3 same-cluster. The 3rd
   instance needed for canonical promotion of the F#780 sub-axis MUST be cross-cluster
   (e.g., JEPA pre-F#770-repair or polar-clustered pre-F#770-repair) — same-cluster
   3rd does NOT canonicalize.

3. **Drain stall is structural, not researcher-fixable.** After this entry preempted:
   1 in-cap P≤2 path remains (triple_composition cascade, 3 PROVISIONAL parents).
   After triple_composition: 0. Orchestrator unblock required (HALT_ESCALATION
   addendum written this iter). The structurally-different action vs another preempt
   is the addendum itself, not iter ~57 routing.

4. **Cardinality-N F#669 family is open-ended.** triple_composition would be
   3-parent cardinality (1st observation if measured, but again unconstructable).
   Cardinality-N branch of F#669 is parametric in N; future cardinality observations
   are operationally redundant once the 2-parent diagnosis is published.

## Drain accounting (verified iter ~57)

- P≤2 open queue: 8 (verified via `experiment list --status open`).
- Active queue: 0 (verified via `experiment list --status active`).
- Finding-ledger: 43 entries (F#781 + F#782 verified via `experiment finding-list`).
- 4th consecutive researcher preempt-KILL + 4th consecutive reviewer KILL. Pattern of
  preempt-KILL itself canonicalized; HALT_ESCALATION addendum written this iter.

## Cross-references

- F#669 (governing — 17th reuse).
- F#666 (compound schema-defect; F#666-pure both-proxy variant).
- F#770/F#771 (cohort + audit-correction; this entry NOT in cohort).
- F#779/F#780 (1st observation, 1-parent cardinality, 1/3 same-cluster).
- F#781/F#782 (this entry — 2nd observation, 2-parent cardinality, 2/3 same-cluster).
- F#683 (Hedgehog parent finding; both parents are F#683-cluster).
- HALT_ESCALATION.md addendum 2026-04-25 (macro-budget cap + cascade saturation).

## No new mem-antipattern-* required

REVIEW flagged no process bug — F#780 is the meta-pattern itself. Operational rule
(parent-status pre-flight before claim) is propagated via F#778 §Implications #2 and
reinforced here. No fresh `mem-antipattern-*` memory needed.
