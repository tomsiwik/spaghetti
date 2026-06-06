# PAPER.md — exp_rdt_act_halting_throughput

## Verdict
**KILLED (preemptive, dependency-unfulfilled)**

## Summary
ACT halting on looped Gemma 4 (Universal Transformers / Graves ACT) requires
a trained loop-LoRA Gemma 4 base that exhibits depth-adaptive quality. Parent
`exp_rdt_loop_lora_gemma4` is smoke-provisional: target KCs K1740/K1741/K1742
are untested and no trained loop-LoRA artifact exists. All four child KCs
(K1745/K1746/K1747/K1748) transitively require parent's target behavioral
claim to be SUPPORTED. Preemptive-kill is the correct action.

## Prediction-vs-measurement table
| KC | Claim | Prediction (if measured now) | Measurement | Status |
|----|-------|------------------------------|-------------|--------|
| K1745 | ≥80% simple queries halt at T=1 | Vacuous (halter gradients have no signal without depth-adaptive loss landscape) | not measured | dependency-unfulfilled |
| K1746 | ≥70% hard queries use T≥3 | Cannot measure: requires ∂loss/∂T < 0 on hard queries, which is parent K1742 | not measured | dependency-unfulfilled |
| K1747 | tok/s ≥90% of base | Conditioned on halter discriminating | not measured | dependency-unfulfilled |
| K1748 | hard-query quality ≈ fixed-T=5 | Requires T=5 > T=1 on hard queries (parent K1740/K1742) | not measured | dependency-unfulfilled |

## Assumptions
- Parent's smoke-provisional state (K1740/K1741/K1742 untested) accurately
  captured in DB on 2026-04-19 (verified via `experiment get`).
- CLI status `killed` on parent is a label artifact (smoke ⇒ rule-4 ⇒
  CLI-killed); disk verdict is PROVISIONAL, not scientific kill.

## Antipattern flagged
`preempt-child-KCs-require-parent-target-claim-unverified` — new sub-axis
candidate under F#513 / F#558 dependency-chain family. Rule: when a child
experiment's KCs all transitively require the parent's *target* claim to
be SUPPORTED, but parent produced only scaffolding/smoke/provisional,
preempt the child. Reclaim after parent `_full` follow-up establishes the
target claim at full scale.

## Unblock path
1. Queue `exp_rdt_loop_lora_gemma4_full` (macro, P1 — logged in parent
   LEARNINGS.md).
2. Train loop-LoRA on real GSM8K + MATH + MMLU at full scale.
3. Fit saturating-exp in T; verify K1740 (+5pp GSM8K-Hard) and K1742
   (R² > 0.90).
4. Redesign this experiment with access to a trained loop-LoRA checkpoint.
   Then add scaffolding KCs (halter wiring, cumulative halt probability
   monotonicity) + reuse target KCs K1745-K1748.

## Precedent
- F#513 (MemoryLLM vs Online LoRA — dependency killed).
- F#558 (P11.G0 GRPO — dependency-chain block).
- F#667 (RDT LTI primitive, SUPPORTED).
- F#668 (RDT scaffolding provisional).

## Skill invocation
Not applicable — no platform code written. Per PLAN.md Part 2, `/mlx-dev`
required for MLX code; preemptive-kill path has none.
