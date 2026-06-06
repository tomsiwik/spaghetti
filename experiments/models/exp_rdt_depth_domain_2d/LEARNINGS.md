# exp_rdt_depth_domain_2d — LEARNINGS.md

## Outcome
Preemptive-kill. 2nd reuse of F#669 (preempt-child-KCs-require-parent-target-claim-unverified).

## What we learned

1. **F#669 now has two precedents** (exp_rdt_act_halting_throughput and this).
   One more occurrence warrants promotion from sub-axis to standalone finding.

2. **Double-parent-unfulfilled is a legitimate variant**. Previous precedent
   had a single smoke-provisional parent; here both parents fail independently
   (one smoke-only, one Phase-1 aborted). Same rule applies: if any KC requires
   trained artifacts from any unfulfilled parent, it cannot be measured.

3. **K1752 was already dead on arrival** — F#571 supersedes Room Model for N>1
   regardless of parent state. Future 2D-composition experiments should drop
   Room Model identity clauses by default.

4. **Unblock cost is macro-compute-bound**. The parent `exp_rdt_loop_lora_gemma4_full`
   follow-up is a real macro training run (GSM8K+MATH, T=1..6, full eval).
   That's outside the P≤2 drain budget — this branch of research is paused
   until the follow-up is scheduled.

## Follow-up tickets (unchanged from parent)

- `exp_rdt_loop_lora_gemma4_full` (macro P1): real task training + eval.
- `exp_method_composition_k_saturation v2` (macro P1): structural fix to
  Phase-1 teacher gate.
- `exp_rdt_depth_domain_2d_v2` (macro P2): can re-open once both parents resolved.

## State after this iteration

- P≤2 open reduces from 3 → 2 (rdt_depth_domain_2d drained).
- Remaining: `exp_p9_cispo_adapter_rl` (P2 macro, dep-floor on RL infra),
  `exp_p9_self_evolution_scaffold` (P2 macro, dep-floor on RL infra).
- Both p9 are likely terminal structural floor — cannot drain without external
  RL infrastructure / multi-iteration compute budget.
