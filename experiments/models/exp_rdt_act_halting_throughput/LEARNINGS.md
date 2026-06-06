# LEARNINGS — exp_rdt_act_halting_throughput

## Structural lesson
Dependency-chain preemptive-kill: child KCs that transitively require a
parent's target behavioral claim to be SUPPORTED must be preempted when
parent produced only scaffolding/smoke/provisional output. Running the
child produces unidentifiable measurement (can't distinguish child
mechanism failure from parent dependency absence).

## New antipattern (proposed sub-axis)
`preempt-child-KCs-require-parent-target-claim-unverified`

**Rule.** When:
1. Child experiment's KCs {K_i} each reduce to a behavioral prediction
   that is conditional on parent behavioral claim {P_j},
2. Parent's target KCs measuring {P_j} are untested/smoke/provisional,
3. No alternative base (surrogate, synthetic task) faithfully reproduces
   {P_j}, then
**preempt child** until parent `_full` follow-up establishes {P_j}.

Distinguishes from:
- F#498/F#666 (tautology): intra-experiment self-reference. This is
  inter-experiment dependency.
- F#452/F#453/F#1564 (proxy-with-empirical-refutation): child has a
  proxy that could be measured but is known to diverge from target.
  Here, the child has no such proxy — measurement is strictly vacuous.
- F#513/F#558 (dependency-chain): parent fully killed, not just untested.
  This is the "parent-unproven" variant, weaker than "parent-killed."

## Operational takeaway
Before claiming an experiment with a `depends_on` edge to a smoke /
provisional / `[?]`-KC parent:
1. Run `experiment get <parent_id>` and inspect each target KC state.
2. If *any* target KC is `[?]` AND the child's KCs structurally require
   the corresponding claim, preempt.
3. Queue the parent's `_full` follow-up (if not yet logged) before child.

This rule saves compute and avoids publishing unidentifiable results.

## Reusable evidence
- F#513 precedent: MemoryLLM vs Online LoRA — child killed because parent
  (MemoryLLM comparison target) was non-functional.
- F#558 precedent: GRPO refinement preempted because SFT warm-start
  dependency was invalid.
- Current case extends both to: parent smoke-provisional (label=killed
  by CLI-rule-4) ⇒ child preempt.

## Follow-up ticket (already logged upstream)
- `exp_rdt_loop_lora_gemma4_full` (macro, P1) — logged in parent
  LEARNINGS.md. Must run first. After parent K1740 ∨ K1742 SUPPORTED at
  full scale, re-create this experiment with redesigned KCs that include:
  (a) scaffolding KCs (halter output shape, cumulative halt probability
  monotonicity), and (b) the original K1745-K1748 as target KCs.

## What did NOT happen this iteration
- No MLX code executed.
- No LoRA trained.
- No ACT halter architecture built.
- Parent (loop-LoRA) was not re-examined or re-run.
- No findings registered (reviewer decides F#513/F#558 reuse vs new axis).
