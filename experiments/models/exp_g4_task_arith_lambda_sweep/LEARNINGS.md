# LEARNINGS — exp_g4_task_arith_lambda_sweep

**Verdict:** KILLED_PREEMPTIVE (5-theorem stack, defense-in-depth)
**Cohort:** audit-2026-04-17 / composition-bug / g4-gemma4
**Position:** 18th cohort preempt this session (17th composition-bug, 1 scale-safety)

## Core Finding
K1608 ("lambda=0.5 task-arithmetic best within 2pp on MMLU-Pro, N=5")
is not falsifiable on the current repo state. Three independent blockers —
adapter inventory shortfall (3/5), success_criteria=[], KC under-specification
(0/5 adjudicatable keywords) — each alone prevents SUPPORTED. Two reinforcers
(iter-budget 41.84 min > 30 min; F#164 BitNet-2B MLP PPL non-transfer to Gemma 4
v_proj MMLU-Pro) add depth.

## Why
F#164 self-caveat literally states "lambda > 0.5 not tested" yet K1608 sweeps
λ ∈ {0.67, 1.0}; K1608 also swaps base (BitNet-2B MLP → Gemma 4 v_proj) and
metric (PPL → MMLU-Pro task accuracy where r≈0.08). Cited-finding structurally
cannot support K1608 as stated. Operator unblock (SC add + 2 domain adapters
OR N=3 re-scope + K1608 pinning) is the only path; no hyperparameter or code
tweak unblocks.

## Implications for Next Experiment
1. **Drop audit-2026-04-17 cohort claims entirely** until P11.ADAPTER-REBUILD
   + SC pinning lands operator-side. Any cohort claim that returns drains via
   the same 5-theorem stack — no learning remaining.
2. **F#164 non-transfer one-liner** is registered as ap-017 reusable preempt (e)
   alongside F#306/F#13-14/F#44/F#45.
3. **Pivot candidates (non-cohort P≤2):** exp_g4_polar_scale_invariance,
   exp_g4_single_domain_vproj_think, exp_g4_activation_bounds_vproj. If P≤2
   exhausts → RESEARCH_BACKLOG_DRAINED per objective success criteria.
4. **Non-blocking runner patch owed:** T5 keyword check should scan the
   referenced experiment's MATH.md for base-model scope, not the condensed
   finding-get summary. Cosmetic; does not affect current verdict.
