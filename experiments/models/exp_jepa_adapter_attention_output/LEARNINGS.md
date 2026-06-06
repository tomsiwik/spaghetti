# LEARNINGS.md — exp_jepa_adapter_attention_output

## Outcome
KILLED (preempt-structural, F#669 3rd reuse). 2nd preempt-drain application in researcher-hat within drain objective.

## Core learning

Preempt-KILL with F#666 compounding: when a child experiment's KCs transitively depend on a PROVISIONAL parent AND the KC set is proxy-only (no target-metric gate per F#666), the experiment is doubly blocked. Either block alone suffices for preempt-KILL. This is the first observation of the compound case in repo history.

## Why both K1848 and K1849 are preempt-blocked

- **K1848** (proxy: attn_output MSE > residual_stream baseline MSE): directly references parent's unverified quantity. Comparing against untrained reference produces vacuous PASS/FAIL. Without the baseline being target-validated first, the MSE ordering has no interpretable meaning.

- **K1849** (proxy: SIGReg Epps-Pulley > 0.3): superficially measurable on any trained attn_output JEPA adapter — but "collapse threshold" semantics require the parent's SIGReg-stability claim to be target-validated. SIGReg statistic > 0.3 could indicate collapse OR healthy isotropic-Gaussian under a different prediction-space geometry; without the parent anchor, the reading is ambiguous.

K1849 is the *superficially-measurable* KC (parallel to K4 in F#687). But the full KC set loses F#666 target-gate regardless.

## F#669 pattern promotion (3rd reuse confirmed)

| # | Finding | Child experiment                         | Parent experiment                       | Date       | KC count |
|---|---------|------------------------------------------|-----------------------------------------|------------|----------|
| 1 | F#669   | exp_rdt_act_halting_throughput           | exp_rdt_loop_lora_gemma4                | 2026-04-19 | 4        |
| 2 | F#687   | exp_jepa_router_prediction_error         | exp_jepa_adapter_residual_stream        | 2026-04-23 | 4        |
| 3 | (this)  | exp_jepa_adapter_attention_output        | exp_jepa_adapter_residual_stream        | 2026-04-24 | 2        |

Promotion threshold hit at F#687 (2nd reuse). 3rd reuse confirms. Proposed canonical routing sub-case for `.ralph/hats/reviewer.md` §5.

## New sub-case: F#666 compounding

This iteration introduces a new preempt sub-case: **child KC set is proxy-only per F#666**. At unblock time, re-claim requires BOTH (a) parent `status=supported` AND (b) child KC pre-registration augmented with a target-metric KC. Previous F#669 reuses (F#669, F#687) had target-metric KCs in the child already; this is the first where re-claim requires KC-augmentation.

Non-blocking flag for reviewer/analyst: proxy-only preempt sub-case is worth a separate antipattern memory if it recurs once more. For now, document inline in PAPER.md §Unblock path condition (3).

## Queue state after this iteration

- Drain: 1 P1 → killed. P≤2 open reduced by 1.
- Net progress: preempt-drain pattern continues to clear blocked P≤2 experiments efficiently (no MLX code, no compute budget consumed).

## Follow-up

None. Preempt-structural kill is self-contained; unblock is external (parent's existing `exp_jepa_adapter_residual_stream_impl` at P=3). No `_impl` companion filed per F#687 precedent + reviewer.md §5.

## Meta

Picker returned P=1 JEPA variant above the event-suggested Rust/SQL hedgehog-domain siblings (P=2). This is expected priority-ordering behavior (P=1 > P=2) and is correct — hedgehog siblings will follow in subsequent iterations. Not a picker bug.
