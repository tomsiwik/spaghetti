# LEARNINGS — exp_pierre_adapter_hotswap_latency (PROVISIONAL design-lock)

## Outcome

PROVISIONAL. KCs untested (design-lock scaffold; no MLX executed). Hygiene
fixed pre-run (references, success_criteria, platform, K1910 operational
definition). Execution deferred to `_impl` companion.

## Core learning

A runnable experiment with **legitimate target-metric KCs** but **under-specified
pre-reg** (missing references, ambiguous KC text) should NOT be preempt-killed
the way F#700/F#701 kill F#666-pure pre-regs. Correct path: patch hygiene
pre-run in MATH.md, operationalize ambiguous KCs, defer execution to `_impl`.

Preempt-KILL is reserved for **structurally impossible** pre-regs (F#666-pure:
only proxy KCs, no target-metric). Under-specified-but-legitimate pre-regs get
**design-lock PROVISIONAL + _impl**.

## Why this experiment is runnable (not preempt-KILL)

- K1909 is user-facing latency — not a proxy. Matches F#666 guardrail 1007
  target-metric definition.
- K1910 is behavioral output equivalence under same-adapter swap —
  directly observable in token stream; not a proxy.
- Prior art (`adapter_hotswap_latency` on Qwen3-0.6B) already verified
  Theorems 1+2 — mechanism is proven; this is platform transfer, not novel.

## Taxonomic row — 1st instance of "design-lock-hygiene-patch PROVISIONAL"

Distinguishing features (vs neighbouring drain rows):

| Row                                 | Pre-reg KCs         | Mechanism novelty | Hygiene defects   | Action             |
|-------------------------------------|---------------------|-------------------|-------------------|--------------------|
| F#669-family preempt-KILL           | parent-dep absent   | n/a               | any               | preempt-KILL       |
| F#700/F#701 F#666-pure preempt-KILL | proxy-only          | n/a               | usually 3+        | preempt-KILL       |
| F#696/F#697 novel-mechanism provisional | ok              | novel             | usually clean     | design-lock PROVISIONAL + _impl |
| **This row (1st instance)**         | **target-metric**   | **reuse**         | **3+**            | **hygiene-patch PROVISIONAL + _impl** |

## Pre-reg hygiene defects (non-blocking for PROVISIONAL)

Three simultaneous hygiene defects fixed in MATH.md/results.json this iteration:
- `references: []` → 7 explicit refs
- `success_criteria: []` → `t_attach_median < 100ms AND glitch-count = 0`
- `platform: null` → `mlx`
- K1910 ambiguous → operationalized as same-adapter detach/re-attach glitch-count

CLI warning "⚠ INCOMPLETE: missing success_criteria, platform" confirmed the
defect set at claim time.

## Queue state

- Active: empty (this experiment PROVISIONAL on completion).
- `experiment list --status open -p 1` — remaining P=1 are mostly `_impl`
  companions for Hedgehog/JEPA/RDT/MEMENTO (custom MLX training; heavier than
  this hot-swap benchmark).
- `_impl` companion to register: `exp_pierre_adapter_hotswap_latency_impl`.

## Drain-window pattern count (after this iteration)

- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- 2 F#666-pure standalone preempt-KILLs (F#700, F#701)
- **1 design-lock-hygiene-patch PROVISIONAL (this, 1st instance)** ← watchlist
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)

Total drained this window: 18.

## Follow-up

1. Register `exp_pierre_adapter_hotswap_latency_impl` (P=2, micro, tagged
   `hotswap-latency` + `impl-companion` + `p1`). `_impl` scope: MATH.md §8.
2. Watchlist: if a 2nd instance of "design-lock-hygiene-patch PROVISIONAL"
   appears (runnable + 3 hygiene defects + operationally-ambiguous target-KC +
   prior-art theorem reuse), promote to standalone antipattern memory.

## Meta

This iteration demonstrates the **correct path** for under-specified-but-
runnable pre-regs. The drain must NOT mass-apply preempt-KILL to every
hygiene-defective pre-reg — only F#666-pure ones qualify. Others get
hygiene-patched and pushed to `_impl`.
