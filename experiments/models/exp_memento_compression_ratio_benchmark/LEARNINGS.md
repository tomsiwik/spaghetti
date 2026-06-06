# LEARNINGS.md — exp_memento_compression_ratio_benchmark

## Outcome
KILLED (preempt-structural, F#669 4th reuse). 3rd preempt-drain application in researcher-hat within drain objective.

## Core learning

Preempt-KILL with **F#666-compliant** KC set: when a child experiment's KCs transitively depend on a PROVISIONAL parent BUT the KC set is properly target-gated (≥1 target metric per F#666), the unblock path simplifies. Re-claim requires only `parent.status=supported`; no KC-augmentation needed.

This is the **complementary** sub-case to F#698 (which compounded preempt with F#666 violation). Together F#698 + this finding span the matrix:

| Parent target-verified? | KC F#666-compliant? | Verdict          |
|-------------------------|---------------------|------------------|
| no                      | no                  | F#698 (compound) |
| no                      | yes                 | this (simple)    |
| yes                     | no                  | F#666-only KILL  |
| yes                     | yes                 | runnable         |

## Why both K1850 and K1851 are preempt-blocked

- **K1850** (proxy: compression ratio < 3x): undefined absent a model that performs compression. No Gemma-4-MEMENTO checkpoint exists publicly (paper authors released Qwen3/Phi-4/Olmo 3 only; Gemma 4 is not among them). Computing base-vs-base would yield 1.0x by identity (uncompressed/uncompressed) — antipattern-t.

- **K1851** (target: compressed-context accuracy < 85% of full-context on GSM8K): "compressed-context" arm requires the MEMENTO block-mask attention loop with mementos in KV channel. No trained model = no compressed-context = no measurement. Substituting "shorter context window" would be antipattern-t silent objective swap.

Both KCs preempt-blocked by absence of Gemma-4-MEMENTO checkpoint. Single root cause, two affected KCs.

## F#669 pattern promotion (4th reuse)

| # | Finding | Child experiment                          | Parent experiment                | Date       | KC count | F#666 compound |
|---|---------|-------------------------------------------|----------------------------------|------------|----------|----------------|
| 1 | F#669   | exp_rdt_act_halting_throughput            | exp_rdt_loop_lora_gemma4         | 2026-04-19 | 4        | no             |
| 2 | F#687   | exp_jepa_router_prediction_error          | exp_jepa_adapter_residual_stream | 2026-04-23 | 4        | no             |
| 3 | F#698   | exp_jepa_adapter_attention_output         | exp_jepa_adapter_residual_stream | 2026-04-24 | 2        | yes (compound) |
| 4 | (this)  | exp_memento_compression_ratio_benchmark   | exp_memento_gemma4_replication   | 2026-04-24 | 2        | no             |

Promotion to canonical was confirmed at F#698 (3rd reuse). 4th reuse re-confirms; the routing is now routine drain operation.

## Sub-observations worth tracking (non-blocking)

1. **Misleading "standalone" framing.** Parent's notes claimed "No dependency on full replication" — materially false because no Gemma-4-MEMENTO checkpoint exists publicly. This is the first observation. If a 2nd misleading "standalone" framing creates a preempt-blocked child, formalize as antipattern memory.

2. **Single-parent-multi-preempt-child pattern.** `exp_jepa_adapter_residual_stream` (F#682) preempted TWO children (F#687 + F#698). When a PROVISIONAL parent has many proposed children, claim-time inspection of parent status would prevent fanning out preempt-children. Tool ergonomics hint, not antipattern.

3. **F#669 family complete sub-case matrix coverage.** F#698 + this finding now span the (parent-unverified, F#666-compliance) 2x2 matrix. Drain efficiency is high: preempt-children clear quickly with no compute consumed.

## Queue state after this iteration

- Drain: 1 P=1 → killed. P≤2 open reduced by 1.
- Net progress: preempt-drain pattern continues to clear blocked P≤2 experiments efficiently.
- F#669-family preempt-KILLs this drain window: 6 (F#669, F#671, F#672, F#687, F#698, this).

## Follow-up

None. Preempt-structural kill is self-contained; unblock is external (parent's existing `exp_memento_gemma4_replication_impl` at P=3). No `_impl` companion filed per F#687/F#698 precedent + reviewer.md §5.

## Meta

Picker returned this P=1 MEMENTO experiment above the event-suggested Hedgehog Rust/SQL P=2 siblings. Expected priority-ordering (P=1 > P=2). Hedgehog domain Rust/SQL siblings will follow in subsequent iterations.

Honored `mem-antipattern-impl-follow-up-delegation`: explicitly considered whether to file `_impl` and decided NO per reviewer.md §5 (preempt-structural KILL does not spawn `_impl`). Decision documented in REVIEW-adversarial.md "Novel-mechanism 4-part check (NOT applicable)" section.
