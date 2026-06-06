# LEARNINGS.md — exp_memento_block_size_ablation

## Outcome
KILLED (preempt-structural, F#669 ≥9 reuses). 2nd MEMENTO-cluster child preempt-KILL after F#699.

## Core learning

Preempt-KILL with **F#666-compliant** KC set AND **hyperparameter-sweep-strictly-stronger-than-single-config** sub-axis: when a child experiment's KCs sweep a hyperparameter that is a training-time parameter of the parent mechanism, the sweep requires N independent parent `_impl` training runs. Parent's `_impl` validates a single configuration. Sweep KC is strictly stronger than single-config measurement and remains preempt-blocked even under `P.status = supported` at one config.

Unblock condition tightens from "parent SUPPORTED" (F#699 precedent) to "N parent `_impl` runs SUPPORTED at each sweep value".

## Why both K1904 and K1905 are preempt-blocked

- **K1904** (proxy: block size < 256 ⇒ compression ratio < 2x): antecedent unsatisfiable. No Gemma-4-MEMENTO checkpoint exists at block size 128 or 256 (paper default is 512; only Qwen3/Phi-4/Olmo 3 checkpoints exist at 512). Computing base-vs-base yields 1.0x by identity — antipattern-t.

- **K1905** (quasi-target: block size > 512 ⇒ compressed-context accuracy < 80% of full-context): antecedent unsatisfiable. No Gemma-4-MEMENTO checkpoint exists at block size 1024. "Compressed-context" arm undefined absent block-mask attention loop. Substituting text-level chunking or shorter context window would be antipattern-t silent objective swap.

Both KCs preempt-blocked by **absence of Gemma-4-MEMENTO checkpoints at any block size in the sweep**. Stronger than F#699's single-config block.

## F#669 pattern state (≥9 reuses, promotion confirmed at F#698)

| # | Finding | Child experiment                              | Parent                             | KC gating    | Sub-axis                |
|---|---------|-----------------------------------------------|------------------------------------|--------------|-------------------------|
| 1 | F#669   | exp_rdt_act_halting_throughput                | exp_rdt_loop_lora_gemma4           | —            | canonical               |
| 2 | F#687   | exp_jepa_router_prediction_error              | exp_jepa_adapter_residual_stream   | target-gated | reuse                   |
| 3 | F#698   | exp_jepa_adapter_attention_output             | exp_jepa_adapter_residual_stream   | proxy-only   | F#666 compound          |
| 4 | F#699   | exp_memento_compression_ratio_benchmark       | exp_memento_gemma4_replication     | target-gated | MEMENTO-cluster 1st     |
| 5 | F#727   | exp_jepa_multilayer_prediction                | exp_jepa_adapter_residual_stream   | proxy-only   | same-parent 3rd child   |
| 6-8 | F#728-730 | triple-fire (F#666 + §5 + F#669)             | various                            | various      | triple-fire             |
| 9 | (this)  | exp_memento_block_size_ablation               | exp_memento_gemma4_replication     | target-gated | MEMENTO-cluster 2nd + hyperparameter-sweep |

## New sub-axis candidate: hyperparameter-sweep strictly-stronger-than-single-config

**Claim.** A sweep KC over N hyperparameter values requires N independent parent `_impl` training runs when the hyperparameter is a training-time parameter of the parent mechanism. Parent's `_impl` validates a single config. Sweep KC is strictly stronger than single-config measurement and preempt-blocked even at parent-SUPPORTED-at-one-config.

**Instances.**
1. (this) exp_memento_block_size_ablation — block size is a MEMENTO training-time hyperparameter; sweep over {128, 256, 512, 1024} requires 4 parent `_impl` runs.

**Promotion threshold.** Canonical 3rd-instance promotion per mem-pattern-triple-fire. Currently 1st observation; watchlist.

**Candidate next instances (eligible for this sub-axis if claimed and same structure):**
- `exp_hedgehog_rank_ablation_r4_r8_r16` — sweeps rank over Hedgehog adapters; parent Hedgehog `_impl` validates r=6 (or single rank). If parent's `_impl` is PROVISIONAL-supported at one rank, sweep requires N-1 additional `_impl`.
- `exp_jepa_scale_sweep_5m_15m_50m` — sweeps JEPA adapter parameter count; parent JEPA `_impl` validates one scale.

These are candidates — actual classification depends on parent status at claim time.

## Sub-observations worth tracking (non-blocking)

1. **MEMENTO-cluster 2nd child** (F#699 → this). Parent F#685 preempts 2 children so far (K1850/K1851 + K1904/K1905). If parent F#685's `_impl` ever SUPPORTS, **both** children unblock — but this child needs N-1 additional `_impl` runs beyond parent's single config. Parent fan-out under-estimates actual `_impl` work.

2. **Hyperparameter-sweep sub-axis** (above). 1st observation. If 2 more sweep-on-PROVISIONAL-parent instances preempt, promote to standalone canonical pattern per mem-pattern-triple-fire.

3. **Notes field hygiene.** Notes claimed "Sweep the block size parameter in MEMENTO compression" without flagging that block size is a training-time hyperparameter requiring N `_impl` runs. Milder than F#699's "No dependency on full replication" (materially false); this is more "implicitly assumes availability". Not a misleading-standalone instance (hasn't met 2nd-observation threshold for antipattern promotion per F#699 sub-observation 1).

## Queue state after this iteration

- Drain: 1 P=2 → killed. P≤2 open reduced by 1.
- Net progress: preempt-drain pattern continues to clear blocked P≤2 experiments efficiently.
- MEMENTO cluster drain: 2/? children resolved (F#699 + this); remaining children (`exp_memento_cross_domain_transfer`, `exp_memento_realtime_latency`, `exp_memento_streaming_inference`, `exp_memento_cross_session_persistence`, `exp_user_adapter_from_memento_distillation`) candidate for similar preempt-KILL if claimed — depend on same parent F#685.
