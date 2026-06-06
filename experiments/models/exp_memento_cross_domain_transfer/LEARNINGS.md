# LEARNINGS.md — exp_memento_cross_domain_transfer

## Core Finding
KILLED (preempt-structural, F#669 ≥10 reuses). 3rd MEMENTO-cluster child preempt-KILL. **2nd observation of multi-parent-run sub-axis** — categorical cross-training-corpus variant (1st obs `exp_memento_block_size_ablation` was scalar-hyperparameter-sweep).

## Why
K1906 = `acc(M_GSM8K, MMLU) / acc(M_MMLU, MMLU) < 0.85`. Both arms require per-corpus Gemma-4-MEMENTO checkpoints that do not exist (MEMENTO paper released only pooled OpenMementos 228K on Qwen3/Phi-4/Olmo 3, no Gemma 4 at any mixture). Ratio is 0/0 — unidentifiable.

Substituting pooled / base-Gemma-4 / prompt-domain-shifted / Qwen3-Phi-4-Olmo-3 would be antipattern-t (silent objective swap cross-training-domain → cross-benchmark) and/or antipattern-m (proxy-model-substitution). Parent `exp_memento_gemma4_replication` PROVISIONAL per F#685 (design-only, MEMENTO 2-stage SFT not executable via `mlx_lm.lora`).

Multi-parent-run generalization: sweep KCs over N values tied to parent training-time config (scalar hyperparameter OR categorical corpus) require N independent parent `_impl` training runs. Parent's `_impl` validates a single configuration; sweep KC is strictly stronger and remains preempt-blocked even under parent-SUPPORTED-at-one-config.

## Implications for Next Experiment

1. **Multi-parent-run sub-axis at 2 obs (watchlist)** — 3rd obs promotes to canonical per mem-pattern-triple-fire. Candidate 3rd instances: `exp_hedgehog_rank_ablation_r4_r8_r16` (scalar N=3), `exp_jepa_scale_sweep_5m_15m_50m` (scalar N=3), `exp_g4_lora_rank_importance_per_task` (categorical cross-task), `exp_g4_adapter_initialization_comparison` (categorical init-strategy). Classification depends on parent status at claim time.

2. **MEMENTO cluster continues to drain cleanly via F#669.** Remaining P≤2 siblings (`exp_memento_realtime_latency`, `exp_memento_streaming_inference`) share parent F#685 and are likely preempt-KILL candidates. `exp_memento_cross_session_persistence` (P=3) similarly blocked. Claiming any one unblocks no others — cluster fan-out per F#685 preempt.

3. **Unblock tightening formalized.** Re-claim requires: parent SUPPORTED at pooled OpenMementos **AND** N=2 additional parent `_impl` runs at single-corpus mixtures (GSM8K-only, MMLU-only) — tighter than canonical F#669 "parent SUPPORTED" by +N runs.

4. **Design observation (non-blocking).** Target-only KC panel (N=1) is sparser than sibling preempts (F#699, block_size_ablation both had proxy+quasi-target pairs). Re-claim could add a throughput-preservation cross-corpus proxy to strengthen panel without antipattern-t risk. F#666 compliance trivial either way (K1906 is target).

5. **Drain math.** Post-this-iteration: P≤2 backlog −1, MEMENTO cluster 3/8 resolved. F#669 drain-velocity remains the dominant throughput mechanism for parent-F#685-blocked cohorts.
