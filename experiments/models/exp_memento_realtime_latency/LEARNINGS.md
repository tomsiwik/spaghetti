# LEARNINGS.md — exp_memento_realtime_latency

## Core Finding
KILLED (preempt-structural, F#669 ≥11 reuses). 4th MEMENTO-cluster child preempt-KILL. **Single-config engineering-target-only** (no new sub-axis canonicalization; multi-parent-run sub-axis remains at 2 obs).

## Why
K1907 = per-block compression latency > 50ms; K1908 = streaming > 2x batch latency. Both require a callable Gemma-4-MEMENTO block-compression forward pass that does not exist. Latency at 50ms granularity is dominated by Metal kernel selection, MLX lazy-eval boundaries, KV layout in unified memory, and tile-size compile decisions — strictly empirical, not derivable from architecture alone. Parent `exp_memento_gemma4_replication` PROVISIONAL per F#685; MEMENTO 2-stage SFT + block-mask attention not executable via `mlx_lm.lora`. MEMENTO paper authors released only Qwen3/Phi-4/Olmo 3 checkpoints on pooled OpenMementos 228K (no Gemma 4 at any mixture); paper Table 4 reports throughput, not per-block 50ms-granularity latency.

Substituting base Gemma 4 inference latency, Qwen3/Phi-4/Olmo 3 timings as Gemma 4 stand-in, hand-written block-mask reimpl without trained weights, or paper-throughput / block-count back-derivation all violate antipattern-t and/or antipattern-m.

## Implications for Next Experiment

1. **Multi-parent-run sub-axis stays at 2 obs (watchlist).** This experiment did NOT advance the sub-axis (single-config measurement, not a sweep). Canonical promotion still pending 3rd observation per mem-pattern-triple-fire. Active candidates: `exp_hedgehog_rank_ablation_r4_r8_r16` (scalar N=3), `exp_jepa_scale_sweep_5m_15m_50m` (scalar N=3), `exp_g4_lora_rank_importance_per_task` (categorical cross-task), `exp_g4_adapter_initialization_comparison` (categorical init-strategy).

2. **MEMENTO cluster drains cleanly via F#669.** 4/8 children resolved. Remaining P≤2 sibling: `exp_memento_streaming_inference` (likely preempt-KILL same parent F#685; another runtime-property variant). P=3 sibling `exp_memento_cross_session_persistence` similarly blocked.

3. **Engineering-target-only KC panel — micro-pattern 2nd obs (watchlist, NOT canonical).** F#738 was 1st obs (behavioral target-only, accuracy ratio); this is 2nd obs (engineering target-only, latency thresholds). Both target-only panels are F#666-compliant trivially (vacuous quantification — no proxy to pair). 3rd obs would canonicalize the "target-only KC under preempt-KILL is structurally distinct from compound-KC preempt-KILL" pattern. Below triple-fire threshold; logged here, not promoted.

4. **Parent-extension distinct from sibling-_impl pattern.** Unblock for K1907/K1908 requires not just parent SUPPORTED but also a parent-side instrumentation tightening (per-block latency hook + streaming-mode forward path) — parent's K1802 measures batch throughput only and does not surface the measurement API K1907/K1908 need. This is a parent-_impl extension, not a new child _impl, distinct from F#738's "N=2 additional per-corpus _impl runs" unblock condition. Sub-distinction worth noting for future engineering-target preempt-KILLs: the unblock may be a parent-extension rather than parent-replication.

5. **Drain math.** Post-this-iteration: P≤2 backlog −1, MEMENTO cluster 4/8 resolved. F#669 drain-velocity remains the dominant throughput mechanism for parent-F#685-blocked cohorts. `exp_memento_streaming_inference` is the next likely MEMENTO-cluster preempt-KILL candidate (also runtime-property; could be same single-config class as this, OR could fold into K1908's streaming regime if scope overlaps — to be assessed at claim time).
