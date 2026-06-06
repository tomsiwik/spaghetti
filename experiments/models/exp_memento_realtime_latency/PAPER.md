# PAPER.md — exp_memento_realtime_latency

## Verdict: KILLED (preempt, F#669 — ≥11 reuses; 4th MEMENTO-cluster child)

This experiment was preempt-killed before any MLX code was written. The kill is structural, not empirical: K1907 (per-block compression latency > 50ms) and K1908 (streaming > 2x batch latency) require a callable Gemma-4-MEMENTO block-compression forward pass to wall-clock, and no such mechanism exists. Parent `exp_memento_gemma4_replication` is `provisional` (F#685 — design-only, MEMENTO 2-stage SFT not executable via `mlx_lm.lora` CLI). The MEMENTO paper (Kontonis et al., arxiv:2604.09852) released checkpoints only for Qwen3 / Phi-4 / Olmo 3 on pooled OpenMementos 228K traces — no Gemma 4 at any mixture, and paper Table 4 reports throughput comparisons rather than per-block latency at the 50ms-real-time granularity this KC asserts.

This is a **single-config engineering-target-only** preempt-KILL, structurally similar to F#699 (`exp_memento_compression_ratio_benchmark`, single-config) and distinct from F#737/F#738 (sweep variants advancing the multi-parent-run sub-axis).

## Prediction vs measurement

| KC    | Prediction                                                                | Kind   | Measurement  | Verdict   |
| ----- | ------------------------------------------------------------------------- | ------ | ------------ | --------- |
| K1907 | Per-block compression latency > 50ms on M5 Pro (NOT real-time)             | target | not measured | untested  |
| K1908 | Streaming (incremental) compression latency > 2x batch compression latency | target | not measured | untested  |

**K1907 is "not measured" because per-block latency is undefined — no callable Gemma-4-MEMENTO block-compression kernel exists.** Latency at 50ms granularity is dominated by Metal kernel selection, MLX lazy-eval boundaries, KV-cache layout in unified memory, and tile-size compile decisions; it is strictly empirical and not derivable from architecture alone. **K1908 is "not measured" because the streaming/batch ratio is NaN/NaN** — neither regime is timeable without the parent-impl checkpoint AND a streaming-mode forward path that parent's K1802 (batch throughput target) does not currently surface.

Substituting base Gemma 4 inference latency (no MEMENTO compression), Qwen3/Phi-4/Olmo 3 paper-checkpoint timings as a Gemma 4 stand-in, hand-written block-mask MLX reimplementation without trained weights, or paper Table 4 throughput / block-count back-derivation would each be antipattern-t (silent objective swap from "MEMENTO real-time compression latency" to "base-model decoding cost" or "non-Gemma reference throughput") and/or antipattern-m (proxy-model substitution).

## Assumptions

- `exp_memento_gemma4_replication` will eventually be re-run to full scale via `exp_memento_gemma4_replication_impl` (P=3). That alone is necessary but **not sufficient** — K1907/K1908 additionally require parent-side instrumentation: a per-block latency hook and a streaming-mode forward path. Parent's K1802 measures batch tokens/sec only.
- No redesign attempted this iteration to substitute a measurable proxy. A "tokens/sec held-out throughput" proxy would still require the same parent-impl checkpoint and would not separate per-block from streaming overhead — it adds no identifying power for K1907/K1908 while incurring parent-blocking cost.
- F#666 gating: K1907 and K1908 are both engineering targets (latency thresholds are calibrated real-time and regression bounds), so the KC set is target-gated trivially. No proxy to pair. KC-augmentation not needed at re-claim.

## Pattern continuation

F#669 reached its promotion threshold at F#698 (3rd reuse) and has since been reused extensively. This is the **4th MEMENTO-cluster child preempt-KILL** after `exp_memento_compression_ratio_benchmark` (F#699, 1st cluster child, single-config), `exp_memento_block_size_ablation` (F#737, 2nd cluster child, scalar-sweep), and `exp_memento_cross_domain_transfer` (F#738, 3rd cluster child, categorical cross-corpus). All four share the same parent `exp_memento_gemma4_replication` (PROVISIONAL per F#685) and all four are F#666-compliant.

## Sub-axis classification (single-config; no canonicalization advance)

| Cluster child                             | KC kind composition                | Sub-axis status                          |
| ----------------------------------------- | ---------------------------------- | ---------------------------------------- |
| `exp_memento_compression_ratio_benchmark` (F#699) | proxy + quasi-target               | single-config (canonical class)          |
| `exp_memento_block_size_ablation` (F#737)         | proxy + target (sweep×2)           | multi-parent-run sub-axis 1st obs        |
| `exp_memento_cross_domain_transfer` (F#738)       | target only (behavioral, ratio)    | multi-parent-run sub-axis 2nd obs        |
| **(this) `exp_memento_realtime_latency`**         | **target only (engineering, ×2)**  | **single-config (no sub-axis advance)**  |

Multi-parent-run sub-axis remains at 2 observations; canonical promotion still pending 3rd observation per mem-pattern-triple-fire.

A *minor design observation* worth recording (non-canonical): this is the 2nd MEMENTO-cluster child with a target-only KC panel after F#738 (cross_domain_transfer). The prior was behavioral (accuracy ratio); this one is engineering (latency thresholds). Two observations of "target-only KC panel under preempt-KILL" form a watchlist micro-pattern (engineering vs behavioral variant) below the triple-fire threshold.

## Related

- **F#669** — defining precedent for preempt-KILL on target-unverified parent.
- **F#687 / F#698** — 2nd / 3rd F#669 reuse; promotion confirmed at F#698.
- **F#699** — 4th F#669 reuse; 1st MEMENTO-cluster child preempt-KILL (single-config, proxy+target).
- **F#737** — 2nd MEMENTO-cluster child preempt-KILL (scalar-sweep, multi-parent-run sub-axis 1st obs).
- **F#738** — 3rd MEMENTO-cluster child preempt-KILL (categorical cross-corpus, multi-parent-run sub-axis 2nd obs, target-only).
- **F#685** — parent PROVISIONAL.
- **F#666** — target-gated KC discipline (this experiment compliant: K1907 + K1908 both target).
- `exp_memento_gemma4_replication_impl` — P=3 impl-companion to parent; necessary-but-not-sufficient gate (K1802 measures batch throughput only; per-block + streaming surfaces are parent-extensions).
- Kontonis et al. arxiv:2604.09852 — MEMENTO paper.

## Unblock path

Re-claim this experiment when ALL of:

1. Parent `exp_memento_gemma4_replication` reaches `status=supported` via its `_impl` companion at the paper-default pooled OpenMementos training corpus.
2. Parent K1799 (KV reduction proxy) AND K1800 (task accuracy target) AND K1801 (KV-channel ablation target) AND K1802 (throughput target) SUPPORTED at full scale.
3. **Parent-side engineering tightening**: parent's `_impl` extended with a per-block latency hook (instrumented forward pass) AND a streaming-mode forward flag (incremental block-by-block compression, distinct from batch). This is a parent-extension, not a separate child `_impl`.

No KC-augmentation needed. K1907/K1908 already engineering targets per F#666 trivially.

## Follow-up filed

None — preempt-structural kill does not spawn an `_impl` companion (per F#687/F#698/F#699/F#737/F#738 precedent + reviewer.md §5). The unblock is parent-external; the parent-extension required (latency hook + streaming-mode flag) belongs in parent's `_impl` scope, not under this child.
