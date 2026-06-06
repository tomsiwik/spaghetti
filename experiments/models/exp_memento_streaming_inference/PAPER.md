# PAPER.md — exp_memento_streaming_inference

## Verdict: KILLED (preempt, F#669 — ≥12 reuses; 5th MEMENTO-cluster child)

This experiment was preempt-killed before any MLX code was written. The kill is structural, not empirical: K1939 (streaming-vs-batch task-accuracy parity < 90%) and K1940 (per-block streaming compression latency > 20ms) require a callable Gemma-4-MEMENTO forward pass with an INLINE streaming-mode path during decoder generation, and no such mechanism exists. Parent `exp_memento_gemma4_replication` is `provisional` (F#685 — design-only, MEMENTO 2-stage SFT not executable via `mlx_lm.lora` CLI). The MEMENTO paper (Kontonis et al., arxiv:2604.09852) released checkpoints only for Qwen3 / Phi-4 / Olmo 3 on pooled OpenMementos 228K traces — no Gemma 4 at any mixture, and paper Table 4 reports offline-batch throughput rather than streaming-during-inference per-block latency at the 20ms granularity K1940 asserts.

This is a **single-config mixed-target-only** preempt-KILL — the **3rd target-only KC panel observation** in MEMENTO-cluster, **CANONICALIZING** the "target-only KC panel under preempt-KILL on PROVISIONAL parent" micro-pattern at three distinct sub-forms (pure-behavioral F#738 + pure-engineering F#739 + mixed this).

## Prediction vs measurement

| KC    | Prediction                                                                  | Kind   | Measurement  | Verdict   |
| ----- | --------------------------------------------------------------------------- | ------ | ------------ | --------- |
| K1939 | Streaming-MEMENTO downstream-task accuracy < 90% of batch-MEMENTO accuracy   | target | not measured | untested  |
| K1940 | Per-block streaming compression latency > 20ms on M5 Pro (NOT real-time)    | target | not measured | untested  |

**K1939 is "not measured" because the streaming/batch task-accuracy ratio is NaN/NaN** — neither regime is computable without a trained Gemma-4-MEMENTO checkpoint. Structurally identical to F#738 K1906's `acc(M_GSM8K, MMLU) / acc(M_MMLU, MMLU)` parity ratio, which was unidentifiable under preempt for the same reason. **K1940 is "not measured" because per-block streaming latency is undefined** — no callable inline streaming-mode forward path exists. Latency at 20ms granularity is dominated by Metal kernel selection, MLX lazy-eval boundaries, KV-cache layout in unified memory, tile-size compile decisions, AND inline integration with the decoder loop; it is strictly empirical and not derivable from architecture alone.

Substituting PPL parity for K1939's task-accuracy parity would replace a behavioral target with a weakly-correlated proxy (F#666 r≈0.08 between PPL and task quality) — antipattern-i (KC-measures-wrong-object). Substituting base Gemma 4 inference latency (no MEMENTO compression) for K1940's per-block streaming latency, Qwen3/Phi-4/Olmo 3 paper-checkpoint timings as a Gemma 4 stand-in, or hand-written block-mask MLX reimplementation without trained weights would each be antipattern-t (silent objective swap) and/or antipattern-m (proxy-model substitution).

## Assumptions

- `exp_memento_gemma4_replication` will eventually be re-run to full scale via `exp_memento_gemma4_replication_impl` (P=3). That alone is necessary but **not sufficient** — K1939/K1940 additionally require parent-side instrumentation strictly beyond F#739's unblock surface: an INLINE streaming-mode forward path with decoder-loop integration (not just offline-streaming), per-block latency hook during inline generation, KV-cache mutation hooks during generation, and a dual-regime (streaming + batch) downstream-task evaluation harness on the same held-out eval set.
- No redesign attempted this iteration to substitute a measurable proxy. A "PPL on held-out eval" proxy would still require the same parent-impl checkpoint AND would replace a behavioral target with a weakly-correlated proxy — adding no identifying power for K1939's streaming-vs-batch task-accuracy parity claim while introducing antipattern-i.
- F#666 gating: K1939 (behavioral) and K1940 (engineering) are both targets. Threshold 90% for K1939 is the calibrated streaming-mode regression bound (10pp accuracy degradation tolerance). Threshold 20ms for K1940 is the calibrated inline-streaming bound (tighter than F#739 K1907's 50ms because streaming-during-inference must keep up with the decoder's token-emission rate, not just human-perceptible 20Hz interactivity). KC-augmentation not needed at re-claim.

## Pattern continuation — CANONICALIZATION at 3rd obs

F#669 reached its promotion threshold at F#698 (3rd reuse) and has since been reused extensively. This is the **5th MEMENTO-cluster child preempt-KILL** after `exp_memento_compression_ratio_benchmark` (F#699), `exp_memento_block_size_ablation` (F#737), `exp_memento_cross_domain_transfer` (F#738), and `exp_memento_realtime_latency` (F#739). All five share the same parent `exp_memento_gemma4_replication` (PROVISIONAL per F#685) and all five are F#666-compliant.

The "target-only KC panel under preempt-KILL on PROVISIONAL parent" micro-pattern reaches **3rd-observation canonicalization** at this experiment, with three distinct sub-forms now attested:

| Sub-form                           | Cluster child                              | KCs                                                    |
| ---------------------------------- | ------------------------------------------ | ------------------------------------------------------ |
| **Pure-behavioral**                | `exp_memento_cross_domain_transfer` (F#738) | K1906 (cross-corpus accuracy parity ratio)              |
| **Pure-engineering**               | `exp_memento_realtime_latency` (F#739)      | K1907 (per-block latency) + K1908 (streaming/batch ratio)|
| **Mixed-behavioral-engineering**   | (this) `exp_memento_streaming_inference`    | K1939 (acc parity) + K1940 (per-block streaming latency)|

Per `mem-pattern-triple-fire`, three distinct sub-forms across three cluster children promotes the micro-pattern. Future MEMENTO-cluster (and structurally analogous PROVISIONAL-parent-cluster) children with target-only KC panels can cite this finding as the canonical reference rather than re-deriving the unidentifiability theorem.

## Sub-axis classification (single-config; canonical class for target-only-KC-panel)

| Cluster child                                    | KC kind composition                        | Sub-axis status                                        |
| ------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------ |
| `exp_memento_compression_ratio_benchmark` (F#699) | proxy + quasi-target                       | single-config (canonical compound-KC class)             |
| `exp_memento_block_size_ablation` (F#737)         | proxy + target (sweep×2)                   | multi-parent-run sub-axis 1st obs                       |
| `exp_memento_cross_domain_transfer` (F#738)       | target only (behavioral, ratio)            | multi-parent-run sub-axis 2nd obs / target-only 1st obs |
| `exp_memento_realtime_latency` (F#739)            | target only (engineering, ×2)              | single-config / target-only 2nd obs                     |
| **(this) `exp_memento_streaming_inference`**      | **target only (behavioral + engineering)** | **single-config / target-only 3rd obs (CANONICAL)**     |

Multi-parent-run sub-axis remains at 2 observations — this experiment is single-config and does NOT advance that sub-axis.

## Related

- **F#669** — defining precedent for preempt-KILL on target-unverified parent.
- **F#687 / F#698** — 2nd / 3rd F#669 reuse; promotion confirmed at F#698.
- **F#699** — 4th F#669 reuse; 1st MEMENTO-cluster child preempt-KILL (single-config, proxy+target).
- **F#737** — 2nd MEMENTO-cluster child preempt-KILL (scalar-sweep, multi-parent-run sub-axis 1st obs).
- **F#738** — 3rd MEMENTO-cluster child preempt-KILL (categorical cross-corpus, target-only behavioral 1st obs).
- **F#739** — 4th MEMENTO-cluster child preempt-KILL (single-config engineering, target-only 2nd obs).
- **F#685** — parent PROVISIONAL.
- **F#666** — target-gated KC discipline (this experiment compliant: K1939 + K1940 both target).
- **`mem-pattern-triple-fire`** — promotion threshold for cross-instance micro-patterns; satisfied at 3rd obs of "target-only KC panel under preempt-KILL on PROVISIONAL parent" with this experiment.
- `exp_memento_gemma4_replication_impl` — P=3 impl-companion to parent; necessary-but-not-sufficient gate (K1802 measures batch throughput only; inline streaming surface is parent-extension strictly beyond F#739's offline-streaming surface).
- Kontonis et al. arxiv:2604.09852 — MEMENTO paper.

## Unblock path

Re-claim this experiment when ALL of:

1. Parent `exp_memento_gemma4_replication` reaches `status=supported` via its `_impl` companion at the paper-default pooled OpenMementos training corpus.
2. Parent K1799 (KV reduction proxy) AND K1800 (task accuracy target) AND K1801 (KV-channel ablation target) AND K1802 (throughput target) SUPPORTED at full scale.
3. **Parent-side engineering tightening (strict superset of F#739's)**: parent's `_impl` extended with (a) an INLINE streaming-mode forward path with decoder-loop integration (compressing blocks block-by-block as new tokens are generated, not just offline-streaming on existing traces), (b) per-block latency hook during inline generation, (c) KV-cache mutation hooks during generation, and (d) a dual-regime downstream-task evaluation harness that runs under both regimes (streaming + batch) on the SAME held-out eval set with comparable scoring.

No KC-augmentation needed. K1939/K1940 already targets per F#666 trivially.

## Follow-up filed

None — preempt-structural kill does not spawn an `_impl` companion (per F#687/F#698/F#699/F#737/F#738/F#739 precedent + reviewer.md §5). The unblock is parent-external; the parent-extension required (inline streaming-mode + per-block hook + KV-cache mutation + dual-regime eval harness) belongs in parent's `_impl` scope, not under this child.
