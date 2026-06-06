# MATH.md — exp_memento_streaming_inference (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. No `run_experiment.py` MLX implementation is attempted because the parent dependency is in a state that makes the KCs structurally untestable: streaming-vs-batch accuracy parity and per-block streaming compression latency are runtime properties of MEMENTO block-mask attention forward passes, and no Gemma-4-MEMENTO forward pass exists to measure (parent F#685 PROVISIONAL).

This is the **5th MEMENTO-cluster preempt-KILL** in the drain window, following:
1. `exp_memento_compression_ratio_benchmark` (F#699, 4th F#669 reuse — single-config; static KV-memory target).
2. `exp_memento_block_size_ablation` (F#737, ≥9th F#669 reuse — multi-parent-run sub-axis 1st obs, scalar-sweep).
3. `exp_memento_cross_domain_transfer` (F#738, ≥10th F#669 reuse — multi-parent-run sub-axis 2nd obs, categorical cross-corpus, target-only behavioral).
4. `exp_memento_realtime_latency` (F#739, 11th F#669 reuse — single-config engineering-target-only).
5. (this) `exp_memento_streaming_inference` — **single-config mixed-target-only** (behavioral parity + engineering latency); CANONICALIZES the "target-only KC panel under preempt-KILL" micro-pattern at 3rd obs per `mem-pattern-triple-fire`.

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — MEMENTO is 2-stage SFT + block-mask attention, not LoRA. Block size default 512 per paper.
- Parent dependency: `exp_memento_gemma4_replication` (status `provisional`, F#685).
- Sibling precedents: `exp_memento_compression_ratio_benchmark` (F#699), `exp_memento_block_size_ablation` (F#737), `exp_memento_cross_domain_transfer` (F#738), `exp_memento_realtime_latency` (F#739).
- Datasets: N/A — neither downstream-task accuracy (K1939) nor per-block latency (K1940) is reachable; no eval set is loaded.

## §1 Preempt-KILL theorem

**Theorem (inter-experiment target unverifiability, mixed-target-only variant).** Let `C` denote child experiment `exp_memento_streaming_inference` with kill criteria K = {K1939 (target, behavioral: streaming-MEMENTO downstream-task accuracy < 90% of batch-MEMENTO downstream-task accuracy), K1940 (target, engineering: streaming compression latency > 20ms per block)}. Let `P` denote parent experiment `exp_memento_gemma4_replication`.

K1939 and K1940 each require empirical execution of a Gemma-4-MEMENTO forward pass — specifically:

- **K1939** (streaming-vs-batch accuracy parity): downstream-task accuracy of `M_streaming` (MEMENTO compressing blocks DURING ongoing inference, inline as tokens are generated) divided by downstream-task accuracy of `M_batch` (MEMENTO compressing blocks AFTER trace generation, offline). Requires loading a trained Gemma-4-MEMENTO checkpoint into MLX, executing it under two scheduling regimes on a held-out evaluation set, and computing the parity ratio.
- **K1940** (per-block streaming latency): wall-clock time-to-compress one MEMENTO block during inline (streaming) inference on M5 Pro Metal. Requires the same checkpoint with a streaming-mode forward path that compresses blocks block-by-block as they are generated, not after-the-fact.

Both KCs are **dynamic** and **strictly empirical**. K1939 depends on the trained MEMENTO weights producing accuracy at all (which requires `P.status = supported`); K1940 depends on Metal kernel selection, MLX lazy-eval boundaries, KV-cache layout in unified memory, and tile-size compile decisions, none derivable analytically.

Required precondition: a Gemma-4-MEMENTO checkpoint loadable into MLX with a callable streaming (inline-during-inference) forward path.

1. **No public MEMENTO checkpoint exists for Gemma 4 at any training mixture.** MEMENTO paper (Kontonis et al., arxiv:2604.09852, Apr 2026) released checkpoints for Qwen3 / Phi-4 / Olmo 3 only, on the paper's pooled OpenMementos 228K-trace dataset. No Gemma-4-MEMENTO at any mixture.
2. **MEMENTO 2-stage SFT + block-mask attention is not executable via `mlx_lm.lora` CLI** (parent F#685 finding). Even if a checkpoint existed, the runtime would need a custom MLX implementation of segment-and-summarize trace generation and block-mask-aware attention, neither of which exists in `mlx_lm` ≤ 0.31.
3. **Streaming-during-inference is a stronger regime than F#739 K1908.** F#739 K1908 compared streaming-vs-batch *compression* latency on existing traces. K1940 requires compressing blocks INLINE while the model is generating new tokens — a tighter coupling between the compression kernel and the decoder loop. Parent's K1802 (batch throughput target) does not exercise this scheduling regime; parent's `_impl` would need a streaming-mode forward path AND inline KV-cache mutation hooks (a strict superset of the F#739-unblock surface) to surface K1940's measurement.
4. K1939 (behavioral accuracy parity) is structurally identical to F#738's K1906 form: a *parity ratio* between two configurations of the same parent design where neither side is computable while parent ∈ {provisional, open}. F#738 measured `acc(M_GSM8K, MMLU) / acc(M_MMLU, MMLU)`; K1939 measures `acc(M_streaming, eval) / acc(M_batch, eval)`. Both reduce to NaN/NaN under preempt — accuracy ratio 0/0 unidentifiable per F#738 evidence.

If `P.status ∈ {provisional, open}` — i.e. no Gemma-4-MEMENTO checkpoint exists — then:

- **K1939**: streaming and batch accuracies are both undefined (no trained MEMENTO mechanism to evaluate). Ratio is NaN/NaN — unidentifiable.
- **K1940**: per-block streaming latency is undefined (no kernel to time, no streaming-mode path even if a checkpoint existed). Comparison to 20ms threshold is `NaN > 20ms` — unidentifiable.

Additionally, even under `P.status = supported`, K1940 requires parent-side instrumentation strictly beyond F#739's unblock surface: parent's `_impl` would need a streaming-mode forward path INLINE with the decoder loop (not just batch vs offline-streaming), and inline KV-cache mutation hooks for per-block timing during generation. K1939 requires both regimes (streaming AND batch) to be runnable on the same checkpoint with comparable evaluation harness — one further parent-extension to expose the regime flag at inference time.

∴ Testing K1939 and K1940 while `P.status ≠ supported|proven` produces unidentifiable samples on both. **QED.**

### §1.1 F#666 gating

- K1939 = **target** (downstream-task accuracy parity is the behavioral claim being made; threshold 90% is the calibrated regression bound for streaming-mode acceptability).
- K1940 = **target** (per-block streaming latency on M5 Pro is the engineering claim; threshold 20ms is the calibrated real-time interactivity bound, tighter than F#739 K1907's 50ms because streaming-during-inference must keep up with the decoder, not just real-time perception).
- No proxy KC present. The KC set is therefore F#666-compliant **trivially** by vacuous quantification — F#666 requires every *proxy* KC to be paired with a target; a target-only KC set satisfies this rule unconditionally.

A defensible target-only design: K1939 measures the behavioral claim directly (does streaming preserve task accuracy?); K1940 measures the engineering claim directly (is streaming fast enough?). There is no behavioral proxy that could be paired (PPL is a textbook proxy but cleanly weaker than downstream-task accuracy; cosine similarity between streaming and batch hidden states is a structural proxy but does not certify task accuracy — see F#666 r≈0.08 between PPL and task quality). A "PPL on a held-out evaluation set" proxy would still need the same parent-impl checkpoint, so it would not unblock the experiment; it would only add structural complexity without identifying-power gain.

### §1.2 Sub-axis classification — CANONICALIZATION at 3rd obs of target-only KC panel

This is a **single-config mixed-target-only** preempt-KILL — same single-config structural class as F#699 and F#739 (no parent-config sweep), but the KC panel composition differs:

| Cluster child                                       | KC kind composition                                | Target-only KC-panel form                |
| --------------------------------------------------- | -------------------------------------------------- | ---------------------------------------- |
| `exp_memento_compression_ratio_benchmark` (F#699)   | proxy + quasi-target                               | N/A (compound, not target-only)          |
| `exp_memento_block_size_ablation` (F#737)           | proxy + target (sweep×2)                           | N/A (compound, not target-only)          |
| `exp_memento_cross_domain_transfer` (F#738)         | target only (behavioral parity ratio)              | **1st obs**: pure-behavioral             |
| `exp_memento_realtime_latency` (F#739)              | target only (engineering ×2)                       | **2nd obs**: pure-engineering            |
| **(this) `exp_memento_streaming_inference`**        | **target only (behavioral + engineering)**         | **3rd obs**: mixed (CANONICALIZES)       |

Per `mem-pattern-triple-fire`, the "target-only KC panel under preempt-KILL on PROVISIONAL parent" micro-pattern reaches **3rd-observation canonicalization** at this experiment, with three distinct sub-forms now attested: pure-behavioral (F#738), pure-engineering (F#739), mixed-behavioral-engineering (this). The promotion finding records the canonicalization and characterizes the three sub-forms.

Multi-parent-run sub-axis remains at 2 obs (F#737 + F#738) — this experiment is single-config and does NOT advance that sub-axis.

## §2 Prior art

- **F#669** (2026-04-19) — defining precedent for preempt-KILL on target-unverified parent.
- **F#685** (2026-04-23) — parent PROVISIONAL: design-only, 4 target-gated KCs untested, MEMENTO 2-stage SFT + block-mask attention not executable via `mlx_lm.lora` CLI.
- **F#699** (2026-04-24) — 4th F#669 reuse, **1st MEMENTO-cluster** child preempt-KILL: `exp_memento_compression_ratio_benchmark`. F#666-compliant (proxy+target).
- **F#737** (2026-04-24) — **2nd MEMENTO-cluster** child preempt-KILL: `exp_memento_block_size_ablation`. Multi-parent-run sub-axis 1st obs (scalar-sweep).
- **F#738** (2026-04-24) — **3rd MEMENTO-cluster** child preempt-KILL: `exp_memento_cross_domain_transfer`. Multi-parent-run sub-axis 2nd obs (categorical cross-corpus). 1st target-only KC panel observation (pure-behavioral). Direct K1939 structural analogue: accuracy parity ratio NaN/NaN under preempt.
- **F#739** (2026-04-24) — **4th MEMENTO-cluster** child preempt-KILL: `exp_memento_realtime_latency`. Single-config engineering-target-only. 2nd target-only KC panel observation (pure-engineering). Direct K1940 structural analogue: per-block latency strictly empirical, parent-extension required even at P.status=supported.
- **F#666** — target-gated KC discipline. This experiment satisfies F#666 trivially (K1939+K1940 both target).
- `mem-pattern-triple-fire` — promotion threshold for cross-instance micro-patterns; satisfied at 3rd obs of "target-only KC panel under preempt-KILL on PROVISIONAL parent" with this experiment.
- Kontonis et al. arxiv:2604.09852 — MEMENTO paper. Authors released Qwen3 / Phi-4 / Olmo 3 checkpoints on pooled OpenMementos (228K traces); no Gemma 4 at any mixture. Paper does not report streaming-during-inference latency at the 20ms-per-block granularity K1940 asserts (Table 4 throughput is offline batch).

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                                                            | Kind   | Measurement status                |
| ----- | ------------------------------------------------------------------------------------------------ | ------ | --------------------------------- |
| K1939 | Streaming-MEMENTO downstream-task accuracy < 90% of batch-MEMENTO accuracy (NOT preserving task) | target | untested (preempt-blocked, F#669) |
| K1940 | Per-block streaming compression latency > 20ms (NOT real-time during inline inference)            | target | untested (preempt-blocked, F#669) |

KC semantics note: both KCs are written as **failure** thresholds (the experiment FAILS — i.e. streaming MEMENTO does not preserve task accuracy / is not real-time-deployable — if these conditions are met). Pass requires accuracy ≥ 90% parity AND per-block latency ≤ 20ms. The 20ms threshold is tighter than F#739 K1907's 50ms because streaming-during-inference must keep up with the decoder's token-emission rate, not just human-perceptible 20Hz interactivity.

## §4 Unblock condition

Re-claimable when **all four** of:

1. Parent `exp_memento_gemma4_replication` reaches `status=supported` at full scale via its `_impl` companion (already filed P=3 as `exp_memento_gemma4_replication_impl`), with K1799+K1800+K1801+K1802 SUPPORTED at the pooled OpenMementos training corpus.
2. Parent's `_impl` exposes a callable streaming-mode forward path (compressing blocks INLINE during decoding, not just offline-batch). This is a strict superset of F#739's unblock surface (per-block hook + offline streaming flag) — it additionally requires inline integration with the decoder loop and KV-cache mutation hooks during generation.
3. Parent's `_impl` exposes a downstream-task evaluation harness that can run under both regimes (streaming + batch) on the SAME held-out evaluation set, with task accuracy measurement comparable across regimes (same prompts, same scoring).
4. Parent's K1800 (task accuracy target) measurement on the chosen evaluation set is established as the batch-mode reference point, so K1939's parity ratio has a defined denominator.

**No KC-augmentation needed** at re-claim: K1939 and K1940 are already targets per F#666. A "PPL on held-out eval" proxy would not unblock — it requires the same parent-impl checkpoint and adds no identifying power for streaming-vs-batch task-accuracy parity (PPL is weakly correlated with task accuracy per F#666 r≈0.08).

Alternatively, the experiment scope could be **reduced** at re-claim to evaluate base Gemma 4 inference latency (no MEMENTO compression) as a "real-time baseline" — but that measures the *base model's* decoding cost, not the *MEMENTO streaming compression mechanism*'s per-block cost. Antipattern-t risk (silent objective swap from "streaming MEMENTO compression latency" to "base Gemma 4 decoding cost"). Avoided.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#687/F#698/F#699/F#737/F#738/F#739 precedent + reviewer.md §5. The unblock condition is parent-external: parent's existing `exp_memento_gemma4_replication_impl` (P=3) is a necessary-but-not-sufficient gate; additional parent-side instrumentation (inline streaming-mode forward path with decoder-loop integration + dual-regime evaluation harness) would be required even post-parent-SUPPORTED. This is a parent-extension chain strictly beyond F#739's, not a new child `_impl`.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt:
- Measuring base Gemma 4 inference latency (no MEMENTO compression) and labeling it the "streaming MEMENTO compression latency" target — would measure base-model decoding cost, not the per-block streaming compression mechanism (antipattern-t).
- Comparing base Gemma 4 task accuracy with and without context truncation as a streaming-vs-batch proxy — context-truncation is not the MEMENTO mechanism (antipattern-t; substitutes algorithm).
- Loading the paper's pooled OpenMementos checkpoint (Qwen3/Phi-4/Olmo 3) and timing its block-compression kernel as a Gemma 4 stand-in — would substitute architectures with different attention shapes, KV layouts, and Metal kernel costs (antipattern-t AND antipattern-m proxy-model).
- Using a synthetic block-mask attention reimplementation in pure MLX (without trained MEMENTO weights) and timing it as a "streaming MEMENTO latency" measurement — would measure a hand-written kernel's cost, not the trained mechanism's cost (antipattern-t).
- Reading paper Table 4 throughput numbers and dividing-by-block-count to back-derive per-block streaming latency — would substitute reported offline-batch throughput on Qwen3/Phi-4/Olmo 3 with assumed Gemma 4 + M5 Pro streaming-during-inference behavior; cross-architecture, cross-hardware, AND cross-regime extrapolation invalid (antipattern-t AND antipattern-m).
- Substituting PPL parity for K1939's task-accuracy parity — would replace a behavioral target with a proxy target, weakly correlated per F#666 (antipattern-i KC measures wrong object).

All six shortcuts would replace the streaming MEMENTO mechanism the KCs measure with a proxy.

## §7 Anti-pattern scan

Composition-math: N/A (no composition).
LORA_SCALE: N/A (no LoRA).
shutil.copy: N/A (no code).
Hardcoded `"pass": True`: N/A (no code, `all_pass: false` written).
Eval truncation producing base=0%: N/A (no eval).
Proxy-model substitution: explicitly rejected in §6 (Qwen3/Phi-4/Olmo 3 not used as Gemma 4 stand-in).
KC measures wrong object: K1939/K1940 correctly identify streaming-vs-batch task-accuracy parity and per-block streaming latency of the MEMENTO mechanism (not a proxy), but the mechanism that produces them doesn't exist on this platform → preempt-KILL.
N=smoke reported as full: N/A (no N; `is_smoke: false`).
Tautological routing: N/A (no routing in this experiment).
Thinking-mode truncation: N/A (no eval).
File-existence cache: N/A (no code).
Copy-paste scaffolding: Scaffold derived from `exp_memento_realtime_latency` (4th MEMENTO preempt-KILL, closest sibling) but variant-specific sections (mixed-target-only KC panel, streaming-during-inference regime, 3rd-obs canonicalization) are rewritten, not copy-pasted. KCs and parent-extension requirements distinct from F#739 (F#940's streaming surface is a strict superset of F#739's K1908).
