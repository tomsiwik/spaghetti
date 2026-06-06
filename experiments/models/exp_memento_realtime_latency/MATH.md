# MATH.md — exp_memento_realtime_latency (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. No `run_experiment.py` MLX implementation is attempted because the parent dependency is in a state that makes the KCs structurally untestable: per-block latency and streaming-vs-batch latency are runtime properties of MEMENTO block-mask attention forward passes, and no Gemma-4-MEMENTO forward pass exists to measure.

This is the **4th MEMENTO-cluster preempt-KILL** in the drain window, following:
1. `exp_memento_compression_ratio_benchmark` (F#699, 4th F#669 reuse — canonical single-config; static KV-memory target).
2. `exp_memento_block_size_ablation` (F#737, ≥9th F#669 reuse — 1st observation of the multi-parent-run sub-axis, scalar-hyperparameter-sweep variant).
3. `exp_memento_cross_domain_transfer` (F#738, ≥10th F#669 reuse — 2nd observation of the multi-parent-run sub-axis, categorical cross-training-corpus variant).
4. (this) `exp_memento_realtime_latency` — **single-config engineering-target-only** variant; minor design observation in §1.2 (no new sub-axis canonicalization).

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — MEMENTO is 2-stage SFT + block-mask attention, not LoRA. Block size default 512 per paper.
- Parent dependency: `exp_memento_gemma4_replication` (status `provisional`, F#685).
- Sibling precedents: `exp_memento_compression_ratio_benchmark` (F#699), `exp_memento_block_size_ablation` (F#737), `exp_memento_cross_domain_transfer` (F#738).
- Datasets: N/A — latency benchmarks are model-internal; no eval set is loaded.

## §1 Preempt-KILL theorem

**Theorem (inter-experiment target unverifiability, runtime-engineering variant).** Let `C` denote child experiment `exp_memento_realtime_latency` with kill criteria K = {K1907 (target: per-block compression latency > 50ms on M5 Pro), K1908 (target: streaming compression latency > 2x batch compression latency)}. Let `P` denote parent experiment `exp_memento_gemma4_replication`.

K1907 and K1908 each require empirical wall-clock measurement of a Gemma-4-MEMENTO forward pass — specifically:

- **K1907** (per-block latency): time-to-compress one MEMENTO block (segment-and-summarize trace generation + block-mask KV insertion) on the actual model. Requires loading a trained Gemma-4-MEMENTO checkpoint into MLX, running its block-compression kernel, and recording per-block wall-clock on M5 Pro Metal.
- **K1908** (streaming vs batch): time-to-compress N blocks one-at-a-time (incremental, used during real-time inference) versus all-at-once (batch, used during training/offline preparation). Requires the same checkpoint executed under two scheduling regimes and a wall-clock comparison.

Both KCs are **dynamic** (kernel-dependent, hardware-specific) and **strictly empirical** — neither can be derived analytically from architecture alone. Latency depends on Metal kernel selection, MLX lazy-eval boundaries, KV-cache layout in unified memory, and tile-size decisions made at compile time. Even an analytical FLOP-budget estimate is insufficient: per-block latency at 50ms granularity is dominated by memory-bandwidth and dispatch overhead, not arithmetic — both are kernel-implementation-dependent and cannot be predicted from the paper's algorithm spec.

Required precondition: a Gemma-4-MEMENTO checkpoint loadable into MLX with a callable block-compression forward path.

1. **No public MEMENTO checkpoint exists for Gemma 4 at any training mixture.** MEMENTO paper (Kontonis et al., arxiv:2604.09852, Apr 2026) released checkpoints for Qwen3 / Phi-4 / Olmo 3 only, on the paper's pooled OpenMementos 228K-trace dataset. No Gemma-4-MEMENTO at any mixture; no per-block kernel exposed in `mlx_lm` for the paper's architecture-specific block-mask attention.
2. **MEMENTO 2-stage SFT + block-mask attention is not executable via `mlx_lm.lora` CLI** (parent F#685 finding). Even if a checkpoint existed, the runtime would need a custom MLX implementation of the segment-and-summarize trace generation loop and block-mask-aware attention, neither of which exists in `mlx_lm` ≤ 0.31.
3. Parent's `_impl` (`exp_memento_gemma4_replication_impl`, P=3) would, if SUPPORTED, validate behavioral KCs (K1799 KV reduction, K1800 task accuracy, K1801 KV-channel ablation, K1802 throughput) at the paper-default pooled mixture. **K1802 (throughput) is parent-validated; K1907/K1908 are strictly stronger** — per-block latency at 50ms granularity is finer-grained than throughput tokens-per-second, and streaming-vs-batch is an additional regime comparison parent's K1802 does not exercise.

If `P.status ∈ {provisional, open}` — i.e. no Gemma-4-MEMENTO checkpoint exists — then:

- **K1907**: per-block compression latency is undefined; no kernel to time. Comparison to 50ms threshold is `NaN > 50ms` — unidentifiable.
- **K1908**: streaming vs batch latency ratio is `NaN / NaN` — unidentifiable.

Additionally, even under the stronger pre-condition `P.status = supported` (parent fully validated at pooled mixture), K1907 remains the natural finer-grained successor to parent's K1802 (throughput target) — re-claim is straightforward at that point. K1908 (streaming regime) would still require a parent-extension to surface the incremental compression API; parent's K1802 measures batch throughput only.

∴ Testing K1907 and K1908 while `P.status ≠ supported|proven` produces unidentifiable samples on both. **QED.**

### §1.1 F#666 gating

- K1907 = **target** (latency on M5 Pro is the engineering claim being made; threshold 50ms is the calibrated real-time bound).
- K1908 = **target** (streaming-vs-batch ratio is the engineering claim; threshold 2x is the calibrated regression bound).
- No proxy KC present. The KC set is therefore F#666-compliant **trivially** by vacuous quantification — F#666 requires every *proxy* KC to be paired with a target; a target-only KC set satisfies this rule unconditionally.

A defensible target-only design: both KCs measure engineering properties (latency), where the target IS the measurement. There is no behavioral proxy that could be paired (PPL, accuracy, cosine — none of these proxy for latency). A "tokens/sec on a held-out distribution" proxy would still need the same parent-impl checkpoint, so it would not unblock the experiment; it would only add structural complexity without identifying-power gain.

### §1.2 Sub-axis classification (single-config, engineering-target-only — minor design observation)

This is a **single-config preempt-KILL** in the same structural class as F#699 (`exp_memento_compression_ratio_benchmark`): the KCs measure properties of a single parent configuration (default block size 512 at pooled-mixture training), not a sweep over parent configurations. **No new sub-axis observation**; the multi-parent-run sub-axis (block_size_ablation 1st obs, cross_domain_transfer 2nd obs) is **not** advanced by this experiment.

A *minor design observation* worth recording (non-canonical, no triple-fire rule applies):

| Cluster child                          | KC kind composition                | F#666 status                          |
| -------------------------------------- | ---------------------------------- | ------------------------------------- |
| `exp_memento_compression_ratio_benchmark` (F#699) | proxy + quasi-target               | F#666-compliant (compound)            |
| `exp_memento_block_size_ablation` (F#737)         | proxy + target (sweep×2)           | F#666-compliant (compound, sweep)     |
| `exp_memento_cross_domain_transfer` (F#738)       | target only (behavioral, ratio)    | F#666-compliant (vacuous)             |
| **(this) `exp_memento_realtime_latency`**         | **target only (engineering, ×2)**  | **F#666-compliant (vacuous)**         |

Engineering-target-only KC sets satisfy F#666 trivially because no proxy is being asserted. This is the 2nd MEMENTO-cluster child with a target-only panel after F#738; the prior was behavioral (accuracy ratio), this one is engineering (latency thresholds). Two observations is below the triple-fire canonicalization threshold; logged as a watchlist micro-pattern, not promoted.

## §2 Prior art

- **F#669** (2026-04-19) — defining precedent for preempt-KILL on target-unverified parent.
- **F#685** (2026-04-23) — parent PROVISIONAL: design-only, 4 target-gated KCs untested, MEMENTO 2-stage SFT + block-mask attention not executable via `mlx_lm.lora` CLI.
- **F#699** (2026-04-24) — 4th F#669 reuse, **1st MEMENTO-cluster** child preempt-KILL: `exp_memento_compression_ratio_benchmark`. F#666-compliant (proxy+target). Single-config static-KV-memory target.
- **F#737** (2026-04-24) — **2nd MEMENTO-cluster** child preempt-KILL: `exp_memento_block_size_ablation`. F#666-compliant. 1st observation of multi-parent-run sub-axis (scalar-sweep variant).
- **F#738** (2026-04-24) — **3rd MEMENTO-cluster** child preempt-KILL: `exp_memento_cross_domain_transfer`. F#666-compliant (target-only). 2nd observation of multi-parent-run sub-axis (categorical cross-corpus variant).
- **F#698** (2026-04-24) — 3rd F#669 reuse + first F#666 compound sub-case.
- **F#666** — target-gated KC discipline. This experiment satisfies F#666 trivially (K1907+K1908 both target).
- Kontonis et al. arxiv:2604.09852 — MEMENTO paper. Authors released Qwen3 / Phi-4 / Olmo 3 checkpoints on pooled OpenMementos (228K traces); no Gemma 4 at any mixture. Paper Table 4 reports throughput comparisons but not per-block latency at the 50ms-real-time granularity this KC asserts.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                                        | Kind   | Measurement status                |
| ----- | ---------------------------------------------------------------------------- | ------ | --------------------------------- |
| K1907 | Per-block compression latency > 50ms on M5 Pro (NOT real-time)               | target | untested (preempt-blocked, F#669) |
| K1908 | Streaming (incremental) compression latency > 2x batch compression latency   | target | untested (preempt-blocked, F#669) |

KC semantics note: both KCs are written as **failure** thresholds (the experiment FAILS — i.e. mechanism is not real-time / streaming has unacceptable overhead — if these conditions are met). Pass requires latency ≤ 50ms per block AND streaming-overhead ≤ 2x. Threshold 50ms picks 20Hz human-perceptible interactivity; 2x is a soft regression bound below which streaming-mode is engineering-deployable.

## §4 Unblock condition

Re-claimable when **both** of:

1. Parent `exp_memento_gemma4_replication` reaches `status=supported` at full scale via its `_impl` companion (already filed P=3 as `exp_memento_gemma4_replication_impl`), with K1799+K1800+K1801+K1802 SUPPORTED at the pooled OpenMementos training corpus.
2. Parent's `_impl` exposes a callable per-block compression API and an incremental (streaming) inference path. K1802 (throughput target) on parent measures batch tokens/sec, not per-block wall-clock or streaming-vs-batch ratio. The parent `_impl` would need a small extension (latency-instrumented forward pass + a streaming-mode flag) to surface the measurement surface K1907/K1908 require. This is a parent-side engineering tightening, not a separate `_impl` companion under this child.

**No KC-augmentation needed** at re-claim: K1907 and K1908 are already targets per F#666. A "tokens/sec held-out throughput" proxy would not unblock — it requires the same parent-impl checkpoint and adds no identifying power for per-block / streaming claims.

Alternatively, the experiment scope could be **reduced** at re-claim to evaluate base Gemma 4 inference latency without MEMENTO compression as a "real-time baseline" — but that measures the *base model's* latency, not the *MEMENTO mechanism's* per-block compression cost. Antipattern-t risk (silent objective swap from "MEMENTO real-time compression latency" to "base Gemma 4 inference latency"). Avoided.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#687/F#698/F#699/F#737/F#738 precedent + reviewer.md §5. The unblock condition is parent-external: parent's existing `exp_memento_gemma4_replication_impl` (P=3) is a necessary-but-not-sufficient gate; additional parent-side instrumentation (per-block latency hook + streaming-mode forward path) would be required even post-parent-SUPPORTED. This is a parent-extension, not a new child `_impl`.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt:
- Measuring base Gemma 4 inference latency (no MEMENTO compression) and labeling it the "MEMENTO real-time compression latency" target — would measure base-model decoding cost, not the per-block compression mechanism (antipattern-t).
- Loading the paper's pooled OpenMementos checkpoint (Qwen3/Phi-4/Olmo 3) and timing its per-block kernel as a Gemma 4 stand-in — would substitute architectures with different attention shapes, KV layouts, and Metal kernel costs (antipattern-t AND antipattern-m proxy-model).
- Using a synthetic block-mask attention reimplementation in pure MLX (without the trained MEMENTO weights) and timing it as a "MEMENTO latency" measurement — would measure a hand-written kernel's cost, not the trained mechanism's cost (antipattern-t; the kernel selection a real implementation makes depends on weight shapes and quantization, not just topology).
- Reading paper Table 4 throughput numbers and dividing-by-block-count to back-derive per-block latency — would substitute reported throughput on Qwen3/Phi-4/Olmo 3 with assumed Gemma 4 + M5 Pro behavior; cross-architecture and cross-hardware extrapolation invalid (antipattern-t AND antipattern-m).

All four shortcuts would replace the per-block / streaming MEMENTO mechanism the KCs measure with a proxy.

## §7 Anti-pattern scan

Composition-math: N/A (no composition).
LORA_SCALE: N/A (no LoRA).
shutil.copy: N/A (no code).
Hardcoded `"pass": True`: N/A (no code, `all_pass: false` written).
Eval truncation producing base=0%: N/A (no eval).
Proxy-model substitution: explicitly rejected in §6 (Qwen3/Phi-4/Olmo 3 not used as Gemma 4 stand-in).
KC measures wrong object: K1907/K1908 correctly identify per-block / streaming-vs-batch latency of the MEMENTO mechanism (not a proxy), but the mechanism that produces the latency doesn't exist on this platform → preempt-KILL.
N=smoke reported as full: N/A (no N; `is_smoke: false`).
Tautological routing: N/A (no routing in this experiment).
Thinking-mode truncation: N/A (no eval).
File-existence cache: N/A (no code).
Copy-paste scaffolding: Scaffold derived from `exp_memento_cross_domain_transfer` (3rd MEMENTO preempt-KILL) but variant-specific sections (engineering-target latency vs categorical-corpus behavioral) are rewritten, not copy-pasted. KCs and parent-extension requirements distinct.
