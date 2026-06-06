# MATH.md — exp_pierre_adapter_cache_prefill (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. No `run_experiment.py` MLX implementation is attempted because the parent dependency is in a state that makes both KCs structurally untestable: pre-fill-during-idle-time latency reduction and pre-fill memory overhead are runtime properties of a multi-adapter serving harness on `pierre-g4e4b` with an idle-time hook, and no such harness — nor the `pierre-g4e4b` model it would run under, nor the idle-time pre-fill scheduler — exists on this platform.

This is the **2nd Pierre-serving-cluster child preempt-KILL** (parent F#570, same cluster as F#740 `exp_pierre_multi_adapter_serving_throughput`). It is the **13th F#669 reuse** overall, following:

1. `exp_memento_compression_ratio_benchmark` (F#699, 1st MEMENTO child — single-config static-KV target).
2. `exp_memento_block_size_ablation` (F#737, 2nd MEMENTO child — scalar-sweep; multi-parent-run sub-axis 1st obs).
3. `exp_memento_cross_domain_transfer` (F#738, 3rd MEMENTO child — categorical cross-corpus; multi-parent-run sub-axis 2nd obs; target-only-behavioral KC panel 1st obs).
4. `exp_memento_realtime_latency` (F#739, 4th MEMENTO child — single-config engineering-target; target-only-engineering KC panel 2nd obs).
5. `exp_pierre_multi_adapter_serving_throughput` (F#740, 1st Pierre-serving child — serving-config spot-measurement at N∈{3,5}; target-only-KC-panel CANONICALIZED via cross-cluster triple-fire).
6. (this) `exp_pierre_adapter_cache_prefill` — **Pierre-serving cluster 2nd child; single-config target-only-engineering (latency-reduction-% + memory-overhead-GB on one idle-time-prefill config); post-canonical reuse of target-only-KC-panel; 1st within-cluster reuse of target-only-KC-panel in Pierre-serving.**

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627) is the Pierre-stack base; a trained `pierre-g4e4b` composite (base + adapter-cache layer + idle-time-prefill scheduler) does **not** exist on disk or in HF cache (per F#570 precondition T1C). **Not loaded.**
- Adapter targets: N/A — K1913/K1914 measure pre-fill latency-hiding and pre-fill-cache memory overhead, not a single adapter's targeting. Single-stack serving would use `v_proj + o_proj` per F#627; adapter-cache pre-fill is a scheduler layer between the serve-request handler and the multi-adapter dispatch layer that does not exist (F#570 T2 — body schema is single-adapter `str`).
- Parent dependency: `exp_prod_mlxlm_integration` (status `killed`, F#570 — 5 preconditions T1B/T1C/T2/T3/DEP all fail).
- Grandparent dependency: `exp_prod_pip_package_pierre` (status `killed`, cited in F#570 DEP).
- Sibling precedents (F#669 family): F#699, F#737, F#738, F#739 (MEMENTO cluster); F#740 (Pierre-serving cluster, sibling); F#655, F#657 (F#570-parent children via F#652 software-infra-unbuilt route, non-F#669 framing).
- Datasets: N/A — latency and memory benchmarks are harness-internal; no eval set is loaded.

## §1 Preempt-KILL theorem

**Theorem (inter-experiment target unverifiability, adapter-cache-prefill variant).** Let `C` denote child experiment `exp_pierre_adapter_cache_prefill` with kill criteria K = {K1913 (target: pre-fill does not reduce first-token latency by > 20%), K1914 (target: pre-fill memory overhead > 2GB)}. Let `P` denote parent experiment `exp_prod_mlxlm_integration`.

K1913 and K1914 each require empirical measurement of a runtime property of a serving harness on `pierre-g4e4b` with an idle-time adapter-cache pre-fill scheduler:

- **K1913** (first-token latency reduction ≥ 20%): `1 - TTFT(prefill-warm) / TTFT(cold)` under the same prompt distribution, hardware state, and user-think-time profile. Requires (a) loading `pierre-g4e4b` base, (b) an adapter-cache pre-fill scheduler that detects idle time between requests and pre-loads the next-likely adapter set, (c) instrumenting time-to-first-token (TTFT) on both the cold path (no pre-fill) and the warm path (cache hit after pre-fill), and (d) a realistic multi-turn serving workload to generate the idle intervals.
- **K1914** (cache memory overhead > 2GB): RSS + unified-memory peak delta introduced by the pre-fill cache versus a no-cache baseline on the same harness. Requires the same harness with instrumented memory probing (`psutil.Process().memory_info().rss` and Metal unified-pool size) on both configurations and a comparable cold-path baseline for attribution.

Both KCs are **dynamic** (harness-dependent, scheduler-dependent, kernel-dependent) and **strictly empirical** — neither can be derived analytically from adapter-config files alone. First-token latency reduction depends on the adapter's loading cost (quantization-path-dependent), the idle-time detection policy (fixed-interval vs workload-driven), the cache replacement strategy (LRU vs heuristic), and the prompt distribution that determines whether the next-likely adapter hypothesis matches reality. Cache memory overhead depends on whether adapters are pre-decoded or stored in packed safetensor form, on Metal Heap pinning behavior for the cache region, and on allocator fragmentation interactions with the base-model KV cache.

Required precondition: a serving harness that (i) loads `pierre-g4e4b`, (ii) accepts multi-adapter selection per-request (to make pre-fill meaningful — if there is only one adapter it is always resident, no pre-fill needed), (iii) exposes an idle-time hook into its request-loop, (iv) exposes TTFT and memory instrumentation on both cold and warm paths, and (v) supports at least two adapter configurations so pre-fill has something to pre-fill.

Per **F#570** (2026-04-18, verified by source inspection of `mlx_lm/server.py:1155,1236` and filesystem state) the parent `exp_prod_mlxlm_integration` is **KILLED** with 5 independent preconditions failed. Mapped onto this child:

1. **T1B failure — no loader plugin API.** `mlx-lm 0.31.2` has only static `mlx_lm.utils.load(path, adapter_path)` (single `adapter_path`), no `mlx_lm.loaders` / `mlx_lm.plugins` / `mlx_lm.providers` entry-point group. **Blocks K1913+K1914 because a registered multi-adapter loader is the minimum substrate for any pre-fill cache to hook into.**
2. **T1C failure — no `pierre-g4e4b` checkpoint.** 0 matches in `~/.cache/huggingface/hub`; no `micro/models/pierre-g4e4b/` dir. **Blocks K1913+K1914 because there is no base model to serve or to measure TTFT against.**
3. **T2 failure — single-adapter body schema.** `mlx_lm/server.py:1236` validates `body["adapter"]` as a single `str`. **Blocks K1913+K1914 because pre-fill is only meaningful when there are multiple adapter identities the server can anticipate; a single-adapter-per-request schema renders pre-fill a no-op (the single adapter is always resident).**
4. **T3 failure — adapter safetensors missing.** `exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` are absent (only `adapter_config.json` exists). **Blocks K1913 cold-vs-warm contrast and K1914 cache-loaded-adapter memory attribution: even a single stack cannot be instantiated, let alone a cache of alternates.**
5. **DEP failure — `exp_prod_pip_package_pierre` KILLED.** The `pip install pierre` UX path is itself preempt-killed upstream. **Blocks the headline `pierre serve --cache-prefill on` CLI that this experiment's K1913/K1914 would exercise.**

Additionally, even under the stronger pre-condition `P.status = supported` (in-tree wrapper built OR upstream mlx-lm fork landed), K1913/K1914 remain strictly stronger than the scope F#570 targets:

- F#570's K1651 targets a single registered loader serving `pierre-g4e4b` with an adapter set — it does not require the serve-loop to expose an idle-time hook for pre-fill scheduling.
- F#570's K1652 targets OpenAI `extra_body` adapter selection pass-through — single-adapter-per-request semantics; pre-fill requires at least two distinct adapter identities the scheduler can choose between for pre-loading.
- F#570's K1653 targets throughput parity with direct Pierre — single-stack comparison, no pre-fill axis, no TTFT cold-vs-warm contrast.

Therefore a parent-extension is required: (a) idle-time hook in the serve-loop (`on_idle`, `before_next_request`, or equivalent); (b) adapter-cache scheduler with a defined pre-fill policy (next-likely-adapter hypothesis, eviction strategy); (c) TTFT instrumentation on both cold and warm paths; (d) memory-overhead probe that isolates the cache region from the base model and KV cache. This is a **parent-extension**, not parent-replication nor a new `_impl` companion under this child.

If `P.status ∈ {killed, open, provisional}` — i.e. no multi-adapter serving harness exists on `pierre-g4e4b` — then:

- **K1913**: `TTFT(warm)` and `TTFT(cold)` are both undefined (no harness, no base model, no idle-time hook). Reduction ratio is `(NaN - NaN) / NaN` → unidentifiable.
- **K1914**: memory overhead of the cache is undefined — no cache layer exists, no `pierre-g4e4b` base to load, and no adapter safetensors to pre-fill into a hypothetical cache. Comparison to 2GB threshold is `NaN > 2GB` → unidentifiable.

∴ Testing K1913 and K1914 while `P.status ≠ supported|proven` plus a parent-extension (idle-time hook + cache scheduler + TTFT and memory-delta instrumentation) produces unidentifiable samples on both. **QED.**

### §1.1 F#666 gating

- K1913 = **target** (first-token latency reduction percentage on M5 Pro is the engineering claim; the 20% floor is the calibrated user-perceived-responsiveness bound — below it, pre-fill does not pay for its added complexity vs accepting the cold-path TTFT).
- K1914 = **target** (cache memory overhead in GB on M5 Pro 48GB is the engineering claim; the 2GB ceiling is the calibrated headroom-budget bound — above it, the cache displaces KV-cache residency and regresses multi-request throughput).
- No proxy KC present. The KC set is therefore F#666-compliant **trivially** by vacuous quantification — F#666 requires every *proxy* KC to be paired with a target; a target-only KC set satisfies the rule unconditionally.

A defensible target-only design: both KCs measure engineering properties (latency-reduction-%, memory-overhead-bytes), where the target IS the measurement. There is no behavioral proxy that could be paired (PPL, accuracy, cosine — none of these proxy for pre-fill latency-hiding or cache memory). A "warm-path PPL stability vs cold-path" proxy would still need the pre-fill cache loaded on the harness, so it would not unblock the experiment; it would only add structural complexity without identifying-power gain. (Further: §6 rejects TTFT-without-cache-layer as antipattern-t silent swap.)

### §1.2 Sub-axis classification (single-config idle-time pre-fill — target-only engineering)

This is a **single-config spot measurement** on one idle-time-prefill configuration, with two distinct engineering metrics (latency-reduction-% and memory-overhead-GB) measured on the same config. Structurally this is closest to **F#699 (single-config, one N)** and **F#739 (single-config engineering-target ×2, MEMENTO cluster)** — NOT a sweep, NOT a spot-measurement at multiple N (unlike F#740's N∈{3,5}), NOT a categorical cross-corpus (unlike F#738).

Classification: **single-config target-only engineering (2 distinct metrics on one config)** — the same sub-axis variant as F#739. This is the **2nd observation of single-config target-only engineering** (1st was F#739 in MEMENTO cluster). Does **not** advance the canonical multi-parent-run sub-axis counter (remains at 2 obs: F#737 scalar-sweep + F#738 categorical). The single-config-target-only-engineering variant itself is not yet a watchlist pattern — would need 1+ more distinct observation beyond F#739 to suggest a cluster; this experiment is a candidate 2nd observation but conservatively reviewer-call.

| Cluster child                                              | Parent cluster     | KC kind composition              | Sub-axis variant                         | F#666 status                 |
| ---------------------------------------------------------- | ------------------ | -------------------------------- | ---------------------------------------- | ---------------------------- |
| `exp_memento_compression_ratio_benchmark` (F#699)          | MEMENTO            | proxy + quasi-target             | single-config                            | F#666-compliant (compound)   |
| `exp_memento_block_size_ablation` (F#737)                  | MEMENTO            | proxy + target (sweep×2)         | scalar hyperparameter sweep              | F#666-compliant (compound)   |
| `exp_memento_cross_domain_transfer` (F#738)                | MEMENTO            | target only (behavioral, ratio)  | categorical cross-corpus                 | F#666-compliant (vacuous)    |
| `exp_memento_realtime_latency` (F#739)                     | MEMENTO            | target only (engineering, ×2)    | single-config engineering                | F#666-compliant (vacuous)    |
| `exp_pierre_multi_adapter_serving_throughput` (F#740)      | Pierre-serving     | target only (engineering, ×2)    | serving-config spot-measurement at N∈{3,5} | F#666-compliant (vacuous)    |
| **(this) `exp_pierre_adapter_cache_prefill`**              | **Pierre-serving** | **target only (engineering, ×2)** | **single-config idle-time pre-fill**     | **F#666-compliant (vacuous)** |

### §1.3 Target-only-KC-panel-under-preempt-KILL: post-canonical reuse

The pattern was **canonicalized at F#740** (3rd obs, cross-cluster triple-fire: F#738 behavioral/MEMENTO + F#739 engineering/MEMENTO + F#740 engineering/Pierre-serving). This experiment is the **4th observation** and the **1st within-cluster reuse inside the Pierre-serving cluster** (both F#740 and this are engineering target-only in Pierre-serving, but on distinct sub-axis variants: F#740 = N-spot-measurement, this = single-config idle-time pre-fill).

**Why the within-cluster reuse matters:** it confirms that the canonical form is cluster-portable, not just cross-cluster-portable. Two distinct sub-axis variants (N-spot-measurement and single-config idle-time pre-fill) inside the same parent cluster both exhibit target-only engineering KC panels with F#666-vacuous-compliance. Post-canonicalization, no further promotion is needed; this observation strengthens the canonical form's within-cluster generalizability.

**What it is NOT:** a new canonicalization event. The pattern is already canonical per F#740. This is counting/tally only.

### §1.4 Pierre-serving-cluster consolidation (emergent)

With 2 Pierre-serving-cluster F#669 children (F#740 + this) and remaining open P≤2 children still tagged `serving`/`p1` (`exp_pierre_adapter_hotswap_latency_impl` at P=2 plus the `_impl` companions of parent F#570's scope), consolidation into a **single preempt-cluster learning** is plausible. Recommendation carried forward from F#740 LEARNINGS §Next claims and Analyst synthesis: if reviewer agrees, remaining Pierre-serving children can be resolved as a batch under one consolidated finding, rather than filing one finding per child. This experiment files its own F#669 13th-reuse finding for now; reviewer may elect to demote individual findings into a single Pierre-serving-consolidation finding at a future iteration.

## §2 Prior art

- **F#669** (2026-04-19) — defining precedent for preempt-KILL on target-unverified parent.
- **F#570** (2026-04-18) — parent KILLED: 5 preconditions (T1B/T1C/T2/T3/DEP) all fail. mlx-lm 0.31.2 has no plugin/loader API; Pierre server path requires in-tree wrapper or upstream fork. Verified by source inspection of `mlx_lm/server.py:1155,1236` and filesystem state.
- **F#655** (prior) — 8th F#502/F#646 hit: ap-017 §s4 T5-K under F#652, parent `exp_prod_mlxlm_integration` KILLED. Confirms Pierre-serving parent's downstream blocking.
- **F#657** (prior) — 28th composition-bug under F#652 software-infra-unbuilt route. Also Pierre-serving-parent child via F#652 framing (not F#669).
- **F#699** (2026-04-24) — 1st MEMENTO-cluster F#669 child preempt-KILL.
- **F#737** (2026-04-24) — 2nd MEMENTO-cluster F#669 child preempt-KILL; 1st obs multi-parent-run sub-axis (scalar sweep).
- **F#738** (2026-04-24) — 3rd MEMENTO-cluster F#669 child preempt-KILL; 2nd obs multi-parent-run sub-axis (categorical cross-corpus); 1st obs target-only-behavioral KC panel.
- **F#739** (2026-04-24) — 4th MEMENTO-cluster F#669 child preempt-KILL; single-config; 2nd obs target-only-engineering KC panel.
- **F#740** (2026-04-24) — 1st Pierre-serving-cluster F#669 child preempt-KILL; serving-config spot-measurement at N∈{3,5}; 3rd obs target-only-KC-panel (cross-cluster triple-fire) → **canonical**.
- **F#666** — target-gated KC discipline. This experiment satisfies F#666 trivially (K1913+K1914 both target).
- **F#702** — parent-availability hygiene (success_criteria empty on this experiment's record — noted, not patched in this iteration since preempt-KILL supersedes hygiene correction).

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                                | Kind   | Measurement status                |
| ----- | -------------------------------------------------------------------- | ------ | --------------------------------- |
| K1913 | Pre-fill doesn't reduce first-token latency by > 20% (FAIL condition) | target | untested (preempt-blocked, F#669) |
| K1914 | Pre-fill memory overhead > 2GB (FAIL condition)                      | target | untested (preempt-blocked, F#669) |

KC semantics note: both KCs are written as **failure** thresholds (the experiment FAILS — i.e. adapter-cache pre-fill is not deployable — if latency reduction ≤ 20% OR cache overhead > 2GB). Pass requires TTFT reduction ≥ 20% AND cache memory overhead ≤ 2GB. Threshold 20% picks the user-perceived-responsiveness bound; 2GB picks the KV-cache-coexistence headroom bound on 48GB unified memory.

## §4 Unblock condition

Re-claimable when **all** of:

1. Parent `exp_prod_mlxlm_integration` reaches `status=supported` via resolution of all 5 F#570 preconditions: (a) T1B — an in-tree wrapper OR an upstream mlx-lm plugin API lands (multi-file change; not a config tweak); (b) T1C — a `pierre-g4e4b` composite lands on disk or in HF cache; (c) T2 — body-schema extended from single-adapter `str` to multi-adapter list OR per-request multi-adapter dispatch; (d) T3 — trained adapter safetensors exist on disk for math/code/medical domains; (e) DEP — `exp_prod_pip_package_pierre` resolves (supported OR the wrapper path is validated without pip-install).
2. **Parent-extension** beyond F#570's scope: (i) idle-time hook in the serve-loop (e.g. `on_idle`, `before_next_request`, or `asyncio.sleep`-based scheduler) with well-defined semantics for when pre-fill may run; (ii) adapter-cache layer with defined pre-fill policy (next-likely-adapter hypothesis) and eviction strategy (LRU, explicit-pin, or other); (iii) TTFT instrumentation distinguishing cold path (cache miss) from warm path (cache hit); (iv) memory-overhead probe that isolates the cache region from the base model and KV cache for clean attribution.

**No KC-augmentation needed** at re-claim: K1913 and K1914 are already targets per F#666. Substituting TTFT-without-cache-layer (measuring only the cold path) would answer a different engineering question (baseline TTFT, not pre-fill latency reduction) — antipattern-t silent swap, rejected in §6.

Alternatively, the experiment scope could be **reduced** at re-claim to cache-overhead-only (drop K1913) if the idle-time hook takes significantly longer to land than the cache layer itself, but that would only answer half the question the current KCs ask; defer and re-file as a distinct narrower experiment if needed.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#687/F#698/F#699/F#737/F#738/F#739/F#740 precedent + reviewer.md §5. The unblock condition is parent-external (F#570 resolution) plus parent-extension (idle-time hook + cache scheduler + TTFT and memory-delta instrumentation); both are at the Pierre-serving-infrastructure layer, not under this child.

Findings to file at completion:
1. **F#669 13th reuse** — cumulative counter advance.
2. **2nd Pierre-serving-cluster F#669 child** — within-cluster reuse after F#740; single-config idle-time pre-fill variant (distinct from F#740's N-spot-measurement).
3. **Target-only-KC-panel-under-preempt-KILL 4th observation** — post-canonical; 1st within-cluster reuse in Pierre-serving cluster; tally-only (pattern already canonicalized at F#740).
4. **Pierre-serving-consolidation watchlist** — reviewer may elect to consolidate remaining Pierre-serving children into one finding if the backlog continues to preempt-KILL identically.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt:
- Running a single-stack `pierre-g4e4b` serve with no cache layer and labeling baseline TTFT as "cold path vs warm path" — with no cache, there is no warm path; measurement would trivially pass K1913 at 0% reduction or fail it at 0% reduction depending on interpretation, both devoid of information (antipattern-t).
- Pre-loading a single adapter at serve startup and timing TTFT for requests that use that single adapter — this measures adapter-at-startup latency, not idle-time-prefill-of-next-likely-adapter latency (antipattern-t silent swap: pre-fill is about the *next* adapter, not the *current* one; the distinction is the entire mechanism under test).
- Using `os.fork`-based warmup to duplicate an in-memory adapter state and declaring the duplicate as "pre-filled cache" — fork-COW does not exercise Metal unified-memory allocator for adapter-cache regions; memory overhead measurement would be a lower bound that underestimates by the amount of demand-paged tensors (antipattern-t).
- Substituting `gemma-4-e4b-it-4bit` base (no Pierre wrapper) for `pierre-g4e4b` and attaching a custom pre-fill hook outside `mlx_lm.server` — bypasses the body-schema path K1913's first-token latency actually runs through (F#570 T2); what is measured is function-call latency of a bespoke harness, not serve-request latency (antipattern-m proxy-model-substitution).
- Using `torch` to prefetch safetensors into OS page cache and reporting the reduced load-time as "pre-fill latency reduction" — OS page cache is not the MLX unified-memory pool; prefetched bytes still pay Metal upload cost when the model actually loads them (antipattern-t AND antipattern-m).
- Back-deriving K1914 cache overhead from `os.path.getsize(adapter.safetensors)` — ignores tensor deserialization overhead, Metal Heap pinning, and allocator fragmentation interactions with the base-model KV cache; engineering claim requires wall-clock memory-pressure measurement (antipattern-t).

All six shortcuts would replace the adapter-cache-prefill mechanism the KCs measure with a proxy or substitute.

## §7 Anti-pattern scan

Composition-math: N/A (no composition).
LORA_SCALE: N/A (no LoRA).
shutil.copy: N/A (no code).
Hardcoded `"pass": True`: N/A (no code, `all_pass: false` written).
Eval truncation producing base=0%: N/A (no eval).
Proxy-model substitution: explicitly rejected in §6 (`gemma-4-e4b-it-4bit` base without Pierre wrapper, `torch` prefetch into OS page cache, `os.fork` duplicate-as-cache all rejected as stand-ins).
KC measures wrong object: K1913/K1914 correctly identify first-token latency reduction from idle-time pre-fill and pre-fill cache memory overhead on the `pierre-g4e4b` multi-adapter serving harness with an idle-time hook (not a proxy), but the harness itself doesn't exist on this platform → preempt-KILL.
N=smoke reported as full: N/A (no N; `is_smoke: false`).
Tautological routing: N/A (no routing in this experiment; K1913's pre-fill hypothesis is a cache-layer policy decision, not adapter-routing).
Thinking-mode truncation: N/A (no eval).
File-existence cache: N/A (no code).
Copy-paste scaffolding: Scaffold derived from `exp_pierre_multi_adapter_serving_throughput` (F#740, 1st Pierre-serving preempt-KILL sibling) but cluster-child-specific sections (K1913/K1914 pre-fill mechanism vs K1911/K1912 concurrent throughput+memory; single-config sub-axis vs N-spot-measurement sub-axis; post-canonical reuse vs canonicalization event) are rewritten, not copy-pasted. Parent-extension requirements differ (idle-time hook + cache scheduler vs concurrency scheduler + N≥5 residency). F#570 preconditions inherited verbatim since parent state has not changed.
