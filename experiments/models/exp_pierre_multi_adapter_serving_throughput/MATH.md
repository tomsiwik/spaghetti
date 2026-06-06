# MATH.md — exp_pierre_multi_adapter_serving_throughput (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. No `run_experiment.py` MLX implementation is attempted because the parent dependency is in a state that makes both KCs structurally untestable: N=3 concurrent-stack throughput-ratio and N=5 concurrent-stack peak-memory are runtime properties of a multi-adapter serving harness on `pierre-g4e4b`, and no such harness — nor the `pierre-g4e4b` model it would run under — exists on this platform.

This is the **1st Pierre-serving-cluster child preempt-KILL** (new cluster — parent F#570, distinct from the MEMENTO cluster). It is the **12th F#669 reuse** overall, following:

1. `exp_memento_compression_ratio_benchmark` (F#699, 1st MEMENTO child — single-config static-KV target).
2. `exp_memento_block_size_ablation` (F#737, 2nd MEMENTO child — scalar-sweep; multi-parent-run sub-axis 1st obs).
3. `exp_memento_cross_domain_transfer` (F#738, 3rd MEMENTO child — categorical cross-corpus; multi-parent-run sub-axis 2nd obs; target-only-behavioral KC panel 1st obs).
4. `exp_memento_realtime_latency` (F#739, 4th MEMENTO child — single-config engineering-target; target-only-engineering KC panel 2nd obs).
5. (this) `exp_pierre_multi_adapter_serving_throughput` — **Pierre-serving cluster, target-only-engineering KC panel 3rd obs (cross-cluster triple-fire), spot-measurement-at-N∈{3,5} serving-config class.**

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627) is the Pierre-stack base; a trained `pierre-g4e4b` composite (base + multi-adapter runtime-dispatch) does **not** exist on disk or in HF cache (per F#570 precondition T1C). **Not loaded.**
- Adapter targets: N/A — K1911/K1912 measure concurrent-serving throughput and memory, not a single adapter's targeting. Single-stack serving would use `v_proj + o_proj` per F#627; multi-stack concurrent serving needs a multi-adapter dispatch layer that does not exist (F#570 T2 — body schema is single-adapter `str`).
- Parent dependency: `exp_prod_mlxlm_integration` (status `killed`, F#570 — 5 preconditions T1B/T1C/T2/T3/DEP all fail).
- Grandparent dependency: `exp_prod_pip_package_pierre` (status `killed`, cited in F#570 DEP).
- Sibling precedents (F#669 family): F#699, F#737, F#738, F#739 (MEMENTO cluster); F#655, F#657 (F#570-parent children via F#652 software-infra-unbuilt route, non-F#669 framing).
- Datasets: N/A — throughput and memory benchmarks are harness-internal; no eval set is loaded.

## §1 Preempt-KILL theorem

**Theorem (inter-experiment target unverifiability, multi-adapter-serving variant).** Let `C` denote child experiment `exp_pierre_multi_adapter_serving_throughput` with kill criteria K = {K1911 (target: N=3 concurrent-adapter-stack throughput < 50% of single-stack), K1912 (target: memory > 40GB at N=5 concurrent-adapter-stacks, exceeds M5 Pro 48GB)}. Let `P` denote parent experiment `exp_prod_mlxlm_integration`.

K1911 and K1912 each require empirical measurement of a runtime property of a multi-adapter concurrent-serving harness on `pierre-g4e4b`:

- **K1911** (N=3 throughput ratio): `tok/s(N=3 concurrent stacks) / tok/s(N=1 single stack)` under the same prompt distribution and hardware state. Requires (a) loading `pierre-g4e4b` base, (b) loading 3 trained adapter sets into runtime memory simultaneously, (c) dispatching per-request among them under realistic concurrency (≥3 concurrent clients), (d) wall-clocking aggregate tokens/sec, and (e) a comparable N=1 baseline on the same harness.
- **K1912** (N=5 peak memory): RSS + unified-memory peak at N=5 concurrently loaded adapter stacks during an active serving workload. Requires the same harness at N=5 with instrumented memory probing (`psutil.Process().memory_info().rss` and Metal unified-pool size).

Both KCs are **dynamic** (harness-dependent, concurrency-scheduler-dependent, kernel-dependent) and **strictly empirical** — neither can be derived analytically from adapter-config files alone. Throughput at N=3 depends on Metal dispatch scheduling, KV-cache sharing policy, adapter hot-swap vs resident strategy, request-batching discipline, and quantization-path selection. Peak memory at N=5 depends on whether adapters are resident-all (5× baseline) or LRU-paged (≤ 2× baseline), and on the unified-memory allocator's fragmentation behavior.

Required precondition: a serving harness that (i) loads `pierre-g4e4b`, (ii) accepts multi-adapter selection per-request, (iii) can hold N≥5 adapter stacks concurrently, and (iv) exposes throughput and memory instrumentation.

Per **F#570** (2026-04-18, verified by source inspection of `mlx_lm/server.py:1155,1236` and filesystem state) the parent `exp_prod_mlxlm_integration` is **KILLED** with 5 independent preconditions failed. Mapped onto this child:

1. **T1B failure — no loader plugin API.** `mlx-lm 0.31.2` has only static `mlx_lm.utils.load(path, adapter_path)` (single `adapter_path`), no `mlx_lm.loaders` / `mlx_lm.plugins` / `mlx_lm.providers` entry-point group. **Blocks K1911+K1912 because a registered multi-adapter loader is the minimum substrate for concurrent-stack serving.**
2. **T1C failure — no `pierre-g4e4b` checkpoint.** 0 matches in `~/.cache/huggingface/hub`; no `micro/models/pierre-g4e4b/` dir. **Blocks K1911+K1912 because there is no base model to serve.**
3. **T2 failure — single-adapter body schema.** `mlx_lm/server.py:1236` validates `body["adapter"]` as a single `str`. **Blocks K1911+K1912 because per-request multi-adapter selection (or concurrent distinct adapter requests) is not dispatchable by the server body schema.**
4. **T3 failure — adapter safetensors missing.** `exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` are absent (only `adapter_config.json` exists). **Blocks K1911 baseline and K1912 resident-set: even a single stack cannot be instantiated, let alone three or five.**
5. **DEP failure — `exp_prod_pip_package_pierre` KILLED.** The `pip install pierre` UX path is itself preempt-killed upstream. **Blocks the headline `pierre serve --adapters {a,b,c}` CLI that N=3 concurrency requires.**

Additionally, even under the stronger pre-condition `P.status = supported` (in-tree wrapper built OR upstream mlx-lm fork landed), K1911/K1912 remain strictly stronger than the scope F#570 targets:

- F#570's K1651 targets a single registered loader serving `pierre-g4e4b` with an adapter set — it does not require the server to hold **N≥2** adapter stacks **concurrently** nor to dispatch per-request among them.
- F#570's K1652 targets OpenAI `extra_body` adapter selection pass-through — single-adapter-per-request semantics, not N concurrent distinct stacks.
- F#570's K1653 targets throughput parity with direct Pierre — single-stack comparison, no concurrency axis.

Therefore a parent-extension is required: (a) body-schema extension to list-of-adapter per request or to per-request-routing across N resident stacks; (b) concurrency scheduler for per-request adapter-stack dispatch; (c) memory-probe instrumentation at N=5. This is a **parent-extension**, not parent-replication nor a new `_impl` companion under this child.

If `P.status ∈ {killed, open, provisional}` — i.e. no multi-adapter serving harness exists on `pierre-g4e4b` — then:

- **K1911**: `tok/s(N=3)` is undefined (no N=3 path), and `tok/s(N=1)` baseline is also undefined (F#570 T3 — no adapter safetensors to load a single stack either). Ratio is `NaN / NaN` → unidentifiable.
- **K1912**: memory at N=5 is undefined — no N=5 loader exists and no `pierre-g4e4b` base to load into memory. Comparison to 40GB threshold is `NaN > 40GB` → unidentifiable.

∴ Testing K1911 and K1912 while `P.status ≠ supported|proven` plus a parent-extension (multi-adapter body-schema + concurrency scheduler + memory instrumentation) produces unidentifiable samples on both. **QED.**

### §1.1 F#666 gating

- K1911 = **target** (throughput ratio on M5 Pro is the engineering claim; the 50% floor is the calibrated deployment bound for multi-tenant serving — below it, concurrent serving is not cost-viable vs request-queueing a single stack).
- K1912 = **target** (peak memory in GB on M5 Pro 48GB is the engineering claim; the 40GB ceiling is the calibrated OS-headroom bound — above it, the harness OOMs under background OS load).
- No proxy KC present. The KC set is therefore F#666-compliant **trivially** by vacuous quantification — F#666 requires every *proxy* KC to be paired with a target; a target-only KC set satisfies the rule unconditionally.

A defensible target-only design: both KCs measure engineering properties (throughput ratio, peak memory), where the target IS the measurement. There is no behavioral proxy that could be paired (PPL, accuracy, cosine — none of these proxy for serving throughput or peak memory). A "per-stack PPL stability at N=3" proxy would still need all 3 trained adapter stacks loadable in the same runtime, so it would not unblock the experiment; it would only add structural complexity without identifying-power gain. (Further: §6 rejects N=1-serial-proxy as an antipattern-t silent swap.)

### §1.2 Sub-axis classification (spot-measurement-at-N∈{3,5}, target-only-engineering — cross-cluster triple-fire)

This is a **spot-measurement at two concurrency depths** (N=3 for K1911, N=5 for K1912, with an implicit N=1 baseline for K1911's ratio). It is neither a single-config (N fixed at one value) nor a canonical multi-point sweep (K1911/K1912 measure distinct metrics at different N points, not the same metric across a range). Structurally, it is a **2-point serving-config spot-measurement** — closer in spirit to F#737 (scalar-sweep) than F#699 (single-config), but more sparse: no full sweep, no curve. I classify it as its own variant: **"serving-config spot-measurement (target-only engineering)"**. Conservative: this **does not automatically advance** the multi-parent-run sub-axis — canonicalization of that sub-axis is reviewer-call per mem-pattern-triple-fire.

A *minor design observation* worth recording (non-canonical for serving-config-spot-measurement as sub-axis; canonicalizing for target-only-KC-panel-under-preempt-KILL micro-pattern — see §1.3):

| Cluster child                                              | Parent cluster   | KC kind composition              | Sub-axis variant                   | F#666 status                 |
| ---------------------------------------------------------- | ---------------- | -------------------------------- | ---------------------------------- | ---------------------------- |
| `exp_memento_compression_ratio_benchmark` (F#699)          | MEMENTO          | proxy + quasi-target             | single-config                      | F#666-compliant (compound)   |
| `exp_memento_block_size_ablation` (F#737)                  | MEMENTO          | proxy + target (sweep×2)         | scalar hyperparameter sweep        | F#666-compliant (compound)   |
| `exp_memento_cross_domain_transfer` (F#738)                | MEMENTO          | target only (behavioral, ratio)  | categorical cross-corpus           | F#666-compliant (vacuous)    |
| `exp_memento_realtime_latency` (F#739)                     | MEMENTO          | target only (engineering, ×2)    | single-config engineering          | F#666-compliant (vacuous)    |
| **(this) `exp_pierre_multi_adapter_serving_throughput`**   | **Pierre-serving** | **target only (engineering, ×2)** | **serving-config spot-measurement at N∈{3,5}** | **F#666-compliant (vacuous)** |

### §1.3 Target-only-KC-panel-under-preempt-KILL: triple-fire canonicalization (cross-cluster)

The prior watchlist state (pre-this-experiment) tracked 2 observations of a "target-only KC panel (F#666-compliant by vacuous quantification) on an F#669 child" micro-pattern, both within the MEMENTO cluster:

1. **F#738** — target-only behavioral (accuracy ratio, MEMENTO cluster).
2. **F#739** — target-only engineering (latency ×2, MEMENTO cluster).

This experiment is the **3rd observation**, cross-cluster (Pierre-serving cluster, parent F#570). Cross-cluster independence is the strongest form of triple-fire: the micro-pattern is not confined to a single parent's idiosyncrasies.

**Promotion:** target-only-KC-panel-under-preempt-KILL canonicalizes at this observation (3rd obs, 1 cross-cluster). Recorded as `F#669 target-only-panel canonical variant` in §5 findings dispatch.

**Why it matters:** engineering targets (throughput, memory, latency) and behavioral targets (accuracy ratio) both admit F#666-compliance via vacuous quantification because neither requires a pairable proxy. Proxies would, in every observed case, require the same parent-impl precondition and add no identifying power. Canonicalizing this variant saves future researchers from synthesizing a spurious proxy just to meet F#666's pairing rule by form rather than intent.

## §2 Prior art

- **F#669** (2026-04-19) — defining precedent for preempt-KILL on target-unverified parent.
- **F#570** (2026-04-18) — parent KILLED: 5 preconditions (T1B/T1C/T2/T3/DEP) all fail. mlx-lm 0.31.2 has no plugin/loader API; Pierre server path requires in-tree wrapper or upstream fork. Verified by source inspection of `mlx_lm/server.py:1155,1236` and filesystem state.
- **F#655** (prior) — 8th F#502/F#646 hit: ap-017 §s4 T5-K under F#652, parent `exp_prod_mlxlm_integration` KILLED. Confirms Pierre-serving parent's downstream blocking.
- **F#657** (prior) — 28th composition-bug under F#652 software-infra-unbuilt route. Also Pierre-serving-parent child via F#652 framing (not F#669).
- **F#699** (2026-04-24) — 1st MEMENTO-cluster F#669 child preempt-KILL.
- **F#737** (2026-04-24) — 2nd MEMENTO-cluster F#669 child preempt-KILL; 1st obs multi-parent-run sub-axis (scalar sweep).
- **F#738** (2026-04-24) — 3rd MEMENTO-cluster F#669 child preempt-KILL; 2nd obs multi-parent-run sub-axis (categorical cross-corpus); 1st obs target-only-behavioral KC panel.
- **F#739** (2026-04-24) — 4th MEMENTO-cluster F#669 child preempt-KILL; single-config; 2nd obs target-only-engineering KC panel.
- **F#666** — target-gated KC discipline. This experiment satisfies F#666 trivially (K1911+K1912 both target).
- **F#702** — parent-availability hygiene (success_criteria empty on this experiment's record — noted, not patched in this iteration since preempt-KILL supersedes hygiene correction).

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                                           | Kind   | Measurement status                |
| ----- | ------------------------------------------------------------------------------- | ------ | --------------------------------- |
| K1911 | N=3 concurrent adapter stacks throughput < 50% of single-stack                  | target | untested (preempt-blocked, F#669) |
| K1912 | Memory usage > 40GB at N=5 stacks (exceeds M5 Pro 48GB)                         | target | untested (preempt-blocked, F#669) |

KC semantics note: both KCs are written as **failure** thresholds (the experiment FAILS — i.e. concurrent serving is not deployable — if the ratio drops below 50% or memory exceeds 40GB). Pass requires throughput ≥ 50% of single-stack at N=3 AND peak memory ≤ 40GB at N=5. Threshold 50% picks the cost-viability bound (below which request-queueing a single stack dominates); 40GB picks the OS-headroom bound on 48GB unified memory.

## §4 Unblock condition

Re-claimable when **all** of:

1. Parent `exp_prod_mlxlm_integration` reaches `status=supported` via resolution of all 5 F#570 preconditions: (a) T1B — an in-tree wrapper OR an upstream mlx-lm plugin API lands (multi-file change; not a config tweak); (b) T1C — a `pierre-g4e4b` composite lands on disk or in HF cache; (c) T2 — body-schema extended from single-adapter `str` to multi-adapter list OR per-request multi-adapter dispatch; (d) T3 — trained adapter safetensors exist on disk for math/code/medical domains; (e) DEP — `exp_prod_pip_package_pierre` resolves (supported OR the wrapper path is validated without pip-install).
2. **Parent-extension** beyond F#570's scope: (i) concurrency scheduler holds N≥5 resident adapter stacks simultaneously in unified memory (or documents an LRU-paged policy that still satisfies the sub-ms per-request dispatch latency K1911's ratio implicitly requires); (ii) memory-probe instrumentation surfaces RSS + Metal-unified-pool peak under active serving workload; (iii) concurrency driver exercises ≥3 concurrent clients with realistic prompt distribution for K1911's throughput measurement.

**No KC-augmentation needed** at re-claim: K1911 and K1912 are already targets per F#666. Substituting a serial N=1-then-N=3-then-N=5 timing (no concurrency) would answer a different engineering question (sequential stack-switching cost, not concurrent-serving cost) — antipattern-t silent swap, rejected in §6.

Alternatively, the experiment scope could be **reduced** at re-claim to N=2 concurrent stacks only (drop N=5 memory KC), but that would not answer the multi-tenant serving question the current KCs ask; defer and re-file as a distinct narrower experiment if needed.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#687/F#698/F#699/F#737/F#738/F#739 precedent + reviewer.md §5. The unblock condition is parent-external (F#570 resolution) plus parent-extension (multi-adapter scheduler + memory instrumentation); both are at the Pierre-serving-infrastructure layer, not under this child.

Findings to file at completion:
1. **F#669 12th reuse** — cumulative counter advance.
2. **Target-only-KC-panel-under-preempt-KILL canonical variant** — triple-fire with cross-cluster independence (F#738 + F#739 + this); promotes watchlist → canonical.
3. **1st Pierre-serving-cluster F#669 child** — new cluster beyond MEMENTO; no pattern-triple-fire yet for this cluster specifically.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt:
- Running a single-stack `pierre-g4e4b` serve and declaring 100% throughput-ratio by single-stack-equivalence — trivially meets K1911 by dodging the concurrency axis (antipattern-t).
- Using N=1-sequential timings of three different single-stack serves and labeling the sum as "N=3 concurrent throughput" — measures sequential stack-switching cost, not concurrent-serving cost (antipattern-t silent swap).
- Loading 5 untrained `adapter_config.json` shells (no safetensors) and probing memory — adapter-config alone allocates no runtime tensors; the measurement would underestimate by 5× or more (antipattern-t AND antipattern-m proxy-model).
- Substituting `gemma-4-e4b-it-4bit` base (no Pierre wrapper) for `pierre-g4e4b` and timing its concurrent-serving throughput via `mlx_lm.server` — `mlx_lm.server` does not dispatch per-request adapter selection (F#570 T2); what it measures is base-model serving without adapter math (antipattern-m proxy-model-substitution).
- Using vLLM / llama.cpp / another server that does accept multi-adapter requests, and reporting their numbers as `pierre-g4e4b` multi-adapter throughput — cross-framework quantization paths and Metal-kernel selections differ; paper-equivalent comparisons are not transferable (antipattern-m).
- Back-deriving K1912 memory from an analytical `5 * adapter_size + base_size` sum — ignores unified-memory allocator fragmentation, KV-cache footprint scaling with concurrency, and Metal Heap pinning; engineering claim requires wall-clock measurement (antipattern-t).

All six shortcuts would replace the concurrent multi-adapter serving mechanism the KCs measure with a proxy or substitute.

## §7 Anti-pattern scan

Composition-math: N/A (no composition).
LORA_SCALE: N/A (no LoRA).
shutil.copy: N/A (no code).
Hardcoded `"pass": True`: N/A (no code, `all_pass: false` written).
Eval truncation producing base=0%: N/A (no eval).
Proxy-model substitution: explicitly rejected in §6 (`gemma-4-e4b-it-4bit` base and vLLM/llama.cpp not used as `pierre-g4e4b` stand-ins).
KC measures wrong object: K1911/K1912 correctly identify concurrent-serving throughput and peak memory of the `pierre-g4e4b` multi-adapter harness (not a proxy), but the harness itself doesn't exist on this platform → preempt-KILL.
N=smoke reported as full: N/A (no N; `is_smoke: false`).
Tautological routing: N/A (no routing in this experiment; K1911's concurrent dispatch is scheduler-level, not adapter-routing).
Thinking-mode truncation: N/A (no eval).
File-existence cache: N/A (no code).
Copy-paste scaffolding: Scaffold derived from `exp_memento_realtime_latency` (4th MEMENTO preempt-KILL) but cluster-specific sections (Pierre-serving F#570 preconditions vs MEMENTO F#685 paper-unavailability) are rewritten, not copy-pasted. Parent-extension requirements and sub-axis classification distinct. Cross-cluster triple-fire §1.3 is new to this experiment.
