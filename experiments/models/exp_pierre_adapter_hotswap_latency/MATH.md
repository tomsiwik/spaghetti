# MATH.md — Pierre Adapter Hot-Swap Latency on Gemma 4 E4B

## §0. Platform skills (design-lock scope)

Scope: this iteration delivers a **design-lock PROVISIONAL** with a scaffold `run_experiment.py`
that does NOT execute MLX platform code. No MLX arrays are allocated, no model is loaded.
Per PLAN.md §1012: platform skills (`/mlx-dev`, `/fast-mlx`) MUST be invoked in the
paired `_impl` execution. Canonical preempt disclosure: "Not invoked — deferred to `_impl`."

## §1. Problem

Pierre serves adapter stacks for routed queries. When the router changes domain
mid-conversation, the adapter set must be replaced before the next forward pass.

**Pre-reg hygiene fixes (applied in this iteration):**
- `references: []` → now explicit (Theorem reuse from `adapter_hotswap_latency`,
  Finding #388, Finding #275, Hu et al. 2106.09685).
- `success_criteria: []` → now explicit (§4 below).
- `platform: null` → now `mlx`.
- `experiment_dir: null` → now `micro/models/exp_pierre_adapter_hotswap_latency/`.

## §2. Background (cited, not analogized)

1. **Prior Pierre/adapter-hotswap finding (reused theorems).** `adapter_hotswap_latency`
   on Qwen3-0.6B measured:
   - `t_inject_only = 0.260 ± 0.017 ms` (Theorem 1: `O(n_layers)` Python attribute
     overhead + mx.eval no-op on pre-materialized arrays).
   - Swap overhead on TTFT ≈ −7.4 ms (within noise); Theorem 2: MLX lazy eval
     rebuilds graph per `__call__`, no adapter-cache invalidation penalty.
   - `t_inject_only` = 0.26 ms on 28 layers × 2 projections = 56 attribute assignments.

2. **Finding #388** — M2P forward 5.31 ± 0.20 ms on Qwen3-0.6B at 67.2% BW
   utilization. Sets the achievable per-forward floor on this architecture class.

3. **Hu et al. arxiv:2106.09685** — LoRA. Adapter B swap is a reference
   reassignment, not a parameter merge; base weights untouched.

4. **Finding #275** — norm-rescaled Euclidean composition. Informs what B-matrix
   sets look like in the multi-adapter case (r=6, v_proj+o_proj for Gemma 4
   per F#627).

5. **Pierre pierre.py (stable, 265 loc).** `attach_adapter(model, frozen_A, adapter_B,
   domain_idx, alpha)` wraps target modules with `RuntimeLoRA`. `detach_adapters`
   undoes the wrap. Both are O(n_layers × |ADAPTER_TARGETS|) Python work + one
   `mx.eval(model.parameters())`.

## §3. Theorem 1 (reused from `adapter_hotswap_latency`) — Attach cost is O(n_layers · T) Python overhead

**Statement.** For Gemma 4 E4B with n_layers = 34 and |ADAPTER_TARGETS| = 7
(q/k/v/o_proj + gate/up/down_proj), the cost of `attach_adapter` for one domain
when A and B are already materialized in unified memory is bounded by:

    t_attach ≤ n_layers · T · (t_attr_set + t_wrap_ctor) + t_eval_noop

with `t_attr_set ≈ 1 μs`, `t_wrap_ctor ≈ 5–10 μs`, `t_eval_noop ≈ 10–100 μs`.

**Proof.** Identical to `adapter_hotswap_latency` §Theorem 1. `attach_adapter`
executes two nested loops: `for li in range(n_layers): for key in ADAPTER_TARGETS`.
Each inner iteration performs `RuntimeLoRA(base, A, B, alpha)` construction
and an `update_modules` call. All matrices are pre-materialized. `mx.eval` at
loop exit is a graph-traversal no-op on materialized arrays. □

**Quantitative prediction (Gemma 4 E4B, 4-bit quant, M5 Pro):**

    t_attach ≤ 34 × 7 × (1 + 10) μs + 100 μs
             ≤ 2.7 ms (upper bound)

Mid-point prediction: **0.6–1.2 ms** for a full 7-target attach (vs 0.26 ms on
Qwen3-0.6B with 28 layers × 2 targets = 56 ops; here we have 34 × 7 = 238 ops,
so scale by 4.25× → 1.1 ms).

## §4. Theorem 2 (reused) — No swap overhead on TTFT via MLX lazy eval

**Statement.** Under MLX lazy evaluation semantics, `t_TTFT_after_swap ≈ t_TTFT_baseline`
(within measurement noise); swap does not add measurable overhead.

**Proof.** Identical to `adapter_hotswap_latency` §Theorem 2. Every forward
pass constructs a fresh computation graph from current module state. There is
no persistent cache keyed on adapter identity; no invalidation occurs on swap. □

**Prediction.** Swap overhead (K1910 proxy) on next-token logits ≈ 0 ms.

## §5. Kill-criteria (target-gated per F#666)

KCs from the pre-reg are retained verbatim; K1910 is operationalized (required
for reviewer checklist (u) / (f)).

### K1909 — Adapter hot-swap latency > 100ms ⇒ FAIL

**Metric type**: target (user-facing wall-clock latency).
**Operational definition.** `t_attach_median` over `BENCH_RUNS=20` runs of
`attach_adapter(model, frozen_A, adapter_B, domain_idx=i, alpha=8.0)` followed
by `detach_adapters(model)` on N=5 synthetic adapter sets. Attach is measured
from the first Python statement inside `attach_adapter` to the return of
`mx.eval(model.parameters())`.

**Predicted**: t_attach_median ∈ [0.6, 2.7] ms (Theorem 1, §3). **PASS**.

### K1910 — Hot-swap during generation produces > 1 token glitch ⇒ FAIL

**Metric type**: target (behavioral output divergence).
**Operational definition.** Define a "token glitch" as a next-token argmax
change caused purely by the swap operation (not by intended adapter change).
Operationally:
1. Prepare prompt `P` of length 64 tokens.
2. With adapter_B_0 attached, generate 16 tokens greedy → `T_0`.
3. At position k ∈ {1,2,4,8}, call `detach_adapters(model)` then immediately
   `attach_adapter(... adapter_B_0 ...)` (same adapter!). Resume greedy generation.
   → `T_swap(k)`.
4. Glitch-count = Σ_k |{i : T_0[i] ≠ T_swap(k)[i]}|.
5. FAIL if glitch-count > 1 across all 4 positions (i.e., bitwise-exact same-adapter
   detach/re-attach must not change the generated tokens).

Rationale: detach+re-attach of the *same* adapter must be a no-op semantically.
Any deviation indicates lossy swap (caching, numerical drift, dtype cast).

**Predicted**: glitch-count = 0 (Theorem 2, §4 + MLX lazy-eval determinism on
pre-materialized arrays). **PASS**.

### Success criteria (formerly empty)

- `t_attach_median < 100ms` AND `glitch-count = 0` across all 4 positions ⇒ SUPPORTED.
- `t_attach_median ≥ 100ms` OR `glitch-count > 1` ⇒ KILLED.
- Scaffold-only (this iteration, no Gemma 4 loaded) ⇒ PROVISIONAL.

## §6. Scope integrity & F#666 routing

This experiment's KCs are **both target-metrics** (user-perceived latency,
behavioral output equivalence under same-adapter swap). No proxy-only KC is
present. F#666 guardrail applied as a **structural check**, not as a
preempt-kill reason. Reviewer.md §5 preempt-structural clause does **not**
apply here — this is a legitimate runnable experiment, not an F#666-pure
preempt-kill candidate.

No scope-swap shortcuts are considered: KC thresholds (100 ms, 1 glitch)
remain verbatim from pre-reg.

## §7. Why design-lock PROVISIONAL (not full run this iteration)

Three legitimate reasons, each independent:

1. **Pre-reg hygiene needed operational patching.** K1910 "token glitch" was
   undefined in the original pre-reg. Defining it inside the running code would
   have violated KC-lock discipline (PLAN.md §1). Safer to define in MATH.md
   first (this document), then register an `_impl` that measures it.

2. **Platform-skill discipline (PLAN.md §1012).** Full MLX platform code MUST
   be written under `/mlx-dev` + `/fast-mlx` invocation. This iteration does not
   invoke them (no MLX is executed); execution is deferred to `_impl`.

3. **Prior art already measured Theorems 1+2 on Qwen3-0.6B.** The remaining
   question is Gemma 4 E4B transfer. Measurement is the `_impl` contribution.

## §8. Paired `_impl` experiment (to be registered on handoff)

Companion id: `exp_pierre_adapter_hotswap_latency_impl` (micro, P=2, tag
`hotswap-latency`, `impl-companion`, `p1`). Execution scope:

1. Invoke `/mlx-dev` + `/fast-mlx` before writing any MLX code.
2. `mlx_lm.load("mlx-community/gemma-4-e4b-it-4bit")`.
3. Synthesize N=5 random-initialized adapter_B sets of shape matching
   F#627 (r=6, v_proj+o_proj, 34 layers). A-matrices via partitioned QR
   (F#562, Grassmannian orthogonality).
4. Measure K1909 (attach-median over 20 runs) and K1910 (same-adapter
   detach/re-attach determinism at positions {1,2,4,8}).
5. Write results.json, PAPER.md with prediction-vs-measurement table.

## §9. References

- **Prior in-repo art (theorem source):** `micro/models/adapter_hotswap_latency/`
  (Theorems 1 & 2 verified on Qwen3-0.6B; smoke inject = 0.26 ms, overhead ≈ 0).
- **Finding #388** — M2P forward 5.31 ms on Qwen3-0.6B (67.2% BW utilization).
- **Finding #275** — norm-rescaled Euclidean adapter composition.
- **Finding #627** — Gemma 4 E4B LoRA targets = v_proj + o_proj.
- **Finding #562** — Grassmannian A-matrix orthogonality on Gemma 4 native dims.
- **Finding #666** — target-gated KC rule; applied as structural check, not
  preempt-kill reason, since both KCs here are already target-metrics.
- **Hu et al. arxiv:2106.09685** — LoRA reparameterization.
- **PLAN.md Part 2** — Pierre P1 vision; runtime LoRA on frozen base;
  target hardware M5 Pro 48GB.
