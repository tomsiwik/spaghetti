# MATH.md — Pierre Adapter Hot-Swap Latency IMPL on Gemma 4 E4B

## §0. Platform skills (invoked, PLAN.md §1012)

`/mlx-dev` and `/fast-mlx` were invoked before this file was written, per guardrail 1012
and the paired parent experiment's §8 scope directive. Subsequent MLX code in
`run_experiment.py` is trusted accordingly.

## §1. Scope

Paired execution companion to the design-lock parent
`exp_pierre_adapter_hotswap_latency` (status=provisional, F#702). This IMPL
measures the two KCs (K1953, K1954) that the parent left operational-but-unrun.

All theorems and predictions are **inherited verbatim** from the parent MATH.md.
No new theorem is introduced. No KC is edited. This file only pins
the measurement recipe and failure-mode classification.

## §2. Inherited theorems (parent MATH.md §3, §4)

- **Theorem 1** — `attach_adapter` cost is O(n_layers · T) Python overhead,
  bounded by `t_attach ≤ n_layers · T · (t_attr_set + t_wrap_ctor) + t_eval_noop`
  with all A/B arrays pre-materialized.
  For Gemma 4 E4B with `n_layers = 42` (measured on
  `mlx-community/gemma-4-e4b-it-4bit`; parent §3 cited 34 in error) and
  `T = |TARGET_KEYS| = 2` (`self_attn.v_proj`, `self_attn.o_proj` per F#627),
  the upper bound is:
      `t_attach ≤ 42 · 2 · (1 + 10) μs + 100 μs ≤ 1.0 ms`
  Mid-point prediction: **0.4–0.9 ms**.
  (Parent §3 used T=7; this IMPL restricts to F#627-proven T=2 per parent §8.
  Gemma 4 E4B layout is `model.layers`, not `model.model.layers`; this IMPL
  ships local attach/detach wrappers that honor the correct path without
  modifying `pierre.pierre`.)

- **Theorem 2** — Under MLX lazy-eval semantics, same-adapter detach + re-attach
  is a semantic no-op on forward output. Predicted glitch-count = 0.

## §3. Kill-criteria (verbatim from pre-reg; F#666 target-gated structurally)

### K1953 — `t_attach_median` over 20 runs > 100 ms ⇒ FAIL
- **Type**: target-metric (user-facing wall-clock latency).
- **Operational**:
  `t_attach[i] = time(attach_adapter(model, frozen_A, adapter_B[d_i], d_i, α=8.0))`
  for 20 runs; `d_i` rotates through N=5 synthesized domains;
  each run is preceded by `detach_adapters(model)` and followed by `mx.eval(model.parameters())`.
  Measurement is the median.
- **Predicted PASS**: median ∈ [0.3, 0.7] ms ≪ 100 ms.

### K1954 — same-adapter detach/re-attach glitch-count > 1 ⇒ FAIL
- **Type**: target-metric (behavioral output divergence).
- **Operational**:
  1. Build prompt `P` of length ≤ 64 tokens; tokenize.
  2. `attach_adapter(model, frozen_A, adapter_B[0], 0, α=8.0)`. Greedy-generate
     16 tokens with KV cache ⇒ `T_0`.
  3. For `k ∈ {1, 2, 4, 8}`: reset cache, re-attach `adapter_B[0]`, prefill,
     greedy-generate `k` tokens, then `detach_adapters(model)` and immediately
     `attach_adapter(..., adapter_B[0], ...)`, resume greedy generation to 16
     total ⇒ `T_swap(k)`.
  4. `glitch_count = Σ_k |{i ∈ [0,16) : T_0[i] ≠ T_swap(k)[i]}|`.
- **Predicted PASS**: glitch-count = 0 (Theorem 2).

Both KCs are first-order target-metrics (latency, token-identity) — F#666
guardrail is satisfied structurally, not via a paired proxy.

## §4. F#666 routing

Both KCs are target-metrics. No proxy-only KC is present. This is **NOT** an
F#666-pure preempt-kill candidate. Reviewer §5 preempt-structural clause does
not apply.

## §5. Scope-preservation (researcher hat §4)

- Base model: `mlx-community/gemma-4-e4b-it-4bit` (matches parent §8; NOT
  proxied to E2B or a larger variant).
- Adapter targets: `self_attn.v_proj`, `self_attn.o_proj` per F#627 (NOT the
  full 7-target set from `pierre.pierre.ADAPTER_TARGETS`).
- Rank: `r = 6` per F#627 canonical.
- α = 8.0 per pre-reg (F#328/F#330-safe: ≤ 8).
- A-matrix initialization: partitioned QR of `(in_features, N·r)` random normal
  ⇒ N domain-orthogonal blocks of `(in_features, r)` (F#562 Grassmannian).
- B-matrix initialization: `N(0, 0.01²)` random (non-zero so the adapter has
  measurable contribution; K1954 is still a no-op under same-B re-attach).
- Bench budget: `BENCH_RUNS = 20`, `WARMUP = 3`, `GEN_TOKENS = 16`,
  `SWAP_POSITIONS = {1,2,4,8}`.

## §6. Antipattern scan (reviewer checklist, researcher hat §4)

- (m) Proxy-model substitution → NO. E4B 4-bit loaded as stated.
- (t) Scope-swap-on-error → code raises on OOM/API error; no silent
  `q→v` or r-down substitution.
- Composition-math bug / LORA_SCALE overrun / shutil.copy-as-adapter /
  hardcoded "pass":True / eval-template truncation → NONE (no training,
  no eval template, no composition).
- Batch discipline: N=1 impl, no sweep to stage.

## §7. References

- Parent: `micro/models/exp_pierre_adapter_hotswap_latency/MATH.md` §3–§8.
- Prior art: `micro/models/adapter_hotswap_latency/MATH.md` (Theorems 1,2
  verified on Qwen3-0.6B, `t_inject_only = 0.260 ± 0.017 ms` at 28×2 = 56 ops).
- F#388 — M2P forward 5.31 ms on Qwen3-0.6B (67.2% BW utilization).
- F#627 — Gemma 4 E4B LoRA targets = v_proj + o_proj.
- F#562 — partitioned-QR Grassmannian A-matrices.
- F#275 — norm-rescaled Euclidean composition (adapter B-matrix shape source).
- F#666 — target-gated KC rule (structural check here; both KCs are targets).
- F#702 — parent provisional finding (design-lock reason).
- Hu et al. arxiv:2106.09685 — LoRA reparameterization.
- PLAN.md Part 2 — target platform M5 Pro 48GB, Gemma 4 E4B base.
