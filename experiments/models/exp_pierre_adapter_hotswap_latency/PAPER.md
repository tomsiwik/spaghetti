# PAPER.md — Pierre Adapter Hot-Swap Latency (Gemma 4 E4B)

**Verdict: PROVISIONAL (design-lock; execution deferred to `_impl` companion).**

## Prediction vs. Measurement

| KC    | Metric                                  | Predicted (MATH.md)     | Measured | Status     |
|-------|-----------------------------------------|-------------------------|----------|------------|
| K1909 | `t_attach_median` (ms)                  | [0.6, 2.7] (Theorem 1)  | untested | deferred   |
| K1910 | glitch-count (same-adapter detach/re-attach) | 0 (Theorem 2)      | untested | deferred   |

No rows measured this iteration — design-lock scaffold only. All predictions
derive from reused theorems in the prior `adapter_hotswap_latency` experiment
(Qwen3-0.6B), transferred structurally to Gemma 4 E4B.

## Why PROVISIONAL (not SUPPORTED, not KILLED)

Three independent reasons (full reasoning in MATH.md §7):

1. **Pre-reg hygiene.** Original pre-reg had `references: []`, `success_criteria: []`,
   `platform: null`, and K1910 was operationally undefined. Fixing KC
   operationalization after data comes in would violate KC-lock discipline
   (PLAN.md §1). This iteration defines the KC in MATH.md; `_impl` measures
   against the locked definition.
2. **Platform-skill discipline.** PLAN.md §1012 requires `/mlx-dev` + `/fast-mlx`
   invocation before writing MLX platform code. This iteration does NOT invoke
   them (no MLX is executed); canonical preempt disclosure: "Not invoked —
   deferred to `_impl`."
3. **Prior-art theorem reuse.** Theorems 1 & 2 from `adapter_hotswap_latency`
   (Qwen3-0.6B) already establish the mechanism: attach is O(n_layers · T)
   Python overhead + mx.eval no-op; swap adds no TTFT penalty under MLX lazy
   eval. Gemma 4 E4B transfer is a measurement deliverable, not a novel
   mechanism.

## F#666 routing

Both KCs (K1909 latency, K1910 output-equivalence) are **target-metrics**:

- K1909 — user-perceived wall-clock latency IS the product claim.
- K1910 — behavioral output equivalence under same-adapter swap is directly
  observable in the generated token stream.

No proxy-only KC is present. Reviewer.md §5 preempt-structural clause does NOT
apply here. This is NOT an F#700/F#701-style F#666-pure preempt-kill case — it
is a legitimate PROVISIONAL pending `_impl`.

## Pre-reg hygiene fixes applied

| Field              | Original    | Fix                                                     |
|--------------------|-------------|---------------------------------------------------------|
| `references`       | `[]`        | 7 refs: prior hotswap art, F#388, F#275, F#627, F#562, F#666, Hu 2021 |
| `success_criteria` | `[]`        | `t_attach_median < 100 ms AND glitch-count = 0`         |
| `platform`         | `null`      | `mlx`                                                   |
| `experiment_dir`   | `null`      | `micro/models/exp_pierre_adapter_hotswap_latency/`      |
| K1910 definition   | ambiguous   | glitch-count = Σ_k ‖{i : T_0[i] ≠ T_swap(k)[i]}‖ for k∈{1,2,4,8} |

## Unblock path (paired `_impl` companion)

New experiment to register on handoff:

- **id**: `exp_pierre_adapter_hotswap_latency_impl`
- **priority**: 2 (micro)
- **tags**: `hotswap-latency`, `impl-companion`, `p1`, `serving`
- **depends_on**: `exp_pierre_adapter_hotswap_latency` (this record)
- **scope**: MLX execution of KC measurement per MATH.md §8, post
  `/mlx-dev`+`/fast-mlx` invocation.

## Taxonomic note

This PROVISIONAL differs from the F#696/F#697 "novel-mechanism design-lock"
subfamily: K1909/K1910 rely on **reused** prior theorems, not a new mechanism.
It differs from F#700/F#701 "F#666-pure preempt-kill" because both KCs are
target-metrics, not proxies. The correct taxonomic row is:

> **design-lock-hygiene-patch PROVISIONAL** — pre-reg was runnable but
> structurally under-specified; hygiene fixed here, execution deferred.

If a second instance of this shape appears in the drain window, promote to
an antipattern memory (currently: 1st observed instance).
