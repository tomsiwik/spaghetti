# MATH.md — Adapter Hot-Swap Latency on Qwen3-0.6B

## Problem

In the M2P serving architecture, domain adapters are precomputed B-matrix sets
stored in unified memory. When a request arrives for a different domain, the current
B matrices must be replaced before the next forward pass. This experiment measures:

1. How long does the reference swap take? (inject_lora_b cost)
2. Is there any overhead to the first token after a swap vs. no swap?
3. What is the total "adapter switch" latency visible to the user?

## Background

**Finding #388** (exp_m2p_generation_speed): M2P forward = 5.31ms,
full pipeline = 11.34ms, BW utilization = 268.7 GB/s (67.2% of 400 GB/s peak).

**Hu et al. 2106.09685** (LoRA): LoRA parameterizes W + BA where A is fixed
after initialization and B is computed per-request by the M2P hypernetwork.

**Apple Silicon Unified Memory Architecture**: CPU and GPU share the same physical
memory. There is no PCIe bus transfer for arrays already in RAM. MLX lazy evaluation
builds a fresh computation graph on each `__call__`; no "KV cache for adapters."

## Definitions

- B-matrix set: `{B_q^(l), B_v^(l)}` for l = 0..n_layers−1, each B of shape (rank, d_out)
  Total params per domain: 2 × n_layers × rank × d_out ≈ 2 × 28 × 4 × 1536 ≈ 344K params
  Memory: 344K × 2 bytes (float16) ≈ 688KB per domain

- inject_lora_b: Python loop over n_layers, sets `lora_b` attribute on q_proj + v_proj,
  then calls `mx.eval(model.parameters())`.

- Swap latency (K951 target): wall time from "start inject_lora_b" to "first token generated"

## Theorem 1 — Reference Swap Cost Is O(n_layers) Python Overhead

**Theorem:** For an adapter B-matrix set already in unified memory, the cost of
`inject_lora_b(model, B_q_new, B_v_new)` is bounded by:

  t_inject ≤ n_layers × t_attr_set + t_eval_noop

where t_attr_set ≈ 1μs (Python attribute assignment),
t_eval_noop ≈ 10–100μs (mx.eval on pre-materialized arrays — effectively a no-op),
n_layers = 28.

**Proof:** 
Step 1: `inject_lora_b` executes a Python `for` loop of n_layers iterations.
Each iteration performs two attribute assignments: `layer.self_attn.q_proj.lora_b = B_q`.
This is a Python object attribute write: O(1) per assignment, ~1μs in CPython.

Step 2: `mx.eval(model.parameters())` traverses the parameter tree and schedules
evaluation. For arrays that are ALREADY materialized (as our precomputed B matrices
and the frozen base model weights are), mx.eval is a no-op per array.
New B matrices are pre-materialized: computed once via M2P forward and `mx.eval`-ed.
Cost: graph traversal only, O(total_params) in list length but O(1) in compute.

Step 3: No memory transfer occurs. B matrices are already in unified memory.
There is no GPU transfer, no allocation, no copy.

Quantitative prediction:
  t_inject ≤ 28 × 2 × 1μs + 100μs = 156μs < 0.2ms

**QED**

## Theorem 2 — First Token After Swap = Baseline (No Overhead)

**Theorem:** The time-to-first-token (TTFT) after an adapter swap equals the
baseline TTFT (no swap), within measurement noise:

  t_TTFT_after_swap ≈ t_TTFT_baseline

**Proof:**
MLX lazy evaluation: each call to `model(tokens)` constructs a fresh computation
graph from the current values of model attributes. There is no "adapter cache" that
becomes invalid when `lora_b` changes.

When `lora_b` is updated between calls, the NEXT call simply uses the new array
in its computation graph. The new B matrices are already in unified memory;
reading them during the forward pass costs exactly |B| / BW = 688KB / 268.7 GB/s ≈ 0.003ms.

Since |B_new| = |B_old| (same shape), reading the new adapter costs the same as
reading the old adapter. The forward pass BW budget is unchanged.

Therefore: t_TTFT_after_swap = t_TTFT_baseline + O(0.003ms) ≈ t_TTFT_baseline

**QED**

## Theorem 3 — K951 Feasibility (Total Swap < 50ms)

**Theorem:** K951 (total swap latency < 50ms) will PASS.

**Proof:**
  t_total_swap = t_inject + t_TTFT

From Theorem 1: t_inject < 0.2ms
From Theorem 2: t_TTFT ≈ t_TTFT_baseline
From Finding #388: t_TTFT ≈ prefill_time for prompt of length T

For T=64 tokens, prefill dominates first-token generation.
From Finding #388: model BW utilization = 268.7 GB/s; model = 4-bit Qwen3-0.6B.
Base model size = 0.6B params × 0.5 bytes/param (4-bit) ≈ 300MB.
Prefill time for T=64: ≈ 300MB / 268.7 GB/s × (prefill_factor) ≈ 1–5ms range.

Conservative upper bound for TTFT: 20ms (10× safety margin).

t_total_swap < 0.2ms + 20ms = 20.2ms << 50ms

K951 threshold: 50ms. Predicted margin: > 2.5×.

**QED**

## Quantitative Predictions

| Metric                   | Predicted Value       | Basis                          |
|--------------------------|----------------------|--------------------------------|
| t_inject_only (ms)       | < 0.2ms              | Theorem 1 (Python overhead)    |
| t_TTFT_baseline (ms)     | < 20ms               | Finding #388 (BW bound)        |
| t_TTFT_after_swap (ms)   | ≈ t_TTFT_baseline    | Theorem 2 (no overhead)        |
| Swap overhead (ms)       | < 0.5ms              | Theorem 1 + Theorem 2          |
| Total swap latency (ms)  | < 20ms               | Theorem 3                      |
| K951 status              | PASS                 | Predicted margin > 2.5×        |

## Kill Criteria

**K951**: mean(t_inject + t_TTFT_after_swap) < 50ms → K951 PASS if met

Note: K951 tests the USER-VISIBLE latency: the time from "switch domain" decision
to first new-domain token appearing. This is the product-level metric.
