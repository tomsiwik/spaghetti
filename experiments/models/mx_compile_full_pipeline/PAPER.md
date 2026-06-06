# mx.compile Full Pipeline: Research Digest

## Hypothesis

Compiling the routing+merge+forward pipeline with mx.compile will yield >20% throughput improvement by eliminating Python dispatch overhead and enabling kernel fusion.

## What This Experiment Tested

Six phases testing mx.compile at increasing granularity:
1. Baseline tok/s (no compile, naive LoRA vs addmm)
2. Compiled routing head (small matmul + sigmoid)
3. Compiled LoRA delta (single projection and 7-projection block)
4. Compiled multi-adapter LoRA (N=2, N=5 fixed)
5. Dynamic adapter selection (recompilation cost, padding strategy)
6. End-to-end: real BitNet-2B-4T generation with compiled LoRA wrappers

## Key References

- MLX compile docs (ml-explore/mlx)
- exp_benchmark_composition_latency_sweep: 2.3-2.4x compile speedup on pre-merge
- exp_inference_speed_10x: 97.2 tok/s addmm, 172 tok/s base, async_eval double-buffering

## Empirical Results

### End-to-End Generation (100 tokens, greedy, M5 Pro)

| Approach | Internal tok/s | Wallclock tok/s | Speedup vs Naive |
|----------|---------------|-----------------|-------------------|
| Base (no adapter) | 171.7 | 153.5 | -- |
| Naive LoRA (uncompiled) | 88.1 | 82.0 | 1.00x |
| addmm LoRA (uncompiled) | 97.2 | 90.4 | 1.10x |
| addmm + compile LoRA | 97.3 | 90.2 | 1.10x |

**mx.compile adds 0.1% over addmm alone -- effectively zero.**

### Component-Level Microbenchmarks

| Component | Uncompiled (ms) | Compiled (ms) | Speedup |
|-----------|----------------|---------------|---------|
| Routing head (d=2560, N=5) | 0.187 | 0.170 | 1.10x |
| LoRA single proj (naive) | 0.173 | 0.842 | 0.21x (SLOWER) |
| LoRA single proj (addmm) | 0.225 | 0.181 | 1.24x |
| LoRA 7-proj block | 0.215 | 0.222 | 0.97x |
| Multi-adapter N=2 | 0.209 | 0.479 | 0.44x (SLOWER) |
| Multi-adapter N=5 | 0.438 | 0.949 | 0.46x (SLOWER) |

### Dynamic Adapter Selection

| Strategy | Result |
|----------|--------|
| Pre-compile per N | Works. First call: 49-60ms compilation overhead. Cached: <1ms. |
| Pad to N_max=5 with zero gates | Works. Constant ~0.2-0.8ms regardless of active N. No recompilation. |

### Kill Criteria Assessment

- **K1 (#258): Compilation fails on dynamic adapter selection** -- **PASS**. Both strategies (pre-compile per N and padding) work correctly. Compilation overhead is 49-60ms per unique N, amortized across all tokens.
- **K2 (#259): No speedup (< 5% improvement)** -- **PASS (marginal)**. +10.4% vs naive LoRA. But this comes entirely from addmm, not compile. Compile itself: +0.1%.
- **S1 (#26): >20% throughput improvement** -- **FAIL**. Maximum improvement is 10.4% (addmm only), 10.5% (addmm + compile).

## Why mx.compile Does Not Help for Generation

### The double-buffering explanation

mlx_lm's generate loop uses `mx.async_eval` to pipeline graph construction with GPU execution:

```python
# From mlx_lm/generate.py
next_y = _step(y)
mx.async_eval(next_y)
# Python overhead of building next graph happens WHILE GPU executes current token
yield current_y
```

This means Python dispatch overhead is already hidden behind GPU compute. mx.compile eliminates dispatch overhead, but that overhead is already free (masked by async_eval). The GPU is the bottleneck, not Python.

### Why microbenchmarks show speedup but E2E does not

In microbenchmarks, `mx.eval` is synchronous: Python waits for GPU, then dispatch overhead adds to total time. In the generation loop, `mx.async_eval` runs Python and GPU in parallel. The dispatch overhead exists but is concurrent with GPU work.

The 1.24x speedup on single addmm projection in microbenchmark = 0.04ms saved per call. Across 210 projections = ~8ms saved. But at 97 tok/s (10.3ms/tok), the generation loop already overlaps ~8ms of graph construction with ~10ms of GPU work. Adding compile just makes the already-hidden overhead even more hidden.

### Why compile sometimes makes things SLOWER

Several cases showed compile slower than uncompiled:
- **Naive LoRA compiled: 0.21x** (4.9x slower) -- The naive path (`h @ B * scale + base_y`) has more operations for compile to trace and fuse. The compilation overhead per call exceeds the small graph's execution time.
- **Multi-adapter N=2: 0.44x** (2.3x slower) -- The compiled function traces through the loop, creating a larger graph. The fusion benefit cannot overcome the trace overhead for such small matmuls (rank-16).

This is consistent with the fast-mlx guide: "Recompilation is relatively expensive and should only be done if there is sufficient work over which to amortize the cost." Rank-16 LoRA matmuls are too small to justify compilation overhead.

### Why the pre-merge experiment showed 2.4x speedup

exp_benchmark_composition_latency_sweep showed 2.3-2.4x from compile because:
1. It operates on (2560, 2560) matrices, not tiny (1, 16) -> (1, 2560) LoRA projections
2. It uses synchronous `mx.eval` timing (no async overlap)
3. The per-adapter dispatch overhead is proportionally larger for the loop over N adapters accumulating into W

Pre-merge compilation is valuable for session-start merging. Runtime LoRA compilation is not valuable for generation.

## Limitations

1. Single adapter tested (N=1 for E2E). Multi-adapter E2E may differ.
2. Sequence length fixed at 100 tokens. Longer contexts untested.
3. Only tested generation (seq_len=1 per step). Prefill (variable seq_len) may benefit differently.
4. Did not test compiling the full model forward pass (blocked by KV cache constraint).

## What Would Kill This

Already effectively killed: mx.compile provides no measurable improvement for LoRA-augmented generation on M5 Pro. The improvement was predicted to come from eliminating Python dispatch overhead, but mlx_lm's async_eval double-buffering already eliminates this overhead at the generation loop level. The only measurable improvement (10.4%) comes from addmm op fusion, which is available without mx.compile.

## Recommendation

1. **Use addmm for LoRA serving** (already done) -- this is the only real speedup available (+10.4%).
2. **Do NOT compile LoRA wrappers** -- zero benefit, adds complexity and potential recompilation pitfalls.
3. **DO compile pre-merge operations** -- 2.4x speedup for session-start adapter merging (proven in prior experiment).
4. **The generation speed ceiling on M5 Pro is set by memory bandwidth** (273 GB/s), not Python overhead. Further improvements require reducing model bytes read per token (quantization, pruning, sparsity).
