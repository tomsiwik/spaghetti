# Learnings: exp_mx_compile_full_pipeline

## Core Finding

mx.compile provides zero additional throughput improvement for LoRA-augmented token generation when mlx_lm's async_eval double-buffering is already active. The 10.4% improvement vs naive LoRA comes entirely from `mx.addmm` op fusion, not from graph compilation. Component-level microbenchmarks show up to 1.24x speedup, but this vanishes in the E2E generation loop because Python dispatch overhead is already hidden behind GPU compute via `mx.async_eval`.

## Why This Happened (Literature-Grounded)

### async_eval already eliminates the overhead compile targets

mlx_lm's generate loop (generate.py) uses `mx.async_eval` to pipeline graph construction with GPU execution. While the GPU processes token N, Python builds the graph for token N+1. This means:
- Python dispatch overhead (~40us per LoRA projection, ~6.7ms across 168 projections) runs concurrently with GPU work (~10.3ms per token at 97 tok/s)
- mx.compile eliminates dispatch overhead, but that overhead is already free (overlapped)
- The GPU is the bottleneck (memory-bandwidth-bound at 74.2% utilization), not Python

This is the same mechanism described in MLX's fast guide: async_eval "pipelines graph construction with computation."

### Microbenchmark vs E2E divergence is expected

Microbenchmarks use synchronous `mx.eval`, where Python waits for GPU. In this regime, dispatch overhead adds directly to wall time, so compile helps (1.24x for single addmm projection). In E2E generation, async_eval overlaps dispatch with compute, making the dispatch time invisible.

This is a well-known performance measurement pitfall: component-level improvements that vanish at system level because the component is not on the critical path. Amdahl's law: if dispatch is 0% of E2E critical-path time (because it's overlapped), infinite speedup on dispatch yields 0% system improvement.

### Small rank-16 matmuls make compile overhead significant

For tiny operations like (1, 2560) @ (2560, 16) = (1, 16), the compilation trace time exceeds the execution time. This explains why several compile cases were SLOWER (0.21x-0.46x):
- Compilation trace: ~50ms first call (amortized)
- Per-call: graph dispatch still has constant overhead that dominates sub-0.2ms operations

The fast-mlx guide explicitly warns: "Recompilation is relatively expensive and should only be done if there is sufficient work over which to amortize the cost."

## Confirming Evidence

1. **Our exp_inference_speed_10x**: Noted 97.2 tok/s addmm, 74.2% bandwidth utilization. The remaining 25.8% is not Python overhead but memory controller inefficiency (bank conflicts, refresh cycles). Compile cannot fix hardware bandwidth utilization.

2. **Our exp_benchmark_composition_latency_sweep**: 2.3-2.4x compile speedup on pre-merge. This used synchronous eval on large (2560, 2560) matrices where dispatch overhead is proportionally significant. Confirms compile works when operations are large enough to amortize trace cost AND timing is synchronous.

3. **MLX fast guide (references/fast-mlx-guide.md)**: "Compiling graphs with mx.compile can make them run a lot faster. But there are some sharp-edges." Lists shape-dependent recompilation and closure capture as key issues. Our experiment confirms: the sharp edges dominate for small LoRA matmuls in async generation.

## Contradicting Evidence

1. **vllm-mlx (arXiv 2601.19139)**: Reports significant speedups from graph optimization on MLX. However, vllm-mlx targets batched serving with multiple requests, where graph fusion across batch elements provides real compute savings. Our single-request scenario does not benefit similarly.

2. **The 2.4x pre-merge compile speedup from our own prior experiment** could be interpreted as evidence that compile should help here too. The difference: pre-merge uses synchronous eval on large matrices, while generation uses async eval on tiny rank-16 projections.

## Alternative Approaches

### 1. Reduce bytes-per-token (the actual bottleneck)
At 74.2% bandwidth utilization, the ceiling is ~232 tok/s (273 GB/s / 1.22 GB model). Approaches:
- Sparse-BitNet (arXiv 2603.05168): 42% natural sparsity in ternary weights could reduce effective model size
- Attention-only LoRA: 126.7 tok/s proven (fewer adapter parameters to read per token)
- Model distillation to smaller size

### 2. Batch multiple tokens (speculative decoding)
Compile could help if we process multiple tokens per step (speculative decoding), where the larger batch amortizes per-step overhead. The graph would be larger, making compile's fusion more valuable.

### 3. Custom Metal kernels for fused LoRA
Instead of compile-level fusion, a custom Metal kernel that fuses BitLinear + LoRA in one pass could eliminate the base_y intermediate materialization. This is a hardware-level optimization that compile's graph rewriting cannot achieve because BitLinear is already a custom kernel opaque to the compiler.

## Implications for Next Experiments

1. **Do not pursue further compile optimization for generation.** The bottleneck is memory bandwidth, not Python dispatch. This is a dead end for this use case.

2. **addmm is the right optimization for runtime LoRA.** The 10.4% improvement is the full benefit available from operation-level fusion. Already deployed.

3. **Pre-merge compile (2.4x) remains valuable** for session-start adapter merging. The architectural split is confirmed: pre-merge (compiled, synchronous) for always-on adapters, runtime LoRA (addmm, async) for routed experts.

4. **The speed ceiling for M5 Pro with BitNet-2B-4T is approximately:**
   - Base: 172 tok/s (74.2% BW utilization)
   - Single adapter (addmm): 97 tok/s
   - Attn-only adapter: 127 tok/s
   - Further improvement requires reducing model or adapter bytes, not improving dispatch.
