# Metal Kernel Profiling: Dispatch Overhead Analysis

## 0. Failure Mode & Impossibility Structure

### Failure Mode: Misattributing Slowdown to Wrong Bottleneck

The degenerate behavior: assuming the 3.5x gap between theoretical and measured
throughput is caused by Metal kernel dispatch overhead, when it might be memory
bandwidth, Python overhead, or MLX graph scheduling.

**What mathematical structure determines the bottleneck?**

For a transformer forward pass, the throughput is bounded by:
```
T_total = max(T_compute, T_memory, T_dispatch, T_python)
```

Each component:
- T_compute = FLOPs / GPU_peak_TFLOPS
- T_memory = bytes_transferred / memory_bandwidth
- T_dispatch = n_kernels * dispatch_latency_per_kernel
- T_python = Python interpreter overhead + MLX graph construction

**Theoretical throughput for BitNet-2B-4T single token decode:**

Model specs: 30 layers, d=2560, d_ffn=6912, n_heads=32, n_kv_heads=8
Per-layer ops (ternary weights = integer adds, not FP MACs):
- QKV projection: 3 * d * d = 3 * 2560 * 2560 = 19.7M ops
- O projection: d * d = 6.6M ops
- Gate + Up projection: 2 * d * d_ffn = 2 * 2560 * 6912 = 35.4M ops
- Down projection: d_ffn * d = 17.7M ops
- Attention (seq_len=1): negligible
- Total per layer: ~79.3M ops
- Total 30 layers: ~2.38G ops

M5 Pro memory bandwidth: ~273 GB/s (estimated from M4 Pro scaling)
Model size in memory: ~1.7GB (packed ternary)
KV cache per token: 30 * 2 * 8 * 80 * 2 = 76.8 KB (bf16, GQA)

Memory-bound prediction (single token):
- Weight reads: 1.7 GB (must read all weights)
- T_memory = 1.7 GB / 273 GB/s = 6.2 ms
- Theoretical throughput: ~161 tok/s

**Measured throughput: ~45 tok/s (from prior experiments)**
**Gap: 161 / 45 = 3.6x**

This gap could come from:
1. Non-overlapped weight reads (serial layer execution)
2. Kernel dispatch overhead (n_kernels ~ 300+ per forward pass)
3. Python/MLX graph overhead
4. Cache thrashing (1.7GB model doesn't fit in GPU cache)
5. Unpacking overhead (ternary → bf16 happens at each layer)

## 1. Profiling Strategy

### 1.1 MLX Native Profiling

MLX provides `mx.disable_compile()` / `mx.enable_compile()` and timing hooks.
More importantly, `mx.compile` can fuse operations.

**Key measurement: number of `mx.eval` calls per forward pass.**
Each `mx.eval` forces a synchronization point. Excess evals are the most
common source of unnecessary dispatch overhead.

### 1.2 Per-Component Timing

Break the forward pass into components and time each:
1. Token embedding lookup
2. Per-layer: attention (QKV, softmax, output)
3. Per-layer: FFN (gate, up, down, activation)
4. Per-layer: norms (RMSNorm)
5. Final norm + LM head projection
6. Python overhead (graph construction vs execution)

### 1.3 Compilation Impact

Test with and without `mx.compile`:
- No compile: each op dispatches separately
- With compile: fused graph, fewer dispatches
- Measure: dispatch count reduction and wall-clock speedup

## 2. Predictions

| Bottleneck | Expected Contribution | Evidence |
|-----------|---------------------|----------|
| Memory bandwidth | ~50% of gap | Model doesn't fit in cache |
| Ternary unpacking | ~20% of gap | bf16 unpacking per layer |
| Kernel dispatch | ~15% of gap | 300+ kernels per forward |
| Python/graph overhead | ~10% of gap | Interpreter between ops |
| Other (norms, etc.) | ~5% of gap | Small matrices, dispatch-bound |

Prediction: `mx.compile` should close 20-40% of the gap by fusing operations
and reducing dispatch count. The remaining gap is memory-bandwidth-bound.
