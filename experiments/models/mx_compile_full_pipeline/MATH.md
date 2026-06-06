# MATH.md: mx.compile Full Pipeline Optimization

## 1. Mechanism Definition: mx.compile

### What mx.compile computes

`mx.compile(f)` performs **graph-level tracing and fusion** of an MLX function `f`. On first invocation with a given set of input shapes, it:

1. **Traces**: Executes `f` symbolically, recording the DAG of MLX operations
2. **Fuses**: Merges compatible elementwise operations into single Metal kernels
3. **Eliminates**: Removes intermediate tensor materializations where possible
4. **Caches**: Stores the compiled Metal compute pipeline for reuse

Mathematically, if `f` computes a sequence of operations:

```
y = g_n(g_{n-1}(...g_1(x)))
```

where each g_i is an MLX op, then mx.compile produces a fused function f* that computes the same result but with fewer kernel launches and intermediate buffers:

```
y = f*(x)  where  f*(x) = g_n(g_{n-1}(...g_1(x)))  [same output, fewer dispatches]
```

The compilation is **shape-dependent by default**: if input shapes change, recompilation occurs. `shapeless=True` enables shape-polymorphic compilation but with caveats (no data-dependent shapes, no Python control flow on array values).

### Compilation modes

1. **Shape-dependent** (default): Recompiles on shape change. Safe for fixed-shape inference (single token generation after prefill).
2. **Shapeless** (`shapeless=True`): One compilation serves all shapes. Riskier: no data-dependent control flow, no `.item()` inside compiled region.

### What gets fused

- **Elementwise chains**: `x * scale + bias` -> single kernel
- **Matmul + elementwise**: `(x @ A) * scale` -> fused launch
- **Reduction chains**: Multiple reductions can share memory
- **Dead code elimination**: Unused intermediate results are pruned

### What does NOT get fused

- **Custom Metal kernels** (BitLinear's `make_bitlinear_kernel`): Already a fused custom kernel, cannot be further optimized by the graph compiler
- **KVCache objects**: Not trees of arrays, so functions taking KVCache as arguments cannot be compiled directly
- **Data-dependent control flow**: `if x.item() > 0` inside compiled region causes trace divergence

## 2. Why mx.compile Helps (Overhead Analysis)

### Python dispatch overhead

Each MLX operation incurs:
- Python function call overhead: ~1-5 us
- MLX graph node creation: ~0.5 us
- Metal kernel dispatch: ~5-15 us

For a single LoRA forward pass (`y = base(x) + (x @ A) @ B * scale`):
- 4 operations x ~10 us dispatch = ~40 us Python overhead per layer
- 24 layers x 7 projections = 168 projections
- 168 x 40 us = ~6.7 ms total Python dispatch overhead

At 100 tok/s, each token takes 10 ms. So Python dispatch is ~67% of token time. Compilation eliminates most of this by fusing the graph into fewer kernel launches.

### Memory traffic reduction

Without compilation, intermediate tensors are materialized:
```
h = x @ A          # materializes (1, r), writes r*2 bytes
lora = h @ B       # materializes (1, d_out), writes d_out*2 bytes
scaled = lora * s  # materializes (1, d_out), writes d_out*2 bytes
y = base_y + scaled # materializes (1, d_out), writes d_out*2 bytes
```

With compilation, `h @ B * s + base_y` can be fused:
```
y = addmm(base_y, h, B, alpha=s)  # single write of d_out*2 bytes
```

For d_out=2560, bf16: saves 3 x 5120 = 15,360 bytes per projection.
Across 168 projections: saves ~2.5 MB per token of memory traffic.

At 273 GB/s bandwidth: 2.5 MB / 273 GB/s = ~9 us saved per token.
Not huge, but additive with dispatch savings.

## 3. What Breaks mx.compile

### Dynamic shapes cause recompilation

If sequence length varies (prefill vs generation), each new shape triggers recompilation (~50-200 ms). For autoregressive generation:
- Prefill: shape (1, seq_len) -- varies per prompt
- Generation: shape (1, 1) -- FIXED after prefill

**Key insight**: Generation phase has FIXED shapes (batch=1, seq=1). This is the ideal case for mx.compile. Prefill varies but happens once.

### KV cache constraint

The full model.__call__ takes `cache` as argument. mlx_lm's KVCache is a Python object wrapping arrays. mx.compile cannot trace through arbitrary Python objects.

**Workaround strategies tested in this experiment:**
1. Compile individual COMPONENTS (LoRA forward, routing head) that take pure array arguments
2. Compile the LoRA delta computation separately from the base forward
3. Test if mlx_lm's generate loop already handles this optimally via async_eval

### Closure capture pitfall

```python
@mx.compile
def lora_forward(x):
    return x @ self.A @ self.B  # self.A captured in closure -> included in trace
```

If `self.A` was computed (not loaded), the full computation of A gets re-executed on every call. Fix: pass as explicit input or use `inputs=[self.A, self.B]`.

### Constants trigger recompilation

Python scalars used as constants in compiled functions trigger recompilation if they change:
```python
@mx.compile
def f(x, scale):  # if scale is Python float, recompiles on change
    return x * scale
```

Fix: make `scale` an `mx.array`.

## 4. Component-Level Compilation Strategy

### Component A: Routing Head
```
route(x) = sigmoid(x @ W_route + b_route)  # (1, d) -> (1, N_experts)
```
- Input: x in R^{1 x d}, W_route in R^{d x N}, b_route in R^N
- Output: gates in [0, 1]^N
- Shape: FIXED during generation (1, d) -> (1, N)
- Compilable: YES (pure array function, no side effects)
- Expected speedup: Moderate (small matmul, dispatch-bound)

### Component B: LoRA Delta Application
```
lora(x, A, B, scale) = (x @ A) @ B * scale  # or addmm variant
```
- Input: x in R^{1 x d_in}, A in R^{d_in x r}, B in R^{r x d_out}
- Output: delta in R^{1 x d_out}
- Shape: FIXED during generation
- Compilable: YES
- Expected speedup: 10-30% (fuses 3 ops into 1-2)

### Component C: Multi-Adapter LoRA
```
multi_lora(x, base_y, As, Bs, gates, scale) = base_y + sum_i(gate_i * (x @ A_i) @ B_i * scale)
```
- Compilable: YES if N is fixed (loop unrolled at trace time)
- Dynamic N: Requires recompilation per unique N
- Expected speedup: 20-50% (eliminates N kernel dispatches)

### Component D: Full Forward Pass
- NOT directly compilable (KV cache)
- Can compile sub-components within the forward pass

## 5. Complexity Analysis

### Compilation cost (one-time)
- Tracing: O(num_ops) in the function -- typically < 1 ms for our components
- Metal pipeline creation: ~10-50 ms
- Total: ~50-200 ms per unique shape

### Runtime per-call
- Compiled: ~1-2 kernel launches per fused region
- Uncompiled: ~4-7 kernel launches per LoRA projection

### Memory
- Compilation cache: O(num_unique_shapes * compiled_pipeline_size)
- No additional runtime memory vs uncompiled (same outputs)

## 6. Worked Example: Single LoRA Layer at d=2560, r=16

**Uncompiled path:**
```
x: (1, 2560) bf16           # 5 KB
h = x @ A: (1, 16) bf16     # dispatch #1, materializes 32 bytes
lora = h @ B: (1, 2560) bf16 # dispatch #2, materializes 5 KB
scaled = lora * 20.0         # dispatch #3, materializes 5 KB
y = base_y + scaled          # dispatch #4, materializes 5 KB
```
4 dispatches, 3 intermediates = ~15 KB intermediate traffic, ~40 us dispatch

**Compiled path (ideal fusion):**
```
y = addmm(base_y, x @ A, B, alpha=20.0)  # 1-2 dispatches, 0 intermediates
```
1-2 dispatches, 0 intermediates = ~0 KB intermediate traffic, ~10 us dispatch

**Expected improvement per layer:** ~30 us saved
**Across 168 projections:** ~5 ms saved per token
**At baseline 97 tok/s (10.3 ms/tok):** potential improvement to ~190 tok/s if dispatch is the ONLY bottleneck

**Reality check:** Much of the 10.3 ms is actual compute (matmul in BitLinear), not dispatch. The dispatch fraction is what determines the ceiling of mx.compile benefit.

## 7. Prior Art

- **exp_benchmark_composition_latency_sweep**: mx.compile gave 2.3-2.4x on pre-merge (synthetic bf16 matmuls). But pre-merge operates on large (2560, 2560) matrices where dispatch overhead is proportionally larger relative to compute.
- **exp_inference_speed_10x**: Noted KV cache prevents full model compilation. Did not test component-level compilation.
- **mlx_lm generate_step**: Already uses `mx.async_eval` for double-buffering (hides Python overhead behind GPU compute). This may already capture some of what mx.compile would provide.
- **fast-mlx guide**: "Compiling graphs with mx.compile can make them run a lot faster." Recommends avoiding closure capture and recompilation.

## 8. Kill Criteria Mapping

- **K1 (#258): Compilation fails on dynamic adapter selection** -- FAILS if mx.compile raises error when adapter count changes between calls, or if recompilation cost exceeds the runtime saved.
- **K2 (#259): No speedup (< 5% improvement)** -- FAILS if compiled tok/s <= 1.05x uncompiled tok/s for any component or end-to-end.
- **S1 (#26): >20% throughput improvement** -- PASSES if compiled pipeline achieves >= 1.2x tok/s vs uncompiled baseline.

**What would kill this hypothesis:**
If mlx_lm's async_eval double-buffering already hides all Python dispatch overhead (i.e., GPU is always busy), then mx.compile cannot help -- the bottleneck is compute, not dispatch. This is the most likely kill scenario given 74.2% bandwidth utilization.
