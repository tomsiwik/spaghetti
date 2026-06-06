# Batched LoRA Dispatch via Stacked Matmul: Proof Verification Report

## Theorem

**Theorem 1 (Numerical Equivalence).** Stacking K adapter A matrices into (K, r, d)
and computing a batched matmul produces identical output to sequential per-adapter
dispatch. Verified: MSE = 0 for stacked, < 1e-13 for addmm (within bf16 epsilon).

**Theorem 2 (Latency Reduction).** Stacking reduces matmul count from 2K to 2 per
module, saving (K-1) kernel launches. The speedup exists when saved dispatch overhead
exceeds batching overhead.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|-------------------------|----------|--------|
| Stacked MSE < 1e-6 (Thm 1) | MSE = 0.0 (exact) | YES |
| Concat MSE < 1e-6 | MSE < 1.2e-14 | YES |
| Addmm MSE < 1e-6 | MSE < 3.3e-14 | YES |
| No speedup at K=1 (identity) | Stacked: 0.99x (prod) | YES |
| Speedup at K=5 (prod) 1.2-2.5x | Stacked: 1.02x (prod) | NO |
| Speedup at K=5 (micro) | Stacked: 1.75x (micro) | YES (micro only) |
| Isolated matmul speedup | 0.98-1.01x (all K) | NO |
| Memory < 3 GB | 17.3 GB (synthetic fp32 model) | N/A (wrong baseline) |

## Hypothesis

Batching K adapter projections into a single stacked matmul achieves >= 85 tok/s
while maintaining numerical equivalence to sequential dispatch.

**Verdict: KILLED.** Stacking provides no speedup at production scale (d=2560, L=30).

## What This Model Is

Four strategies for computing runtime LoRA adapter projections were compared:

1. **Sequential** (v3 baseline): Python loop over K adapters, each doing `x@A.T` then `@B.T`
2. **Stacked**: Stack all A matrices into (K, r, d), single batched matmul via broadcast
3. **Concat**: Concatenate A matrices into (K*r, d), single large matmul, then split for B
4. **Addmm**: Sequential with `mx.addmm` fusion (fuses base_out + scale * lora in one op)

## Key References

- Punica BGMV (Chen et al., 2310.18547) -- fused CUDA kernel for multi-adapter dispatch
- S-LoRA (Sheng et al., 2311.03285) -- scales to thousands of concurrent adapters
- Finding #76: mx.compile redundant for generation (async_eval already hides dispatch)
- Finding #288: v3 48% overhead from bf16 matmul, not dispatch overhead
- Finding #300: Memory bandwidth is the bottleneck, not dispatch count

## Empirical Results

### Numerical Equivalence (K770 PASS)

All four strategies produce identical or near-identical output:

| K | Stacked MSE | Concat MSE | Addmm MSE |
|---|-------------|------------|-----------|
| 1 | 0.0 | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 | 2.6e-14 |
| 3 | 0.0 | 0.0 | 2.6e-14 |
| 5 | 0.0 | 1.2e-14 | 3.3e-14 |

Stacked approach achieves exact numerical equivalence (MSE = 0.0) because MLX performs
the same operations in the same order with the same accumulator. Addmm has epsilon
differences from fused multiply-add rounding.

### Micro-Scale Speed (d=128, L=4) -- Mechanism Works in Principle

| K | Sequential (ms) | Stacked (ms) | Concat (ms) | Addmm (ms) | Best Speedup |
|---|-----------------|--------------|-------------|-------------|-------------- |
| 1 | 0.594 | 0.643 | 0.620 | 0.503 | 1.18x (addmm) |
| 2 | 0.823 | 0.776 | 0.796 | 0.690 | 1.19x (addmm) |
| 3 | 1.057 | 0.781 | 0.960 | 0.890 | 1.35x (stacked) |
| 5 | 1.359 | 0.779 | 1.219 | 1.216 | 1.75x (stacked) |

At micro scale, stacked shows clear benefit at K>=3 (1.35-1.75x). This is because
Python loop overhead is a significant fraction of total compute when the model is tiny.
Addmm consistently provides ~18% improvement via fused multiply-add.

### Production-Scale Speed (d=2560, L=30) -- NO Improvement

| K | Sequential (ms) | Stacked (ms) | Concat (ms) | Addmm (ms) | Best Speedup |
|---|-----------------|--------------|-------------|-------------|-------------- |
| 1 | 53.33 | 53.64 | 53.46 | 52.74 | 1.01x (addmm) |
| 2 | 54.19 | 54.75 | 54.48 | 53.96 | 1.00x (addmm) |
| 3 | 55.60 | 55.85 | 55.96 | 55.29 | 1.01x (addmm) |
| 5 | 58.04 | 56.70 | 58.33 | 57.88 | 1.02x (stacked) |

At production scale, ALL strategies perform within 2% of sequential. The stacking
mechanism that provides 1.75x at micro scale provides only 1.02x at production scale.

### Isolated Matmul Benchmark -- Definitive Null Result

| Scale | K | Sequential (us) | Stacked (us) | Concat (us) |
|-------|---|-----------------|--------------|-------------|
| micro | 1 | 156 | 157 | 156 |
| micro | 5 | 158 | 159 | 156 |
| prod | 1 | 156 | 157 | 158 |
| prod | 5 | 160 | 162 | 158 |

The isolated matmul benchmark removes all model overhead and measures pure matmul
performance. Result: ALL strategies take ~157 us regardless of K, d, or batching.

**This is the definitive kill signal.** MLX's Metal backend already parallelizes
sequential matmul operations via lazy evaluation. The computation graph is built
lazily and dispatched as a single graph to the GPU. Stacking the matrices into a
batched tensor does not reduce the actual GPU work or memory bandwidth.

## Why It Failed: Root Cause Analysis

The hypothesis assumed that sequential Python loops create sequential GPU kernel
launches. This is FALSE for MLX. MLX's lazy evaluation model means:

1. **All matmuls in the loop are recorded into a computation graph** without executing
2. **The graph is dispatched to Metal as a batch** when `mx.eval()` is called
3. **The GPU executes all operations concurrently** (pipeline parallelism)

This is equivalent to what Punica BGMV does with fused CUDA kernels, except MLX does
it automatically via lazy evaluation. The Punica optimization is necessary on CUDA
because PyTorch uses eager execution (each op launches a kernel immediately). MLX's
lazy model eliminates this bottleneck by design.

The micro-scale speedup (1.75x at K=5) exists because at d=128, the total GPU compute
is so small (~0.4ms) that Python loop overhead (~0.1ms) is 25% of total time. At
production scale (d=2560, L=30), total compute is ~55ms and Python overhead is <0.5ms
(< 1%), so eliminating the loop provides negligible benefit.

## Implications

1. **MLX lazy evaluation is already the "kernel fusion" we were looking for.** The
   Punica BGMV pattern is only useful for eager-execution frameworks (PyTorch CUDA).
   On MLX, the framework provides this optimization for free.

2. **The v3 48% overhead (Finding #288) is from raw bf16 matmul cost, not dispatch.**
   Reducing dispatch count from K*2 to 2 per module cannot improve speed because
   dispatch is free under lazy eval. The overhead is from the additional memory reads
   for adapter weights (A and B matrices).

3. **addmm provides consistent ~1% improvement** at production scale by fusing the
   addition into the matmul. This is marginal but free.

4. **Speed improvements require reducing memory bandwidth**, not reducing operation
   count. Approaches: ternary adapters (Finding #288's suggestion), adapter pruning
   (Finding #304: attn-only for prose), or fewer active adapters.

## Limitations

- Synthetic fp32 model, not real BitNet-2B-4T ternary base. Absolute tok/s numbers
  are NOT comparable to v3's 73 tok/s. Only relative speedups are meaningful.
- No KV cache (single-token generation without cache). Real inference with KV cache
  may have different overhead proportions.
- Single sequence (batch=1). Larger batches may shift the compute/bandwidth balance.

## What Would Kill This

Already killed. The definitive evidence is the isolated matmul benchmark showing
zero speedup at any scale or K value. MLX lazy evaluation makes batching redundant.

## Kill Criteria Assessment

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| K769: Speed | >= 85 tok/s | 19.0 tok/s (synthetic model, not comparable to v3) | FAIL* |
| K770: MSE | < 1e-6 | 0.0 (exact) | PASS |
| K771: Memory | < 3 GB | 17.3 GB (synthetic fp32 model) | FAIL* |

*K769 and K771 fail on absolute thresholds, but these are not comparable to v3
because the experiment uses a synthetic fp32 model (3.9B params) instead of the
real ternary BitNet-2B-4T (1.18 GB packed). The meaningful signal is the relative
speedup, which is 0.99-1.02x at production scale -- definitively NO improvement.

**Verdict: KILLED.** Stacking/batching LoRA dispatch provides no speed improvement
on MLX because lazy evaluation already batches operations automatically.
