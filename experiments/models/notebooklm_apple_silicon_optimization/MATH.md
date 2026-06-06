# Quantitative Analysis: MLX Optimization for Ternary Models on M5 Pro

## A. The Bandwidth-Bound Regime: Why Ternary Inference Cannot Be Compute-Bound

### Roofline Model for Ternary Inference

**Definition (Roofline Bound).** For any operation with arithmetic intensity
I (FLOPs/byte), peak memory bandwidth B_mem (GB/s), and peak compute C_peak (FLOP/s):

  Throughput <= min(C_peak, I * B_mem)

The crossover point (ridge point) occurs at I_ridge = C_peak / B_mem.

**Theorem 1 (Ternary Inference is Bandwidth-Bound at Batch=1).**
For a ternary model with N parameters, generating one token requires reading
all N weights. Ternary matmul replaces multiply-accumulate with
addition/subtraction, yielding approximately N FLOPs per token (additions only,
no multiplications). The arithmetic intensity is:

  I_ternary = N_flops / N_bytes = N / (N * 1.58/8) = 8/1.58 = 5.06 FLOPs/byte

For the M5 Pro:
- B_mem = 307 GB/s (official spec; our measured peak is 268.6 GB/s)
- C_peak (GPU Neural Accelerator) >= 4x M4 Pro AI compute

  I_ridge = C_peak / B_mem >> 5.06 for any reasonable GPU TFLOPS figure

Since I_ternary = 5.06 << I_ridge, ternary inference at batch=1 is
*unconditionally* memory-bandwidth-bound. QED.

**Corollary 1.1.** The theoretical maximum generation speed at batch=1 is:

  tok/s_max = B_mem / model_size

For BitNet-2B-4T on M5 Pro (measured bandwidth 268.6 GB/s, model 1.18 GB):
  tok/s_max = 268.6 / 1.18 = 227.6 tok/s

For M5 Pro official bandwidth (307 GB/s):
  tok/s_max = 307 / 1.18 = 260.2 tok/s

**Corollary 1.2 (Batch crossover).** Batched inference increases arithmetic
intensity by factor b (batch size), since b forward passes share one weight read:

  I_batched = b * I_ternary = 5.06b FLOPs/byte

The system becomes compute-bound when 5.06b > I_ridge. For standard float models
on M5 Pro GPU (~20 GPU cores, each with Neural Accelerator):

If we estimate the M5 Pro GPU at ~20 TFLOPS for standard compute (the 4x claim
is for "AI compute" which may reference the Neural Accelerator's tensor throughput):

  b_crossover = I_ridge / 5.06 = (20e12 / 307e9) / 5.06 = 65.1 / 5.06 ~ 13

So batch > 13 crosses into compute-bound territory. For single-user generation
(batch=1), this is irrelevant. For serving multiple concurrent requests,
batching becomes the primary throughput lever.

### Measured Performance vs. Theoretical Bounds

| Metric | Theoretical | Measured | Utilization |
|--------|------------|----------|-------------|
| Peak bandwidth | 307 GB/s | 268.6 GB/s | 87.5% |
| tok/s (at 268.6 GB/s) | 227.6 | 165.6 | 72.7% |
| tok/s (at 307 GB/s) | 260.2 | 165.6 | 63.6% |
| Forward pass (seq=1) | ~4.4 ms* | 6.04 ms | 72.8% |

*Theoretical: 1.18 GB / 268.6 GB/s = 4.39 ms

The 27% gap between measured bandwidth and measured throughput is due to:
1. KV cache reads (small but nonzero at short sequences)
2. Activation computation (RMSNorm, RoPE, softmax, SiLU)
3. Metal command buffer dispatch overhead
4. Python interpreter overhead (~10.8% measured)

## B. Memory Budget Model

**Theorem 2 (Adapter Memory Scaling).**
For N adapters with rank r on a model with hidden dimension d and L layers,
each with p projections:

  Memory_adapter = 2 * L * p * d * r * dtype_bytes  (A and B matrices)

For BitNet-2B-4T (d=2560, L=30, p=7 projections, rank=16, bf16):
  Memory_adapter = 2 * 30 * 7 * 2560 * 16 * 2 = 43,008,000 bytes = 41.0 MB theoretical

Measured: 45.2 MB (10.2% overhead from MLX allocator alignment).

**Corollary 2.1.** Total memory for N adapters:
  Memory_total = Memory_base + N * Memory_adapter + Memory_routing + Memory_KV

For M5 Pro 48GB (40GB usable after OS):
  N_max = (40GB - 1.18GB - 0.082KB * N - KV) / 45.2MB

At 2GB KV budget: N_max = (40 - 1.18 - 2) / 0.0452 = 813 adapters

Measured N_max = 853 (Finding #77), consistent within allocator variance.

## C. Bandwidth Utilization Analysis

### Why 73% and Not Higher

The gap between peak bandwidth (268.6 GB/s measured) and effective throughput
(195.6 GB/s effective = 1.18 GB * 165.6) has identifiable components:

| Overhead Source | Estimated Impact | Evidence |
|----------------|-----------------|----------|
| Python/tokenizer | 10.8% | Finding: wall-clock vs internal timing |
| KV cache reads | ~5% at short seq | Grows with sequence length |
| Activation compute | ~5% | RMSNorm, RoPE, softmax, SiLU |
| Metal dispatch | ~5% | Per-layer eval boundary overhead |
| Unexplained | ~2% | Cache effects, allocator |

### Path to Higher Utilization

**Proposition 3.** To approach 90% bandwidth utilization (242 GB/s effective,
~205 tok/s), the following are necessary:

1. Eliminate Python overhead: ~11% recovery -> ~184 tok/s
2. Reduce dispatch overhead via mx.compile: already gives 21% at seq=1 -> ~166 tok/s (current)
3. The remaining gap requires either:
   a. Fused BitLinear kernels (eliminate activation/norm round-trips to memory)
   b. Speculative decoding (amortize fixed overhead over multiple tokens)
   c. Batched inference (increase arithmetic intensity)

## D. Quantitative Predictions

### D1: M5 Pro Bandwidth Improvement over M4 Pro

M5 Pro: 307 GB/s official
M4 Pro: 120 GB/s (from research report [ref 2,3])

Predicted improvement ratio: 307/120 = 2.56x

For same model (BitNet-2B-4T at 1.18 GB):
- M4 Pro theoretical: 120/1.18 = 101.7 tok/s
- M5 Pro theoretical: 307/1.18 = 260.2 tok/s
- At 73% utilization: M4 Pro ~74 tok/s, M5 Pro ~190 tok/s

Our measured 165.6 tok/s suggests we are not yet exploiting the full 307 GB/s
(only reaching 268.6 GB/s peak in microbenchmarks).

### D2: GPU Neural Accelerator Impact

The M5 Pro has Neural Accelerators integrated in each GPU core, claiming 4x
AI compute vs M4 Pro. However, for bandwidth-bound workloads, this extra
compute does NOT improve single-batch generation speed. The Neural Accelerators
become relevant for:

1. Batched inference (batch > ~13 to cross ridge point)
2. Prefill (prompt processing, which IS compute-bound)
3. Speculative decoding verification steps

### D3: Optimal Serving Configuration

| Scenario | Predicted tok/s | Memory | Bottleneck |
|----------|----------------|--------|------------|
| Base only (batch=1) | 166-205 | 1.18 GB | Bandwidth |
| +1 adapter (addmm) | 97 | 1.22 GB | Adapter overhead |
| +1 adapter (attn-only) | 127 | ~1.20 GB | Reduced adapter |
| Batch=4, base only | ~650 | 1.18 GB + 4*KV | Still BW-bound |
| Batch=16, base only | ~2000* | 1.18 GB + 16*KV | Approaching compute |

*Theoretical if batch fits memory and compute is available.

## E. Assumptions and Breaking Conditions

1. **Bandwidth-bound assumption** holds for batch=1 generation. Breaks at
   batch > ~13 where compute becomes the bottleneck. The Neural Accelerators
   then become the critical resource.

2. **Model fits in memory** assumption. The 1.18 GB packed model easily fits.
   At N=500 adapters (23.86 GB), the system may experience memory pressure
   affecting bandwidth utilization.

3. **MLX overhead is minimal** assumption. Measured at 10.8% for Python,
   5% for eval boundaries. Could increase with complex routing logic.

4. **No memory wiring** assumption. Our measurements do not use
   `sysctl iogpu.wired_limit_mb`. Wiring memory could improve bandwidth
   utilization by preventing page faults during inference.

## F. Worked Example: Single Token Generation

For BitNet-2B-4T on M5 Pro generating one token:

1. Read all weights: 1.18 GB / 268.6 GB/s = 4.39 ms (theoretical)
2. KV cache read: ~0.2 ms (at seq_len=100, 30 layers * 2560 * 2 * 100 * 2 bytes / 268.6 GB/s)
3. Activation compute: ~0.3 ms (30 * (RMSNorm + RoPE + softmax + SiLU))
4. LM head: 2.53 ms (measured, largest single component — tied embedding, 2560 * 151936 * 2 bytes)
5. Metal dispatch: ~0.3 ms total (30 layers * ~0.01 ms/dispatch)
6. Total: ~7.7 ms (theoretical), measured 6.04 ms

The measured time being FASTER than our itemized estimate suggests MLX's lazy
evaluation successfully overlaps weight reads with compute (measured in profiling:
sum-of-parts = 135% of total, indicating ~35% overlap).

## G. Complexity and Architecture Connection

### FLOPs per Token

For a standard transformer with d_model, n_layers, n_heads:
  FLOPs_float = 2 * N_params (multiply + accumulate)
  FLOPs_ternary = N_params (addition only, no multiply)

BitNet-2B-4T: ~2B parameters -> ~2 GFLOP/token (ternary) vs ~4 GFLOP/token (float)

### Arithmetic Intensity Comparison

| Model Type | Bits/Weight | Bytes/Weight | AI (FLOPs/byte) | Regime (M5 Pro) |
|-----------|------------|-------------|-----------------|-----------------|
| FP16 | 16 | 2 | 1.0 | Bandwidth-bound |
| INT4 | 4 | 0.5 | 4.0 | Bandwidth-bound |
| Ternary (1.58-bit) | 1.58 | 0.2 | 5.06 | Bandwidth-bound |
| Ternary batch=16 | 1.58 | 0.2 | 80.9 | Compute-bound |

All single-batch LLM inference is bandwidth-bound on M5 Pro. The advantage
of ternary is not "faster compute" but "fewer bytes to read" -- 10x fewer
than FP16, 2.5x fewer than INT4.
