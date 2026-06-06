# Sparse-BitNet: Research Digest

## Hypothesis

Exploiting the natural ~42% zero fraction in BitNet-2B-4T ternary weights via
sparse operations could yield free inference speedup by skipping zero-valued
weight positions during matmul.

**Result: KILLED (K2 FAIL)**. Sparse approaches are slower than the native packed
BitLinear kernel on Apple Silicon. The packed uint8 format with fused Metal kernel
is already bandwidth-optimal.

## What This Experiment Tests

BitNet-2B-4T stores weights as ternary {-1, 0, +1} packed 4 values per byte in
uint8. The Sparse-BitNet paper (arxiv 2603.05168) observed ~42% zeros in trained
ternary models. This experiment tests whether exploiting that sparsity can speed
up inference on Apple Silicon by:

1. Measuring the actual zero fraction in BitNet-2B-4T (Phase 1)
2. Benchmarking multiple sparse matmul strategies against the native kernel (Phase 2-3)
3. Testing across both attention and MLP layers (Phase 3b)

## Key References

- Sparse-BitNet (arxiv 2603.05168): natural 42% sparsity in ternary weights
- exp_inference_speed_10x: 172 tok/s base, 74.2% bandwidth utilization, packed kernel
- exp_memory_budget_analysis: 1.18 GB base model in packed uint8

## Empirical Results

### Phase 1: Natural Sparsity Measurement

| Metric | Value |
|--------|-------|
| Overall zero fraction | **42.21%** |
| Range (per layer) | 36.1% - 60.1% |
| Attention (q_proj) zero fraction | 42-51% |
| MLP zero fraction | 38-60% |
| Layer 1 MLP (highest) | ~60% zeros |
| K3 threshold (30%) | **PASS** |

The 42% claim from Sparse-BitNet is confirmed for BitNet-2B-4T. Attention Q/K
projections tend to be sparser (45-50%) than V projections (36-44%). Early MLP
layers (1-3) are notably sparser (45-60%), suggesting embedding-adjacent layers
need fewer active weights.

**Bug note:** Initial run showed 28.89% due to incorrect unpacking. The Metal
kernel encoding is `(byte & 3) - 1`, not a separate mapping table. After fixing:
0 -> -1, 1 -> 0, 2 -> +1. The 1-encoding (zero value) was being miscounted.

### Phase 2-3: Benchmark (q_proj, 2560x2560)

| Method | Time (us) | Speedup vs Native | Memory |
|--------|-----------|-------------------|--------|
| Native BitLinear (packed uint8) | 241.4 | 1.000x (baseline) | 1.64 MB |
| Sparse masked (2x bf16 matmuls) | 261.9 | 0.922x (8% SLOWER) | 26.21 MB |
| Compiled sparse masked | 258.3 | 0.934x (7% SLOWER) | 26.21 MB |
| Unpacked bf16 matmul | 186.1 | 1.297x (30% faster) | 13.11 MB |
| Compiled unpacked bf16 | 186.2 | 1.297x (30% faster) | 13.11 MB |

### Phase 3b: MLP Layers (larger, 6912x2560)

| Layer | Native (us) | Unpacked bf16 (us) | Ratio |
|-------|------------|---------------------|-------|
| gate_proj | 272.7 | 291.5 | 0.936x (SLOWER) |
| up_proj | 176.3 | 289.7 | 0.609x (64% SLOWER) |
| down_proj | 180.9 | 289.2 | 0.626x (59% SLOWER) |

### Kill Criteria Assessment

| Criterion | Result | Evidence |
|-----------|--------|----------|
| K1: PPL degradation > 5% | N/A (exact) | max abs diff = 1.0 on [-167, 207] range (bf16 rounding) |
| K2: No wall-clock speedup | **FAIL** | Sparse masked 7% slower, unpacked bf16 8x more memory |
| K3: Zero fraction < 30% | **PASS** | 42.21% zeros confirmed |
| S1: >= 1.3x speedup | **FAIL** | Best: 1.297x on q_proj only, but 8x memory cost |

## Why It Failed

### 1. Bandwidth Bound, Not Compute Bound

At 172 tok/s, BitNet-2B-4T already uses 74.2% of the M5 Pro's 273 GB/s memory
bandwidth. The bottleneck is moving data to the GPU, not computing on it.
Packed uint8 (2 bits/weight) transfers 4x less data than bf16. Any approach
that unpacks weights increases data transfer and slows inference.

### 2. Packed Kernel Handles Zeros for Free

The Metal kernel's inner loop does `sum += v[j] * ((w & 3) - 1)`. When the
encoded value is 1 (zero weight), it computes `v[j] * 0 = 0`. This multiply-by-zero
costs the same as multiply-by-one on SIMD hardware -- there is no branch, no skip,
just a zero that gets accumulated. The "sparsity" is invisible to the ALU pipeline.

### 3. Sparse Approaches Destroy Memory Efficiency

| Format | Storage per logical weight | Ratio vs packed |
|--------|--------------------------|-----------------|
| Packed uint8 | 0.25 bytes | 1.0x |
| Unpacked bf16 | 2.0 bytes | 8.0x |
| Sparse masks (2x bf16) | 4.0 bytes | 16.0x |
| Sparse CSR indices | ~3.0 bytes | 12.0x |

Every sparse format tested is larger than the packed representation. When
bandwidth is the bottleneck, larger = slower.

### 4. The Surprising q_proj Result

Unpacked bf16 matmul is 1.3x faster than the native kernel on q_proj (2560x2560).
This is likely because Metal's bf16 matmul kernel is more optimized than the
custom ternary kernel (which uses float32 accumulation and a non-standard access
pattern with 4-row interleaving). However, this comes at 8x memory cost
(13 MB vs 1.6 MB per layer), and the advantage disappears on larger MLP layers
where the native kernel's bandwidth efficiency dominates.

At full-model scale: switching to bf16 would increase model size from 1.18 GB
to ~9.4 GB. At 273 GB/s bandwidth, this limits throughput to 273/9.4 = 29 tok/s
vs the current 172 tok/s. The 1.3x per-layer speedup is overwhelmed by 6x more
data transfer.

## Limitations

1. **Single-token inference only.** Batch inference with B>1 might change the
   compute/bandwidth ratio, but BitNet's use case is interactive generation.
2. **No custom sparse Metal kernel.** A purpose-built kernel that reads packed
   uint8 but skips zero positions might avoid the memory expansion. However, the
   SIMD architecture penalty for irregular access patterns makes this unlikely
   to help.
3. **Only tested layer 0.** Other layers might have different characteristics,
   though the MLP benchmark confirms the trend across layer types.

## What Would Kill This

Already killed. No path to speedup exists on Apple Silicon because:
- The packed uint8 format is already more compact than any sparse representation
- Metal's SIMD pipeline handles zeros for free (no branch prediction penalty)
- The inference bottleneck is memory bandwidth, not ALU utilization
- Any format change that increases data size reduces throughput

## Key Learnings

1. **Natural sparsity is 42.21% in BitNet-2B-4T** -- confirming Sparse-BitNet.
   However, this sparsity cannot be exploited on GPU hardware that lacks native
   sparse matmul support.

2. **The packed uint8 BitLinear kernel is optimal for Apple Silicon.** Do not
   attempt to "optimize" it with sparse approaches. The 2-bit packing is the
   most compact possible representation for ternary weights.

3. **bf16 matmul is faster per-layer for small matrices** (1.3x on 2560x2560)
   but loses at model scale due to 8x memory expansion. This is consistent with
   the exp_inference_speed_10x finding that pre-merge (bf16 unpack) is slower
   than runtime LoRA.

4. **Sparsity-based acceleration requires hardware support.** Apple's Neural
   Engine or future GPU architectures with sparse tensor cores could change
   this conclusion, but current Metal GPU cannot benefit.
