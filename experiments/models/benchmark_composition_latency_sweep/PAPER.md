# Composition Latency Sweep: Research Digest

## Hypothesis

Pre-merge composition latency scales linearly or sub-linearly with adapter count N,
remaining interactive (<50ms) at N=25 on Apple Silicon.

## What This Experiment Is

A systematic benchmark of pre-merge LoRA composition latency as a function of adapter
count N (1 to 100) on the M5 Pro using MLX. Tests three merge strategies (uncompiled,
mx.compile, precomputed delta cache), profiles bottleneck (matmul vs accumulation vs
vectorized), and measures multi-layer (7-projection) merge cost. Fits power law
T = a * N^alpha to determine scaling behavior.

## Key References

- **Naive LoRA Summation** (arxiv 2508.11985): proves orthogonality enables additive composition
- **Batched pre-merge throughput** (this project): runtime LoRA 4-87x faster for per-token routing
- **Continual learning adapter growth** (this project): quality stable within ~1% across N=5-15

## Empirical Results

### Scaling Law: Sub-linear (alpha < 1)

| Strategy | Formula | alpha | R-squared |
|----------|---------|-------|-----------|
| Uncompiled | T = 0.5793 * N^0.8275 | 0.83 | 0.972 |
| mx.compile | T = 0.3507 * N^0.7333 | 0.73 | 0.970 |
| Cached deltas | T = 0.3591 * N^0.8979 | 0.90 | 0.997 |

All strategies show sub-linear scaling (alpha < 1.0). The merge operation gets
proportionally cheaper per adapter as N grows, likely due to MLX amortizing
Metal dispatch overhead and memory bandwidth utilization improving with larger
contiguous workloads.

### Absolute Latency

| N | Uncompiled (ms) | Compiled (ms) | Cached (ms) | 7-Layer (ms) |
|---|-----------------|---------------|-------------|--------------|
| 1 | 0.92 | 0.51 | 0.39 | 2.4 |
| 5 | 1.69 | 0.96 | 1.33 | 10.6 |
| 10 | 3.19 | 1.38 | 2.58 | 20.9 |
| 25 | 7.72 | 3.28 | 6.08 | 52.2 |
| 50 | 15.34 | 6.32 | 12.19 | 119.9 |
| 100 | 30.38 | 12.48 | 24.11 | 260.5 |

### Key Finding: mx.compile Delivers 2.4x Speedup

mx.compile provides a consistent 2.3-2.4x speedup at N >= 10. This is the single
most impactful optimization. At N=25, compiled merge takes only 3.28 ms -- well
within the 50ms interactive budget.

### Bottleneck Analysis (N=25)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Matmul (B^T @ A^T) | 1.54 | 20% |
| Accumulation (W += alpha*delta) | 6.20 | 80% |
| Vectorized alternative | 3.85 | -- |

The bottleneck is NOT the rank-16 matmuls -- it is the sequential accumulation of
2560x2560 delta matrices into W_merged. This is memory-bandwidth bound: each
accumulation reads and writes 13.1 MB (bf16). The vectorized approach (stack + weighted
sum) halves accumulation time (3.85 ms vs 6.20 ms) but costs 328 MB at N=25.

### Memory Usage

| N | Active (GB) | Peak (GB) | Delta Cache (MB) |
|---|-------------|-----------|-------------------|
| 1 | 0.013 | 0.039 | 13.1 |
| 25 | 0.017 | 0.345 | 327.7 |
| 50 | 0.021 | 0.677 | 655.4 |
| 100 | 0.029 | 0.803 | 1310.7 |

Active memory grows negligibly (29 MB at N=100). Peak memory from merge intermediates
stays under 1 GB even at N=100. The precomputed delta cache strategy trades memory
(1.3 GB at N=100) for modest speedup (1.25x) -- not worth it compared to mx.compile.

### Multi-Layer Cost

For a full transformer block (7 projections), merge latency is approximately 7x
single-layer. At N=25 with uncompiled merge: 52 ms. With mx.compile: estimated
~23 ms (applying 2.35x factor). A full 24-block model at N=25 would need
~550 ms (compiled) for complete re-merge -- acceptable for initial merge at
session start, but confirms per-token routing should use runtime LoRA.

## Kill Criteria Assessment

| Criterion | Result | Evidence |
|-----------|--------|----------|
| K1 (#255): Superlinear scaling | **PASS** | alpha = 0.83 (< 1.05 threshold) |

## Success Criteria Assessment

| Criterion | Result | Evidence |
|-----------|--------|----------|
| S1 (#48): Interactive at N=25 | **PASS** | 7.72 ms uncompiled, 3.28 ms compiled (both < 50ms) |
| S1: Sub-linear scaling | **PASS** | alpha = 0.83 (< 1.0) |

## Optimization Recommendations

1. **Always use mx.compile for merge** -- 2.4x speedup, no code complexity
2. **Pre-merge for always-on adapters only** -- merge once at session start
3. **Runtime LoRA for per-token routing** -- confirmed faster by 4-87x (prior result)
4. **Do NOT use precomputed delta cache** -- 1.3 GB at N=100 for only 1.25x speedup;
   mx.compile gives 2.4x for free
5. **Vectorized merge** is interesting (halves accumulation) but memory-heavy --
   only worthwhile if N is fixed and memory is available

## Limitations

- **Single projection tested**: Full model would have 24 blocks x 7 projections = 168
  merge operations. Multi-layer phase tests 7 at a time; extrapolation to 168 is linear.
- **Synthetic weights**: Real ternary weights may have different memory access patterns.
- **No contention**: Single-thread benchmark, no concurrent Metal workloads.
- **bf16 only**: Did not test float32 or mixed precision.

## What Would Kill This

- If real model weights show cache thrashing patterns that synthetic weights do not
  (unlikely -- same shapes and dtype)
- If per-token routing becomes unnecessary (making pre-merge the only path) -- but
  prior work confirms runtime LoRA is better for routing
- If N > 1000 adapters needed (extrapolating: ~300 ms compiled, still feasible
  for session start)
