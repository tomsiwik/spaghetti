# Memory Budget Analysis: Research Digest

## Hypothesis

On a 48 GB Apple Silicon machine, the BitNet-2B-4T base model (1.18 GB packed)
leaves sufficient memory to hold 500+ ternary LoRA adapters simultaneously,
validating the vision of massive adapter pools on consumer hardware.

## What This Experiment Is

A systematic memory budget analysis combining theoretical derivation and empirical
measurement on MLX. We compute exact byte counts for every component of the
SOLE serving pipeline (base model, adapters, router, routing heads, KV cache,
activations), then validate with actual MLX memory profiling at N=10, 50, 100, 500.

## Key References

- **Prior experiment:** micro/models/memory_optimized_serving/ (1.22 GB baseline)
- **S-LoRA** (Sheng et al., 2311.03285): concurrent LoRA serving for thousands of adapters
- **BitNet b1.58** (Ma et al., 2402.17764): ternary architecture enabling 8x weight compression

## Empirical Results

### Base Model Memory

| Component | Measured |
|-----------|----------|
| BitNet-2B-4T (BitLinear packed) | 1,178.6 MB |
| After forward pass | 1,179.3 MB |
| Forward pass peak | 1,185.8 MB |

The base model occupies only 1.18 GB -- 8x smaller than the 4.8 GB bf16 equivalent.
This is the fundamental enabler of massive adapter pools.

### Per-Adapter Memory (Measured)

| N | Per Adapter (MB) | Per Head (MB) | Total (MB) | Alloc Time |
|---|------------------|---------------|------------|------------|
| 10 | 45.19 | 0.080 | 452.7 | 0.1s |
| 50 | 44.38 | 0.082 | 2,268 | 0.6s |
| 100 | 44.87 | 0.081 | 4,541 | 1.3s |
| 500 | 45.24 | 0.082 | 22,709 | 7.2s |

Per-adapter cost: **45.2 MB** (bf16 A+B matrices, rank 16, 30 layers x 7 targets).
This is 4.4% above the theoretical 43.3 MB due to MLX allocator overhead (~1.9 MB
per adapter from page alignment, metadata, and Metal buffer headers).

Per-routing-head cost: **82 KB** (2-layer MLP, 41K params, bf16). Negligible.

### Practical Maximum N (40 GB usable)

| Seq Length | N_max | Memory at N=100 | Memory at N=500 |
|------------|-------|-----------------|-----------------|
| 256 | **853** | 5.73 GB | 23.86 GB |
| 2048 | **850** | 5.87 GB | 24.00 GB |
| 8192 | **840** | 6.34 GB | 24.47 GB |

KV cache has negligible impact: going from seq=256 to seq=8192 reduces N_max by
only 13 adapters (1.5%).

### Forward Pass Peak Memory (with routing + composition)

| Active Adapters (k) | Memory (MB) | Peak (MB) | LoRA Overhead |
|---------------------|-------------|-----------|---------------|
| k=1 | 1,227 | 1,253 | negligible |
| k=2 | 1,273 | 1,299 | negligible |
| k=5 | 1,408 | 1,434 | negligible |

Forward pass peak memory adds only ~25 MB above steady state, regardless of k.
Runtime LoRA computation creates transient intermediate tensors that are immediately
freed. The bottleneck is stored adapter weights, not computation.

### Theoretical vs Measured

| Metric | Theoretical | Measured | Ratio |
|--------|-------------|----------|-------|
| Base model | 1,178.6 MB | 1,178.6 MB | 1.000x |
| Per adapter (bf16) | 43.3 MB | 45.2 MB | 1.044x |
| Per head | 82 KB | 82 KB | 1.000x |
| N_max (seq=256) | 895 | 853 | 0.953x |

The 4.4% overhead is consistent MLX allocator cost (page alignment + Metal buffer
metadata). The theoretical model is accurate within 5%.

### Memory Budget at Key Milestones

| Configuration | Memory | % of 40 GB |
|---------------|--------|------------|
| Base only | 1.18 GB | 3.0% |
| + 10 adapters + heads | 1.63 GB | 4.1% |
| + 50 adapters + heads | 3.45 GB | 8.6% |
| + 100 adapters + heads | 5.73 GB | 14.3% |
| + 500 adapters + heads | 23.86 GB | 59.6% |
| + 853 adapters + heads | 39.86 GB | 99.7% |

### With int8 B-Matrices (Theoretical, Validated at Reconstruction Error 3e-04)

| Metric | bf16 A+B | int8 B + bf16 A |
|--------|----------|-----------------|
| Per adapter | 45.2 MB | ~33.2 MB |
| N_max (seq=256) | 853 | ~1,165 |

int8 quantization of B matrices (validated in memory_optimized_serving with
negligible PPL impact) would increase capacity by 37%.

### Kill Criteria Assessment

- **K1 (261): "< 100 adapters fit in 48GB"** --> **PASS**. N_max = 853 at seq=256.
  That is 8.5x above the 100-adapter kill threshold. Even at the longest context
  (seq=8192), N_max = 840.

### Success Criteria Assessment

- **S1 (27): ">500 adapters fit in 48GB"** --> **PASS**. N_max = 853 at seq=256.
  At N=500: 23.86 GB used (59.6% of budget), leaving 16 GB headroom.

## Key Findings

1. **BitNet-2B-4T's 1.18 GB footprint is the enabler.** A standard 2B model at bf16
   would consume ~4.8 GB, cutting adapter capacity from 853 to ~813. The ternary base
   provides 3.6 GB of extra adapter budget compared to bf16.

2. **Adapter memory scales linearly with no surprises.** Per-adapter cost is a
   constant 45.2 MB from N=10 to N=500. No fragmentation amplification, no
   non-linear scaling, no hidden costs.

3. **Routing infrastructure is negligible.** 500 routing heads = 41 MB (0.1% of
   budget). The shared router adds 1.3 MB. Routing does not constrain adapter capacity.

4. **KV cache is not the bottleneck.** Even at seq=8192 (629 MB KV cache), it
   reduces adapter capacity by only 1.5%.

5. **Forward pass overhead is negligible.** Peak memory during runtime LoRA
   computation adds only ~25 MB regardless of k active adapters. The entire
   routing + composition + forward pipeline runs within the stored adapter budget.

## Limitations

1. **Synthetic adapters only.** We allocate random bf16 tensors with correct shapes
   but do not load real trained adapters from disk. Real adapter loading includes
   additional transient memory for file I/O buffers. Expected impact: <1%.

2. **No concurrent request profiling.** A production server handling multiple
   simultaneous requests would need separate KV caches per request, potentially
   significant at batch > 1.

3. **MLX allocator behavior may vary.** The 4.4% overhead measured is specific to
   the current MLX version. Future versions may change allocation granularity.

4. **Platform reports 52 GB.** The M5 Pro reports 52 GB total via `mx.device_info()`,
   not 48 GB. We use 40 GB usable (8 GB system reservation). Actual usable memory
   may be higher, increasing N_max proportionally.

5. **Adapter format is bf16 (not ternary-packed).** Runtime LoRA requires bf16 for
   the addmm operation. A custom Metal kernel for ternary matmul could reduce B
   matrix memory 16x, but does not exist yet.

## What Would Kill This

- **At micro scale:** If MLX memory fragmentation causes OOM well before theoretical
  limit. Our measurement at N=500 (22.7 GB) shows no fragmentation effects.

- **At production scale:** If multi-request serving with KV cache per request
  consumes most of the adapter budget (batch=8 at seq=2048 = 1.26 GB KV cache).

- **Format constraint:** If a future mechanism requires adapters in a format larger
  than bf16 (e.g., fp32 for gradient accumulation during online learning).
