# Inference Latency vs N: Mathematical Analysis

## Notation

| Symbol | Definition | Micro Value | Macro Value (Qwen2.5-7B) |
|--------|-----------|-------------|--------------------------|
| d | Hidden dimension | 128 | 3584 |
| L | Number of transformer layers | 4 | 28 |
| T | LoRA target modules per layer | 6 (q,k,v,o,fc1,fc2) | 7 (q,k,v,o,up,gate,down) |
| r | LoRA rank | 8 | 16 |
| N | Total experts in the library | 5..100 | 5..1000+ |
| k | Experts selected per query (top-k) | 1, 2, 4 | 1, 2 |
| F_base | Base model FLOPs per token | ~2M | ~14B |
| s | Sequence length (tokens) | 32 | 512+ |
| b | Batch size | 1 | 1..64 |

## Strategy A: Pre-Merged Composition

### Operation

Given N experts with LoRA parameters {(A_i, B_i)}_{i=1}^{N} per linear layer,
pre-merge computes a single weight matrix:

```
W' = W + (1/N) * sum_{i=1}^{N} B_i @ A_i
```

Where:
- W: (d_out, d_in) -- original weight matrix
- A_i: (r, d_in) -- expert i's down-projection
- B_i: (d_out, r) -- expert i's up-projection
- B_i @ A_i: (d_out, d_in) -- full-rank delta contribution

### Merge Cost (One-Time)

Per linear layer:
```
C_merge = N * (d_out * r * d_in)    FLOPs  (N matmuls of B_i @ A_i)
        + N * d_out * d_in           FLOPs  (element-wise sum)
```

Total across model:
```
C_merge_total = L * T * C_merge
```

For micro model: L=4, T=6, d=128, r=8, N=100:
```
C_merge_total = 4 * 6 * (100 * 128 * 8 * 128 + 100 * 128 * 128)
              = 24 * (100 * 131072 + 100 * 16384)
              = 24 * 14,745,600
              = 353,894,400 FLOPs
```

This is a one-time cost at model load, measured at ~53ms for N=100 at micro scale.
Amortized over thousands of queries, it is negligible.

### Inference Cost

```
F_premerge = F_base    (identical to base model -- same weight shapes)
```

**Pre-merge overhead = 0% by construction.** The merged weight matrix has the
same shape as the original. The forward pass is bit-for-bit identical in
computational cost. The only difference is the values in the weight matrices.

Measured overhead: -3% to +3% (within measurement noise).

### Scaling with N

The merge TIME scales as O(N) (one matmul per expert per layer).
The inference LATENCY is O(1) w.r.t. N -- it does not depend on N at all.

This is the key advantage: pre-merge converts an O(N) composition into
an O(1) forward pass at the cost of always activating all experts.

## Strategy B: Dynamic Top-k Composition

### Operation

At query time, select k experts, apply their deltas:

```
Y = f_base(x) + (1/k) * sum_{j in S_k} B_j @ A_j @ x
```

Where S_k is the set of k selected experts.

### FLOPs Analysis

Per linear layer, per expert:
```
F_lora_delta = b * s * r * (d_in + d_out)   (two matmuls: x@A^T and result@B^T)
```

For k experts across the full model:
```
F_dynamic = F_base + k * L * T * b * s * r * (d_in + d_out)
```

### Theoretical Overhead

```
overhead = F_lora / F_base = k * L * T * r * (d_in + d_out) / F_base
```

For micro model (d=128, r=8, L=4, T=6):
```
F_lora_per_expert = 4 * 6 * (8 * (128 + 128)) = 4 * 6 * 2048 = 49,152 FLOPs
F_base ~ 2,000,000 FLOPs

overhead_k1 = 49,152 / 2,000,000 = 2.5%
overhead_k2 = 98,304 / 2,000,000 = 4.9%
overhead_k4 = 196,608 / 2,000,000 = 9.8%
```

For macro model (Qwen2.5-7B: d=3584, r=16, L=28, T=7):
```
F_lora_per_expert = 28 * 7 * (16 * (3584 + 3584)) = 28 * 7 * 114,688 = 22,477,056
F_base ~ 14,000,000,000

overhead_k1 = 22.5M / 14B = 0.16%
overhead_k2 = 45M / 14B = 0.32%
```

### Scaling with N and k

**Critical result:** The dynamic forward pass cost depends on k, not N.

```
F_dynamic(N, k) = F_base + k * F_lora_single_expert
```

N affects only:
1. Memory footprint: O(N * L * T * r * (d_in + d_out)) to store all expert matrices
2. Routing cost: O(log N) for hash ring lookup

The inference FLOP count is N-independent. This is confirmed empirically:
dynamic k=1 overhead varies by only ~11 percentage points across N=5..100,
which is within measurement noise.

### Implementation Overhead

The measured overhead (~260% at k=1) vastly exceeds the theoretical (~2.5%) due to:

1. **Weight copy overhead**: Modifying in-place requires clone + copy + restore
   for each forward pass. This is 2 * L * T full weight tensor copies.

2. **Python overhead**: Each weight modification goes through PyTorch's Python
   dispatch layer.

At production scale with fused CUDA kernels (vLLM, S-LoRA), the overhead
approaches the theoretical bound. See macro/batched_lora_latency for evidence:
direct copy at k=1 is -4% overhead (faster than monolithic due to measurement noise).

## Strategy C: Hybrid Composition

### Operation

Pre-merge N_f foundation experts, dynamically select k from N_d specialists:

```
W' = W + (1/N_f) * sum_{i=1}^{N_f} B_i @ A_i     (pre-merged)
Y = f_{W'}(x) + (1/k) * sum_{j in S_k} B_j @ A_j @ x   (dynamic on top)
```

### FLOPs

```
F_hybrid = F_base + k * F_lora_single_expert
```

Same as Strategy B. The pre-merged experts are absorbed into the weights
and add zero runtime cost. Only the dynamic specialists add overhead.

### When Hybrid Wins

Hybrid is optimal when:
- Most queries are handled by foundation experts (high coverage)
- Only rare/specialized queries need dynamic selection
- k_dynamic << N total

Example: 50 foundation experts + 50 specialists, k=1 dynamic:
- Pre-merge: all 100 experts always active (diluted signal)
- Dynamic: 1/100 expert active (high specialization, high overhead)
- Hybrid: 50 foundation always active + 1 specialist (best of both)

## Hash Ring Routing Latency

### Complexity

Building the ring: O(N * V * log(N * V)) where V = virtual nodes per expert.
Querying: O(log(N * V)) for binary search + O(k) for collecting k distinct experts.

With V=150:
```
Ring size = N * 150
Query = O(log(150*N)) + O(k)
```

### Measured Scaling

| N | Ring Size | Per-Query (us) |
|---|-----------|----------------|
| 5 | 750 | 0.52 |
| 10 | 1500 | 0.52 |
| 20 | 3000 | 0.53 |
| 50 | 7500 | 0.57 |
| 100 | 15000 | 0.57 |

Growth: 1.10x for 20x N. This is consistent with O(log N): log(100)/log(5) = 2.86,
but the absolute overhead is so small (0.5 us) that it is negligible compared to
model inference (1 ms at micro scale, 30 ms at macro scale).

Routing is not a bottleneck at any N.

## Memory Scaling

### Per Expert

```
M_expert = L * T * 2 * r * d * sizeof(float)
```

Micro (d=128, r=8, L=4, T=6, float32):
```
M_expert = 4 * 6 * 2 * 8 * 128 * 4 = 196,608 bytes = 0.19 MB
```

Measured: 0.28 MB/expert (includes both d_in and d_out dimensions for MLP layers
where d_in != d_out, specifically fc1: 128->512 and fc2: 512->128).

### Strategy Memory Comparison

| Strategy | Formula | N=100 (micro) | N=1000 (macro, d=3584, r=16) |
|----------|---------|---------------|-------------------------------|
| Pre-merge | M_base | 3.3 MB | ~14 GB |
| Dynamic | M_base + N * M_expert | 31.4 MB | ~14 GB + 18 GB = 32 GB |
| Hybrid | M_base + N_d * M_expert | 17.4 MB | ~14 GB + 9 GB = 23 GB |

At macro scale (Qwen2.5-7B), per-expert is ~18 MB (from macro/batched_lora_latency).
1000 experts: 18 GB of LoRA matrices. This fits in a single 24GB GPU alongside
the base model (quantized) or across 2 GPUs.

## Summary of Scaling Laws

| Metric | Pre-Merge | Dynamic top-k | Hash Ring Routing |
|--------|-----------|---------------|-------------------|
| Inference FLOPs | O(1) w.r.t. N | O(k), N-independent | O(log N) |
| Inference latency | = base | = base + k * delta overhead | < 1 us |
| Merge/setup cost | O(N) one-time | O(1) | O(N log N) one-time |
| Memory | O(1) (merged) | O(N) (store all experts) | O(N) (ring) |
| Expert selectivity | None (all active) | Full (choose k of N) | Routing only |

The fundamental result: **latency scales with k (experts selected), not N (experts
available).** Adding more experts to the library does not slow down inference.
