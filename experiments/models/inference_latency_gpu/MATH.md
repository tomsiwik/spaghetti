# GPU Latency Validation: Mathematical Analysis

## Notation

| Symbol | Definition | Value (Qwen2.5-7B on RTX 4090) |
|--------|-----------|-------------------------------|
| d | Hidden dimension | 3584 |
| d_ff | FFN intermediate dimension | 18944 |
| L | Number of transformer layers | 28 |
| T | LoRA target modules per layer | 7 (q,k,v,o,up,gate,down) |
| r | LoRA rank | 16 |
| N | Total experts in the library | 5, 10, 20, 50 |
| k | Experts selected per query | 1, 2 |
| s | Sequence length (tokens) | 64, 256 |
| b | Batch size | 1 |

## Pre-Merge: Theoretical Zero Overhead

### Why Pre-Merge Costs Exactly Zero at Inference

After merging N experts into the base weights:

```
W' = W + (1/N) * sum_{i=1}^{N} B_i @ A_i
```

The resulting `W'` has **identical shape** to `W`. The forward pass uses standard
`Y = X @ W'^T`, which is the same GEMM kernel as the base model. There is no
additional computation, no branching, no memory indirection.

Therefore:
```
F_premerge = F_base        (exactly, not approximately)
Latency_premerge = Latency_base    (within measurement noise)
```

### Merge Cost (One-Time, Offline)

Per linear layer, the merge requires N rank-r outer products:
```
C_merge_per_layer = N * (2 * d_out * r * d_in)    FLOPs
```

For Qwen2.5-7B (7 target modules per layer, 28 layers):
- Attention (q,k,v,o): d_in = d_out = 3584
- MLP (up,gate): d_in = 3584, d_out = 18944
- MLP (down): d_in = 18944, d_out = 3584

Per layer, per expert:
```
C_attn = 4 * 2 * 3584 * 16 * 3584 = 1,310,720,000 FLOPs
C_mlp  = 3 * 2 * 18944 * 16 * 3584 = 6,533,480,000 FLOPs  (approx, mixed dims)
C_per_expert_per_layer = C_attn + C_mlp ~ 7.84B FLOPs
```

Total merge for N=50:
```
C_merge_total = 28 * 50 * 7.84B = 10.98T FLOPs
```

On RTX 4090 (82.6 TFLOPS FP16): ~133ms theoretical, ~1-5s practical
(memory-bound, not compute-bound).

This is a **one-time cost** amortized over all subsequent queries.

### Why Overhead Should Be Within Noise

The GPU kernel dispatch for `W'` is identical to `W`:
- Same tensor shapes
- Same CUDA kernel (cuBLAS GEMM)
- Same memory access pattern
- Same compute intensity

Any measured overhead (positive or negative) is measurement noise from:
1. GPU clock boosting variation
2. Memory cache state differences
3. OS scheduling jitter

**Kill threshold: 5%** -- anything above this indicates an unexpected
architectural issue (e.g., denormalized values from averaging many small deltas,
cache pollution from the merge process affecting subsequent kernels).

## Dynamic Top-k: Theoretical Overhead

### PEFT Adapter Forward Pass

When using PEFT's LoRA adapter (not merged), each linear layer computes:
```
Y = X @ W^T + (alpha/r) * X @ A^T @ B^T
```

This adds two extra matmuls per LoRA target per layer:
```
F_lora_per_target = 2 * b * s * r * d    (A: d_in -> r, B: r -> d_out)
```

### Expected GPU Overhead (PEFT, Not Fused)

For k=1, seq_len=256, Qwen2.5-7B:
```
F_lora_total = k * L * T * 2 * b * s * r * d_avg
             = 1 * 28 * 7 * 2 * 1 * 256 * 16 * ~8000
             = 12.9B FLOPs
```

F_base (7B model, ~14B FLOPs per token * 256 tokens):
```
F_base ~ 3,584B FLOPs for 256 tokens
```

Theoretical overhead: 12.9B / 3584B = **0.36%**

However, PEFT implementation overhead includes:
- Python dispatch per adapter module (not fused into base GEMM)
- Separate small GEMMs (r=16 is tiny, GPU underutilized)
- Memory overhead for intermediate activations

Expected measured overhead: **5-30%** (implementation-bound, not compute-bound).

### N-Independence Argument

The key theoretical prediction: dynamic overhead depends on k, not N.

At inference time, only k adapters participate in the forward pass.
The other N-k adapters are stored in GPU memory but never touched.
Therefore:

```
F_dynamic(N, k) = F_base + k * F_lora_single
```

N appears nowhere in the forward pass computation. N affects only:
1. GPU memory consumption: O(N * r * d * L * T) bytes
2. Routing decision: O(log N) for hash ring (negligible)
3. **Potential cache effect**: Large N means more adapter tensors in GPU memory,
   potentially competing with base model weights for cache space

Kill criterion: If the slope of overhead vs N at fixed k exceeds 0.1%/expert,
there is an unexpected N-dependent cost (likely cache pollution).

## Memory Analysis

### Per-Expert Storage (float16)

```
M_expert = L * T * 2 * r * d_avg * 2 bytes
         = 28 * 7 * 2 * 16 * ~8000 * 2
         = 200,704,000 bytes
         ~ 191 MB (float16)
```

Wait -- this includes both attention and MLP with different dimensions. More precisely:

Per layer (float16):
- q_proj: A(16,3584) + B(3584,16) = 2 * 16 * 3584 * 2 = 229,376 bytes
- k_proj: same = 229,376
- v_proj: same = 229,376
- o_proj: same = 229,376
- gate_proj: A(16,3584) + B(18944,16) = (16*3584 + 18944*16) * 2 = 1,209,344
- up_proj: same = 1,209,344
- down_proj: A(16,18944) + B(3584,16) = same = 1,209,344

Per layer total: 4 * 229,376 + 3 * 1,209,344 = 4,545,536 bytes = 4.33 MB
Per expert (28 layers): 28 * 4.33 MB = **121.3 MB**

At N=50: 50 * 121 MB = **6.1 GB** for all adapter weights
Base model (FP16): ~14 GB
Total: ~20 GB (fits in 24GB 4090)

At N=50 with 4-bit base: ~3.5 GB base + 6.1 GB adapters = 9.6 GB (comfortable)

## Summary of Predictions

| Metric | N=5 | N=10 | N=20 | N=50 |
|--------|-----|------|------|------|
| Pre-merge overhead | ~0% | ~0% | ~0% | ~0% |
| Pre-merge merge time | ~0.1s | ~0.2s | ~0.5s | ~1-5s |
| Dynamic k=1 overhead | ~10-30% | ~10-30% | ~10-30% | ~10-30% |
| Dynamic k=2 overhead | ~20-60% | ~20-60% | ~20-60% | ~20-60% |
| Memory (adapters only) | 0.6 GB | 1.2 GB | 2.4 GB | 6.1 GB |

The critical prediction: **overhead is constant across N** for both strategies.
Pre-merge is constant by construction. Dynamic is constant because only k adapters
participate regardless of N.
