# Memory-Optimized Serving: Mathematical Foundations

## 1. Mechanism Definition

### Ternary Packing (BitLinear)

BitNet-2B-4T stores weights as ternary values {-1, 0, +1} packed 4 per uint8 byte.

For a linear layer with dimensions (out_features, in_features):
- **Packed storage**: `ceil(out_features/4) * in_features` bytes (uint8)
- **Dense bf16 equivalent**: `out_features * in_features * 2` bytes

Packing ratio: **8x** compression vs bf16.

Extraction from packed byte `b`:
```
w0 = (b & 0x03) - 1       # bits [0:1]
w1 = ((b >> 2) & 0x03) - 1  # bits [2:3]
w2 = ((b >> 4) & 0x03) - 1  # bits [4:5]
w3 = ((b >> 6) & 0x03) - 1  # bits [6:7]
```

The Metal kernel in `mlx_lm` reads packed uint8 directly, computes the matrix-vector product in-kernel. No intermediate bf16 tensor is materialized.

### Runtime LoRA on BitLinear

For a BitLinear layer with output y = BitLinear(x), we add:

```
y_adapted = BitLinear(x) + (x @ A) @ B * scale
```

Where:
- `x in R^{batch x in_features}` (bf16)
- `A in R^{in_features x r}` (Grassmannian skeleton, bf16)
- `B in R^{r x out_features}` (trained adapter, bf16)
- `scale = 20.0` (LoRA scaling factor)

FLOPs per layer: `2 * batch * in_features * r + 2 * batch * r * out_features`
At r=16, d=2560: `2 * 1 * 2560 * 16 + 2 * 1 * 16 * 2560 = 163,840` FLOPs
vs base BitLinear: `2 * 1 * 2560 * 2560 = 13,107,200` FLOPs
LoRA overhead: **1.25%** of base computation.

### Memory Budget

For the complete serving pipeline:

| Component | Formula | Size |
|-----------|---------|------|
| Packed ternary weights | `sum(ceil(out/4) * in)` for 210 layers | 521.0 MB |
| Non-ternary params (embed, norm, LM head) | bf16 tensors | 657.6 MB |
| **Base model total** | packed + bf16 | **1,178.6 MB** |
| 1 adapter B matrices (bf16) | `30 layers * 7 targets * r * d_avg * 2B` | 21.9 MB |
| 1 domain A matrices (bf16) | `30 * 7 * d_avg * r * 2B` | 21.4 MB |
| KV cache (seq=256, 1 batch) | negligible at short seq | ~3.5 MB |
| Activations (inference) | `batch * seq * d * 2B` | ~1.3 MB |
| **Total with 1 adapter** | | **~1,224 MB** |

## 2. Why It Works

### Key Insight: BitLinear Metal Kernel Avoids bf16 Unpack

The previous 10.98 GB measurement came from **unpacking all ternary weights to bf16** before inference. This 8x bloat:
- Packed: 521 MB
- Unpacked bf16: 4,827 MB (measured)

The BitLinear Metal kernel reads uint8 packed weights directly and computes `y = W_ternary @ x` without materializing the dense matrix. This is the core mechanism that enables sub-2 GB serving.

### LoRA as Additive Correction

Runtime LoRA adds `(x @ A) @ B * scale` to the BitLinear output. Since this is purely additive:
1. No modification to base weights needed
2. Adapters can be swapped by changing A, B pointers (zero-copy)
3. Multiple adapters compose linearly: `sum_i scale_i * (x @ A_i) @ B_i`

### Adapter Quantization Safety

Int8 quantization of B matrices introduces negligible error:
- Per-tensor symmetric quantization: `B_q = round(B / scale)`, `scale = max(|B|) / 127`
- Mean reconstruction error: 8.2e-05
- Worst-case max error: 3.09e-04
- At 4x compression (43.7 MB -> 10.9 MB), quality impact is unmeasurable

## 3. What Breaks It

### Scaling to Many Concurrent Adapters

With N adapters loaded simultaneously:
- Memory: `1,178.6 + N * (21.9 + 21.4)` MB (bf16 B + A matrices)
- At N=5: 1,178.6 + 216.5 = 1,395 MB (still under 2 GB)
- At N=25: 1,178.6 + 1,082.5 = 2,261 MB (still under 3 GB)
- At N=50: 1,178.6 + 2,165 = 3,344 MB (exceeds 3 GB target)

With int8 B matrices: N=50 would be 1,178.6 + 50*(10.9+21.4) = 2,793 MB (under 3 GB).

### KV Cache at Long Sequences

KV cache scales as `O(batch * seq_len * n_layers * n_kv_heads * d_head * 2)`.
For BitNet-2B-4T: 30 layers, 8 KV heads, d_head=80, bf16:
- seq=256: 30 * 8 * 80 * 256 * 2 * 2 = 19.7 MB
- seq=2048: 157.3 MB
- seq=8192: 629.1 MB

At seq=8192, total would be ~1,854 MB (still under 2 GB, but KV cache becomes significant).

### Pre-merge Approach Costs 4x Memory

Unpacking BitLinear to bf16 nn.Linear for pre-merge costs 4,827 MB (measured).
This is the **wrong** strategy for memory optimization. Pre-merge requires the dense unpack because you need to add the LoRA delta to the weight matrix. Runtime LoRA avoids this entirely.

## 4. Assumptions

1. **BitLinear Metal kernel performance is adequate.** Measured at 82 tok/s, which is 32% of theoretical bandwidth limit (255 tok/s). Adequate for interactive serving.

2. **Adapter quality is preserved through bf16 cast.** Adapters trained in fp32, served in bf16. PPL difference between pre-merge (3.74) and runtime LoRA (3.75) is 0.3% -- negligible.

3. **Grassmannian A matrices are required.** These add 21.4 MB per domain. If A matrices could be computed on-the-fly (random seed), this drops to near-zero. Current approach stores them.

## 5. Complexity Analysis

| Operation | FLOPs | Memory | Notes |
|-----------|-------|--------|-------|
| Base forward (BitLinear) | O(d^2 * seq * L) | 1,178.6 MB (constant) | Metal kernel, packed weights |
| Runtime LoRA (1 adapter) | O(d * r * seq * L) | +43.3 MB | 1.25% FLOPs overhead |
| Runtime LoRA (k adapters) | O(k * d * r * seq * L) | +43.3k MB | Linear in k |
| Adapter swap | O(1) | 0 | Pointer change |
| KV cache | 0 (pre-computed) | O(seq * L * h_kv * d_h) | Grows with sequence |

## 6. Worked Example

At micro scale (d=64, r=4, L=4 layers, 7 targets per layer):

**Packed base weights:**
- Per layer: 7 * ceil(64/4) * 64 = 7 * 16 * 64 = 7,168 bytes
- All layers: 4 * 7,168 = 28,672 bytes = 28 KB

**One adapter (bf16):**
- B matrices: 4 * 7 * 4 * 64 * 2 = 14,336 bytes = 14 KB
- A matrices: 4 * 7 * 64 * 4 * 2 = 14,336 bytes = 14 KB
- Total: 28 KB

**Total serving memory:** 28 KB + 28 KB = 56 KB (at toy scale).
Ratio: adapter/base = 28/28 = 1.0 (adapters same size as base at r=d/16).

At production scale (d=2560, r=16): adapter/base = 43.3/1178.6 = 3.7% of base.

## 7. Connection to Architecture

### Pre-merge vs Runtime LoRA Decision

This experiment proves the memory-optimal strategy depends on context:

| Scenario | Strategy | Memory | Overhead |
|----------|----------|--------|----------|
| Always-on adapters (instruction) | Pre-merge at session start | 4.8 GB (bf16) | 0% per-token |
| Dynamic routing (per-query) | Runtime LoRA on BitLinear | 1.2 GB | 1.25% per-token |
| Memory-constrained deployment | Runtime LoRA mandatory | 1.2 GB | 1.25% per-token |

For the SOLE architecture on Apple Silicon:
1. Load BitLinear base (1.18 GB)
2. Load top-k adapter B matrices on demand (~22 MB each in bf16, ~11 MB in int8)
3. Load corresponding A matrices from skeleton (~21 MB each)
4. Apply as runtime LoRA wrappers
5. Swap adapters between queries by replacing B and A pointers

Total at k=3: 1,178.6 + 3*(21.9 + 21.4) = 1,309 MB = **1.31 GB**.
Total at k=5 (all domains): 1,178.6 + 5*(21.9 + 21.4) = 1,395 MB = **1.40 GB**.

This is **3.4x smaller than Qwen2.5-3B** (4.7 GB at bf16) while providing domain-specialized quality.

### References
- S-LoRA (2311.03285): concurrent serving of thousands of LoRA adapters
- CLA (2405.12981): cross-layer attention for KV cache reduction (not needed at current memory budget)
- bitnet.cpp: CPU inference for ternary models, 45 tok/s on M2
