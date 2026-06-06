# Memory Budget Analysis: Mathematical Foundations

## 1. Mechanism Definition

### Problem Statement

Given 48 GB unified memory on M5 Pro (~40 GB usable after OS/system), how many
ternary LoRA adapters can coexist alongside BitNet-2B-4T base model, router,
routing heads, and KV cache?

### Memory Components

Total memory budget:
```
M_total = M_base + M_adapters(N) + M_router + M_heads(N) + M_kv(seq) + M_activations
```

Where N = number of adapters, seq = sequence length.

#### (a) Base Model: M_base

BitNet-2B-4T stores ternary weights packed 4 per uint8 byte.

For a linear layer W in R^{out x in}:
- Packed storage: ceil(out/4) * in bytes (uint8)
- Dense bf16: out * in * 2 bytes

BitNet-2B-4T architecture (30 transformer layers, d=2560, d_ffn=6912):
- 210 BitLinear layers (7 per transformer layer: q, k, v, o, gate, up, down)
- Non-ternary parameters: token_embedding (151936 x 2560), final_norm, lm_head

Measured values (from prior experiment):
- Packed ternary weights: 521.0 MB
- Non-ternary params (bf16): 657.6 MB
- **M_base = 1,178.6 MB**

#### (b) Per-Adapter Memory: m_adapter

Each LoRA adapter has two matrices per target layer:
- A in R^{in x r}: projection down (from Grassmannian skeleton, shared per domain)
- B in R^{r x out}: trained adapter weights

At rank r=16, for the full model (30 layers x 7 targets):

**Format 1: bf16 (current runtime LoRA format)**
Per target: A = in * r * 2 bytes + B = r * out * 2 bytes
Total per adapter (A+B): ~43.3 MB

Note: A matrices come from the Grassmannian skeleton. If N adapters share the
same set of A matrices (one A per layer-target, not per domain), then:
- A skeleton (shared): ~21.4 MB (constant, independent of N)
- B per adapter: ~21.9 MB

m_adapter_bf16 = 21.9 MB (B only, A shared from skeleton)

**Format 2: Ternary-packed B (projected)**
B values are {-1, 0, +1}, packed 4 per byte:
Per target B: r * ceil(out/4) bytes
Total per adapter B: sum over 210 targets of r * ceil(out_i / 4) ~ 1.38 MB

But runtime LoRA computation requires bf16 unpacking for matrix multiply.
Storage is 1.38 MB, but serving requires unpacking to bf16.

**Format 3: int8 B (validated)**
Per target B: r * out bytes
Total per adapter B: ~10.9 MB
Reconstruction error: 3.09e-04 (negligible for PPL)

#### (c) Grassmannian Skeleton: M_skeleton

The A matrices are shared across all adapters (one A per layer-target position).
For each of the 210 targets, A in R^{in_i x r}:
- bf16: sum(in_i * r * 2) ~ 21.4 MB
- This is paid ONCE regardless of N adapters.

However, if each adapter needs a DIFFERENT A matrix (true Grassmannian: each
adapter gets its own orthogonal A), then:
- M_skeleton(N) = N * 21.4 MB

From the architecture: Grassmannian AP generates N orthonormal A matrices.
Each adapter i has its own A_i. So the skeleton cost scales with N:
- M_skeleton = N * 21.4 MB (bf16)

Wait -- this is critical. Let me verify. The Grassmannian skeleton pre-computes
N orthonormal A matrices so that A_i^T A_j approx 0. Each adapter needs its own A_i.

So the true per-adapter cost in bf16 is:
**m_adapter_bf16 = 21.9 MB (B) + 21.4 MB (A) = 43.3 MB**

#### (d) Router: M_router

Shared router (per-token Gumbel-sigmoid):
- 659K parameters, bf16: 659,000 * 2 = 1.32 MB
- M_router = 1.32 MB

#### (e) Per-Adapter Routing Heads: m_head

Each adapter has a tiny binary routing head:
- 2-layer MLP: Linear(2560, 16) + ReLU + Linear(16, 1)
- Params: 2560*16 + 16 + 16*1 + 1 = 41,009 params
- bf16: 41,009 * 2 = 82.0 KB = 0.082 MB

m_head = 0.082 MB per adapter

#### (f) KV Cache: M_kv(seq)

BitNet-2B-4T: 30 layers, 8 KV heads, d_head = 80:
M_kv = 2 * 30 * 8 * 80 * seq * 2 bytes (K and V, bf16)
     = 76,800 * seq bytes

| seq_len | M_kv |
|---------|------|
| 256     | 19.7 MB |
| 2048    | 157.3 MB |
| 8192    | 629.1 MB |

#### (g) Activations: M_act

During forward pass (batch=1, seq=256):
- Hidden states: 256 * 2560 * 2 = 1.3 MB
- Intermediate FFN: 256 * 6912 * 2 = 3.5 MB
- Peak during one layer: ~10 MB
- Not accumulated across layers (inference, no grad)

M_act ~ 10 MB (constant)

### Total Memory Formula

```
M_total(N, seq) = M_base + N * m_adapter + M_router + N * m_head + M_kv(seq) + M_act
```

Where m_adapter depends on storage format:

| Format | m_adapter | m_head | Notes |
|--------|-----------|--------|-------|
| bf16 A+B | 43.3 MB | 0.082 MB | Full precision runtime LoRA |
| bf16 B only (shared A) | 21.9 MB | 0.082 MB | Only if A is truly shared |
| int8 B + bf16 A | 32.3 MB | 0.082 MB | int8 B validated |
| ternary-packed B + bf16 A | 22.8 MB | 0.082 MB | Requires unpack for compute |

### Theoretical Maximum N

With M_available = 40,000 MB (40 GB usable):

```
N_max = (M_available - M_base - M_router - M_kv - M_act) / (m_adapter + m_head)
```

| Format | N_max (seq=256) | N_max (seq=2048) | N_max (seq=8192) |
|--------|----------------|-----------------|-----------------|
| bf16 A+B | 895 | 891 | 880 |
| int8 B + bf16 A | 1,197 | 1,193 | 1,179 |
| bf16 B (shared A) | 1,763 | 1,757 | 1,736 |

All scenarios: N_max >> 500. Theoretical answer is clear.

### Measured Maximum N

Empirical measurement shows 4.4% overhead from MLX Metal allocator:
- Per adapter measured: 45.2 MB (vs 43.3 MB theoretical)
- Per head measured: 82 KB (matches theoretical)

| Seq Length | N_max (measured) | N_max (theoretical) | Overhead |
|------------|-----------------|--------------------|---------|
| 256 | 853 | 895 | 4.7% |
| 2048 | 850 | 891 | 4.6% |
| 8192 | 840 | 880 | 4.5% |

The overhead is consistent and small (4.4-4.7%), coming from:
- Metal buffer page alignment (4096-byte boundaries)
- Per-allocation metadata (~8-16 bytes per tensor)
- MLX graph tracking overhead

## 2. Why It Works

### Ternary Compression Advantage

BitNet-2B-4T base is 1.18 GB (vs 4.8 GB at bf16) because ternary weights pack
8x denser. This leaves 38.8 GB for adapters.

At 43.3 MB per adapter (worst case bf16): 38,800/43.3 = 896 adapters.
At 22 MB per adapter (bf16 B, shared A): 38,800/22 = 1,763 adapters.

The key insight: the base model is so small (1.18 GB vs typical 4-7 GB for 2B
models) that adapter storage dominates the budget, and each adapter is tiny.

### On-Demand Loading vs All-In-Memory

For production serving with per-token routing:
- Router selects top-k adapters (typically k=2)
- Only k adapters need to be in memory for the forward pass
- But routing heads for ALL N adapters must be loaded to make selection decisions
- Routing head memory: N * 0.082 MB (negligible: 500 * 0.082 = 41 MB)

## 3. What Breaks It

### KV Cache at Long Contexts

At seq=8192: KV cache = 629 MB, reducing adapter budget by ~15 adapters (bf16).
At seq=32768 (if supported): KV cache = 2,516 MB, reducing by ~58 adapters.
Not a practical concern for N >> 100.

### Memory Fragmentation

MLX's Metal allocator may fragment unified memory. Theoretical N_max assumes
perfect packing. Practical N_max will be lower due to:
- Allocator overhead (~8 bytes per allocation)
- Page alignment (4096 bytes minimum)
- MLX cache residuals

### Peak Memory During Forward Pass

When computing runtime LoRA for k active adapters:
- Each adapter forward: x @ A_i (batch x seq x r) + result @ B_i (batch x seq x out)
- Intermediate tensors: batch * seq * r * 2 bytes per adapter = negligible
- Peak is dominated by stored adapter weights, not computation intermediates

## 4. Assumptions

1. **Unified memory is 48 GB.** The M5 Pro unified memory pool is shared between
   CPU, GPU, and Neural Engine. We assume 40 GB usable (8 GB for OS/system).
   Justification: CODING_GUIDELINES.md uses the same 8 GB reservation.

2. **Each adapter has its own A matrix.** The Grassmannian skeleton generates N
   distinct orthonormal A matrices. Cost scales linearly with N.
   If violated: could share A matrices, reducing per-adapter cost by ~50%.

3. **Runtime LoRA requires bf16 A and B in memory.** The Metal kernel cannot
   directly multiply ternary-packed B with inputs. Ternary B must be unpacked
   to bf16 for the addmm operation. Storage on disk can be ternary, but
   in-memory serving requires bf16.
   If violated: a custom Metal kernel for ternary matmul could reduce B memory 16x.

4. **Adapters are loaded all at once.** We measure simultaneous memory. In practice,
   on-demand loading from NVMe SSD (~7 GB/s on M5 Pro) can load a 43 MB adapter
   in ~6 ms, making lazy loading practical.

## 5. Complexity Analysis

| Operation | Memory | Time |
|-----------|--------|------|
| Load N adapters | N * m_adapter | N * ~6ms (from SSD) |
| Route (all N heads) | N * 0.082 MB | O(N * d) = negligible |
| Forward (k active) | k * m_adapter + M_base | O(k * d * r * seq * L) |
| Swap adapter | 0 (pointer change) | O(1) |

## 6. Worked Example

At micro scale (d=64, r=4, L=4, 7 targets per layer):

Per adapter:
- A: 4 * 7 * 64 * 4 * 2 = 14,336 bytes = 14 KB
- B: 4 * 7 * 4 * 64 * 2 = 14,336 bytes = 14 KB
- Total: 28 KB

With 1 GB budget for adapters: 1,000,000 / 28 = 35,714 adapters at micro scale.

At production scale (d=2560, r=16, L=30):
- Per adapter: 43.3 MB
- With 38.8 GB budget: 896 adapters

## 7. Connection to Architecture

This analysis validates the SOLE vision of massive adapter pools on consumer
hardware. The BitNet-2B-4T base is small enough (1.18 GB) that 48 GB unified
memory can hold hundreds to thousands of adapters simultaneously.

The practical deployment modes:

| Mode | N loaded | Memory | Use case |
|------|----------|--------|----------|
| Minimal | 5-10 | 1.4-1.6 GB | Single-user, few domains |
| Standard | 50-100 | 3.3-5.5 GB | Multi-domain expert system |
| Maximum | 500-900 | 23-40 GB | Full adapter pool, all domains |

The bottleneck shifts from "how many fit" to "how fast can we route" at N > 100.
Per-adapter routing heads (82K params) are the right design: they scale linearly
and total memory for 500 heads is only 41 MB.
