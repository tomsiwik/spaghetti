# Learnings: Sparse-BitNet Pruning

## Critical Discoveries

### 1. Natural sparsity is 42.21% in BitNet-2B-4T (confirmed)
Sparse-BitNet paper's claim (arxiv 2603.05168) holds. Mean zero fraction across
all 2.08B ternary weights is 42.21%. Range: 36.1% (v_proj) to 60.1% (layer 1 MLP).
Attention Q projections are sparser (42-51%) than V projections (36-44%).

### 2. Packed uint8 kernel cannot be beaten by sparse on Apple Silicon
The fused Metal BitLinear kernel reads packed uint8 (2 bits/weight) and computes
in one pass. Sparse approaches must either:
- Expand to bf16 (8x memory): kills bandwidth advantage at model scale
- Store index structures: 12-16x more memory than packed
When inference is bandwidth-bound (74.2% of 273 GB/s), bigger = slower.

### 3. Metal SIMD handles zeros for free
The kernel inner loop `sum += v[j] * ((w & 3) - 1)` computes zero as
`v * 0 = 0` with identical cost to `v * 1`. No branch, no predication, no penalty.
Sparsity is invisible to the GPU's SIMD pipeline. This is fundamentally different
from CPU/sparse-tensor-core hardware where skipping zeros saves real work.

### 4. bf16 matmul is faster per-layer for small matrices (1.3x on 2560x2560)
But this is deceptive: the custom ternary kernel uses float32 accumulation with
non-standard 4-row strided access. Metal's optimized bf16 gemm beats it per-FLOP.
However at model scale (1.18 GB packed vs ~9.4 GB bf16), the 8x data expansion
drops throughput from 172 to ~29 tok/s. Per-layer speed is irrelevant when
bandwidth is the bottleneck.

### 5. Weight encoding bug: (w & 3) - 1, NOT lookup table
BitNet-2B-4T encodes ternary values as: 0 -> -1, 1 -> 0, 2 -> +1.
The Metal kernel simply subtracts 1 from the 2-bit value. Initial incorrect
decoding (0->0, 1->+1, 2->-1) gave wrong sparsity (28.9%) and large output
errors (max diff 258). Always verify encoding against the actual kernel source.

### 6. Sparsity distribution is non-uniform across layers
Layer 1 MLP has ~60% zeros (highest). Layer 0 MLP has ~38% (lowest among MLP).
Attention Q projections are consistently sparser than V projections. This
correlates with layer function: embedding-adjacent layers and less-critical
projections have more zeros.

## Design Implications for SOLE

- **Do not pursue sparse ternary optimization on Apple Silicon.** Dead end.
- **Packed uint8 is the optimal serving format.** Confirmed again after
  exp_inference_speed_10x.
- **Sparse acceleration requires hardware support** (sparse tensor cores,
  Neural Engine structured sparsity). Future M-series chips may enable this
  but current hardware cannot benefit.
- **The 42% sparsity has value for pruning/compression research** even though
  it cannot be exploited for speed. It confirms that ternary models have
  substantial structural sparsity that could inform adapter design.
