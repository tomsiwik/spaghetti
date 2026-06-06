# MLX Adapter Inference Speed: Mathematical Foundations

## Problem Statement

We measure the overhead of LoRA adapter composition on Apple Silicon's Metal GPU
via the MLX framework. The question: does adapter application on Metal GPU
preserve the O(k) scaling proven on CPU (PyTorch) and CUDA (RTX 4090)?

## Notation

| Symbol | Meaning | Micro value |
|--------|---------|-------------|
| d | hidden dimension | 128 |
| L | number of transformer layers | 4 |
| M | linear layers per block | 6 (Wq, Wk, Wv, Wo, fc1, fc2) |
| r | LoRA rank | 8 |
| N | total experts in library | {1, 2, 4, 8} |
| k | experts applied per forward | {1, 2, ..., min(k,N)} |
| T | sequence length | 32 |
| B | batch size | 1 |
| V | vocabulary size | 256 |

## FLOPs Analysis

### Base model forward pass

For a single transformer block with SDP attention:
- Attention projections (Wq, Wk, Wv, Wo): 4 * 2 * B*T * d * d = 8*B*T*d^2
- MLP (fc1: d->4d, fc2: 4d->d): 2 * 2 * B*T * d * 4d = 16*B*T*d^2
- Total per block: 24*B*T*d^2
- Total L blocks: 24*L*B*T*d^2

Micro: 24 * 4 * 1 * 32 * 128^2 = 50,331,648 FLOPs (~50M)

### LoRA overhead per expert per layer

One LoRA application: x -> B @ (A @ x), where A: (r, d_in), B: (d_out, r).
- A @ x: 2 * B*T * r * d_in
- B @ (A@x): 2 * B*T * d_out * r
- Total: 2*B*T*r*(d_in + d_out)

For the 6 linear layers in one block:
- 4 attention projections (d->d): 4 * 2*B*T*r*2d = 16*B*T*r*d
- MLP fc1 (d->4d): 2*B*T*r*5d = 10*B*T*r*d
- MLP fc2 (4d->d): 2*B*T*r*5d = 10*B*T*r*d
- Total per block: 36*B*T*r*d

Overhead ratio (single expert, single block):
  rho = 36*r*d / (24*d^2) = 36*r / (24*d) = 3r / (2d)

Micro: rho = 3*8 / (2*128) = 24/256 = 9.375%

### Overhead for k experts

If applied sequentially (each expert modifies weights then forward):
  Total overhead for pre-merge: 0 (merged offline)
  Total overhead for runtime: k * rho per block

  With k experts at micro scale: k * 9.375%

  k=1: 9.375%, k=2: 18.75%, k=4: 37.5%, k=8: 75%

### Pre-merge overhead

Pre-merge computes W' = W + (1/N) * sum_i(B_i @ A_i) offline.
Forward pass on W' has identical FLOPs to forward pass on W.
Expected overhead: 0% (within measurement noise).

## MLX-Specific Considerations

### Lazy Evaluation
MLX uses lazy evaluation with explicit synchronization via `mx.eval()`.
Operations are batched into a computation graph and dispatched to Metal in bulk.
This means:
1. LoRA matmul overhead may be absorbed into the graph (lower than PyTorch eager)
2. Timing must bracket `mx.eval()` calls to measure actual Metal execution
3. First invocation compiles Metal shaders; warmup is essential

### Unified Memory
CPU and GPU share the same memory on Apple Silicon. No PCIe transfer overhead.
Expert weight tensors are directly accessible from Metal without copies.
This should make dynamic expert loading cheaper than CUDA (no host->device transfer).

### Buffer Allocation
MLX allocates Metal buffers for intermediate results. With k experts, we need
k additional intermediate buffers per layer. Memory pressure scales O(k*L*r*T*B).

Micro: k * 4 * 8 * 32 * 1 * 4 bytes = k * 4096 bytes (negligible)

## Composition Strategies

### Strategy A: Pre-Merge
W'_l = W_l + (1/N) * sum_{i=1}^{N} B_l^i @ A_l^i

Merge cost: N * M * L * 2*r*(d_in+d_out) FLOPs (one-time)
Inference cost: identical to base model
Memory: identical to base model (experts discarded after merge)

### Strategy B: Runtime Apply (weight modification)
For each forward pass:
1. Save original weights
2. W_l += (1/k) * sum_{j in S} B_l^j @ A_l^j  (for selected set S of k experts)
3. Forward pass
4. Restore original weights

Overhead: k * rho per block (weight modification) + 2*M*L*d^2 (save/restore)

### Strategy C: Runtime Apply (activation path)
For each layer l, for input x:
  y = W_l @ x + (1/k) * sum_{j in S} B_l^j @ (A_l^j @ x)

This avoids weight modification but requires streaming through expert matrices.
Overhead: k * rho per block (same FLOPs, but different memory pattern)

## Expected Results

| Strategy | k | Theoretical overhead | Expected measured |
|----------|---|---------------------|-------------------|
| Pre-merge | -- | 0% | <5% (noise) |
| Runtime | 1 | 9.4% | 10-15% (+ dispatch) |
| Runtime | 2 | 18.8% | 20-30% |
| Runtime | 4 | 37.5% | 40-60% |
| Runtime | 8 | 75.0% | 80-120% |

The gap between theoretical and expected accounts for:
- Metal kernel launch overhead per matmul
- Memory allocation for intermediate LoRA products
- Python dispatch overhead (minimized by MLX lazy eval)

## Kill Criteria (Formal)

**K1: Single-adapter overhead <= 15%**
  Let t_base = mean forward latency with no adapters.
  Let t_1 = mean forward latency with k=1 adapter (runtime apply).
  overhead_1 = (t_1 - t_base) / t_base
  PASS if overhead_1 <= 0.15

**K2: N-adapter overhead is sub-linear or linear in N**
  Let overhead(N) = (t_N - t_base) / t_base for pre-merge with N adapters.
  Fit overhead(N) = alpha * N^beta.
  PASS if beta <= 1.0 (linear or sub-linear).

  For runtime apply at fixed k: overhead should be independent of N (library size).

## Worked Example (d=128, r=8, L=4, T=32)

Base FLOPs per block: 24 * 1 * 32 * 128^2 = 12,582,912
LoRA FLOPs per block (k=1): 36 * 1 * 32 * 8 * 128 = 1,179,648
Ratio: 1,179,648 / 12,582,912 = 9.375%

At k=2: 2 * 9.375% = 18.75%
At k=4: 4 * 9.375% = 37.5%

Pre-merge of N=8 experts (one-time cost):
  8 * 6 * 4 * 2 * 8 * (128 + 128) = 786,432 FLOPs per attn layer
  (plus MLP layers with different dims)
  Total: ~9.4M FLOPs = fraction of one forward pass
