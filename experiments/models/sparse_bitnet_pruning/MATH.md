# Sparse-BitNet: Exploiting Natural Ternary Sparsity

## 1. Mechanism Definition

### Ternary Weight Packing

BitNet-2B-4T stores weights W in {-1, 0, +1} packed as uint8: 4 ternary values
per byte using 2-bit encoding. For a layer with logical shape (out_features, in_features):

- Packed shape: (out_features / 4, in_features) in uint8
- Storage: out_features * in_features * 2 / 8 = out_features * in_features / 4 bytes
- Unpacking: each byte b -> 4 values via ((b >> 0) & 3) - 1, ((b >> 2) & 3) - 1, etc.
  Encoding: {0: -1, 1: 0, 2: +1} (confirmed from Metal kernel source: `(w & 3) - 1`)
- Row layout: packed row `row_idx` maps to output rows at stride out_features/4:
  `out[row_idx + i * (out_features/4)]` for i in {0, 1, 2, 3}

### Dense Ternary MatMul

For input x in R^{B x d_in} and ternary W in {-1, 0, +1}^{d_out x d_in}:

y = x @ W^T

This is computed by the fused Metal kernel `_bitlinear_kernel` which:
1. Reads packed uint8 weights (bandwidth: d_out * d_in / 4 bytes)
2. Unpacks to ternary values in registers
3. Accumulates: y[b,j] = sum_i x[b,i] * W[j,i] where W[j,i] in {-1, 0, +1}
4. Multiplies by weight_scale (per-tensor scalar)

The multiply by W[j,i] reduces to conditional add/subtract/skip since W is ternary.

### Sparse Ternary MatMul (Proposed)

If fraction f of weights are zero, then for each output j:
- nnz_j = d_in * (1 - f) non-zero entries
- y[b,j] = sum_{i: W[j,i]=+1} x[b,i] - sum_{i: W[j,i]=-1} x[b,i]

This skips f * d_in entries per output. With f ~ 0.42 (Sparse-BitNet claim):
- Dense: d_out * d_in multiply-adds
- Sparse: d_out * d_in * (1-f) = 0.58 * d_out * d_in add/subtracts
- Theoretical speedup: 1 / (1 - f) = 1.72x

### Index-Based Sparse Format

For each row j of W, store:
- plus_indices[j]: indices where W[j,i] = +1
- minus_indices[j]: indices where W[j,i] = -1
- y[b,j] = x[b, plus_indices[j]].sum() - x[b, minus_indices[j]].sum()

Storage overhead for indices: d_out * d_in * (1-f) * sizeof(int32) = d_out * d_in * 0.58 * 4
vs packed: d_out * d_in / 4

Ratio: 0.58 * 4 / 0.25 = 9.28x MORE memory for the sparse format!

### Alternative: Masked Dense

y = x @ (W_unpacked * mask) where mask = (W_unpacked != 0)

This requires unpacking W to bf16 first:
- Unpacked: d_out * d_in * 2 bytes (bf16)
- Packed: d_out * d_in / 4 bytes
- Memory increase: 8x

## 2. Why It Should Work (Theory)

The Sparse-BitNet paper (arxiv 2603.05168) observes that pre-trained ternary models
like BitNet b1.58 have ~42% zero weights naturally. This is NOT pruning -- the zeros
emerge from training with STE (straight-through estimator).

For bandwidth-bound inference (our case: 74.2% of 273 GB/s), skipping computation
on zeros could reduce the effective weight transfer if a sparse format is more compact
than packed uint8. However:

**Critical analysis:** The packed uint8 format already achieves 2 bits per weight.
A sparse format must encode POSITIONS of non-zero values, which costs at least
log2(d_in) bits per non-zero entry. For d_in = 2560, that's ~12 bits per entry.
At 58% non-zero rate: 0.58 * 12 = 6.96 bits per logical weight vs 2 bits packed.

**The sparse format is 3.5x LESS compact than packed ternary.**

## 3. What Breaks It

### Bandwidth Analysis (the real bottleneck)

At 172 tok/s single-token generation on M5 Pro (273 GB/s bandwidth):
- Model size: 1.18 GB packed
- Per-token weight transfer: 1.18 GB (must stream all weights for each token)
- Bandwidth required: 1.18 GB * 172 = 203 GB/s (74.2% utilization)

Any sparse format that increases model size will DECREASE tok/s because we're
bandwidth-bound. The fused Metal kernel already handles zeros efficiently inside
the SIMD pipeline -- a zero ternary value just means "add 0" which costs the
same as "add 1" on the GPU ALU.

### GPU Architecture Mismatch

Metal's GPU uses SIMD groups of 32 threads executing in lockstep. Sparse access
patterns cause:
1. **Divergent memory access:** Irregular indices prevent coalesced reads
2. **Thread divergence:** Different threads process different numbers of non-zeros
3. **Occupancy loss:** Irregular workload prevents optimal warp scheduling

The fused packed kernel reads sequential uint8 values -> maximum memory coalescing.
Sparse gather from irregular positions loses this property.

### Diminishing Returns for Ternary

In FP16 matmul, skipping a multiply-add saves significant ALU time. In ternary
matmul, the "multiply" is already free (conditional add/subtract). Sparsity only
saves the add/subtract operation, which is the cheapest possible ALU op.

## 4. Kill Criteria Mapping

- **K1 (PPL degradation > 5%):** If sparse implementation is mathematically exact
  (skip only zeros), K1 passes trivially. However, any approximation (e.g., block
  sparsity that zeroes out near-zero blocks) could fail.

- **K2 (no wall-clock speedup):** Most likely failure mode. The packed uint8 kernel
  is already bandwidth-optimized. Sparse formats increase memory footprint.
  The GPU's SIMD architecture penalizes irregular access patterns.
  Prediction: K2 FAIL.

- **K3 (natural zero fraction < 30%):** Must verify empirically by unpacking
  weights. If Sparse-BitNet's 42% holds for BitNet-2B-4T, K3 passes.

## 5. Complexity Analysis

| Format | Storage (per layer) | Bandwidth | ALU ops |
|--------|-------------------|-----------|---------|
| Packed uint8 | d_out * d_in / 4 | Optimal (sequential) | d_out * d_in (add/sub/skip) |
| Sparse indices | d_out * nnz * 4 | Poor (random gather) | d_out * nnz (add/sub only) |
| Unpacked bf16 mask | d_out * d_in * 2 | 8x packed | d_out * d_in (with branch) |

For q_proj (2560 x 2560): packed = 1.64 MB, sparse indices = 15.2 MB, unpacked = 13.1 MB.

## 6. Worked Example (q_proj, layer 0)

- d_out = 2560, d_in = 2560
- Packed: (640, 2560) uint8 = 1,638,400 bytes = 1.56 MB
- Assume f = 0.42: nnz = 2560 * 2560 * 0.58 = 3,801,088
- Sparse CSR: row_ptr (2561 * 4B) + col_idx (3.8M * 4B) + values (3.8M * 1B)
  = 10,240 + 15,204,352 + 3,801,088 = 19,015,680 bytes = 18.1 MB
- Ratio: sparse/packed = 11.6x MORE memory

## 7. Connection to Architecture

This experiment tests whether the natural sparsity in BitNet-2B-4T (our production
base model) can be exploited for inference speedup. The key connection:

- **Serving (Track D):** Currently at 172 tok/s base, 97 tok/s with adapter.
  Any speedup here directly improves the user experience.
- **The result is almost certainly negative** because the packed uint8 Metal kernel
  is already near bandwidth-optimal (74.2% utilization), and sparse formats
  increase rather than decrease the data transfer requirement.
- **Value of the experiment:** Definitively closing this avenue prevents wasted
  effort on sparse ternary optimization and confirms that the packed format is
  the correct serving strategy.

## References

- Sparse-BitNet (arxiv 2603.05168): natural 42% sparsity in ternary weights
- exp_inference_speed_10x LEARNINGS: 172 tok/s, 74.2% bandwidth utilization
- exp_memory_budget_analysis LEARNINGS: 1.18 GB base model
