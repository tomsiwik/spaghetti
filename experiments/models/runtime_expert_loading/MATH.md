# Runtime Expert Loading: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 896 (Qwen 0.5B) |
| d_ff | MLP intermediate dimension | 4864 (Qwen 0.5B) |
| r | LoRA rank | 16 (production) |
| L | Number of transformer layers | 24 (Qwen 0.5B) |
| N | Number of active experts in the composition | {1, 5, 10, 20, 50} |
| W_l | Base weight matrix at layer l, projection p | varies per projection |
| A_i, B_i | LoRA factors for expert i | A: (r, d_in), B: (d_out, r) |
| M_l | Pre-merged weight at layer l: W_l + sum(B_i A_i) | same as W_l |

## 2. Expert Composition Strategies

### 2.1 Strategy A: Full Recompute

When swapping expert j_old for j_new, recompute all merged weights from scratch:

    M_l = W_l + sum_{i in S} B_i^l A_i^l

where S is the new active set. This touches every projection in every layer.

**FLOPs per projection:** For W of shape (d_out, d_in):
- B_i A_i costs 2 * d_out * r * d_in FLOPs per expert
- Summing N experts: N * 2 * d_out * r * d_in
- Total per projection: O(N * d_out * r * d_in)

**Total FLOPs across all projections per layer:**
- q_proj: 2 * N * 896 * 16 * 896 = 2N * 12.85M
- k_proj: 2 * N * 128 * 16 * 896 = 2N * 1.84M
- v_proj: 2 * N * 128 * 16 * 896 = 2N * 1.84M
- o_proj: 2 * N * 896 * 16 * 896 = 2N * 12.85M
- gate_proj: 2 * N * 4864 * 16 * 896 = 2N * 69.73M
- up_proj: 2 * N * 4864 * 16 * 896 = 2N * 69.73M
- down_proj: 2 * N * 896 * 16 * 4864 = 2N * 69.73M

**Per layer total:** 2N * 238.57M FLOPs
**All 24 layers:** 2N * 5.73B FLOPs

At N=50: ~573B FLOPs. At Apple Silicon ~10 TFLOPS: ~57ms theoretical.

**Memory written:** Total parameters in all projections per layer:
- q: 896*896 = 802,816
- k: 128*896 = 114,688
- v: 128*896 = 114,688
- o: 896*896 = 802,816
- gate: 4864*896 = 4,358,144
- up: 4864*896 = 4,358,144
- down: 896*4864 = 4,358,144
- Per layer: 14.91M params * 4 bytes = 59.6 MB
- All 24 layers: 1.43 GB written

### 2.2 Strategy B: Runtime LoRA (on-the-fly BA computation)

Instead of pre-merging, keep W_l frozen and compute LoRA output at inference:

    y = xW_l^T + sum_{i in S} x A_i^T B_i^T

**Swap cost:** Zero. Just update the active set pointer.

**Per-token overhead per projection:** For input x of shape (seq, d_in):
- x A_i^T: seq * d_in * r FLOPs per expert
- (x A_i^T) B_i^T: seq * r * d_out FLOPs per expert
- Total per expert per projection: seq * r * (d_in + d_out)
- Total for N experts: N * seq * r * (d_in + d_out)

**Per-token per-layer overhead (all 7 projections, seq=1):**
- q: N * 1 * 16 * (896 + 896) = N * 28,672
- k: N * 1 * 16 * (896 + 128) = N * 16,384
- v: N * 1 * 16 * (896 + 128) = N * 16,384
- o: N * 1 * 16 * (896 + 896) = N * 28,672
- gate: N * 1 * 16 * (896 + 4864) = N * 92,160
- up: N * 1 * 16 * (896 + 4864) = N * 92,160
- down: N * 1 * 16 * (4864 + 896) = N * 92,160

**Per-token per-layer:** N * 366,592 FLOPs
**All 24 layers:** N * 8.80M FLOPs per token

At N=10: 88M FLOPs/token. Base model forward is ~1B FLOPs/token.
**Overhead: ~8.8% per token at N=10, ~44% at N=50.**

### 2.3 Strategy C: Incremental Update

When swapping expert j_old for j_new, update each merged weight:

    M_l' = M_l - B_{j_old}^l A_{j_old}^l + B_{j_new}^l A_{j_new}^l

**FLOPs per projection:** 2 * 2 * d_out * r * d_in (two rank-r updates)

This is independent of N -- always exactly 2 expert matmuls per projection.

**Total FLOPs (all projections, all layers):**
- Per layer: 2 * 2 * 238.57M / N_factor = 4 * 238.57M = 954.28M
  (Wait: 238.57M was per-expert. For 2 experts: 2 * 2 * sum(d_out * r * d_in).)
- Per layer: 4 * 119.3M = 477.1M FLOPs
- All 24 layers: 11.45B FLOPs

At Apple Silicon ~10 TFLOPS: ~1.1ms theoretical.

**Memory written:** Same as Strategy A (must write all projections): 1.43 GB.
But computation is O(1) in N, not O(N).

## 3. Theoretical Comparison

| Strategy | Swap FLOPs | Swap Memory Write | Per-token Overhead | Quality |
|----------|-----------|-------------------|-------------------|---------|
| A: Full Recompute | O(N * L * P * r) | 1.43 GB | 0 | Exact |
| B: Runtime LoRA | 0 | 0 | O(N * L * P * r) per token | Exact |
| C: Incremental | O(L * P * r) (constant) | 1.43 GB | 0 | Exact |

Where P = sum over projections of d_out * d_in.

**Key insight:** All three strategies produce mathematically identical results.
The only difference is WHERE the computation happens:
- A: At swap time, proportional to N
- B: At every token, proportional to N
- C: At swap time, independent of N

**Crossover analysis (when B beats C):**
- C costs 11.45B FLOPs at swap time
- B costs N * 8.80M FLOPs per token
- Break-even tokens = 11.45B / (N * 8.80M) = 1301/N tokens
- At N=10: B is cheaper if you generate < 130 tokens after swapping
- At N=50: B is cheaper if you generate < 26 tokens after swapping

For interactive use (generating 100+ tokens per response), Strategy C dominates.

## 4. Kill Criteria Mapping

**K1: Expert swap >100ms.** Strategy C theoretical minimum ~1.1ms. Even with
memory bandwidth limits (Apple Silicon: ~200 GB/s, writing 1.43 GB takes ~7ms),
total should be well under 100ms.

**K2: Quality regression.** All three strategies are mathematically exact.
B_i A_i is the same matrix whether computed at swap time or inference time.
The only risk is floating-point ordering differences, which should be < 1e-5.

## 5. Worked Example (micro scale)

Parameters: d=64, r=4, L=2, N=4, projections=[q,k,v,o] (attn only for simplicity)

**Strategy A (full recompute):**
- Per projection: 4 experts * 2 * 64 * 4 * 64 = 131,072 FLOPs
- 4 projections * 2 layers = 8 projection instances
- Total: 8 * 131,072 = 1,048,576 FLOPs

**Strategy C (incremental, swap 1 expert):**
- Per projection: 2 * 2 * 64 * 4 * 64 = 65,536 FLOPs
- 8 instances: 524,288 FLOPs
- Savings: 2x (since we swap 1 of 4, savings = N/2 = 2x)

**Strategy B (runtime, per token):**
- Per token per projection: 4 * 1 * 4 * (64 + 64) = 2,048 FLOPs
- 8 instances: 16,384 FLOPs per token
- Break-even: 1,048,576 / 16,384 = 64 tokens (vs Strategy A)
