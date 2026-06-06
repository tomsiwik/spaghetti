# MATH.md: Batched LoRA Dispatch via Stacked Matmul on MLX

## Type: Frontier Extension

**Proven result being extended:** v3 factored RuntimeLoRA is the correct architecture
for per-token adapter composition (Finding #288, #300, #301). The factored form
`y = x @ W_base.T + s * sum_i (x @ A_i.T) @ B_i.T` dominates pre-merge by 4-87x
(batched_premerge_throughput experiment). The bottleneck is bf16 matmul cost across
210 modules x K adapters (Finding #288: 48% overhead at rank 16).

**Gap:** Can we reduce matmul count by batching K adapter projections into a single
stacked matmul per module, rather than K sequential matmuls?

---

## Step A: Diagnose the Disease

**Failure mode:** Sequential adapter dispatch creates K*2 matmul kernel launches per
module per token. At K=1, 210 modules = 420 matmuls. At K=2, 840 matmuls. Each matmul
is small (rank r=16), underutilizing GPU compute. The Metal GPU prefers fewer, larger
operations over many small ones.

**Is this the root cause?** Finding #76 says dispatch overhead is hidden by async_eval.
Finding #300 says bandwidth, not dispatch count, is the bottleneck at production scale.
But Finding #288 says the 48% overhead comes from "bf16 matmul at rank 16 across 210
modules" — the raw cost of many small matmuls, not kernel launch overhead per se.

The disease is: **each small matmul (1 x d) @ (d x r) underutilizes the GPU's SIMD
units because the matrix dimensions are too small for efficient tiling.** Stacking K
adapters into one matmul of shape (1 x d) @ (d x K*r) increases the work per kernel
launch, improving GPU utilization.

## Step B: The Right Question

**Wrong:** "How do we reduce dispatch overhead?"
(Finding #76 already showed dispatch is free under async_eval.)

**Right:** "What operation shape maximizes GPU arithmetic intensity for K rank-r
adapter projections on a single input vector?"

The answer comes from basic GPU computation theory: a matmul of shape (M, N) @ (N, P)
has arithmetic intensity proportional to M*P / (M+P). For the sequential case:
each matmul is (1, d) @ (d, r), giving intensity ~ r / (1 + r). For r=16, this is
0.94 — extremely compute-light, bandwidth-dominated.

If we stack K adapters: (1, d) @ (d, K*r), giving intensity ~ K*r / (1 + K*r).
For K=5, r=16: intensity ~ 80/81 = 0.99. The improvement in intensity is marginal
(0.94 -> 0.99), but the real gain is **reducing the number of kernel launches from
K to 1**, which reduces:
1. Python-side loop overhead (iterating over adapters)
2. Metal command buffer submission overhead (per-matmul dispatch)
3. Memory read redundancy (x is loaded from memory K times sequentially; once in stacked)

## Step C: Prior Mathematical Foundations

**Punica BGMV (Chen et al., 2310.18547):** The Batched Gather Matrix-Vector
multiplication kernel fuses K adapter projections into a single GPU kernel. For
inputs x_i with adapter assignment a_i:

```
y_i = x_i @ W_base.T + sum_{k in S_i} s * (x_i @ A_k.T) @ B_k.T
```

Punica's key insight: when all tokens use the SAME adapter set (as in our
per-sequence routing), the adapter projections can be batched by stacking A matrices.

**MLX batched matmul:** MLX supports batched matrix multiplication via
`mx.matmul(X, A_stack)` where A_stack has shape (K, d, r). The operation computes
K independent matmuls in parallel, returning shape (K, ..., r). This maps directly
to the stacked adapter projection.

**Operational intensity model (Williams et al., Roofline):** For an operation with
W arithmetic ops and Q bytes moved:

```
I = W / Q  (operational intensity, FLOPs/byte)
```

The system is compute-bound when I > I_ridge = peak_FLOPS / peak_BW.
For M5 Pro: I_ridge ~ 14 TFLOPS / 273 GB/s ~ 51 FLOPs/byte.

For rank-16 vector-matrix multiply: I = 2*d*r / (2*(d+r)*2) = d*r / (2*(d+r))
At d=2560, r=16: I = 40960 / 5152 = 7.95 FLOPs/byte. This is well below I_ridge,
confirming bandwidth-bound operation. Stacking does NOT change the total FLOPs or
bytes — it just reduces overhead per operation.

## Step D: Proof of Equivalence and Speed Bound

### Theorem 1 (Numerical Equivalence)

**Theorem 1.** Let x in R^{1 x d}, and let {(A_k, B_k)}_{k=1}^K be K adapter pairs
with A_k in R^{r x d} and B_k in R^{d_out x r}. Define:

Sequential form:
```
y_seq = sum_{k=1}^K (x @ A_k.T) @ B_k.T
```

Stacked form: let A_stack in R^{K x r x d} with A_stack[k] = A_k, and
B_stack in R^{K x d_out x r} with B_stack[k] = B_k. Then:
```
H = x @ A_stack.transpose(0, 2, 1)    # broadcast: (1, d) @ (K, d, r) -> (K, 1, r)
Y_k = H @ B_stack.transpose(0, 2, 1)  # (K, 1, r) @ (K, r, d_out) -> (K, 1, d_out)
y_stack = sum_{k=0}^{K-1} Y_k[k]      # (1, d_out)
```

Then y_seq = y_stack (exact arithmetic equality in infinite precision).

*Proof.* For each k:
- Sequential: z_k = (x @ A_k.T) @ B_k.T = x A_k^T B_k^T
- Stacked: H[k] = x @ A_stack[k]^T = x A_k^T (since A_stack[k] = A_k)
- Y_k[k] = H[k] @ B_stack[k]^T = x A_k^T B_k^T = z_k

Therefore sum_k Y_k[k] = sum_k z_k = y_seq. QED.

**Corollary (Finite precision).** In floating-point arithmetic, the difference
|y_seq - y_stack| is bounded by:
```
|y_seq - y_stack| <= K * epsilon_mach * max_k(||A_k|| * ||B_k||) * ||x||
```
where epsilon_mach ~ 2^{-11} for bfloat16. For typical adapter norms (||A||, ||B|| ~ 0.01),
this gives MSE << 1e-6 (K2 kill criterion).

### Theorem 2 (Latency Reduction)

**Theorem 2.** Let T_seq and T_stack be the wall-clock times for sequential and stacked
adapter projection respectively. Under the following model:

```
T_seq = K * (T_dispatch + T_matmul(1, d, r) + T_matmul(1, r, d_out))
T_stack = T_dispatch + T_batched_matmul(K, 1, d, r) + T_dispatch + T_batched_matmul(K, 1, r, d_out) + T_sum
```

where T_dispatch is the per-operation overhead (Metal command buffer + Python loop
iteration) and T_sum is the reduction cost.

If batched matmul achieves perfect parallelism across the K batch dimension:
```
T_batched_matmul(K, 1, d, r) = T_matmul(1, d, r) + alpha * (K-1) * T_matmul(1, d, r)
```
where alpha in [0, 1] measures the overhead per additional batch element (alpha=0
is perfect parallelism, alpha=1 is sequential).

Then:
```
T_stack / T_seq = (2 * T_dispatch + (1 + alpha*(K-1)) * 2 * T_matmul + T_sum) / (K * (T_dispatch + 2 * T_matmul))
```

For K >= 2 and alpha < 1:
```
Speedup = T_seq / T_stack > 1  iff  (K-1) * T_dispatch > alpha * (K-1) * 2 * T_matmul + T_sum
                                iff  T_dispatch > alpha * 2 * T_matmul + T_sum / (K-1)
```

*Proof.* Direct algebraic manipulation. The speedup exists when the saved dispatch
overhead (K-1) * T_dispatch exceeds the batching overhead. QED.

**Quantitative prediction for our setting:**

At production scale (d=2560, r=16, K=1..5):
- The system is bandwidth-bound (Finding #300), so T_matmul ~ bytes_moved / BW
- For x@A.T: bytes = 2*(2560 + 16) * 2 = 10,304 bytes, time ~ 10304 / 273e9 = 0.038 us
- This is too small for meaningful GPU work. The actual time is dominated by kernel
  launch overhead, estimated at ~5-20 us per operation on Metal.

At micro scale (d=128, r=8, K=1..5): we can measure alpha and T_dispatch directly.

**Prediction table:**

| Metric | Predicted | Kill threshold |
|--------|-----------|----------------|
| Numerical MSE (stacked vs sequential) | < 1e-10 (fp32), < 1e-6 (bf16) | K770: MSE > 1e-6 |
| Speed at K=1 | No improvement (stacking is identity) | -- |
| Speed at K=5 | 1.2-2.5x faster (depends on alpha) | K769: < 85 tok/s |
| Memory overhead | < 5% (temporary stack tensors) | K771: > 3 GB |

### Theorem 3 (Amortization Across Modules)

**Theorem 3.** For a transformer with L layers and M modules per layer, the total
adapter overhead is:

Sequential: T_total_seq = L * M * K * (T_dispatch + 2 * T_matmul)
Stacked:    T_total_stack = L * M * (2 * T_dispatch + (1 + alpha*(K-1)) * 2 * T_matmul + T_sum)

The MODULE-LEVEL stacking speedup is:
```
S_module = K * (T_dispatch + 2 * T_matmul) / (2 * T_dispatch + (1+alpha*(K-1)) * 2 * T_matmul + T_sum)
```

This is multiplied by L * M, so even modest per-module speedup accumulates across
the full model (L=30, M=7 for BitNet-2B-4T = 210 module applications).

## Step E: Assumptions and Breaking Conditions

1. **MLX batched matmul is efficient for small inner dims.** If the Metal backend
   serializes the K batch elements, alpha -> 1 and stacking provides no benefit.
   Breaking: we observe no speedup for stacked vs sequential.

2. **Adapter matrices fit in unified memory.** K=5 adapters at rank 16 need
   K * (A_size + B_size) = 5 * (16*2560 + 2560*16) * 2 bytes = 819 KB per module.
   Total across 210 modules: 168 MB. Well within 48 GB.

3. **mx.vmap or manual stacking works on MLX.** MLX may not support vmap for this
   pattern; manual stacking (mx.stack + batched matmul) is the fallback.

4. **The measurement captures real inference, not just isolated matmul.** We must
   benchmark end-to-end token generation, not just the adapter matmul in isolation.

## Step F: Worked Example (d=8, r=2, K=3)

```
x = [1, 0, 1, 0, 1, 0, 1, 0]  # (1, 8)

A_1 = [[0.1, 0, 0.1, 0, 0, 0, 0, 0],    # (2, 8)
       [0, 0.1, 0, 0.1, 0, 0, 0, 0]]
A_2 = [[0, 0, 0, 0, 0.1, 0, 0.1, 0],
       [0, 0, 0, 0, 0, 0.1, 0, 0.1]]
A_3 = [[0.1, 0.1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0.1, 0.1, 0, 0, 0, 0]]

Sequential:
  h_1 = x @ A_1.T = [0.2, 0.2]     # (1, 2)
  h_2 = x @ A_2.T = [0.2, 0.2]
  h_3 = x @ A_3.T = [0.1, 0.1]

Stacked:
  A_stack = stack([A_1, A_2, A_3])  # (3, 2, 8)
  H = x @ A_stack.T                # broadcast: (1, 8) @ (3, 8, 2) -> (3, 1, 2)
  H[0] = [0.2, 0.2]  = h_1  CHECK
  H[1] = [0.2, 0.2]  = h_2  CHECK
  H[2] = [0.1, 0.1]  = h_3  CHECK
```

One matmul op instead of three. Same result.

## Step G: Complexity and Architecture Connection

**FLOPs:** Identical for sequential and stacked (2 * K * d * r per module).
**Memory:** Stacking adds K * r * d for A_stack + K * d_out * r for B_stack.
At d=2560, r=16, K=5: (5 * 16 * 2560 + 5 * 2560 * 16) * 2 bytes = 819 KB per module.
Total temporary: 210 modules * 819 KB = 168 MB. Well within budget.

**Architecture:** This is a drop-in replacement for the inner loop of
`_apply_lora_linear`. No changes to routing, adapter training, or composition logic.

**Connection to production:** Punica BGMV does exactly this on CUDA with fused
kernels. We implement the same principle using MLX's batched matmul (no custom kernels).
S-LoRA (2311.03285) further optimizes with unified memory management, but the core
insight is identical: stack adapters, batch multiply.

---

## Self-Test

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Batched matmul computes K projections in a single kernel launch, eliminating
   the per-adapter dispatch loop overhead. The equivalence is exact (Theorem 1).

2. **Which existing theorem(s) does the proof build on?**
   Punica BGMV (Chen et al., 2310.18547) — stacked adapter projection.
   Roofline model (Williams et al., 2009) — operational intensity analysis.

3. **What specific numbers does the proof predict?**
   MSE < 1e-6 (bf16), < 1e-10 (fp32). Speed improvement 1.2-2.5x at K>=2
   (depends on alpha). No improvement at K=1.

4. **What would FALSIFY the proof?**
   If MLX batched matmul has alpha >= 1 (serializes batch elements), stacking
   provides zero benefit. Also falsified if total overhead is dominated by
   non-adapter computation (base model forward pass).

5. **How many hyperparameters does this approach add?**
   0. The stacking is purely mechanical — no tunable parameters.

6. **Hack check:** No fix stack. This is a single mechanical optimization (replace
   Python loop with batched matmul). The mathematical guarantee is exact equivalence.
