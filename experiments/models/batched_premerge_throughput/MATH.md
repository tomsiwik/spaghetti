# Batched Pre-Merge Throughput: Mathematical Foundations

## Problem Statement

Per-token routing assigns each token in a batch to a potentially different subset
of experts. Pre-merge composition (proven 0% per-token overhead in
`exp_e2e_demo_pipeline_mlx`) folds LoRA deltas into base weights:

W_merged = W_base + sum_{i in S} w_i * alpha * B_i^T @ A_i^T

where S is the active expert set and alpha is the LoRA scaling factor.

The problem: with per-token routing, each token t has its own active set S_t.
A naive implementation merges a new W for every unique S_t. The question is
whether grouping tokens by their expert set and batching the merge is faster.

## 1. Mechanism Definition

### Naive per-token merge

For a batch of T tokens with per-token routing decisions {S_1, ..., S_T}:

```
for t in 1..T:
    W_t = W_base + sum_{i in S_t} w_i * alpha * B_i^T @ A_i^T   # shape: (d_out, d_in)
    y_t = x_t @ W_t^T                                             # shape: (1, d_out)
```

**Cost per layer per token:**
- Merge: |S_t| matmuls of B_i^T @ A_i^T, each O(d_out * r * d_in / r) = O(d_out * d_in)
  Wait -- B_i in R^{r x d_out}, A_i in R^{d_in x r}
  B_i^T @ A_i^T: (d_out x r) @ (r x d_in) = O(d_out * r * d_in)
  But this is rank-r, so it's O(d_out * r + r * d_in) using the factored form
- Actually: delta = alpha * B_i^T @ A_i^T can be done as:
  Step 1: B_i^T @ A_i^T is (d_out, d_in), cost O(d_out * r * d_in)
  OR factored as x @ A_i @ B_i * alpha (runtime LoRA), cost O(d_in * r + r * d_out) per token

Pre-merge cost: k * O(d_out * r * d_in) per unique expert set (k = |S_t|)
Inference cost: O(d_out * d_in) per token (standard matmul, no LoRA overhead)

Runtime LoRA cost: k * O((d_in * r + r * d_out) * T_group) for T_group tokens

### Batched pre-merge

Key insight: Many tokens share the same expert set S. Group them.

Let U = {S_1, ..., S_M} be the M distinct expert sets across all T tokens.
Let T_j = |{t : S_t = S_j}| be the count of tokens using expert set S_j.

```
for j in 1..M:
    W_j = W_base + sum_{i in S_j} w_i * alpha * B_i^T @ A_i^T   # Merge once
    Y_j = X_j @ W_j^T                                             # Batch matmul for T_j tokens
```

**Cost:**
- Merge: M * k * O(d_out * r * d_in) for all M unique sets
- Inference: M * O(T_j * d_out * d_in) = O(T * d_out * d_in) total

### Speedup analysis

Naive total cost = T * [k * C_merge + C_matmul]
Batched total cost = M * k * C_merge + T * C_matmul + C_group

where:
- C_merge = O(d_out * r * d_in) per expert per merge
- C_matmul = O(d_out * d_in) per token
- C_group = O(T * log T) for sorting/grouping tokens

Speedup = T * k * C_merge / (M * k * C_merge + C_group)
       ~= T / M when C_merge dominates

For top-k routing with N experts, choose(N, k) possible expert sets.
In practice M << T because:
- Top-1 routing: M <= N (at most N distinct sets)
- Top-2 routing: M <= N*(N-1)/2 (at most N-choose-2 sets)
- For N=5, k=2: M <= 10

So speedup ~= T/min(M, choose(N,k)) for merge-dominated workloads.

At our micro scale: d_out=d_in=2560, r=16, N=5
- C_merge = 2560 * 16 * 2560 = 104.9M FLOPs per expert
- C_matmul = 2560 * 2560 = 6.6M FLOPs per token
- Merge/matmul ratio = 16x (merge is 16x more expensive per operation)

## 2. Why It Works

The batched approach exploits the **set-deduplication principle**: when T tokens
share M << T distinct expert sets, the expensive merge operation (O(d*r*d) per
expert) is amortized across T/M tokens per set.

This is the same principle behind batch matrix multiplication in transformers:
instead of computing attention per-query, we batch all queries sharing the same
KV cache. Here we batch all tokens sharing the same merged weight matrix.

The mathematical guarantee: since W_j depends only on S_j (the expert set) and
not on the token content, tokens with identical routing decisions produce
identical merged weights. Merging once per unique set is exact (no approximation).

## 3. What Breaks It

**M approaches T (every token has unique expert set):**
- Occurs when N is large and k > 1 with diverse routing
- If M = T, batched = naive (no savings)
- Kill condition: M/T > 0.5 with no throughput gain

**Grouping overhead dominates:**
- Token sorting/indexing costs O(T) with MLX gather operations
- If C_group > (T - M) * k * C_merge, batching is net negative
- Kill criterion K2 tests this directly

**Small batch sizes (T small):**
- At T=1 (autoregressive generation), M=1 always, batching is trivially optimal
  but the overhead of grouping is wasted
- Batching is most beneficial for prompt processing (T=256+)

**mx.compile recompilation:**
- If token group sizes vary, mx.compile may recompile per shape
- Mitigation: pad groups to fixed sizes or use shapeless compilation

## 4. Assumptions

1. **LoRA adapters are pre-loaded in memory** (validated in e2e_demo_pipeline_mlx)
2. **Token routing decisions are available before merge** (per-token routing
   produces assignments first, then we compose)
3. **MLX gather/scatter is efficient** for token regrouping (assumption, to be tested)
4. **Top-k routing with small k** (k=1 or k=2, as proven in per-token routing experiments)
5. **N <= 25 adapters** (proven scaling range)

## 5. Complexity Analysis

| Approach | Merge FLOPs | Inference FLOPs | Grouping | Total |
|----------|------------|-----------------|----------|-------|
| Naive (per-token) | T*k*d_out*r*d_in | T*d_out*d_in | 0 | T*(k*d*r*d + d*d) |
| Batched | M*k*d_out*r*d_in | T*d_out*d_in | O(T) | M*k*d*r*d + T*d*d + O(T) |
| Runtime LoRA | 0 | T*k*(d*r+r*d) + T*d*d | 0 | T*(d*d + 2k*d*r) |

At d=2560, r=16, k=2, T=256, N=5:
- Naive: 256 * 2 * 2560 * 16 * 2560 + 256 * 2560^2 = 53.7G + 1.68G = 55.4G
- Batched (M=10): 10 * 2 * 2560 * 16 * 2560 + 256 * 2560^2 = 2.10G + 1.68G = 3.78G
- Runtime LoRA: 256 * (2560^2 + 2*2*2560*16) = 256 * (6.55M + 163.8K) = 1.72G

**Batched pre-merge: 14.7x faster than naive, but still 2.2x slower than runtime LoRA for inference.**

However, pre-merge produces a standard nn.Linear -- no architectural changes needed,
and subsequent operations (attention, FFN) see a standard weight matrix. Runtime LoRA
requires modifying the forward pass.

## 6. Worked Example (micro scale)

d=64, r=4, N=4 experts, k=2 (top-2 routing), T=8 tokens

Token routing decisions:
- Tokens 0,1,3: experts {0,2}
- Tokens 2,5: experts {1,3}
- Tokens 4,6,7: experts {0,1}

M=3 unique expert sets. Groups: [0,1,3], [2,5], [4,6,7]

Naive: 8 merges of 2 experts each = 16 expert-merges
Batched: 3 merges of 2 experts each = 6 expert-merges
Speedup on merge: 16/6 = 2.67x

Including inference (same for both): depends on merge/matmul ratio.
At d=64, r=4: merge cost = 64*4*64 = 16K, matmul cost = 64*64 = 4K
Merge is 4x more expensive.

Naive total = 8*(2*16K + 4K) = 8*36K = 288K FLOPs
Batched total = 3*2*16K + 8*4K = 96K + 32K = 128K FLOPs + grouping overhead
Speedup = 288/128 = 2.25x (before grouping overhead)

## 7. Connection to Architecture

This experiment tests the efficiency of the **Serve** phase of BitNet-SOLE.

In production, the serving pipeline is:
1. Query arrives (T tokens)
2. Per-token router assigns each token to top-k experts (proven: Gumbel-sigmoid, 0.58% overhead)
3. **This experiment:** Group tokens by expert set, batch merge, process groups
4. Generate output tokens

The e2e_demo_pipeline_mlx showed pre-merge works but used oracle routing (same
expert set for all tokens in a query). Per-token routing (proven in
exp_bitnet_real_data_25_domain_adapters with 92.7% accuracy) creates the
heterogeneous expert set problem this experiment addresses.

Production systems (DeepSeek-V3, Qwen3 MoE) handle this via expert parallelism
across GPUs. Our single-device approach uses batched merge as the analogous
optimization -- different strategy, same principle of amortizing expert activation.

## References

- MoLoRA (arXiv 2603.15965): per-token routing, 1.7B beats 8B
- DeepSeek-V3 (arXiv 2412.19437): expert parallelism for MoE serving
- exp_e2e_demo_pipeline_mlx: pre-merge 0% overhead baseline
- exp_bitnet_real_data_25_domain_adapters: per-token routing at N=24
