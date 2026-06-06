# Pre-Merge Composition Latency: Mathematical Analysis

## 1. Mechanism Definition

Pre-merge composition computes a merged weight matrix from a base weight and N
LoRA adapters before the forward pass:

```
W_merged = W_base + sum_{i=1}^{N} alpha_i * (B_i^T @ A_i^T)
```

Where:
- `W_base in R^{d_out x d_in}` is the base weight (frozen)
- `A_i in R^{d_in x r}` is the frozen Grassmannian down-projection for expert i
- `B_i in R^{r x d_out}` is the trained up-projection for expert i
- `alpha_i in R` is the routing weight for expert i
- `r` is the LoRA rank (16 in our architecture)

For BitNet-2B-4T: `d_in = d_out = 2560`, `r = 16`.

Each adapter delta `Delta_i = B_i^T @ A_i^T` has shape `(d_out, d_in)` = `(2560, 2560)`.

### 1.1 Pre-Merge FLOPs per Adapter

Computing one delta `B_i^T @ A_i^T`:
- `B_i^T`: shape `(d_out, r)` -- free (transpose is a view)
- `A_i^T`: shape `(r, d_in)` -- free (transpose is a view)
- MatMul `(d_out, r) @ (r, d_in)`: `2 * d_out * r * d_in` FLOPs
- At d=2560, r=16: `2 * 2560 * 16 * 2560 = 209,715,200` ~ 210M FLOPs

Scaling the delta: `alpha_i * delta_i`: `d_out * d_in = 6,553,600` ~ 6.6M FLOPs

Adding to W: `W + delta`: `d_out * d_in = 6,553,600` ~ 6.6M FLOPs

**Total per adapter: ~223M FLOPs**

**Total for N adapters: ~223M * N FLOPs** (linear in N)

### 1.2 Forward Pass FLOPs (Independent of N)

For a single token `x in R^{d_in}`:
- `y = x @ W_merged^T`: `2 * d_in * d_out = 13,107,200` ~ 13.1M FLOPs

For T tokens: `2 * T * d_in * d_out` FLOPs.

### 1.3 Memory Bandwidth

The merge is memory-bandwidth bound, not compute-bound, because:
- Each delta `B_i^T @ A_i^T` produces a `(2560, 2560)` matrix = 6.5M elements
- At bf16: 13.1 MB per delta, must be written and added to W_merged
- M5 Pro memory bandwidth: ~273 GB/s

**Merge time lower bound** (memory-bandwidth limited):
- Per adapter: read B_i (16*2560*2 = 81.9 KB) + read A_i (2560*16*2 = 81.9 KB) + write delta (13.1 MB) + read/write W (26.2 MB) ~ 39.5 MB
- At 273 GB/s: 39.5 MB / 273 GB/s = 0.145 ms per adapter
- For N=100: ~14.5 ms (bandwidth-limited lower bound)

## 2. Why Pre-Merge Latency Should Scale Linearly with N

The merge operation is a sequential accumulation:
```
W_merged = W_base
for i in selected_experts:
    W_merged += alpha_i * (B_i^T @ A_i^T)
```

Each step is independent (no data dependencies between adapter deltas).
The operations are purely elementwise after the matmul, so:

**Prediction: T_merge(N) = c_0 + c_1 * N**

Where:
- `c_0` = constant overhead (Metal dispatch, Python loop setup)
- `c_1` = per-adapter merge time (dominated by the rank-r matmul)

This is LINEAR in N. The kill criterion asks whether it grows superlinearly.
Superlinear growth would indicate:
- Cache thrashing as W_merged exceeds L2 cache
- Memory allocation pressure from intermediate tensors
- Metal command buffer overflow

## 3. What Breaks It

**Cache capacity:** W_merged at bf16 = 2560^2 * 2 = 13.1 MB. M5 Pro L2 cache is
~32 MB per cluster. The merged weight fits in L2, but as we accumulate more deltas,
the intermediate delta matrices cycle through cache. At N >> d/r, we may see
cache pressure, but N=100 << N_max=25,600, so this should not be an issue.

**Metal dispatch overhead:** Each adapter requires a matmul kernel launch.
If dispatch latency is ~10 us and N=100, that is 1 ms of dispatch overhead alone.
With `mx.compile`, multiple dispatches can be fused.

**Memory growth:** Each adapter stores A_i and B_i. Total adapter memory:
- Per adapter: (d_in*r + r*d_out) * 2 bytes = (2560*16 + 16*2560) * 2 = 163.8 KB
- N=100: 16.4 MB (trivial on 48 GB)

**Kill condition:** K1 fails if `T_merge(N) = O(N^alpha)` with alpha > 1.
We fit a power law `T = a * N^alpha` and check alpha.

## 4. Connection to Architecture

Pre-merge is the serving strategy for **always-on adapters** (e.g., instruction adapter).
For routed experts, runtime LoRA is 4-87x faster (proven in batched_premerge_throughput).

The critical question is: at what N does pre-merge become too slow for interactive use?
Prior work showed 0% overhead for pre-merge at small N, but never swept to N=100.

The S1 criterion (<50ms overhead at N=25) determines whether pre-merge is viable
for a "personality blend" use case where a user selects 25 always-on adapters.

## 5. Optimization via mx.compile

The merge loop has a fixed computation graph for each N. By compiling the
merge function, MLX can:
1. Fuse the scale+add operations across adapters
2. Eliminate intermediate allocations for delta matrices
3. Potentially vectorize the rank-r matmuls

Expected speedup: 1.5-3x based on prior compiled_merge results from
batched_premerge_throughput experiment.

## 6. Prior Art

- **Naive LoRA Summation** (arxiv 2508.11985): proves orthogonality enables additive
  composition. Our merge is exactly this: W + sum(alpha_i * B_i^T @ A_i^T).
- **LoRA Soups** (arxiv 2410.13025): CAT (concatenate-and-average) composition.
  Shows linear scaling of merge cost with N.
- **Batched pre-merge throughput** (this project): runtime LoRA dominates pre-merge
  by 4-87x for per-token routing. This experiment isolates the pre-merge path
  to find its scaling law and absolute latency.
- **Continual learning adapter growth** (this project): composition quality stable
  within ~1% of base across N=5-15, meaning we WANT to compose many adapters.
