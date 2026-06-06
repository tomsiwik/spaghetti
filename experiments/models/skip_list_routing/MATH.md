# Skip-List Multi-Resolution Routing: Mathematical Foundations

## Notation

| Symbol | Shape / Type | Definition |
|--------|-------------|------------|
| N | scalar | Number of leaf experts at Level 0 |
| L | scalar | Number of coarse levels = floor(log2(N)) |
| d | scalar | Embedding dimension |
| x | (B, T, d) | Input hidden states |
| n_k | scalar | Number of experts at level k: ceil(N / 2^k) |
| R_k | (d, n_k) | Router weight matrix at level k |
| g_k | (d, 1) | Confidence gate weight at level k (for k >= 1) |
| b_k | (1,) | Confidence gate bias at level k |
| p_k(x) | (B, T, n_k) | Softmax routing probabilities at level k |
| c_k(x) | (B, T, 1) | Confidence = sigmoid(g_k^T x + b_k) at level k |
| E_i | CapsuleGroup | Leaf expert i at Level 0 |
| E_{i,k} | virtual | Coarse expert i at Level k (weight-averaged children) |

## Level Structure

Given N leaf experts, the skip list has L+1 levels (Level 0 through Level L):

```
Level L (coarsest):  n_L = ceil(N / 2^L) experts
Level L-1:           n_{L-1} = ceil(N / 2^{L-1}) experts
...
Level 1:             n_1 = ceil(N / 2) experts
Level 0 (finest):    n_0 = N experts
```

For N=8: Level 0 = 8, Level 1 = 4, Level 2 = 2, Level 3 = 1.

**Parentage**: Expert i at Level k covers leaf experts [i * 2^k, (i+1) * 2^k).

## Coarse Expert Construction (Zero Extra Parameters)

Express experts at Level k > 0 are weight-averaged children. For expert i at
Level k:

```
E_{i,k}(x) = (1/m) * sum_{j in children(i,k)} E_{j,k-1}(x)
```

where children(i,k) = {2i, 2i+1} (or just {2i} if 2i+1 >= n_{k-1}).

This recursion bottoms out at Level 0 where E_{i,0} = E_i (actual parameters).

**No extra parameters**: coarse experts reuse leaf expert weights.

## Soft Adaptive Routing

Routing proceeds top-down from Level L to Level 0 with soft level selection.

### Level Weights

At each coarse level k (from L down to 1), a confidence gate decides whether
to stop (use level k's routing) or descend:

```
c_k(x) = sigmoid(g_k^T x + b_k)    in [0, 1]
```

The weight for level k in the output is:

```
w_L(x) = c_L(x)                                        (coarsest level)
w_k(x) = c_k(x) * prod_{j=k+1}^{L} (1 - c_j(x))      (intermediate levels)
w_0(x) = prod_{j=1}^{L} (1 - c_j(x))                  (finest level)
```

**Theorem**: sum_{k=0}^{L} w_k(x) = 1 for all x.

*Proof*: By induction. Define S_k = sum_{j=k}^{L} w_j(x).
- Base: S_L = c_L = w_L. Check.
- Step: S_k = w_k + S_{k+1} = c_k * P_k + S_{k+1}, where P_k = prod_{j=k+1}^{L}(1-c_j).
  S_{k+1} = (1-c_k) was used at previous step... Actually, let's expand directly.
  S_0 = prod(1-c_j) + sum_{k=1}^{L} c_k * prod_{j=k+1}^{L}(1-c_j).
  This is the standard telescoping: P(stop at layer k) + P(pass all) = 1.
  Formally, this is the cascade probability: identical to stick-breaking in
  Dirichlet processes. Each c_k "breaks off" a fraction of remaining probability. QED.

### Per-Level Routing

At level k, given the input x:

```
s_k(x) = x @ R_k^T                         (d,) @ (d, n_k)^T -> (n_k,)
p_k(x) = softmax(s_k(x))                   (n_k,)
selected_k = top_K(p_k)                     K indices
```

Masked probabilities (after top-k):
```
p_k^masked = mask(p_k, selected_k) / sum(mask(p_k, selected_k))
```

### Output

```
output(x) = sum_{k=0}^{L} w_k(x) * [sum_{i=0}^{n_k-1} p_k^masked(x)[i] * E_{i,k}(x)]
```

## Routing Cost Analysis

### Important Caveat

The analysis below describes ROUTING DECISION costs only, and applies to a
**hypothetical hard routing inference mode** where each token is dispatched to
a single level based on confidence threshold. During training, ALL levels are
computed for every token -- there are no actual FLOP savings. The level-weight
distribution indicates *potential* savings under hard inference routing, which
is not implemented or tested in this experiment.

### Training-Time Expert Evaluation Cost (Actual)

During soft training, every level's experts are evaluated. Due to recursive
coarse expert construction, the total leaf expert forward passes per token is:

```
Level 0: N leaf evaluations (direct)
Level 1: N leaf evaluations (N/2 coarse experts, each averaging 2 leaves)
Level 2: N leaf evaluations (N/4 coarse experts, each averaging 4 leaves)
...
Level L: N leaf evaluations (1 coarse expert, averaging all N leaves)
```

Total = N * (L+1) leaf expert forward passes per token.

For N=8, L=3: 8 * 4 = 32 leaf expert forward passes per token.
Flat top-k=2 baseline: 2 leaf expert forward passes per token.
**Training-time expert evaluation is 16x more expensive than flat top-k=2.**

This is a fundamental limitation of the soft routing regime and a serious
scalability concern at larger N.

### Hypothetical Hard Routing Inference Cost

If hard early stopping were implemented (stop at level k when c_k > threshold),
the expected routing decision cost depends on the level-weight distribution:

### Flat Softmax (Baseline)
- Router: one linear projection d -> N, one softmax, one top-k.
- Cost: N * d MADs (multiply-adds) for projection + O(N) for softmax.

### Fixed-Depth Tree (hierarchical_tree)
- Always traverses depth D = log2(N) gates, each costing d+1 MADs.
- Total: D * (d+1) MADs per token.
- For N=8, d=64: 3 * 65 = 195 MADs.

### Skip-List Hard Routing (Hypothetical)
- Under hard routing, a token stops at the first level where confidence
  exceeds a threshold. The expected routing cost is:

```
C_adaptive = sum_{k=0}^{L} w_k * C_k
```

where C_k = cost of routing at levels L through k:

```
C_k = sum_{j=k}^{L} (n_j * d + d + 1)
     = d * sum n_j + (L-k+1) * (d+1)    (routers + confidence gates)
```

For N=8, d=64 with observed level-weight distribution [0.672, 0.126, 0.156, 0.046]:
- C_3 (coarsest): 1*64 + 65 = 129 MADs
- C_2: 129 + 2*64 + 65 = 322 MADs
- C_1: 322 + 4*64 + 65 = 643 MADs
- C_0 (finest): 643 + 8*64 + 65 = 1220 MADs

Expected: 0.672*129 + 0.126*322 + 0.156*643 + 0.046*1220 = 284 MADs

vs Flat: 8*64 = 512 MADs

**Hypothetical routing decision cost reduction: ~45% at micro scale.**
*This applies only to an unimplemented hard routing inference mode.
Actual training computes all levels and is 16x more expensive for expert
evaluation than flat routing.*

## Worked Example (d=64, N=8, K=2)

**Setup**: 8 leaf CapsuleGroups (each with A: 32x64, B: 64x32). Three coarse
levels above.

**Token x** with embedding dimension 64 enters skip list:

1. **Level 3** (1 expert): Router projects x -> 1 score. Confidence gate
   evaluates c_3 = sigmoid(-0.5) = 0.38. Weight w_3 = 0.38. Pass-through = 0.62.

2. **Level 2** (2 experts): Router projects x -> 2 scores, softmax -> [0.6, 0.4].
   Top-2 selects both. Confidence c_2 = sigmoid(1.2) = 0.77. Weight w_2 = 0.62 * 0.77 = 0.48.
   Pass-through = 0.62 * 0.23 = 0.14.

3. **Level 1** (4 experts): Router projects x -> 4 scores. Top-2 selects experts 1,3.
   Confidence c_1 = sigmoid(0.3) = 0.57. Weight w_1 = 0.14 * 0.57 = 0.08.
   Pass-through = 0.14 * 0.43 = 0.06.

4. **Level 0** (8 experts): Gets remaining w_0 = 0.06. Top-2 selects experts 2,5.

**Verify**: w_3 + w_2 + w_1 + w_0 = 0.38 + 0.48 + 0.08 + 0.06 = 1.00.

**Output**: 0.38*(L3 output) + 0.48*(L2 output) + 0.08*(L1 output) + 0.06*(L0 output).

86% of the level weight is assigned to Level 2 or coarser. Only 6% of level
weight reaches fine-grained Level 0. Note: ALL four levels were computed to
produce this output; the weights describe the learned blending, not
computational savings. The effective level-weight depth is:
0.38*1 + 0.48*2 + 0.08*3 + 0.06*4 = 1.82.

## Parameter Overhead

Extra parameters per layer vs flat CapsulePool:
- Flat: 1 router (d x N) = d*N
- Skip: routers at each level (d*n_0 + d*n_1 + ... + d*n_L) + L confidence gates (d+1 each)
  = d * (N + N/2 + N/4 + ... + 1) + L*(d+1)
  = d * (2N - 1) + L*(d+1)
  ~ 2*d*N + L*d

Overhead: d*N (extra routers) + L*d (gates) ~ d*(N + log2(N)).
For d=64, N=8: 64*(8+3) = 704 extra params per layer.
4 layers: 2816 extra = +1.3% vs flat baseline (204,160 params).

## Assumptions

1. **Weight-averaged coarse experts approximate cluster behavior**: Averaging
   children CapsuleGroups produces a meaningful "generalist" expert. This holds
   when children share similar input-output structure (validated by behavioral
   dedup finding: Layer 0 capsules have J=0.527 co-activation).

2. **Confidence gate learns meaningful signal**: The gate must distinguish
   "easy" tokens (coarse sufficient) from "hard" tokens (fine-grained needed).
   At d=64 with character-level names, the token difficulty distribution may be
   narrow, limiting the range of adaptive behavior.

3. **Soft routing during training approximates hard routing at inference**:
   During training, all levels contribute weighted by soft probabilities.
   At inference, one could apply hard early stopping (stop when c_k > threshold).
   The soft-hard gap is bounded by the concentration of level weights.

4. **Level structure matches data complexity hierarchy**: The 2^k grouping
   assumes a power-of-2 hierarchy in expert specialization. Non-power-of-2
   structures (as in frequency-weighted skip lists) might be more efficient
   but add complexity.
