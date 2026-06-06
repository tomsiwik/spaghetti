# LSH Capsule Routing: Mathematical Foundations

## 1. Random-Projection Locality-Sensitive Hashing

### 1.1 Core Idea

Locality-sensitive hashing (LSH) maps similar inputs to the same hash bucket
with high probability. For angular/cosine similarity, the canonical LSH
family uses random hyperplane projections (Charikar 2002, "SimHash"):

```
h(x) = sign(r^T x),   r ~ N(0, I_d)
```

where `P[h(x) = h(y)] = 1 - theta(x,y)/pi` and `theta(x,y)` is the angle
between x and y.

### 1.2 Extension to Expert Routing

Instead of binary hash functions, we use random projections to map tokens
to expert groups. For G groups:

```
R_t in R^{G x d},   R_t[g,j] ~ N(0, 1/d)    (fixed, not learned)
h_t(x) = argmax_g (R_t @ x)_g
```

Each hash function h_t maps every token to exactly one expert group. This
is equivalent to a Voronoi partition of R^d using G random prototype vectors
(the rows of R_t).

### 1.3 Notation

```
d          = embedding dimension (default 64)
G          = number of expert groups (default 8)
T          = number of hash tables (default 4)
k          = number of experts to select per token (default 2)
n_c        = capsules per group (default 32)
P          = G * n_c total capsules (default 256)
R_t        = t-th random projection matrix, R^{G x d}
h_t(x)     = argmax_g (R_t x)_g, the hash function for table t
```

---

## 2. Multi-Table Voting

### 2.1 Vote Accumulation

With T hash tables, each token x receives T votes:

```
v_g(x) = |{ t : h_t(x) = g }|   (vote count for group g)
```

Since each table votes for exactly one group, sum_g v_g(x) = T.

The top-k groups by vote count are selected. Ties are broken by
accumulated projection score:

```
s_g(x) = sum_{t=1}^{T} (R_t @ x)_g    (total projection score)
```

Combined ranking: `rank_g = v_g * C + s_g` where C is a large constant
ensuring vote count dominates.

### 2.2 Routing Weights

After selecting the top-k groups S(x), routing weights are computed
via softmax over accumulated scores (not votes):

```
w_g(x) = exp(s_g(x)) / sum_{g' in S(x)} exp(s_{g'}(x))   for g in S(x)
w_g(x) = 0                                                  for g not in S(x)
```

This gives data-dependent weights without any learned parameters. The
softmax over projection scores provides smooth gradients for the capsule
groups, even though the routing decision itself is non-differentiable.

### 2.3 Why Gradients Flow

The routing decision (which k groups to activate) is fixed for a given
input (determined by hash functions). But the routing weights w_g(x)
and the capsule computations CapsuleGroup_g(x) are both differentiable.

The output is:
```
LSH_MoE(x) = sum_{g in S(x)} w_g(x) * CapsuleGroup_g(x)
```

Gradients flow through:
1. w_g(x) -> s_g(x) -> R_t @ x (but R_t is fixed, so this gives
   gradients w.r.t. x only, which flow backward through the network)
2. CapsuleGroup_g(x) -> A_g, B_g (learned capsule params)

The key insight: LSH routing decides WHICH experts to use (non-differentiable
selection), but HOW MUCH to weight them (softmax over scores) and WHAT they
compute (capsule forward pass) are both differentiable. This is the same
principle as straight-through estimators and the Switch Transformer's
top-1 routing.

### 2.4 Worked Example (T=4, G=4, k=2, d=3)

```
Token x = [1.0, 0.5, -0.3]

R_1 = [[0.2, -0.1, 0.3],    R_1 @ x = [0.21, 0.19, -0.35, 0.15]
       [0.1,  0.2, 0.1],     -> h_1(x) = 0 (group 0 wins)
       [-0.3, 0.1, 0.2],
       [0.0,  0.3, 0.0]]

R_2 = [[-0.1, 0.2, 0.1],    R_2 @ x = [0.07, 0.38, 0.01, -0.16]
       [0.3,  0.1, -0.1],    -> h_2(x) = 1 (group 1 wins)
       [0.1, -0.2, 0.3],
       [0.0, -0.1, -0.2]]

R_3 = [[0.1,  0.3, 0.0],    R_3 @ x = [0.25, 0.09, 0.22, -0.08]
       [0.0,  0.1, 0.1],     -> h_3(x) = 0 (group 0 wins)
       [0.2,  0.0, -0.1],
       [-0.2, 0.1, 0.1]]

R_4 = [[0.3,  0.0, -0.2],   R_4 @ x = [0.36, 0.04, -0.17, 0.21]
       [0.0,  0.1, 0.0],     -> h_4(x) = 0 (group 0 wins)
       [-0.1,-0.1, 0.1],
       [0.2,  0.0, -0.1]]

Votes:  v_0 = 3, v_1 = 1, v_2 = 0, v_3 = 0
Top-2:  S(x) = {0, 1}

Scores: s_0 = 0.21 + 0.07 + 0.25 + 0.36 = 0.89
        s_1 = 0.19 + 0.38 + 0.09 + 0.04 = 0.70

Weights: w_0 = exp(0.89)/(exp(0.89)+exp(0.70)) = 0.548
         w_1 = exp(0.70)/(exp(0.89)+exp(0.70)) = 0.452

Output: 0.548 * CapsuleGroup_0(x) + 0.452 * CapsuleGroup_1(x)
```

---

## 3. Comparison with Learned Softmax Routing

### 3.1 Softmax Router (CapsulePool baseline)

```
s = W_r @ x,      W_r in R^{G x d}    (LEARNED)
w = softmax(s)
select top-k, renormalize
```

Parameters: G * d (learned routing weights)
FLOPs: 2 * G * d (matmul) + G (softmax) + G*log(G) (top-k)

### 3.2 LSH Router

```
for t in 1..T:
    s_t = R_t @ x,    R_t in R^{G x d}    (FIXED random)
    h_t = argmax(s_t)
accumulate votes, select top-k by votes
weights = softmax over accumulated scores
```

Parameters: 0 learned routing parameters
FLOPs: T * (2 * G * d + G) (projections + argmax) + G (accumulate) + k (softmax)
     = T * 2 * G * d + O(T*G)

### 3.3 Asymptotic Comparison

| Metric | Softmax | LSH (T tables) |
|--------|---------|-----------------|
| Routing params | G*d | 0 |
| Routing FLOPs | O(G*d) | O(T*G*d) |
| Selection quality | Learned, optimal | Data-independent, approximate |
| Calibration needed | Yes (100+ steps) | No |

At first glance, LSH routing costs T times more FLOPs than softmax.
However, the critical insight is about SCALING with expert count N:

**Softmax**: Must compute W_r @ x for ALL N experts -> O(N*d).
**LSH with fixed T**: Costs O(T*G*d) where G = N. At micro scale (G=8),
LSH costs T times more than softmax -- a disadvantage.

**Current implementation cost** (dense projection, no hash tables):

| N | Softmax FLOPs | LSH (T=4) FLOPs | Ratio |
|---|---------------|-------------------|-------|
| 8 | 1,024 | 4,096 | 4x more |
| 64 | 8,192 | 32,768 | 4x more |
| 256 | 32,768 | 131,072 | 4x more |
| 1024 | 131,072 | 524,288 | 4x more |

The ratio is always T:1 because both methods scale as O(N*d).

**Theoretical cost with binary LSH** (not implemented): With binary hash
functions h_t(x) = sign(r_t^T x) producing bit vectors, and pre-indexed
hash tables mapping bit patterns to expert buckets, routing would cost
O(T*d + T*B) where B is O(1) bucket lookup. This would give:

| N | Softmax FLOPs | Binary LSH (T=4) FLOPs | Ratio |
|---|---------------|-------------------------|-------|
| 8 | 1,024 | ~260 | 4x less |
| 64 | 8,192 | ~260 | 32x less |
| 256 | 32,768 | ~260 | 126x less |
| 1024 | 131,072 | ~260 | 504x less |

This scaling advantage requires a different hash function family (binary
LSH) and pre-indexed lookup tables. The current experiment validates
routing QUALITY, not routing SPEED at large N.

---

## 4. Parameter Count

### 4.1 Per Layer

```
Attention:  4 * d^2                           (wq, wk, wv, wo)
Routing:    0                                  (no learned routing params)
Capsules:   G * 2 * d * n_c                   (A and B matrices)
```

At d=64, G=8, n_c=32:
```
Attention:  4 * 64^2 = 16,384
Routing:    0
Capsules:   8 * 2 * 64 * 32 = 32,768
Layer total:                    49,152
```

### 4.2 Full Model (4 layers, V=28)

```
Per-layer:    49,152
All layers:   4 * 49,152 = 196,608
Embeddings:   28*64 + 32*64 = 3,840
RMSNorm:      5 * 64 = 320 (norm0 + 4 layers * 2 norms)
              Actually: norm0(64) + 4*(norm1(64) + norm2(64)) = 9*64 = 576
              But RMSNorm has 1 param per dim = 64 each
LM head:      28*64 = 1,792
Total:        ~202,112
```

Comparison to softmax CapsuleMoE:
```
Per-layer:    49,152 + 512 = 49,664 (adds G*d router)
All layers:   4 * 49,664 = 198,656
Total:        ~204,160
```

**LSH saves 2,048 params** (4 layers * 512 router params/layer).
At micro scale this is 1.0% of total. At macro scale with d=896, G=256:
softmax router = 256*896 = 229,376 params/layer. Savings grow linearly.

---

## 5. Expert Utilization

### 5.1 Expected Load Balance

Random projections produce a Voronoi partition of R^d. For truly
isotropic data, each Voronoi cell would capture approximately 1/G of the
data, giving E[f_g] = 1/G by symmetry. However, post-RMSNorm activations
are NOT isotropic -- RMSNorm normalizes L2 norm (projecting onto a sphere
of radius sqrt(d)) but does not equalize directional variance. The data
may concentrate in certain directions, causing systematic load imbalance.

With T tables, the vote distribution for a given token concentrates:
some groups get more votes than others (depending on which Voronoi cells
the token falls in). This creates sharper routing than a single table.

### 5.2 Empirical Observation

The experiment measures utilization as the fraction of tokens routed to
each expert. With random projections, utilization shows significant
imbalance -- evidence that post-RMSNorm activations are anisotropic.

Empirically measured standard deviation from uniform (1/G = 0.125):
- LSH T=4: std = 0.29-0.33 per layer (significant imbalance)
- Softmax (with balance loss): near-uniform utilization (balance loss forces it)
- Softmax (no balance loss): utilization 1.000 for all experts (top-k soft
  weighting assigns nonzero weight to all experts, but weights are sharp)

The imbalance in LSH routing does not hurt quality at micro scale because
the capsule groups adapt to their routing pattern during training. At
larger G, the imbalance could concentrate load on a few experts, creating
both quality and efficiency problems.

---

## 6. Assumptions

1. **Random projections preserve angular locality.** Guaranteed by the
   Johnson-Lindenstrauss lemma for d >= O(log(G)/epsilon^2). At d=64,
   G=8, this is well satisfied.

2. **Token embeddings are approximately norm-normalized after RMSNorm.**
   RMSNorm normalizes the L2 norm (projecting onto a sphere of radius
   sqrt(d)), but does NOT guarantee isotropy (equal variance in all
   directions). Empirically, expert utilization std is 0.29-0.33,
   indicating significant anisotropy. The E[f_g] = 1/G expectation
   from Section 5.1 does not hold in practice.

3. **Routing quality does not require learned routing parameters.**
   The experiment tests this directly. Revised finding: at micro scale
   (G=8, homogeneous data), all routing variants (uniform, LSH, softmax)
   produce statistically indistinguishable results. LSH matches softmax
   within noise, but so does uniform routing -- routing quality is
   irrelevant at this scale.

4. **More hash tables improve routing quality (diminishing returns).**
   Tested with T in {1, 2, 4, 8}. Finding: T=1 already matches
   softmax. No clear ordering among T values (T=2 is marginally best
   at -1.34% vs softmax_no_bal, but not significantly so with p=0.212).

5. **The hash function quality is independent of training progress.**
   Since projections are fixed, routing assignments are stable throughout
   training. The capsule groups learn to specialize to their assigned
   input distribution.
