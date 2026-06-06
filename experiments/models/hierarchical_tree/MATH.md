# Hierarchical Capsule Tree: Mathematical Foundations

## 1. Binary Tree of Capsule Groups

### 1.1 Tree Structure

A **binary capsule tree** of depth D contains:
- 2^D - 1 internal nodes, each with a learned binary gate
- 2^D leaf nodes, each containing a CapsuleGroup

Notation:
```
D          = tree depth (default 3)
L          = 2^D leaf groups (default 8)
I          = 2^D - 1 internal gates (default 7)
n_c        = capsules per leaf group (default 32)
P          = L * n_c total capsules (default 256)
d          = embedding dimension (default 64)
B          = beam width for top-down traversal (default 2)
```

Node indexing (0-based, breadth-first):
```
Root = node 0
Left child of node i  = 2*i + 1
Right child of node i = 2*i + 2
Leaf l corresponds to internal-node path determined by binary expansion of l
```

### 1.2 Internal Gate (Binary Routing Decision)

Each internal node i has a learned binary gate:

```
g_i(x) = sigmoid(w_i^T x + b_i),   w_i in R^d, b_i in R^1
```

g_i(x) in [0, 1] is the probability of routing left. The probability of
routing right is 1 - g_i(x).

Parameters per gate: d + 1 (weight vector + bias).
Total gate parameters per layer: I * (d + 1) = (2^D - 1) * (d + 1).

At D=3, d=64: 7 * 65 = 455 params per layer (2.8% overhead vs flat router's
8 * 64 = 512 params for G=8 softmax router).

### 1.3 Leaf CapsuleGroup

Each leaf l contains a CapsuleGroup with n_c capsules:

```
A_l in R^{n_c x d}    -- detector matrix (rows are a_{l,j}^T)
B_l in R^{d x n_c}    -- expansion matrix (columns are b_{l,j})
CapsuleGroup_l(x) = B_l @ ReLU(A_l @ x)
```

Parameters per leaf: 2 * d * n_c.
Total leaf parameters per layer: L * 2 * d * n_c = 2 * d * P.

At D=3, d=64, n_c=32: 8 * 2 * 64 * 32 = 32,768 params per layer.
This matches flat CapsulePool with G=8, P/G=32: 8 * 2 * 64 * 32 = 32,768.

---

## 2. Leaf Probability Distribution

### 2.1 Path Probability

The probability of reaching leaf l is the product of gate decisions along its
root-to-leaf path. For a depth-D tree, leaf l has a D-step path determined by
the binary representation of l.

Let bit(l, k) = (l >> (D-1-k)) & 1 be the k-th bit (from MSB) of l.
Let node(l, k) be the internal node at depth k on the path to leaf l.

```
P(leaf = l | x) = prod_{k=0}^{D-1} [
    g_{node(l,k)}(x)        if bit(l, k) = 0  (go left)
    1 - g_{node(l,k)}(x)    if bit(l, k) = 1  (go right)
]
```

### 2.2 Normalization Proof

The leaf probabilities form a valid probability distribution (sum to 1):

**Proof by induction on depth.**

Base case (D=1): Two leaves with P(left) = g_0(x), P(right) = 1 - g_0(x).
Sum = g_0(x) + 1 - g_0(x) = 1.

Inductive step: At depth k, the subtree rooted at node i has total probability
mass p_i (from the product of gate decisions above it). Its two children
receive p_i * g_i(x) and p_i * (1 - g_i(x)), summing to p_i. By induction,
each child's subtree distributes its mass correctly among its leaves.

Therefore sum_{l=0}^{L-1} P(leaf = l | x) = 1 for all x. QED.

### 2.3 Worked Example (D=2, 4 leaves)

```
Tree:
       [gate_0]
       /      \
  [gate_1]  [gate_2]
  /    \    /    \
 L0    L1  L2    L3

g_0 = 0.7, g_1 = 0.4, g_2 = 0.8

P(L0) = g_0 * g_1         = 0.7 * 0.4 = 0.28
P(L1) = g_0 * (1-g_1)     = 0.7 * 0.6 = 0.42
P(L2) = (1-g_0) * g_2     = 0.3 * 0.8 = 0.24
P(L3) = (1-g_0) * (1-g_2) = 0.3 * 0.2 = 0.06

Sum = 0.28 + 0.42 + 0.24 + 0.06 = 1.00  (verified)
```

---

## 3. Beam-Search Routing

### 3.1 Algorithm

Given beam width B, select the top-B leaves by probability:

```
1. Compute all L leaf probabilities P(leaf = l | x)
2. Select S = argmax_B { P(leaf = l | x) }   (top-B indices)
3. Renormalize: w_l = P(leaf = l | x) / sum_{l' in S} P(leaf = l' | x)
```

### 3.2 Output Computation

```
TreeMoE(x) = sum_{l in S} w_l * CapsuleGroup_l(x)
```

This is mathematically analogous to the flat CapsulePool with top-k routing,
except the weights w_l come from tree path probabilities instead of a softmax
over a flat linear projection.

### 3.3 Comparison: Tree vs Flat Routing

**Flat softmax routing (CapsulePool):**
```
s = W_r @ x,    W_r in R^{G x d}
w = softmax(s)
select top-k, renormalize
```
Router params: G * d

**Tree binary routing:**
```
for each internal node i: g_i(x) = sigmoid(w_i^T x + b_i)
leaf probs = product of gate decisions along paths
select top-B, renormalize
```
Router params: (2^D - 1) * (d + 1)

At D=3 (L=8), d=64:
- Flat: 8 * 64 = 512 params
- Tree: 7 * 65 = 455 params (11% fewer)

**Key difference**: Flat routing computes G independent scores and normalizes
via softmax. Tree routing computes D=3 sequential binary decisions per path.
The tree imposes a hierarchical structure: leaves that share an internal node
are structurally related (siblings compete for the same probability mass).
This is a structural prior that does not exist in flat routing.

---

## 4. Parameter Count

### 4.1 Per Layer

```
Attention:  4 * d^2         (wq, wk, wv, wo)
Tree gates: (2^D - 1)*(d+1) (binary gates)
Capsules:   2 * d * P       (A and B matrices for all leaves)
```

At D=3, d=64, n_c=32 (P=256):
```
Attention:  4 * 64^2 = 16,384
Tree gates: 7 * 65   = 455
Capsules:   2*64*256  = 32,768
Layer total:           49,607
```

### 4.2 Full Model (4 layers, V=27)

```
Per-layer:    49,607
All layers:   4 * 49,607 = 198,428
Embeddings:   2*27*64 + 32*64 = 5,504
Total:        203,932
```

Comparison to flat CapsuleMoE (G=8, P/G=32):
```
Per-layer:    16,384 + 512 + 32,768 = 49,664
All layers:   4 * 49,664 = 198,656
Total:        204,160
```

**Tree is 228 params smaller** (0.1% difference), because tree gates use
7*(d+1)=455 params vs flat router's 8*d=512. The capsule payload is identical.

---

## 5. Effective Sparsity

### 5.1 Active Capsule Fraction

With beam B=2, L=8 leaf groups:

```
active_ratio_L1 = B / L = 2/8 = 25%
```

After ReLU sparsity (~50% of capsules inactive):
```
effective_ratio = 0.25 * 0.5 = 12.5%
```

Compare flat CapsulePool (G=8, k=2):
```
active_ratio_L1 = k/G = 2/8 = 25%
effective_ratio = 0.25 * 0.5 = 12.5%
```

**Identical sparsity.** Both select 2 of 8 groups, both have ~50% ReLU sparsity.

### 5.2 FLOPs Per Token

Tree routing:
```
FLOPS_gates = I * (2*d + 1) = 7 * 129 = 903
             (each gate: d multiplies + d adds + bias + sigmoid)
FLOPS_leaf_probs = D * L = 3 * 8 = 24
             (multiply D gate outputs per leaf, L leaves)
```

Leaf computation (per selected leaf):
```
FLOPS_leaf = 2 * d * n_c + 2 * d * n_c = 4 * d * n_c
           = 4 * 64 * 32 = 8,192
```

Total:
```
FLOPS_total = FLOPS_gates + FLOPS_leaf_probs + B * FLOPS_leaf
            = 903 + 24 + 2 * 8,192 = 17,311
```

Compare flat (G=8, k=2):
```
FLOPS_router = 2 * d * G = 2 * 64 * 8 = 1,024
FLOPS_group  = 4 * d * (P/G) = 4 * 64 * 32 = 8,192
FLOPS_total  = 1,024 + 2 * 8,192 = 17,408
```

**Routing cost is comparable** (903 vs 1,024 for routing; identical for compute).
Both are dominated by the B=k=2 leaf/group computations.

Note: As in capsule_moe, the micro implementation computes all leaves and
multiplies non-selected ones by zero. Actual FLOP savings require conditional
computation. At large L (64+ leaves), tree routing has a structural advantage:
the tree allows O(D * beam) gate evaluations instead of O(L) score computations,
reducing routing cost from linear to logarithmic in the number of experts.

---

## 6. Auxiliary Losses

### 6.1 Balance Loss

Identical to flat CapsulePool but over leaf probabilities:

```
f_l = mean_{b,t} P(leaf = l | x_{b,t})
L_bal = L * sum_{l=0}^{L-1} f_l^2
```

Minimum at uniform (f_l = 1/L for all l): L_bal_min = 1.

### 6.2 Gate Entropy Loss (Optional)

Encourages sharp binary decisions at each gate:

```
H(leaf distribution) = -sum_l P(leaf=l) * log(P(leaf=l))
L_entropy = mean_{b,t} H(leaf distribution | x_{b,t})
```

Minimizing this pushes toward deterministic routing (low entropy).
We add this with coefficient 0.1 relative to balance loss.

### 6.3 Total Training Loss

```
L_total = L_CE + alpha * sum_{layers} (L_bal + 0.1 * L_entropy)
```

with alpha = 0.01 (standard).

---

## 7. Composition by Weight Averaging

### 7.1 Protocol

The tree structure naturally supports composition. Given domain-specific
fine-tuned trees (attention frozen), compose by:

```
1. Pretrain base model with tree architecture on all data
2. Fine-tune tree gates + leaf capsules per domain (attention frozen)
3. Weight-average domain-specific parameters:
   w_composed = (1/N) * sum_{d=1}^{N} w_d
4. Calibrate gates on mixed data (~100 steps, leaves frozen)
```

### 7.2 Why Tree Composition May Differ From Flat

In flat routing, composition averaging blends all group weights uniformly.
The router must then learn to separate domains from scratch during calibration.

In tree routing, the hierarchical structure provides a natural decomposition:
- Averaging the root gate blends the coarsest split
- Averaging deeper gates blends finer distinctions
- The tree structure itself encodes a prior on which experts are "related"
  (siblings share a parent gate, so they naturally compete)

This hierarchical averaging may preserve more structure than flat averaging,
because related experts (siblings in the tree) are blended together while
unrelated experts (different subtrees) are blended independently.

### 7.3 Composition Gap Analysis

Empirically measured at 3 seeds:
```
Flat:  joint=0.5200, composed=0.5214, gap=+0.26%
Tree:  joint=0.5186, composed=0.5195, gap=+0.17%
```

The tree shows a smaller composition gap (+0.17% vs +0.26%), suggesting
the hierarchical structure provides a marginally better function-space
anchor during composition. However, this difference (0.09pp) is within
noise at 3 seeds and should not be over-interpreted.

---

## 8. Routing Topology: Structural Prior Analysis

### 8.1 Expert Coupling

In flat routing, all G experts compete equally in the softmax. Changing
one expert's relevance affects all others (softmax normalization).

In tree routing, experts are coupled only through their shared gates:
- Leaf 0 and Leaf 1 share gate_1 (direct siblings)
- Leaf 0 and Leaf 4 share only gate_0 (distant cousins)
- Changing Leaf 4's behavior only affects Leaf 0 through gate_0

This is a **locality prior**: experts that are structurally close in the
tree compete more strongly. This mirrors the empirical finding that Layer 0
capsules are shared/redundant (behavioral Jaccard=0.527) while deeper layers
specialize (J<0.05) -- the tree makes this coarse-to-fine structure explicit.

### 8.2 Routing Decision Cost Scaling

| Expert count | Flat (softmax) | Tree (binary gates) |
|---|---|---|
| 8   | 8*d = 512     | 7*(d+1) = 455      |
| 16  | 16*d = 1,024  | 15*(d+1) = 975     |
| 64  | 64*d = 4,096  | 63*(d+1) = 4,095   |
| 256 | 256*d = 16,384| 255*(d+1) = 16,575 |

At small L, costs are similar. At large L, the tree gains an advantage if
only active paths are computed: beam-B traversal evaluates only B*D gates
(not all I gates), reducing routing cost from O(L*d) to O(B*D*d) = O(B*log(L)*d).

At L=256, B=2, D=8: tree evaluates ~16 gates vs 256 flat scores.
