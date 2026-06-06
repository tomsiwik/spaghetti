# KD-Tree Feature-Space Partitioning: Mathematical Foundations

## 1. KD-Tree Split Structure

### 1.1 Notation

```
D          = tree depth (default 3)
L          = 2^D leaf groups (default 8)
I          = 2^D - 1 internal split nodes (default 7)
n_c        = capsules per leaf group (default 32)
P          = L * n_c total capsules (default 256)
d          = embedding dimension (default 64)
B          = beam width for top-k selection (default 2)
T          = temperature parameter (annealed during training)
T_init     = initial temperature (default 1.0)
T_max      = maximum temperature (default 10.0)
```

### 1.2 KD-Tree Split Node

Each internal node i has:
- A projection vector v_i in R^d (learned, not constrained to unit norm)
- A threshold tau_i in R (learned split point)

The split decision is:

```
p_left(x; i) = sigmoid(T * (v_i^T x - tau_i))
```

Where T is the temperature parameter. This is a **soft binary space partition**:
- T = 0: p_left = 0.5 for all x (uniform, no routing information)
- T = 1: standard sigmoid (smooth, gradients flow well)
- T -> infinity: step function at v_i^T x = tau_i (hard partition)

**Comparison with hierarchical_tree (sigmoid gate):**
```
hierarchical_tree: p_left(x; i) = sigmoid(w_i^T x + b_i)
kd_tree_routing:   p_left(x; i) = sigmoid(T * (v_i^T x - tau_i))
```

These are mathematically equivalent when T=1 and b_i = -tau_i. The key difference
is the temperature parameter T, which is extrinsic to the weights and controls
sharpness independently of the learned projection.

### 1.3 Parameters per Split Node

```
v_i in R^d      -- projection direction (d parameters)
tau_i in R^1    -- threshold (1 parameter)
```

Total per node: d + 1 = 65 (at d=64).
Total per layer: I * (d + 1) = 7 * 65 = 455.
Identical to hierarchical_tree (which uses Linear(d, 1, bias=True) = d + 1 params).

---

## 2. Leaf Probability Distribution

### 2.1 Path Probability (identical to hierarchical_tree)

The probability of reaching leaf l is the product of split decisions along its
root-to-leaf path:

```
P(leaf = l | x, T) = prod_{k=0}^{D-1} [
    p_left(x; node(l,k))        if bit(l, k) = 0
    1 - p_left(x; node(l,k))    if bit(l, k) = 1
]
```

where bit(l, k) = (l >> (D-1-k)) & 1 is the k-th routing bit for leaf l.

### 2.2 Temperature Effect on Leaf Distribution

As T increases, the leaf distribution concentrates:

```
T = 1:    H(leaf | x) ≈ D * log(2) = 2.079 nats  (near uniform)
T = 5:    H(leaf | x) << D * log(2)                (concentrated)
T = 10:   H(leaf | x) ≈ 0                          (near deterministic)
```

Normalized entropy = H(leaf | x) / log(L). At T=10 with our micro experiments:
```
Layer 0: 0.048 (nearly deterministic routing)
Layer 2: 0.016 (extremely sharp)
```

Compare hierarchical_tree at the same scale: 0.745 (near uniform).

### 2.3 Geometric Interpretation

At T -> infinity, each leaf owns a **convex polytope** in R^d defined by:

```
Region(l) = { x in R^d : for all k in {0..D-1},
    (-1)^{bit(l,k)} * (v_{node(l,k)}^T x - tau_{node(l,k)}) > 0 }
```

These regions are non-overlapping and cover R^d (binary space partition).
This is the KD-tree property: each split creates a hyperplane, and the
intersection of D hyperplanes defines each leaf's territory.

---

## 3. Temperature Annealing Schedule

### 3.1 Schedule

```
T(s) = {
    T_init                                          if s < 0.2 * S_total
    T_init + (T_max - T_init) * (s - 0.2*S) / (0.8*S)   otherwise
}
```

Where s is the training step and S is total steps.

Rationale:
- First 20%: warm-up with soft routing (T=1). Gradients flow to all leaves,
  allowing the split directions to explore before committing.
- Remaining 80%: linear ramp to T_max. Gradually sharpens routing, forcing
  commitment to partition structure.

### 3.2 Gradient Analysis

The gradient of the split probability with respect to v_i:

```
dp/dv_i = T * sigmoid'(T * (v_i^T x - tau_i)) * x
        = T * p * (1 - p) * x
```

At high T, p is near 0 or 1 for most tokens, so p*(1-p) is near 0.
Only tokens near the decision boundary (v_i^T x ≈ tau_i) receive gradient.
This concentrates learning on the boundary refinement.

---

## 4. Split Diversity Loss

### 4.1 Motivation

Without regularization, multiple split nodes could learn the same projection
direction, producing degenerate partitions where some leaves are empty.

### 4.2 Formulation

For sibling split nodes at the same depth level:

```
L_div = (1 / |pairs|) * sum_{(i,j) siblings} cos^2(v_i, v_j)
```

where cos(v_i, v_j) = (v_i^T v_j) / (||v_i|| * ||v_j|| + eps).

Minimizing L_div encourages orthogonal split directions at each level,
producing a more balanced partition of the embedding space.

### 4.3 Total Auxiliary Loss

```
L_aux = alpha * sum_layers (L_bal + 0.1 * L_entropy + 0.05 * L_div)
```

with alpha = 0.01.

---

## 5. Parameter Count

### 5.1 Per Layer

```
Attention:    4 * d^2            = 4 * 64^2    = 16,384
Split nodes:  (2^D - 1) * (d+1) = 7 * 65      = 455
Capsules:     2 * d * P          = 2 * 64 * 256 = 32,768
Layer total:                                    = 49,607
```

### 5.2 Full Model (4 layers, V=28)

```
Per-layer:    49,607
All layers:   4 * 49,607 = 198,428
Embeddings:   28*64 + 32*64 = 3,840
LM head:      28*64 = 1,792
Total:        ~203,932
```

**Identical to hierarchical_tree** (203,932 params). The split node and
sigmoid gate have the same parameter count: d + 1 per node.

---

## 6. Effective Sparsity and FLOPs

### 6.1 Active Computation

Same as hierarchical_tree: beam B=2 selects 2 of 8 leaves (25% active).
With ReLU sparsity: effective ~12.5%.

### 6.2 Routing FLOPs

At T -> infinity (inference), routing is:
```
For each of D=3 levels:
    project: v_i^T x = d multiplies + d adds
    compare: v_i^T x > tau_i = 1 compare
    branch:  deterministic (no softmax, no top-k)
```

Total: D * (2d + 1) = 3 * 129 = 387 FLOPs per token.
No top-k sort needed (hard binary decisions).

Compare:
- hierarchical_tree: 903 FLOPs (computes all 7 gates + leaf probs)
- flat softmax: 1,024 FLOPs (G scores + softmax + top-k sort)

At inference with hard splits, KD-tree has the lowest routing cost because
it only evaluates D=3 split nodes on the active path (not all 7).

---

## 7. Worked Example (D=2, d=4, T=5)

```
Split nodes:
    v_0 = [0.5, -0.3, 0.1, 0.8], tau_0 = 0.2
    v_1 = [0.1, 0.7, -0.2, 0.1], tau_1 = -0.1
    v_2 = [-0.4, 0.2, 0.6, -0.1], tau_2 = 0.3

Input x = [1.0, 0.5, -0.3, 0.2]:
    v_0^T x = 0.5 - 0.15 - 0.03 + 0.16 = 0.48
    p_left_0 = sigmoid(5 * (0.48 - 0.2)) = sigmoid(1.4) = 0.802
    -> go LEFT with probability 0.802

    v_1^T x = 0.1 + 0.35 + 0.06 + 0.02 = 0.53
    p_left_1 = sigmoid(5 * (0.53 - (-0.1))) = sigmoid(3.15) = 0.959
    -> go LEFT with probability 0.959

Leaf probabilities:
    P(L0) = 0.802 * 0.959 = 0.769
    P(L1) = 0.802 * 0.041 = 0.033
    P(L2) = 0.198 * p_left_2
    P(L3) = 0.198 * (1-p_left_2)

    v_2^T x = -0.4 + 0.1 - 0.18 - 0.02 = -0.50
    p_left_2 = sigmoid(5 * (-0.50 - 0.3)) = sigmoid(-4.0) = 0.018

    P(L2) = 0.198 * 0.018 = 0.004
    P(L3) = 0.198 * 0.982 = 0.194

Sum = 0.769 + 0.033 + 0.004 + 0.194 = 1.000

Top-2 selection: L0 (0.769), L3 (0.194)
Renormalized: L0 = 0.769/0.963 = 0.799, L3 = 0.194/0.963 = 0.201

Entropy: -(0.769*log(0.769) + 0.033*log(0.033) + 0.004*log(0.004) + 0.194*log(0.194))
       = 0.753 nats
Normalized: 0.753 / log(4) = 0.543
```

Note how temperature T=5 already produces sharp routing (76.9% to top leaf),
while the hierarchical_tree at the same scale has entropy 0.745.

---

## 8. Split Direction Analysis

### 8.1 Concentration Metric

For each split node, the "concentration" measures how axis-aligned the projection is:

```
concentration(v_i) = max_j |v_{i,j}| / sum_j |v_{i,j}|
```

At d=64:
- Perfectly axis-aligned: concentration = 1.0 (all weight on one dimension)
- Uniformly spread: concentration = 1/64 = 0.016
- Observed: 0.04-0.07 (slightly above uniform)

The learned projections are NOT axis-aligned at micro scale. They use
distributed representations across many dimensions, similar to the
hierarchical_tree's sigmoid gates. The KD-tree constraint (single projection
direction) is structurally identical to the sigmoid gate when both learn
free projection vectors.

### 8.2 Implications

The KD-tree's advantage over hierarchical_tree is NOT in the projection
structure (both learn arbitrary hyperplanes) but in the temperature annealing,
which forces sharp routing independent of the learned weights. The
hierarchical_tree could achieve equally sharp routing if given the same
temperature mechanism.
