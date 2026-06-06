# Huffman-Shaped Expert Tree: Mathematical Foundations

## 1. Problem Statement

Given a balanced binary tree of L leaf expert groups with uniform depth D = ceil(log2(L)),
reshape the tree so that the **expected routing depth** (average number of gate decisions
per token) is minimized, given empirical leaf activation frequencies.

This is exactly the **minimum redundancy coding** problem solved by Huffman (1952):
assign binary codes (tree paths) to symbols (leaf experts) such that the expected
codeword length (routing depth) is minimized.

---

## 2. Notation

```
L          = number of leaf expert groups
D_bal      = ceil(log2(L)), depth of balanced tree
f_l        = activation frequency of leaf l, sum_l f_l = 1
code_l     = binary path from root to leaf l (Huffman code)
d_l        = len(code_l), depth of leaf l in Huffman tree
H(f)       = -sum_l f_l * log2(f_l), Shannon entropy of frequencies (bits)
E[d]       = sum_l f_l * d_l, expected routing depth
n_c        = capsules per leaf group
d          = embedding dimension
B          = beam width (top-k leaves selected per token)
```

---

## 3. Huffman Tree Construction

### 3.1 Algorithm

1. Create L leaf nodes, each with frequency f_l
2. Insert all nodes into a min-heap ordered by frequency
3. While heap has more than 1 node:
   a. Pop two lowest-frequency nodes (left, right)
   b. Create parent node with freq = f_left + f_right
   c. Push parent back into heap
4. Remaining node is the root

**Complexity**: O(L log L) for construction.

### 3.2 Fundamental Properties

**Property 1 (Optimality)**: Huffman coding minimizes E[d] = sum_l f_l * d_l
over all binary prefix codes for the frequency distribution f.

**Proof**: By Huffman (1952). The greedy algorithm produces an optimal prefix code.
No other binary tree assignment of L leaves can achieve a lower expected depth
for the given frequencies.

**Property 2 (Shannon bound)**: H(f) <= E[d] < H(f) + 1

The expected depth is bounded below by the entropy and above by entropy plus one bit.
When frequencies are powers of 2 (dyadic), E[d] = H(f) exactly.

**Property 3 (Uniform degeneration)**: When f_l = 1/L for all l,
H(f) = log2(L) = D_bal, and the Huffman tree is balanced.

This means Huffman is a strict generalization of balanced trees. With uniform
frequencies, Huffman produces the same tree.

### 3.3 Internal Node Count

**Theorem**: A binary tree with L leaves has exactly L-1 internal nodes.

**Proof**: Every internal node has exactly 2 children. Total edges = 2*(L-1)
(each internal node contributes 2 edges). Total edges also = total nodes - 1
(tree property). So total nodes = 2*(L-1) + 1 = 2L-1, giving L-1 internal nodes.

For L=8: 7 internal nodes, same as balanced D=3 tree. The Huffman tree has the
same number of gates as the balanced tree -- only their arrangement differs.

---

## 4. Leaf Probability Distribution

### 4.1 Path Probability

Each internal node i has a gate: g_i(x) = sigmoid(w_i^T x + b_i).

The probability of reaching leaf l is:

```
P(leaf = l | x) = prod_{(i, bit) in path(l)} [
    g_i(x)        if bit = 0  (go left)
    1 - g_i(x)    if bit = 1  (go right)
]
```

where path(l) is the sequence of (node_index, direction_bit) pairs from root to leaf l.

### 4.2 Normalization (same proof as balanced tree)

**Theorem**: sum_l P(leaf = l | x) = 1 for all x.

**Proof**: By induction on tree structure (identical to balanced tree proof
in hierarchical_tree/MATH.md, Section 2.2). At each internal node, the
probability mass splits into p_left and (1-p_left), preserving the total.
The tree topology (balanced vs Huffman) does not affect this -- it holds
for any binary tree.

### 4.3 Worked Example (8 leaves, heavy skew)

```
Frequencies: f = [0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]
Huffman codes:
  L0 (f=0.35): code=11,   depth=2
  L1 (f=0.20): code=01,   depth=2
  L2 (f=0.15): code=101,  depth=3
  L3 (f=0.10): code=001,  depth=3
  L4 (f=0.08): code=000,  depth=3
  L5 (f=0.05): code=1000, depth=4
  L6 (f=0.04): code=10010, depth=5
  L7 (f=0.03): code=10011, depth=5

E[depth] = 0.35*2 + 0.20*2 + 0.15*3 + 0.10*3 + 0.08*3
         + 0.05*4 + 0.04*5 + 0.03*5
         = 0.70 + 0.40 + 0.45 + 0.30 + 0.24 + 0.20 + 0.20 + 0.15
         = 2.64

Balanced:  E[depth] = 3.0
Reduction: (3.0 - 2.64) / 3.0 = 12.0%
```

---

## 5. Expected Depth Reduction Analysis

### 5.1 Reduction as a Function of Entropy

The depth reduction R is:

```
R = (D_bal - E[d]) / D_bal
```

Since H(f) <= E[d] < H(f) + 1:

```
R > (D_bal - H(f) - 1) / D_bal
R < (D_bal - H(f)) / D_bal
```

For L=8, D_bal=3:
- Uniform (H=3.0): R = 0% (no reduction)
- Moderate (H=2.89): R ~ 2-3%
- Heavy (H=2.58): R ~ 12%
- Extreme (H=2.19): R ~ 26%

### 5.2 Zipf Scaling Law

For Zipf-distributed frequencies f_i proportional to 1/(i+1)^alpha:

| alpha | H (bits, L=8) | E[depth] | Reduction |
|-------|---------------|----------|-----------|
| 0.0   | 3.000         | 3.000    | 0.0%      |
| 0.5   | 2.910         | 2.939    | 2.0%      |
| 1.0   | 2.620         | 2.682    | 10.6%     |
| 1.5   | 2.173         | 2.209    | 26.4%     |
| 2.0   | 1.685         | 1.768    | 41.1%     |
| 3.0   | 0.909         | 1.276    | 57.5%     |

**Scaling with L**: Reduction increases with tree size because D_bal grows
logarithmically but E[d] stays close to H(f):

| L   | D_bal | E[d] (a=1.0) | Reduction |
|-----|-------|--------------|-----------|
| 8   | 3     | 2.68         | 10.6%     |
| 16  | 4     | 3.43         | 14.2%     |
| 32  | 5     | 4.17         | 16.6%     |
| 64  | 6     | 4.89         | 18.5%     |

At production scale with L=256 experts and Zipf(1.0) frequencies,
the expected reduction would be approximately 20-25%.

---

## 6. Parameter Count

### 6.1 Gates

Both balanced and Huffman trees with L leaves have L-1 internal nodes.
Each gate has d+1 parameters (weight vector + bias).

```
Gate params = (L-1) * (d+1)
```

At L=8, d=64: 7 * 65 = 455 params per layer. **Identical to balanced tree.**

### 6.2 Leaf Capsules

Each leaf has n_c capsules, each with 2*d params (a_i and b_i vectors).

```
Capsule params = L * n_c * 2 * d
```

At L=8, n_c=32, d=64: 8 * 32 * 2 * 64 = 32,768 per layer. **Identical to balanced.**

### 6.3 Total (d=64, L=8, n_c=32, 4 layers, V=28)

```
Per-layer:   455 + 32,768 + 16,384(attn) = 49,607
All layers:  4 * 49,607 = 198,428
Embeddings:  28*64 + 32*64 = 3,840
Norms:       5 * 64 = 320 (1 norm0 + 4 * (norm1 + norm2) = 9 norms, but each is d params)
LM head:     28 * 64 = 1,792
Total:       ~204,060
```

Note: The Huffman tree has 204,060 params vs balanced tree's 203,932. The small
difference (128 params) comes from implementation details in norm layer count.
Both are within 0.1% of each other.

---

## 7. Balance Loss for Non-Uniform Trees

### 7.1 Problem

The balanced tree uses L * sum(f_l^2) as balance loss, minimized at uniform
f_l = 1/L. For Huffman trees, we do NOT want uniform utilization -- we want
utilization proportional to the target frequencies.

### 7.2 KL Divergence Loss

We use KL divergence between actual utilization and target frequencies:

```
L_bal = L * KL(f_actual || f_target)
      = L * sum_l f_actual_l * log(f_actual_l / f_target_l)
```

Minimized at f_actual = f_target (KL = 0). This encourages the model to
maintain the frequency distribution that the Huffman tree was designed for.

---

## 8. Conditional Computation Savings

### 8.1 Balanced Tree (All Leaves Computed)

In the micro implementation, all L leaves are computed and non-selected ones
are masked. FLOPS:

```
FLOPS_balanced = L * FLOPS_leaf + routing_overhead
```

### 8.2 Huffman Tree with Conditional Computation

With true conditional computation (not implemented at micro scale), only
the B selected leaves and the gates along their paths are computed:

```
FLOPS_huffman = B * FLOPS_leaf + E[depth] * FLOPS_gate
```

The gate evaluations follow the beam search path, so on average only
E[depth] * B gates are evaluated (not all L-1).

### 8.3 Comparison

For L=8, B=2, with heavy skew (E[d]=2.64):
```
Balanced gates:  7 gate evaluations (all internal nodes)
Huffman gates:   ~2.64 * 2 = 5.28 gate evaluations (amortized)
Saving:          24% fewer gate evaluations
```

At L=64, B=2, with Zipf(1.0) (E[d]=4.89):
```
Balanced gates:  63 gate evaluations
Huffman gates:   ~4.89 * 2 = 9.78 gate evaluations
Saving:          84% fewer gate evaluations
```

Gate computation is small compared to leaf computation, but at large L
the routing cost becomes significant, and Huffman's O(H) scaling dominates
balanced tree's O(L) scaling.

---

## 9. Connection to Pruning

Dead capsule pruning (exp9: 57% dead, 0% quality loss) maps naturally onto
Huffman trees. Dead leaves have frequency f_l = 0 and do not appear in the
Huffman tree at all. The tree self-compresses:

```
Before pruning: L=8 leaves, some with f=0
After pruning:  L'=L-dead leaves, tree is rebuilt
                E[depth] decreases (fewer leaves = shallower tree)
                Gate count = L'-1 (fewer internal nodes)
```

This is the ExpertZIP insight: Huffman coding + dead expert removal =
automatic tree compression. The tree structure itself becomes the
compression artifact.

---

## 10. Key Assumptions

1. **Leaf frequencies are measurable.** We profile them from a trained model.
   At micro scale, frequencies are near-uniform (H=2.999/3.0 bits), providing
   no benefit. At macro scale with diverse data, we expect significant skew.

2. **Frequencies are stationary.** The Huffman tree is built once from profiled
   frequencies. If the data distribution shifts, the tree becomes suboptimal.
   The splay-tree experiment (exp_splay_adaptive_routing) addresses this.

3. **Conditional computation is available.** The depth reduction only translates
   to FLOP savings if the hardware can skip non-selected subtrees. At micro scale
   we compute all leaves regardless.

4. **Tree structure does not constrain learning.** The Huffman tree imposes a
   different coupling structure than the balanced tree (deep leaves have more
   ancestors to share gates with). Empirically, this does not degrade quality
   (delta < 0.5% across all tested distributions).
