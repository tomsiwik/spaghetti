# Splay-Tree Adaptive Routing: Mathematical Foundations

## 1. Splay Bias Mechanism

### 1.1 Setting

We operate on a binary capsule tree of depth D with L = 2^D leaf capsule
groups and I = 2^D - 1 internal gates (as in the parent hierarchical_tree
experiment). Each internal gate i produces:

```
g_i(x) = sigmoid(w_i^T x + b_i),   w_i in R^d, b_i in R^1
```

The splay extension adds a non-parametric bias correction:

```
g_i^splay(x) = sigmoid(w_i^T x + b_i + beta_i)
```

where beta_i in R is the splay bias for gate i, updated at runtime (not by
gradient descent).

### 1.2 Notation

```
D      = tree depth (default 3)
L      = 2^D leaf groups (default 8)
I      = 2^D - 1 internal gates (default 7)
f_l    = EMA frequency of leaf l selection, sum_l f_l = 1
alpha  = splay strength (controls bias magnitude)
gamma  = EMA decay factor (default 0.95)
n_c    = capsules per leaf group (default 32)
d      = embedding dimension (default 64)
B      = beam width (default 2)
```

### 1.3 Frequency Tracking

After each forward pass, leaf frequencies are updated via exponential
moving average:

```
f_l <- gamma * f_l + (1 - gamma) * p_l^{batch}
```

where p_l^{batch} = mean_{b,t} P(leaf = l | x_{b,t}) is the mean
selection probability of leaf l across the current batch.

After update, renormalize: f_l <- f_l / sum_l f_l.

Initial condition: f_l = 1/L for all l (uniform).

### 1.4 Splay Bias Computation

For each internal node i, define left and right subtree frequencies:

```
F_left(i) = sum_{l in left_subtree(i)} f_l
F_right(i) = sum_{l in right_subtree(i)} f_l
```

The splay bias is the log-odds of subtree frequency:

```
beta_i = alpha * log(F_left(i) / F_right(i))
```

**Justification**: For a sigmoid gate, adding log(p/q) to the logit
approximately multiplies the output probability by p/q. This shifts the
gate toward the more frequently used subtree, which is exactly the splay
tree's "move to root" operation in soft form.

### 1.5 Effect on Gate Output

The modified gate output is:

```
g_i^splay(x) = sigmoid(logit_i(x) + alpha * log(F_left(i) / F_right(i)))
```

When alpha = 0, this reduces to the standard gate (no splay).
When alpha -> infinity, the gate becomes deterministic toward the higher-
frequency subtree.

---

## 2. Connection to Splay Trees

### 2.1 Classical Splay Tree

A splay tree (Sleator & Tarjan, 1985) is a self-adjusting binary search
tree where each access triggers rotations that move the accessed element
to the root. Key property: amortized O(log n) per operation, with optimal
working-set performance (recently accessed elements are near the root).

### 2.2 Soft Splay Analogy

Our mechanism does NOT perform tree rotations (which would break learned
weights). Instead, it achieves the same effect through soft bias:

| Splay tree | Our mechanism |
|---|---|
| Zig/zig-zig/zig-zag rotations | Gate bias correction |
| Move accessed node to root | Increase frequency -> increase bias |
| Amortized O(log n) | Same tree depth, but higher probability paths |
| Working-set optimality | EMA frequency tracks working set |

The analogy is that in a classical splay tree, frequently accessed nodes
get shorter paths (fewer comparisons). In our mechanism, frequently
selected leaves get higher gate probabilities along their paths (lower
effective routing cost, since the correct path is more certain).

### 2.3 Working-Set Property

Define the working set W(t, s) as the set of distinct leaves accessed in
the last s routing decisions. The classical splay tree's working-set
theorem guarantees:

```
Total access cost <= O(n log n + sum_t log W(t, s_t))
```

where s_t is the number of accesses since the last access to the t-th
element.

Our soft splay approximation: leaves in the recent working set have higher
f_l (due to EMA), hence higher gate biases along their paths, hence higher
P(leaf = l | x). This means the beam-search routing is more likely to find
them without exploring the full tree.

---

## 3. Parameter Count

### 3.1 Parametric Cost

The splay biases beta_i are computed from runtime statistics (f_l), not
stored as learned parameters. Therefore:

```
Splay params = Hierarchical tree params (identical)
             = (2^D - 1) * (d + 1) + L * 2 * d * n_c  per layer
```

At D=3, d=64, n_c=32:
```
Gates:    7 * 65 = 455
Capsules: 8 * 2 * 64 * 32 = 32,768
Layer:    49,607 (with attention)
Total:    203,932 (4 layers + embeddings)
```

### 3.2 Runtime State Cost

Additional runtime state per layer:
```
Leaf frequencies: L floats = 8 floats
Gate biases:      I floats = 7 floats
Subtree sums:     2 * I floats = 14 floats (computed, not stored)
```

Total: 15 floats per layer = 60 floats for 4 layers = 240 bytes at fp32.
This is negligible (0.06% of model parameters).

---

## 4. Computational Overhead

### 4.1 Per-Step Cost

After each forward pass, the splay update requires:

```
1. Compute mean leaf probabilities: O(B * T * L) -- already available
2. EMA update: O(L) multiplies and adds
3. Normalize frequencies: O(L)
4. Compute subtree sums: O(I * L/2) = O(L * log L) worst case
5. Compute log-odds: O(I) logs and divides
6. Set gate biases: O(I) assigns
```

Total: O(L * log L) per layer per step.

At L=8: ~40 floating point operations per layer per step.
Compared to forward pass FLOPs per layer: ~50,000.
**Splay overhead: <0.1% of forward pass FLOPs.**

### 4.2 Why Measured Overhead is Higher

The measured overhead (+51.5%) is NOT from computation but from Python
interpreter overhead:
- `.tolist()` and `.item()` calls force MLX graph synchronization
- Python-level loops over 8 leaves and 7 gates
- dict/list allocation for diagnostics

In a production C++ implementation, the overhead would be negligible.
At micro scale with MLX's lazy evaluation, the synchronization cost
dominates.

---

## 5. Domain Shift Analysis

### 5.1 Adaptation Dynamics

When data distribution shifts from domain A to domain B:

1. **Step 0**: Reset f_l = 1/L, beta_i = 0 (on_domain_switch)
2. **Step 1-k**: New domain B data flows through tree, leaf selection
   probabilities determined by learned gates (no splay bias yet)
3. **Step k+**: EMA frequencies reflect domain B's routing pattern;
   splay biases begin shifting gates toward B's preferred leaves
4. **Equilibrium**: f_l converges to domain B's stationary leaf distribution

The convergence speed is controlled by gamma (EMA decay):
- gamma = 0.95: half-life = log(0.5) / log(0.95) = 13.5 steps
- gamma = 0.90: half-life = 6.6 steps
- gamma = 0.99: half-life = 69 steps

### 5.2 Comparison to Gradient-Based Adaptation

The static tree adapts through gradient descent on gate parameters:

```
w_i <- w_i - lr * dL/dw_i
```

This is a slow adaptation (requires many gradient steps to shift gate
behavior). The splay mechanism adds a fast adaptation channel:

```
beta_i <- alpha * log(F_left(i) / F_right(i))  [immediate, no gradient]
```

The two channels operate at different timescales:
- Splay: adapts in ~half-life = 13.5 steps (statistical accumulation)
- Gradient: adapts in ~100+ steps (optimization convergence)

Hypothesis: splay provides faster initial adaptation, while gradients
provide better final quality. The combination should dominate either alone.

### 5.3 Why It Did Not Work at Micro Scale

At D=3 with L=8 leaves and domains a-m vs n-z (which share similar
character distributions), several factors suppress the splay advantage:

1. **Small tree**: 8 leaves means the routing "search space" is tiny.
   Gradient descent can recalibrate 7 gates quickly (~50-100 steps).
   Splay's speed advantage is negligible.

2. **Similar domains**: a-m and n-z names share most character frequencies.
   Domain shift is mild, so there is little routing restructuring needed.

3. **Gradient descent is already fast**: At lr=3e-3 with Adam, 200 steps
   is sufficient to recalibrate gates from scratch. The splay bias
   provides ~13 steps of "preview" but this is a small fraction of
   the 200-step adaptation budget.

4. **Python overhead**: The implementation-level overhead (51.5%) masks
   any computational savings from better routing.

---

## 6. Worked Example

### 6.1 Setup

D=2, L=4 leaves, alpha=1.0, gamma=0.9.

Initial: f = [0.25, 0.25, 0.25, 0.25], all beta_i = 0.

### 6.2 After Batch Where Leaf 0 Dominates

Batch selection: p_batch = [0.5, 0.2, 0.2, 0.1]

EMA update:
```
f_0 = 0.9 * 0.25 + 0.1 * 0.5 = 0.275
f_1 = 0.9 * 0.25 + 0.1 * 0.2 = 0.245
f_2 = 0.9 * 0.25 + 0.1 * 0.2 = 0.245
f_3 = 0.9 * 0.25 + 0.1 * 0.1 = 0.235
```

Normalize: sum = 1.0 (already normalized in this case).

Splay biases for tree:
```
       [gate_0]
       /      \
  [gate_1]  [gate_2]
  /    \    /    \
 L0    L1  L2    L3
```

Gate 0: F_left = f_0 + f_1 = 0.520, F_right = f_2 + f_3 = 0.480
  beta_0 = 1.0 * log(0.520 / 0.480) = 0.080

Gate 1: F_left = f_0 = 0.275, F_right = f_1 = 0.245
  beta_1 = 1.0 * log(0.275 / 0.245) = 0.115

Gate 2: F_left = f_2 = 0.245, F_right = f_3 = 0.235
  beta_2 = 1.0 * log(0.245 / 0.235) = 0.042

Effect: Leaf 0's path (left at gate_0, left at gate_1) gets +0.080 + 0.115
= +0.195 total logit boost. Leaf 3's path gets -0.080 - 0.042 = -0.122.
The most frequently used leaf gets the strongest boost.

### 6.3 After 10 Steps of Leaf 0 Dominance

With gamma=0.9 and consistent p_batch = [0.5, 0.2, 0.2, 0.1]:

```
f converges toward p_batch:
f_0 -> 0.5, f_1 -> 0.2, f_2 -> 0.2, f_3 -> 0.1

beta_0 -> log(0.7 / 0.3) = 0.847
beta_1 -> log(0.5 / 0.2) = 0.916
beta_2 -> log(0.2 / 0.1) = 0.693
```

Leaf 0 total boost: 0.847 + 0.916 = 1.763 logits.
This is a significant shift: sigmoid(0 + 1.763) = 0.854 (strong left bias).

---

## 7. Assumptions

1. **EMA is a good proxy for working-set frequency**: Assumes data
   distributions change slowly relative to EMA half-life. Rapid oscillation
   between domains would confuse the frequency tracker.

2. **Log-odds is the right correction form**: Assumes sigmoid gates respond
   well to additive logit corrections. This is exact for the sigmoid
   function but interacts with learned gate weights in complex ways.

3. **Domain shift is the primary non-stationarity**: The experiment tests
   abrupt domain switch. Gradual drift, cyclic patterns, or random
   non-stationarity are not tested.

4. **Splay biases and gradient updates are compatible**: The splay bias
   changes the loss landscape that gradients optimize over. If the bias
   shifts the gate significantly, gradient updates may fight the bias
   (conflicting optimization directions). This interaction is not analyzed.
