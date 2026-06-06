# Split-and-Freeze Protocol: Mathematical Foundations

## 1. Setting

We operate on the binary capsule tree from hierarchical_tree (depth D=3, L=8
leaves, beam B=2). All notation follows MATH.md in that directory.

Additional notation:
```
S_L    = {0,...,3}     left subtree leaf indices
S_R    = {4,...,7}     right subtree leaf indices
G_L    = {1,3,4}       left subtree internal gate indices
G_R    = {2,5,6}       right subtree internal gate indices
g_0    = root gate (domain router after grafting)
theta_F = parameters of the frozen subtree
theta_G = parameters of the grafted subtree
theta_C = parameters eligible for calibration
```

---

## 2. Split Operation

### 2.1 Definition

Given a leaf CapsuleGroup at index l with parameters:
```
A_l in R^{n_c x d},    B_l in R^{d x n_c}
```

The split operation produces two children, each with half the capsules:
```
A_child0 = A_l[0:n_c/2, :] + epsilon_0,     epsilon_0 ~ N(0, sigma^2 I)
A_child1 = A_l[n_c/2:, :] + epsilon_1,      epsilon_1 ~ N(0, sigma^2 I)

B_child0 = B_l[:, 0:n_c/2] + delta_0,       delta_0 ~ N(0, sigma^2 I)
B_child1 = B_l[:, n_c/2:] + delta_1,         delta_1 ~ N(0, sigma^2 I)
```

where sigma = noise_scale (default 0.01) provides symmetry breaking.

### 2.2 Split Preserves Function Approximation (theoretical)

**Note**: This section describes a mathematical property of the split
operation defined in Section 2.1. The KC1 experiment did NOT test this
split operation. Instead, KC1 tested warm-start vs cold-start on
existing leaf pairs (see Section 2.3). The `split_leaf()` function is
implemented but was not invoked by the experiment runner.

Before split, the leaf output is:
```
f_l(x) = B_l @ ReLU(A_l @ x)
       = sum_{j=0}^{n_c-1} b_{l,j} * ReLU(a_{l,j}^T x)
```

After split (ignoring noise), the combined output of the two children
is exactly the parent output if both children are selected:
```
f_child0(x) + f_child1(x) = B_child0 @ ReLU(A_child0 @ x) + B_child1 @ ReLU(A_child1 @ x)
                           = sum_{j=0}^{n_c/2-1} b_{l,j} ReLU(a_{l,j}^T x)
                             + sum_{j=n_c/2}^{n_c-1} b_{l,j} ReLU(a_{l,j}^T x)
                           = f_l(x)
```

The noise breaks symmetry: without it, the parent gate between the two
children would have identical gradients for both sides, preventing
differentiation during fine-tuning.

### 2.3 What KC1 Actually Tests: Warm-Start vs Cold-Start

The KC1 experiment tests a related but distinct question from the split
operation above. Rather than dividing one parent leaf into two half-size
children, KC1 takes two existing sibling leaves (leaves 0 and 1, both
full-size, trained jointly during the base phase) and compares:

**Warm-start**: Keep the base-trained weights for the leaf pair as-is.
These leaves already approximate the data distribution from base training.
Fine-tuning refines this existing approximation.

**Cold-start**: Reinitialize the same leaf pair with random weights
(Xavier/Glorot). Fine-tuning must learn the approximation from scratch.

Both configurations have identical trainable parameters (33,028) and
training budget (200 steps). The hypothesis is that warm-start should
converge faster and/or to a better optimum because the inherited weights
encode useful features. The counter-hypothesis is that inherited features
may create suboptimal local minima, and random initialization might
escape them.

At micro scale, both methods converge to the same quality (-0.03% gap),
suggesting the fine-tuning budget is sufficient to erase the initialization
difference.

---

## 3. Freeze Operation

### 3.1 Definition

A subtree rooted at internal node i is frozen by setting:
```
d(theta_i) / dt = 0     for all theta_i in {gates, leaves} of subtree_i
```

In implementation, this means the parameters are excluded from the
computational graph during backpropagation.

### 3.2 Frozen Branch Degradation Analysis

When a new subtree is grafted alongside a frozen subtree, the frozen
branch's output is preserved exactly:
```
f_frozen(x) is unchanged (same weights, same computation)
```

However, the **effective contribution** of the frozen branch to the
model output changes because:

1. **Root gate redistribution**: The root gate g_0(x) determines how much
   probability mass flows to each subtree. After grafting, g_0 must learn
   to route domain A tokens left (to frozen subtree) and domain B tokens
   right (to grafted subtree). If g_0 fails to learn this routing, domain
   A tokens may be partially routed to the untrained grafted subtree,
   degrading quality.

2. **Beam selection competition**: With beam=2, the top-2 leaves by
   probability are selected. If the grafted subtree's leaves have
   artificially high probabilities (from random gate initialization),
   they may steal beam slots from the frozen subtree's leaves.

3. **Normalization effects**: Selected leaf weights are renormalized to
   sum to 1. If a grafted leaf is selected alongside a frozen leaf,
   the frozen leaf's effective weight is reduced by the renormalization.

### 3.3 Degradation Bound

Let P_A(x) be the probability mass flowing to the frozen left subtree
for a domain A input x. Before grafting, P_A(x) = 1 (all leaves are
trained for domain A). After grafting:
```
P_A(x) = g_0(x)    (probability of going left at root)
```

The degradation depends on how well g_0 learns to assign P_A(x) -> 1
for domain A inputs. With perfect calibration:
```
g_0(x) -> 1 for x in domain A
g_0(x) -> 0 for x in domain B
```

yielding zero degradation. With imperfect calibration:
```
E_{x in A}[1 - g_0(x)] = epsilon
```

where epsilon is the root gate's classification error on domain A.

### 3.4 Calibration Scope Analysis

The v2 diagnostic experiment measured degradation under three calibration scopes:

| Scope | Trainable params | Mean degradation |
|-------|-----------------|-----------------|
| Root gate only | 260 | +13.3% |
| All unfrozen gates | 1,040 | +2.5% |
| All unfrozen gates + leaves | 66,576 | +0.1% |

This shows that:
- Root gate alone cannot learn the domain routing decision reliably
- All-gates calibration brings degradation to the threshold boundary
- Full right-subtree calibration (allowing grafted leaves to adapt to
  the shared representation) eliminates degradation entirely

The key insight: the grafted leaves must adapt their output space to be
compatible with the frozen subtree's expectation of the shared attention
representation. The frozen leaves were trained with a specific attention
output distribution; the grafted leaves must learn to produce outputs
that, when combined by the router, don't interfere with the frozen
branch's contribution.

---

## 4. Parameter Count

### 4.1 Split Experiment (KC1)

Per seed, identical for both split and from-scratch:
```
Trainable params = 4 layers * (
    2 leaves * 2 * d * n_c/leaf    (split pair capsules)
    + 1 gate * (d + 1)             (parent gate)
)
= 4 * (2 * 2 * 64 * 32 + 65)
= 4 * (8192 + 65) = 4 * 8257 = 33,028
```

### 4.2 Freeze Experiment (KC2) — Training Phase

Right subtree + root gate:
```
Trainable params = 4 layers * (
    4 leaves * 2 * d * n_c/leaf    (right subtree capsules)
    + 3 gates * (d + 1)            (right subtree gates)
    + 1 gate * (d + 1)             (root gate)
)
= 4 * (4 * 2 * 64 * 32 + 4 * 65)
= 4 * (16384 + 260) = 4 * 16644 = 66,576
```

### 4.3 Freeze Experiment (KC2) — Calibration Phase

| Config | Params per layer | Total |
|--------|-----------------|-------|
| Root-only | 65 | 260 |
| All unfrozen gates | 4 * 65 = 260 | 1,040 |
| Right-tree full | 16,384 + 260 = 16,644 | 66,576 |

---

## 5. Worked Example (D=2, 4 leaves)

At D=2, d=64, n_c=16:
```
Tree:
       [gate_0]
       /      \
  [gate_1]  [gate_2]
  /    \    /    \
 L0    L1  L2    L3
```

**Split L0 into L0', L1'**:
```
A_L0 in R^{16 x 64}

A_L0' = A_L0[0:8, :] + N(0, 0.01^2)    in R^{8 x 64}
A_L1' = A_L0[8:16, :] + N(0, 0.01^2)   in R^{8 x 64}
```

New gate_1 routes between L0' and L1'. Total output of the split pair:
```
f_split(x) = g_1(x) * (B_L0' @ ReLU(A_L0' @ x))
           + (1 - g_1(x)) * (B_L1' @ ReLU(A_L1' @ x))
```

**Freeze {L0, L1, gate_1} (left subtree), graft new {L2, L3, gate_2}**:
```
Before grafting:
  g_0(x_A) ~ 0.7 (routes mostly left for domain A)
  All leaves trained for domain A

After grafting:
  Left: frozen {L0, L1, gate_1} — unchanged weights
  Right: fresh {L2, L3, gate_2} — random init
  g_0: needs retraining to route A->left, B->right

With root-only calibration (65 params):
  g_0(x_A) -> ? (may not converge to correct routing)

With right-tree calibration (2*2*64*16 + 2*65 = 4226 params):
  g_0(x_A) -> ~1, g_0(x_B) -> ~0
  L2, L3 adapt to domain B
  Degradation on domain A: ~0%
```

---

## 6. Assumptions

1. **Tree structure adequacy**: The binary tree depth (D=3) is sufficient
   for the split/freeze protocol at micro scale. Deeper trees (D=5+)
   would allow more granular splits.

2. **Capsule partition quality**: Splitting capsules by index (first half /
   second half) is arbitrary. Feature-based partitioning (e.g., by
   activation correlation clustering) might produce better splits but
   adds complexity.

3. **Noise scale**: sigma=0.01 is small enough to preserve parent quality
   but large enough for symmetry breaking. Not tuned.

4. **Calibration sufficiency**: 200-400 steps of right-tree calibration
   is sufficient at micro scale. At macro scale with more complex
   routing distributions, more calibration may be needed.

5. **Domain separability**: The binary a-m vs n-z domain split is
   well-separated in character space. With overlapping or adversarial
   domains, the routing problem becomes harder.
