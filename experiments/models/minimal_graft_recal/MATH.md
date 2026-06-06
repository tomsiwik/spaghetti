# Minimal Graft Recalibration: Mathematical Foundations

## 1. Gate Topology After Grafting

### 1.1 Notation (inherited from subtree_grafting/MATH.md)

```
D          = tree depth (3)
L          = 2^D leaf groups (8)
I          = 2^D - 1 internal gates (7)
n_c        = capsules per leaf group (32)
d          = embedding dimension (64)
B          = beam width (2)
N          = number of domains (2)
n_layer    = transformer layers (4)
```

### 1.2 Gate Classification After Grafting

In a depth-3 binary tree, the 7 internal gates decompose into three
functional categories after subtree grafting:

```
                [g_0]  <-- ROOT: domain router
               /      \
         [g_1]          [g_2]  <-- GRAFT-POINT: top of each subtree
         /    \          /    \
      [g_3] [g_4]    [g_5] [g_6]  <-- DEEP: within-subtree routing
      / \    / \      / \    / \
     L0 L1 L2 L3    L4 L5 L6 L7
```

| Category | Gates | Role After Grafting | Source |
|----------|-------|---------------------|--------|
| Root | {0} | Routes between domain subtrees | Base model |
| Graft-point | {1, 2} | Top-level split within each domain subtree | Domain-specific |
| Deep | {3, 4, 5, 6} | Fine-grained routing within subtrees | Domain-specific |

### 1.3 The Interface Mismatch Hypothesis

After grafting, the **interface** between the root gate and each subtree
is the critical failure point. The root gate g_0 was trained on the
base model's joint distribution. The graft-point gates g_1, g_2 were
trained on domain-specific distributions. The function they compose is:

```
P(leaf = l | x) = g_0(x) * g_{parent(l)}(x) * g_{grandparent(l)}(x)
```

The root-to-graft-point interface is where the distribution mismatch
concentrates. The deep gates (g_3-g_6) operate entirely within a
domain subtree on inputs already filtered by g_1 or g_2 -- their
input distribution is less affected by grafting.

---

## 2. Recalibration Strategies

### 2.1 Selective Gate Sets

We define three recalibration gate sets, each a strict superset of
the previous:

```
S_root       = {0}             (1 gate, root only)
S_graft      = {0, 1, 2}      (3 gates, root + graft-point)
S_all        = {0, 1, ..., 6} (7 gates, all internal)
```

### 2.2 Trainable Parameter Counts

Each gate is a linear projection: g_i(x) = sigma(w_i^T x + b_i),
where w_i in R^d, b_i in R^1.

Per gate: d + 1 = 65 parameters.
Per layer: |S| * 65 parameters.
Total (4 layers):

| Strategy | Gates | Params/layer | Total params |
|----------|-------|-------------|-------------|
| Root-only | 1 | 65 | 260 |
| Root+graft-point | 3 | 195 | 780 |
| All-gates | 7 | 455 | 1,820 |

### 2.3 Cost Reduction

The ratio of calibration parameters:

```
cost(S_graft) / cost(S_all) = 780 / 1820 = 3/7 = 0.429
cost(S_root) / cost(S_all) = 260 / 1820 = 1/7 = 0.143
```

Root+graft-point recalibration uses 42.9% of all-gates params (2.3x
cheaper). Root-only uses 14.3% (7.0x cheaper).

---

## 3. Expected Behavior by Gate Category

### 3.1 Root Gate (g_0)

After grafting, the root gate must learn a new function: route tokens
to the correct domain subtree. Its base-model weights were trained on
the joint distribution, but now the left and right subtrees represent
different domains. Recalibrating g_0 is necessary but potentially
insufficient -- it cannot compensate for internal subtree mismatches.

### 3.2 Graft-Point Gates (g_1, g_2)

Each graft-point gate was trained on domain-specific data. After
grafting, g_1 receives the full joint input distribution (filtered only
by g_0), not just domain A inputs. It must handle:

1. Domain A inputs: continue routing as trained (no change needed)
2. Domain B inputs: handle gracefully even though never trained on them

The graft-point gates are the most critical recalibration targets
because they sit at the domain interface boundary.

### 3.3 Deep Gates (g_3-g_6)

Deep gates receive inputs filtered by two gates above them. By the
time an input reaches g_3, it has passed through g_0 and g_1, which
have already made the domain and subgroup decisions. The deep gate's
input distribution is narrower and more domain-specific. Deep gate
recalibration should provide diminishing returns.

---

## 4. Prediction

Based on the interface mismatch analysis:

```
quality(S_root) << quality(S_graft) <= quality(S_all)
```

The root-only strategy should be clearly insufficient (confirmed by
parent experiment: +2.42% gap). Adding graft-point gates should
capture most of the benefit of all-gates recalibration, because the
interface mismatch concentrates at the root-to-subtree boundary.

### 4.1 Quantitative Prediction

From the parent experiment diagnostic:
- Root-only: +2.42% vs weight averaging
- All-gates: +0.67% vs weight averaging

The gap attributable to non-root gates is 2.42% - 0.67% = 1.75%.
If graft-point gates capture most of this, we expect:

```
root+graft-point gap ~= 0.67% + epsilon  (where epsilon << 1.75%)
```

If deep gates contribute equally, each gate recovers 1.75%/6 = 0.29%.
Graft-point gates (2 of 6 non-root) would recover 0.58%, leaving
1.17% of the gap. But the interface hypothesis predicts graft-point
gates recover disproportionately more.

---

## 5. Worked Example (D=2, d=4)

Smaller tree for illustration (3 internal gates, 4 leaves):

```
       [g_0]
       /    \
    [g_1]  [g_2]
    / \    / \
   L0 L1 L2 L3
```

Domain A: left subtree {g_1, L0, L1}
Domain B: right subtree {g_2, L2, L3}

After grafting, suppose:
- g_0 base weights: w_0 = [0.2, -0.1, 0.3, 0.1], b_0 = 0.0
- g_1 domain A weights: w_1 = [0.5, 0.8, -0.2, 0.1], b_1 = -0.3
- g_2 domain B weights: w_2 = [-0.3, 0.1, 0.6, 0.9], b_2 = 0.2

Input x = [1.0, 0.5, -0.5, 0.2]:
- g_0(x) = sigma(0.2 - 0.05 - 0.15 + 0.02) = sigma(0.02) = 0.505
  (near-random domain routing -- needs recalibration)

After root recalibration to route domain A left:
- g_0'(x_A) -> 0.9 (correctly routes A left)
- g_0'(x_B) -> 0.1 (correctly routes B right)

But g_1 still receives some B inputs (when g_0 is uncertain).
Recalibrating g_1 lets it handle these gracefully instead of
making domain-A-specific routing decisions on domain-B inputs.

---

## 6. Generalization to Deeper Trees

For depth D, the gate categories generalize:

```
Root:        {0}                              (1 gate)
Graft-point: {1, 2}                          (2 gates)
Deep:        {3, 4, ..., 2^D - 2}            (2^D - 4 gates)
```

Cost ratio of root+graft-point vs all-gates:

```
3 / (2^D - 1)
```

| Depth | Total gates | Cost ratio | Savings |
|-------|-------------|-----------|---------|
| 3 | 7 | 0.429 | 2.3x |
| 4 | 15 | 0.200 | 5.0x |
| 5 | 31 | 0.097 | 10.3x |
| 6 | 63 | 0.048 | 21.0x |

The savings grow exponentially with depth. At macro scale (depth 5-6),
selective recalibration provides an order-of-magnitude reduction in
calibration parameters.
