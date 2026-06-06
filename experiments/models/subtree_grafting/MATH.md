# Subtree Grafting Composition: Mathematical Foundations

## 1. Tree Structure and Subtree Decomposition

### 1.1 Notation (inherited from hierarchical_tree/MATH.md)

```
D          = tree depth (3)
L          = 2^D leaf groups (8)
I          = 2^D - 1 internal gates (7)
n_c        = capsules per leaf group (32)
d          = embedding dimension (64)
B          = beam width (2)
N          = number of domains (2)
```

### 1.2 Subtree Assignment

A depth-D binary tree decomposes naturally at the root into N=2 subtrees:

```
         [root gate g_0]
         /              \
    LEFT SUBTREE     RIGHT SUBTREE
    (domain A)       (domain B)
```

For D=3:
```
Left subtree:  gates {1, 3, 4},  leaves {0, 1, 2, 3}
Right subtree: gates {2, 5, 6},  leaves {4, 5, 6, 7}
Root gate:     {0}               (domain router)
```

Each subtree is a complete binary tree of depth D-1=2 with:
- 2^(D-1) - 1 = 3 internal gates
- 2^(D-1) = 4 leaf groups
- Total capsules per subtree: 4 * 32 = 128

### 1.3 Partition Completeness

**Claim**: Root gate + left subtree + right subtree = complete tree.

The gate indices partition as:
- Root: {0}
- Left subtree gates: {2i+1 : i in left_subtree_internal} = {1, 3, 4}
- Right subtree gates: {2i+2 : i in right_subtree_internal} = {2, 5, 6}
- Union: {0, 1, 2, 3, 4, 5, 6} = all 7 internal nodes

The leaf indices partition as:
- Left: {0, 1, 2, 3}
- Right: {4, 5, 6, 7}
- Union: {0, 1, ..., 7} = all 8 leaves

The decomposition is exact: no parameter is shared between subtrees (except
through the root gate, which routes between them). QED.

---

## 2. Composition Methods

### 2.1 Weight Averaging

Given N domain-specific parameter sets {theta_d}_{d=1}^N, each trained from
shared base theta_0:

```
theta_avg = (1/N) * sum_{d=1}^{N} theta_d
```

Applied to all tree parameters: gates {g_i : 0 <= i < I} and leaves
{L_j : 0 <= j < L}. This blends routing decisions and capsule weights
uniformly across domains.

**Problem**: Weight averaging a sigmoid gate g_i(x) = sigma(w_i^T x + b_i)
between two domain-specific versions does NOT produce the average of their
routing decisions:

```
sigma((w_A + w_B)/2 * x + (b_A + b_B)/2)  !=  (sigma(w_A*x + b_A) + sigma(w_B*x + b_B)) / 2
```

The sigmoid's nonlinearity means averaged weights produce a different routing
function than either domain learned. This is the "function-space gap" that
weight averaging inherently introduces.

### 2.2 Subtree Grafting

Given domain-specific models trained with subtree assignment:
- Model A trained on domain A data, only left subtree parameters updated
- Model B trained on domain B data, only right subtree parameters updated

Compose by grafting:

```
theta_graft = {
    root gate:      theta_0 (base, then calibrated)
    left subtree:   theta_A (from domain A model, exact)
    right subtree:  theta_B (from domain B model, exact)
}
```

**Key property**: Each domain's internal routing decisions are preserved
exactly. Domain A's gates {g_1, g_3, g_4} and leaves {L_0, L_1, L_2, L_3}
are used without any blending. The root gate g_0 is the only parameter
that needs retraining (to route between the two domain subtrees).

### 2.3 Grafting vs Averaging: Structural Comparison

| Property | Weight Averaging | Subtree Grafting |
|----------|-----------------|------------------|
| Leaf weights | Blended (midpoint) | Preserved exactly |
| Internal gates | Blended (avg sigmoid) | Preserved exactly |
| Root gate | Blended | Retrained |
| Routing decisions | Approximated | Exact per subtree |
| Calibration target | All gates | Root gate (or all gates) |
| Function-space gap | Present (nonlinear avg) | Absent within subtrees |

### 2.4 Why Grafting Might Be Better

Within each subtree, the gates and leaves form a coherent routing-to-compute
pipeline. Gate g_1 learned to split domain A's inputs into subgroups, and
leaves L_0, L_1 learned to handle those subgroups. Grafting preserves this
learned pipeline. Weight averaging destroys it by blending g_1's weights with
domain B's g_1 (which may have learned a completely different split).

### 2.5 Why Grafting Might Be Worse

Grafting constrains each domain to half the tree. With N=2 and L=8, each
domain gets 4 leaf groups instead of 8. Weight averaging gives each domain
access to all 8 leaves during fine-tuning (the full tree), then blends the
results. If a domain needs more than L/N leaves to learn its full pattern
set, grafting is capacity-limited.

---

## 3. Parameter Counts

### 3.1 Trainable Parameters During Fine-Tuning

**Weight averaging (full tree per domain)**:
```
Per domain: I*(d+1) + L*2*d*n_c = 7*65 + 8*2*64*32 = 455 + 32,768 = 33,223 per layer
Total: 4 * 33,223 = 132,892
```

**Subtree grafting (half tree per domain)**:
```
Per domain: 3*(d+1) + 4*2*d*n_c = 3*65 + 4*2*64*32 = 195 + 16,384 = 16,579 per layer
Total: 4 * 16,579 = 66,316
```

Grafting fine-tunes 49.9% fewer parameters per domain (each domain updates
only its assigned subtree).

### 3.2 Calibration Parameters

**Weight averaging (all gates)**:
```
I*(d+1) = 7*65 = 455 per layer, 1,820 total
```

**Subtree grafting, root only**:
```
1*(d+1) = 65 per layer, 260 total
```

**Subtree grafting, all gates** (used in v2 experiment):
```
I*(d+1) = 7*65 = 455 per layer, 1,820 total
```

### 3.3 Total Inference Parameters

Identical for both methods: the composed model has the same architecture.
```
Total: 203,932 params (same as hierarchical_tree)
```

---

## 4. Routing Probability Under Grafting

### 4.1 Domain Routing via Root Gate

After grafting, the root gate g_0 acts as a domain router:
```
P(domain A | x) = g_0(x)         = sigma(w_0^T x + b_0)
P(domain B | x) = 1 - g_0(x)
```

Each token's probability of reaching a left-subtree leaf vs right-subtree
leaf is entirely controlled by g_0. The root gate has d+1 = 65 parameters
to learn this binary domain classification.

### 4.2 Leaf Probability Decomposition

The probability of reaching leaf l decomposes as:
```
P(leaf = l | x) = P(subtree(l) | x) * P(leaf = l | subtree(l), x)
```

For left-subtree leaves (l in {0,1,2,3}):
```
P(leaf = l | x) = g_0(x) * prod_{k=1}^{D-1} gate_decision(l, k, x)
```

The within-subtree probability P(leaf = l | subtree, x) is computed entirely
from the domain's trained gates -- preserved exactly by grafting.

### 4.3 Beam Interaction

With beam=2 and 2 subtrees, the beam can select:
- 2 leaves from the same subtree (both from domain A or both from domain B)
- 1 leaf from each subtree (one from domain A, one from domain B)

The root gate probability determines which case is more likely. After
calibration, g_0 learns to output ~0.5 for ambiguous inputs (using both
domains) and ~0 or ~1 for domain-specific inputs (using one domain).

---

## 5. Worked Example (D=2, N=2)

Smaller tree for illustration:

```
     [g_0]          root gate (domain router)
     /    \
  [g_1]  [g_2]     subtree gates
  /  \   /  \
 L0  L1 L2  L3     leaves

Domain A: left subtree = {g_1, L0, L1}
Domain B: right subtree = {g_2, L2, L3}
```

After domain-specific training:
- Domain A learned: g_1(x) clusters A's data into L0 (consonant-heavy) vs L1 (vowel-heavy)
- Domain B learned: g_2(x) clusters B's data into L2 (short names) vs L3 (long names)

**Weight averaging**: g_1_avg = (g_1_A + g_1_B)/2 -- meaningless blend of
"consonant/vowel split" with whatever domain B learned for g_1 during its
full-tree fine-tuning.

**Subtree grafting**: g_1 = g_1_A exactly. The consonant/vowel split is
preserved. g_2 = g_2_B exactly. The short/long split is preserved. Only
g_0 is retrained to route between the two domain subtrees.

---

## 6. Complexity Analysis

### 6.1 Fine-Tuning Cost

**Weight averaging**: N * (steps_ft * cost_per_step_full_tree)
**Subtree grafting**: N * (steps_ft * cost_per_step_half_tree)

Grafting is ~2x cheaper per fine-tuning step (half the trainable parameters).
At micro scale with small trees, the difference is negligible (both <5s).
At macro scale with L=64+ leaves, the 2x savings compound.

### 6.2 Calibration Cost

**Weight averaging**: steps_cal * cost_per_step (all gates trainable)
**Subtree grafting, root only**: steps_cal * cost_per_step_root_only
**Subtree grafting, all gates**: steps_cal * cost_per_step (all gates trainable)

Root-only calibration is minimal (65 params per layer vs 455). But the v2
experiment showed all-gates calibration produces better results for grafting.

### 6.3 Scaling to N>2 Domains

With N>2 domains, the tree does not naturally decompose into N subtrees
unless L/N is a power of 2. Options:
- N=4, D=3, L=8: 2 leaves per domain (each domain gets a depth-1 subtree)
- N=2, D=4, L=16: 8 leaves per domain (richer per-domain structure)
- Arbitrary N: use a variable-depth split (some domains get deeper subtrees)

This is a limitation of the binary tree structure for non-power-of-2 domains.
The Huffman tree (exp_huffman_pruning) naturally handles unequal splits.
