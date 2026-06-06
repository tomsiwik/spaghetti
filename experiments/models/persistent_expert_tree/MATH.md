# Persistent Expert Tree: Mathematical Foundations (v2)

## 1. Persistent Binary Tree via Path Copying

### 1.1 Definitions

A **persistent binary tree** of depth D contains:
- 2^D - 1 internal nodes (gates), each with parameters in R^{d+1}
- 2^D leaf nodes (expert capsule groups), each with parameters in R^{2*d*n_c}

Notation:
```
D       = tree depth (default 3)
L       = 2^D leaf groups (default 8)
I       = 2^D - 1 internal gates (default 7)
d       = embedding dimension (default 64)
n_c     = capsules per leaf group (default 32)
V       = number of versions stored
```

A **version** v = (gates_v, leaves_v) is a pair of reference lists:
```
gates_v  = [g_0^v, g_1^v, ..., g_{I-1}^v]    -- gate module references
leaves_v = [l_0^v, l_1^v, ..., l_{L-1}^v]    -- leaf module references
```

Two versions can share the same module instance (structural sharing):
```
id(g_i^{v1}) == id(g_i^{v2})  =>  g_i^{v1} and g_i^{v2} are the same object
```

### 1.2 Path Copying

To create version v+1 from version v by updating leaf j:

1. Compute the ancestor path: P(j) = set of internal nodes on root-to-leaf_j path
   - |P(j)| = D for any j
   - P(j) = {node_0, node_1, ..., node_{D-1}} where node_0 = root

2. Copy path nodes:
```
gates_{v+1}[i] = { copy(gates_v[i])   if i in P(j)
                  { gates_v[i]         otherwise (shared reference)

leaves_{v+1}[i] = { copy(leaves_v[i])  if i == j
                   { leaves_v[i]        otherwise (shared reference)
```

3. New nodes: D gates + 1 leaf = D + 1 per single-leaf update.

### 1.3 Batch Update (Multiple Leaves)

For updating leaves J = {j_1, j_2, ..., j_m}:

```
P(J) = union_{j in J} P(j)    -- union of all ancestor paths
```

New nodes = |P(J)| gates + m leaves.

**Worst case**: all leaves updated, P(J) = all internal nodes.
  New nodes = I + L = 2^{D+1} - 1 (full copy).

**Best case**: all leaves share one subtree.
  For m = L/2 = 2^{D-1} leaves in one subtree:
  New nodes = (2^{D-1} - 1) + D-1 + 2^{D-1} = 2^D - 2 + D - 1
  Saving: D+1 nodes vs full copy (the other subtree is shared).

At D=3, updating 4 of 8 leaves (one subtree):
  P(J) = {root, child, grandchild_0, grandchild_1} = 4 gates
  New = 4 gates + 4 leaves = 8 nodes (out of 15 total). Sharing = 7/15 = 47%.

### 1.4 Leaf-Only Fine-Tuning Requirement

Path copying provides structural sharing ONLY when updates are restricted to
specific leaves. The fine-tuning protocol MUST:

1. **Freeze non-tree parameters** (embeddings, attention, norms, lm_head)
2. **Freeze gates** (routing parameters are recalibrated separately)
3. **Freeze non-target leaves** (leaves shared with other versions)
4. **Train only the target leaves** (the path-copied independent copies)

If all parameters are trained (full fine-tuning), every parameter diverges
from base, destroying structural sharing entirely. This was the v1 bug:
full fine-tuning produced 200% overhead with 0% savings vs full copy.

---

## 2. Memory Analysis

### 2.1 Parameter Sizes

Per internal gate: d + 1 parameters (linear projection + bias).
Per leaf group: 2 * d * n_c parameters (A and B matrices).

At d=64, n_c=32:
```
params_gate = 65
params_leaf = 2 * 64 * 32 = 4,096

Total per layer (tree only):
  gates:  7 * 65 = 455
  leaves: 8 * 4,096 = 32,768
  tree total: 33,223
  ratio: leaves/tree = 98.6%
```

### 2.2 Persistent vs Full-Copy vs Flat-Dict Storage

For 4 versions (v0 base, v1 update leaves 0-3, v2 update leaves 4-7,
v3 cross-version compose), each updating m=4 leaves from common base:

**Full copy** (4 complete trees per layer):
  4 * 33,223 = 132,892 per layer

**Flat dict** (3 snapshots of tree params: v0, v1, v2):
  3 * 33,223 = 99,669 per layer

**Persistent (path-copy)**: Count unique module instances across all versions.
  v0: 7 gates + 8 leaves = 15 nodes (base)
  v1: ~5 new gates + 4 new leaves = 9 new nodes (leaves 0-3 + path union)
  v2: ~5 new gates + 4 new leaves = 9 new nodes (leaves 4-7 + path union)
  v3: 7 new gates + 0 new leaves = 7 new nodes (compose: fresh gates, shared leaves)
  Total unique: 15 + 9 + 9 + 7 = 40 nodes (theoretical)
  Actual measured: 38 unique nodes, sharing_ratio = 1.58

**Measured at micro scale (4 layers):**
```
Persistent total:  267,864 params
Full-copy total:   531,568 params
Flat-dict total:   398,676 params
Base (1 tree):     132,892 params

Overhead vs mutable (1 tree):    101.6%
Savings vs full-copy (4 trees):  49.6%
Savings vs flat-dict (3 snaps):  32.8%
```

### 2.3 Why KC2 Fails at Micro

KC2 requires overhead < 15%. With 4 versions updating half the leaves each:

```
overhead = (persistent - base) / base
         = (267,864 - 132,892) / 132,892
         = 101.6%
```

This fails because leaves are 98.6% of tree parameters. Each version that
updates 4 leaves adds ~4 * 4,096 = 16,384 leaf params + ~5 * 65 = 325
gate params = 16,709 new params per version. Three additional versions add
~50,127 new params against a base of 33,223 per layer.

The overhead is dominated by the leaf/tree ratio. As this ratio decreases
(which happens at macro scale with LoRA adapters), path-copying overhead
shrinks proportionally.

### 2.4 Scaling to Macro (LoRA Adapters)

At macro scale, leaves are LoRA adapters (rank r), not full CapsuleGroups:
```
params_adapter = 2 * d * r    (A and B matrices)
params_base_layer = O(d^2)    (attention + MLP)
```

At d=896, r=16:
```
params_adapter = 2 * 896 * 16 = 28,672
params_base_layer ~ 896^2 * 12 ~ 9.6M (Qwen 0.5B)
adapter_fraction = 28672 / 9.6M = 0.3%
```

Persistent overhead per version (m=4 adapters + ~5 gates):
```
delta_macro = 4 * 28672 + 5 * (896+1) = 114688 + 4485 = 119173
overhead_macro = 119173 / 9.6M = 1.2% per version
```

At macro scale, KC2 (<15%) passes trivially: even 12 concurrent versions
would stay under 15% overhead. The 32.8% savings vs flat-dict demonstrated
at micro scale would scale even better (fewer parameters change per version).

---

## 3. Cross-Version Composition

### 3.1 Formalization

Given versions v1 and v2, both derived from base v0, define cross-version
composition by a leaf-version map M: {leaf_idx -> version_id}.

```
leaves_composed[i] = leaves_{M(i)}[i]    -- take leaf i from version M(i)
gates_composed[i]  = fresh copy            -- gates are recalibrated
```

Example: M = {0:v1, 1:v1, 2:v1, 3:v1, 4:v2, 5:v2, 6:v2, 7:v2}
  = left subtree from v1, right subtree from v2.

### 3.2 Why Cross-Version Works

With leaf-only fine-tuning, versions share the same non-leaf parameters
(embeddings, attention, norms, lm_head) exactly. Version differences are
confined to specific leaf modules. This means:

1. **Shared context**: All versions process inputs through identical
   embedding and attention layers. The routing context is the same.

2. **Leaf independence**: Each leaf is an independent expert module.
   Taking leaf i from v1 and leaf j from v2 is well-defined because
   neither leaf depends on the other's parameters.

3. **Gate recalibration**: After composition, gates are retrained on mixed
   data (100 steps). This adapts routing to the new leaf combination.

### 3.3 Four Baselines Compared

1. **Joint training** (upper bound): Train all params on all domains, 500 steps.
2. **Same-version avg**: Average leaf params from v1 and v2, recalibrate gates.
3. **Same-version cherry-pick** (control): All leaves from v1 only, recalibrate.
   Isolates the effect of version-crossing from cherry-picking.
4. **Cross-version cherry-pick**: Leaves 0-3 from v1, 4-7 from v2, recalibrate.

### 3.4 Empirical Validation

Measured across 3 seeds (42, 123, 7):
```
Joint training:                0.5238
Same-version avg (calibrated): 0.5247 (+0.17% vs joint)
Same-version pick (calibrated):0.5276 (+0.72% vs joint)
Cross-version (calibrated):    0.5231 (-0.13% vs joint)
Cross vs Same-pick:            -0.85% (BETTER than same-version)
```

Per-seed cross vs same-pick: +0.08%, -1.65%, -0.97%.
Kill criterion (>5%): PASSES with 60x margin.

Notable: cross-version composition is BETTER than same-version cherry-pick.
This is expected because cross-version gets domain-specialized leaves (A from
v1 trained on A data, B from v2 trained on B data), while same-version
cherry-pick uses v1 leaves that were only trained on domain A data.

---

## 4. Rollback Fidelity

### 4.1 Guarantee with Structural Sharing

With the persistent tree API, `set_version(v)` swaps the module reference
lists to version v's gates and leaves. Since leaf-only fine-tuning does
NOT modify shared nodes (non-target leaves are frozen), rollback is exact:

```
set_version(v0) => output(model, x) == output_v0(x)
```

Measured: max absolute difference = 0.00e+00 across all seeds.

### 4.2 Requirement: Freeze Shared Nodes

Rollback fidelity REQUIRES that shared nodes (those referenced by multiple
versions) are never mutated during training. The protocol enforces this by:

1. Path-copying target leaves (creating independent copies for training)
2. Freezing all non-target leaves (preventing gradient flow to shared nodes)
3. Freezing all non-tree parameters (preserving shared context)

Violation of these constraints (e.g., full model fine-tuning) would corrupt
shared references and destroy rollback fidelity.

---

## 5. Worked Example: D=2, 4 Leaves

### 5.1 Base Tree (v0)
```
        [gate_0]
        /      \
   [gate_1]  [gate_2]
   /    \    /    \
  L0    L1  L2    L3

Params: 3 gates + 4 leaves = 7 nodes
```

### 5.2 Update L0, L1 -> v1 (batch path copy, leaf-only fine-tune)
```
Path({0,1}) = {gate_0, gate_1}  (root and left child)

v1.gates  = [gate_0', gate_1', gate_2]   -- gate_2 shared with v0
v1.leaves = [L0', L1', L2, L3]           -- L2, L3 shared with v0

New nodes: 2 gates + 2 leaves = 4 (out of 7)
Shared:    1 gate + 2 leaves = 3 (43%)
Training:  only L0', L1' receive gradients; L2, L3 frozen
```

### 5.3 Update L2, L3 -> v2 (from v0, leaf-only fine-tune)
```
Path({2,3}) = {gate_0, gate_2}  (root and right child)

v2.gates  = [gate_0'', gate_1, gate_2'']  -- gate_1 shared with v0
v2.leaves = [L0, L1, L2'', L3'']          -- L0, L1 shared with v0

New nodes: 2 gates + 2 leaves = 4
Training:  only L2'', L3'' receive gradients; L0, L1 frozen
```

### 5.4 Cross-Version Compose: L0,L1 from v1, L2,L3 from v2 -> v3
```
v3.gates  = [gate_0''', gate_1''', gate_2''']  -- all fresh (recalibrated)
v3.leaves = [L0', L1', L2'', L3'']              -- shared from v1 and v2

L0', L1' from v1 (domain A specialists)
L2'', L3'' from v2 (domain B specialists)
New: 3 gates only (leaves are shared references)
```

### 5.5 Memory Accounting (d=64, n_c=32)

```
Per gate: 65 params, per leaf: 4096 params

v0: 3*65 + 4*4096 = 195 + 16384 = 16579  (base)
v1: 2*65 + 2*4096 = 130 + 8192  = 8322   (new nodes)
v2: 2*65 + 2*4096 = 130 + 8192  = 8322   (new nodes)
v3: 3*65 + 0*4096 = 195 + 0     = 195    (new gates, shared leaves)

Persistent total: 16579 + 8322 + 8322 + 195 = 33418
Full-copy total:  4 * 16579 = 66316
Flat-dict total:  3 * 16579 = 49737

Overhead vs base:        (33418 - 16579) / 16579 = 101.6%
Savings vs full-copy:    1 - 33418/66316 = 49.6%
Savings vs flat-dict:    1 - 33418/49737 = 32.8%
```
