# Capsule MoE: Mathematical Foundations

## 1. Core Abstraction: Rank-1 Non-Linear Capsules

A **capsule** is the atomic unit of computation. Each capsule `i` consists of two
vectors and a non-linear activation:

```
Capsule_i = (a_i, b_i, activation_fn)
  a_i in R^d       -- input detector (projection vector)
  b_i in R^d       -- output generator (expansion vector)
  activation_fn    -- element-wise non-linearity (ReLU)
```

The output of capsule `i` for input `x in R^d` is:

```
c_i(x) = b_i * ReLU(a_i^T x)
```

This is a **rank-1 non-linear transformation**: `a_i^T x` produces a scalar
activation, the ReLU gates it, and `b_i` expands it back to `R^d`.

### Key Observation

A standard MLP with hidden dimension `H` computes:

```
MLP(x) = W_up * ReLU(W_down * x)
```

where `W_down in R^{H x d}` and `W_up in R^{d x H}`. Decomposing `W_down` into
rows `a_i^T` and `W_up` into columns `b_i`:

```
MLP(x) = sum_{i=1}^{H} b_i * ReLU(a_i^T x)
       = sum_{i=1}^{H} c_i(x)
```

Therefore: **a standard MLP is a dense composition of H capsules**. Our model
makes this composition **sparse and grouped**.

---

## 2. Capsule Pool and Group Structure

### 2.1 Pool Definition

A **capsule pool** contains `P` capsules organized into `G` groups of `P/G`
capsules each:

```
Pool = {Group_1, Group_2, ..., Group_G}
Group_g = {(a_{g,j}, b_{g,j})}_{j=1}^{P/G}
```

In matrix form, each group `g` stores:

```
A_g in R^{(P/G) x d}    -- rows are the a_{g,j}^T vectors
B_g in R^{d x (P/G)}    -- columns are the b_{g,j} vectors
```

### 2.2 Parameter Count

Per group:
```
params_per_group = (P/G) * d + d * (P/G) = 2 * d * (P/G)
```

Total capsule pool per layer:
```
params_pool = G * 2 * d * (P/G) = 2 * d * P
```

Comparison to dense MLP (`W_1 in R^{4d x d}`, `W_2 in R^{d x 4d}`):
```
params_MLP = 4d * d + d * 4d = 8d^2
```

Setting `P = 4d` (same hidden dimension as MLP):
```
params_pool = 2 * d * 4d = 8d^2 = params_MLP
```

**At `P = 4d`, the capsule pool is parameter-equivalent to the dense MLP.**

### 2.3 Micro-Scale Example

At `d = 64, G = 4, P = 256`:
- Capsules per group: `256 / 4 = 64`
- Params per group: `2 * 64 * 64 = 8,192`
- Total pool params: `4 * 8,192 = 32,768`
- Dense MLP params: `8 * 64^2 = 32,768`  (exact match)

---

## 3. Two-Level Routing

### 3.1 Level 1: Group Selection (Learned Router)

A lightweight linear router selects the top-`k_g` groups per token:

```
s = x W_r^T,     W_r in R^{G x d},   s in R^G
p = softmax(s)
```

Top-`k_g` selection with renormalization:

```
threshold = min over top-k_g values of s
mask_g = 1  if s_g >= threshold,  else 0
w_g = (p_g * mask_g) / (sum_g' p_g' * mask_g' + eps)
```

Router parameter overhead: `G * d` per layer.
At `G = 4, d = 64`: `256` parameters -- negligible (0.8% of pool params).

### 3.2 Level 2: Activation Sparsity (Free, Inherent to ReLU)

Within selected groups, ReLU provides **automatic** fine-grained filtering:

```
h_{g,j} = ReLU(a_{g,j}^T x)
```

If `a_{g,j}^T x <= 0`, capsule `j` in group `g` produces zero output. This is
not a routing decision -- it is a consequence of the non-linearity. No additional
parameters or computation are needed.

In a trained network with ReLU activation, empirically ~50% of hidden units are
inactive for any given input (Li et al., 2023 "The Lazy Neuron Phenomenon").
This means Level 2 sparsity is roughly 50% on top of Level 1 selection.

### 3.3 Combined Forward Pass

For a single token `x in R^d`:

```
CapsuleMoE(x) = sum_{g in TopK(s)} w_g * B_g * ReLU(A_g * x)
```

Expanding:

```
CapsuleMoE(x) = sum_{g in TopK(s)} w_g * sum_{j=1}^{P/G} b_{g,j} * ReLU(a_{g,j}^T x)
```

In batched matrix form for selected group `g`:

```
h_g = ReLU(A_g @ x^T)     -- shape (P/G,)   activation of each capsule
out_g = B_g @ h_g          -- shape (d,)      group output
```

Final output:

```
y = sum_{g in TopK} w_g * out_g
```

This is equivalent to a **sparse MLP** where only `k_g * (P/G)` of the `P`
hidden units are potentially active, and ReLU zeroes out ~50% of those.

---

## 4. Effective Sparsity Analysis

### 4.1 Active Parameter Fraction

At any given token, the active parameters are:

```
active = k_g * params_per_group = k_g * 2 * d * (P/G)
total  = G * 2 * d * (P/G) = 2 * d * P
active_ratio_L1 = k_g / G
```

With `k_g = 2, G = 4`: Level 1 ratio = `2/4 = 50%`.

After ReLU sparsity within active groups (~50% neurons inactive):

```
effective_active_ratio = (k_g / G) * 0.5 = 0.25  (25%)
```

### 4.2 FLOPs Per Token

Level 1 routing:
```
FLOPS_router = 2 * d * G     (matrix-vector product + softmax)
```

Level 2 capsule computation (per selected group):
```
FLOPS_group = 2 * d * (P/G) + 2 * d * (P/G)   (A_g @ x + B_g @ h_g)
            = 4 * d * (P/G)
```

Total:
```
FLOPS_total = FLOPS_router + k_g * FLOPS_group
            = 2*d*G + k_g * 4*d*(P/G)
```

At `d=64, G=4, k_g=2, P=256`:
```
FLOPS_router = 2 * 64 * 4 = 512
FLOPS_group  = 4 * 64 * 64 = 16,384
FLOPS_total  = 512 + 2 * 16,384 = 33,280
```

Dense MLP FLOPs:
```
FLOPS_MLP = 2 * d * 4d + 2 * 4d * d = 16 * d^2
          = 16 * 64^2 = 65,536
```

**At k_g=2, G=4: capsule MoE uses ~51% of MLP FLOPs** (before ReLU sparsity)
**-- assuming conditional computation is implemented.**

### 4.3 Implementation Gap: Dense vs. Conditional Execution

The FLOP analysis above assumes that non-selected groups are **skipped** (not
computed). This requires conditional computation: only executing the matmuls for
groups where `mask_g = 1`. The micro-scale implementation does NOT do this.
Instead, it runs all G groups and multiplies non-selected groups by w=0:

```python
for i, group in enumerate(self.groups):
    w = masked_probs[..., i:i+1]  # zero for non-selected groups
    out = out + w * group(x)      # group(x) is still computed
```

This means the **actual FLOPs at micro scale are**:

```
FLOPS_actual = FLOPS_router + G * FLOPS_group    (all groups computed)
             = 512 + 4 * 16,384 = 66,048
```

This is slightly MORE than the dense MLP (65,536) due to the router overhead.
The throughput data confirms this: capsule MoE runs at ~95K tok/s vs GPT's
~128K tok/s (26% slower).

At small G=4, running all groups and zeroing non-selected outputs is cheaper
than the memory indirection cost of sparse dispatch. At large G (64+),
conditional computation via scatter/gather or block-sparse kernels becomes
mandatory for the theoretical FLOP savings to materialize.

---

## 5. Load Balancing Loss

Identical to standard MoE. Let `f_g` be the mean router probability for group
`g` across all tokens in the batch:

```
f_g = mean_{b,t} p_{b,t,g}
```

Balance loss:

```
L_bal = G * sum_{g=1}^{G} f_g^2
```

Minimum at uniform routing (`f_g = 1/G` for all `g`):

```
L_bal_min = G * G * (1/G)^2 = 1
```

Total training loss:

```
L_total = L_CE + alpha * sum_{layers} L_bal
```

with `alpha = 0.01` (standard).

---

## 6. Composition by Concatenation

### 6.1 The Composability Claim

Given a shared pretrained base model `M_base` with attention weights `W_attn`
and capsule groups trained on different domains:

```
Groups_A = {(A_{A,g}, B_{A,g})}_{g=1}^{G}   fine-tuned on domain A (attention frozen)
Groups_B = {(A_{B,g}, B_{B,g})}_{g=1}^{G}   fine-tuned on domain B (attention frozen)
```

A composed pool concatenates the groups:

```
Groups_composed = Groups_A ++ Groups_B    -- 2G groups total
```

The attention weights `W_attn` remain those of `M_base`. The group router
must be recalibrated to handle `2G` groups (with `2*k_g` top-k to maintain
the same active fraction), which requires a brief training phase on mixed
data (~100 steps at micro scale).

### 6.2 Non-Interference Condition

Two capsule group sets do not interfere if their outputs are approximately
additive in the residual stream:

```
||Pool(x; composed) - Pool(x; Groups_A) - Pool(x; Groups_B)|| ~= 0
```

This holds under two conditions:

**Condition 1 (Backbone consistency):** Both group sets were fine-tuned on
the same backbone (same `W_attn`, same embeddings). If the backbone is
different, the capsule groups operate in different representation spaces and
their outputs are not additive. **Experimentally confirmed: independently
trained models fail (+13.5%), shared-base models pass (-0.3%).**

**Condition 2 (Router separation):** The calibrated router assigns domain-A
tokens primarily to Groups_A and domain-B tokens to Groups_B. After ~100
steps of mixed-domain training, the router learns this separation.

### 6.3 Shared-Base Protocol

The validated composition protocol is:

```
1. Train M_base on general data (all parameters trainable)
2. For each domain d:
   a. Initialize M_d from M_base
   b. Freeze attention + embeddings in M_d
   c. Fine-tune only capsule groups on domain d data
3. Compose:
   a. Take W_attn from M_base
   b. Concatenate capsule groups from all {M_d}
   c. Create router with output dim = sum of all group counts
   d. Train router on mixed data for ~100 steps (all other weights frozen)
```

This protocol has cost proportional to O(D) for D domains at fine-tuning
time, and O(D * G * d) for router calibration -- both much cheaper than
joint retraining.

### 6.4 Limitations of the Composability Claim

Composability via concatenation assumes:
1. **The base model is shared and frozen** -- experimentally confirmed as
   necessary. Without this, attention weight averaging destroys both models.
2. Capsule groups only modify the MLP residual stream
3. The router can distinguish domains after concatenation -- confirmed at
   micro scale with ~100 steps of calibration

What remains untested: whether this scales to many domains (D >> 2), whether
truly different domains (not just a-m vs n-z names) compose cleanly, and
whether the router calibration cost grows with G.

---

## 7. Comparison to Standard MoE

| Property | Standard MoE (N=4, k=2) | Capsule MoE (G=4, k_g=2, P=256) |
|---|---|---|
| Expert granularity | Monolithic MLP (8d^2 each) | Rank-1 capsule (2d each) |
| Total MLP params | N * 8d^2 = 32 * d^2 | 2 * d * P = 8d^2 |
| Active MLP params | k * 8d^2 = 16 * d^2 | k_g * 2 * d * (P/G) = 4d^2 |
| Routing overhead | N * d = 4d | G * d = 4d |
| Level-2 sparsity | None | ~50% (ReLU) |
| Expert count | 4 (fixed) | 256 capsules in 4 groups |
| Composability | Requires merging or ensembling | Concatenate groups |

**Key difference**: Standard MoE has 4x the MLP parameters of the baseline
(one full MLP per expert). Capsule MoE is parameter-equivalent to the baseline
but achieves sparsity through two-level routing. This matches the vision:
**same total knowledge, less active compute**.

---

## 8. Full Model Parameter Count

```
P_CapsuleMoE = n_layer * (4d^2 + 2*d*P + G*d) + V*d + T*d + d*V
```

Breaking down per layer: `4d^2` (attention) + `2*d*P` (capsule pool) + `G*d`
(group router).

At default config (`V=28, T=32, d=64, n_layer=4, G=4, P=256`):

```
Attention per layer:  4 * 64^2          =  16,384
Capsule pool/layer:   2 * 64 * 256      =  32,768
Router per layer:     4 * 64            =     256
Embeddings:           V*d + T*d + V*d   = 2*V*d + T*d

Per layer total:      16,384 + 32,768 + 256 = 49,408
All layers:           4 * 49,408 = 197,632
```

Note: The embedding count depends on the actual vocabulary size V. In the code,
`wte` (V, d) and `lm_head` (V, d) are **separate** (not weight-tied), plus
`wpe` (T, d). The default constructor uses V=28, but the arena's CharTokenizer
produces V=27 (26 letters + 1 BOS token). Both are reported below.

```
At V=28 (code default):  + 2*28*64 + 32*64 = 5,632  -> total = 203,264
At V=27 (arena runtime):  + 2*27*64 + 32*64 = 5,504  -> total = 203,136
```

Compare to dense GPT (same V-dependence):
```
Per layer: 12 * 64^2 = 49,152
All layers: 4 * 49,152 = 196,608
At V=28: + 5,632 = 202,240
At V=27: + 5,504 = 202,112
```

**Capsule MoE: 203,136 params vs GPT: 202,112 params (at V=27) -- 0.5% overhead
from group router weights (4 layers * 4 * 64 = 1,024 params).**
