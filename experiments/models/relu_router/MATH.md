# ReLU Router: Mathematical Foundations

## 1. Architecture: A Standard ReLU MLP

The ReLU Router IS a standard two-layer ReLU MLP. The architecture has
zero novelty. The contribution is the composition protocol (Section 5).

### 1.1 MLP Definition

A two-layer MLP with hidden dim P computes:

```
MLP(x) = W_up @ ReLU(W_down @ x)
       = B @ ReLU(A @ x)
```

where:
```
A in R^{P x d}    -- rows are detector vectors a_i^T
B in R^{d x P}    -- columns are expansion vectors b_i
x in R^d           -- input vector
```

Expanded element-wise:

```
y = sum_{i: a_i^T x > 0}  b_i * (a_i^T x)
```

Only neurons with positive pre-activation contribute. This is a standard
property of ReLU MLPs, not a novel observation.

### 1.2 Conceptual Relabeling

We relabel each (a_i, b_i) pair as a "rank-1 capsule" or "self-routing
expert." The ReLU activation a_i^T x > 0 can be viewed as a routing
decision: positive means "fire," zero means "silent." This relabeling
is a change of terminology, not architecture. Every ReLU MLP hidden
neuron is trivially a "self-routing expert" under this framing.

The relabeling becomes non-trivial only when it enables a useful
composition protocol (Section 5).

### 1.3 Prior Art

- **ReMoE (ICLR 2025)**: Uses ReLU on router logits for variable-k expert
  selection. Differs from our work: ReMoE keeps separate large experts and
  uses ReLU in the routing layer, not the computation itself.
- **Union-of-Experts (UoE)**: Shows internal expert activation norms capture
  routing information, making external routers redundant at scale.
- **MoRAM**: Rank-1 associative memory experts self-route via intrinsic keys.
  Closest to our framing but operates in a LoRA adapter context.

---

## 2. Sparsity Properties

### 2.1 Natural ReLU Sparsity

By the "Lazy Neuron Phenomenon" (Li et al., 2023), approximately 50% of
ReLU activations are zero for any given input in trained transformers.

Measured at micro scale (d=64, P=256): 49-52% natural sparsity across layers.

### 2.2 Sparsity Control (UNTESTED at micro scale)

An adaptive L1 mechanism is implemented but does NOT achieve its target
during the 500-step micro experiment. The mechanism is described here for
completeness, but its effectiveness is unvalidated.

```
L_sparsity = lambda_adaptive * mean(activation_frequency)
activation_frequency_i = E_{x in batch}[1{a_i^T x > 0}]
lambda_adaptive = lambda_base * (1 + 5 * clamp(s_target - s_running, -0.5, 0.5))
```

At micro scale:
- s_target was set to 0.50 (revised from 0.75 after review)
- s_running stays at ~50% (the natural ReLU level)
- The adaptive coefficient is ~lambda_base (no meaningful adjustment)
- **The sparsity control mechanism is effectively inert**

This means the model operates as a dense MLP with ~50% natural sparsity.
No compute savings beyond what any ReLU MLP provides for free.

### 2.3 Balance Loss

Penalizes variance in per-capsule activation frequency:

```
L_balance = beta * P * Var(activation_frequency)
```

This prevents routing collapse (few capsules dominate). It functions
correctly at micro scale but is secondary given the inert sparsity control.

### 2.4 Total Auxiliary Loss

```
L_aux = L_sparsity + L_balance
```

Summed across all layers and added to the cross-entropy language modeling loss.

---

## 3. Parameter Count

### 3.1 Per Layer

```
params_capsule = P * d + d * P = 2 * d * P
```

No router weight matrix (saves G * d per layer).

### 3.2 Comparison to Capsule MoE

At P = 4d = 256 (d=64):

| Component | ReLU Router | Capsule MoE |
|-----------|-------------|-------------|
| MLP/capsule pool per layer | 2 * 64 * 256 = 32,768 | 32,768 |
| Router per layer | 0 | 4 * 64 = 256 |
| Attention per layer | 4 * 64^2 = 16,384 | 16,384 |
| Per-layer total | 49,152 | 49,408 |
| All layers (4) | 196,608 | 197,632 |

Router savings: 4 layers * 256 = **1,024 parameters** (0.5% reduction).
This is practically irrelevant at any scale.

### 3.3 Micro-Scale Instance

At V=27 (arena runtime):
```
ReLU Router: 202,112 params
Capsule MoE: 203,136 params
Dense GPT:   202,112 params (exact match)
```

**The ReLU Router is parameter-identical to the dense GPT baseline.**

---

## 4. Composition by Concatenation

### 4.1 The Mathematical Identity

Given a shared pretrained base model M_base and domain-specific MLP weights:

```
(A_1, B_1): fine-tuned on domain 1 (attention frozen)
(A_2, B_2): fine-tuned on domain 2 (attention frozen)
```

Composed MLP:

```
A_composed = [A_1; A_2]    -- vertical stack, shape (2P, d)
B_composed = [B_1, B_2]    -- horizontal stack, shape (d, 2P)
```

Forward pass:

```
y = B_composed @ ReLU(A_composed @ x)
  = [B_1, B_2] @ ReLU([A_1; A_2] @ x)
  = [B_1, B_2] @ [ReLU(A_1 @ x); ReLU(A_2 @ x)]
  = B_1 @ ReLU(A_1 @ x) + B_2 @ ReLU(A_2 @ x)
  = Pool_1(x) + Pool_2(x)
```

**The composed output is the sum of individual pool outputs.** This is
mathematically exact. Verified numerically with max error < 3e-7.

The identity holds because:
1. Vertical stacking of A means each block independently computes its
   inner products before ReLU is applied element-wise
2. ReLU is applied independently to each element: ReLU([a; b]) = [ReLU(a); ReLU(b)]
3. Horizontal B selects and sums the two blocks

### 4.2 Why Exact Summation is Insufficient

The mathematical identity says nothing about whether the SUM is useful.
During training, a single MLP learns to produce outputs with a certain
magnitude and direction distribution. The downstream layers (attention,
layer norm, lm_head) are calibrated for this distribution.

When summing two independently-trained MLPs:
- Output magnitude roughly doubles
- Direction distribution changes unpredictably
- Downstream layers receive inputs from an unseen distribution

This is the core reason zero-shot composition degrades. It is NOT just
a "loudness problem" (relative magnitude between domains). It is an
absolute magnitude problem: the sum of two pools produces outputs in a
different regime than what the rest of the network expects.

Experimental evidence: per-pool scalar calibration (training only 1
scaling factor per pool per layer = 8 total parameters) reduces
degradation from +5.0% to +4.4% -- barely helping. If loudness were the
only issue, scalar calibration would solve it.

### 4.3 ReLU Independence Property

ReLU is unnormalized: each neuron's activation depends only on its own
weight vector, independent of other neurons. Adding new neurons to the
MLP cannot reduce existing activations.

This is the property that makes concatenation mathematically valid.
Compare to softmax, where sum(w_i) = 1 and adding new experts reduces
all existing weights.

### 4.4 Comparison to Weight Averaging

Weight averaging is the standard model merging baseline:
```
A_avg = (A_1 + A_2) / 2,    B_avg = (B_1 + B_2) / 2
```

Advantages over concatenation:
- Same parameter count as individual models (no size increase)
- At micro scale, produces +2.0% degradation (vs +5.0% for concatenation)

Disadvantages:
- Destructive: averages create compromise detectors that match neither domain
- At scale with diverse domains, interference may be severe

The weight averaging result at micro scale (+2.0% vs +5.0%) is a
significant challenge to the concatenation claim. Concatenation's
mathematical exactness does not translate to practical superiority
when the sum of outputs is out-of-distribution for downstream layers.

---

## 5. Theoretical Connections

### 5.1 Content-Addressable Memory

The forward pass h = ReLU(A @ x) can be viewed as a content-addressable
memory lookup. Each row a_i^T of A is a "key." The input x is the
"query." The activation a_i^T x measures key-query alignment. ReLU
gates the result: keys that align positively are retrieved.

This is structurally similar to Product Key Memory (Lample et al., 2019):
```
PKM:     v = sum_i softmax(k_i^T x) * v_i
ReLU:    v = sum_i ReLU(a_i^T x) * b_i
```

The connection to Modern Hopfield Networks (Ramsauer et al., 2021) is
superficial: the defining property of modern Hopfield networks is
exponential storage capacity via polynomial/exponential energy functions.
ReLU provides a linear energy function with linear storage capacity.
Calling this a "Hopfield Network" strips away the defining property and
leaves only the trivial claim that it does key-value lookup.

---

## 6. Worked Numerical Example

At d=4, P=8 (toy scale):

```
A = [[1, 0, 0, 0],    -- neuron 0: fires on positive x[0]
     [0, 1, 0, 0],    -- neuron 1: fires on positive x[1]
     [0, 0, 1, 0],    -- neuron 2: fires on positive x[2]
     [0, 0, 0, 1],    -- neuron 3: fires on positive x[3]
     [-1, 0, 0, 0],   -- neuron 4: fires on negative x[0]
     [0, -1, 0, 0],   -- neuron 5: fires on negative x[1]
     [1, 1, 0, 0],    -- neuron 6: fires on x[0]+x[1] > 0
     [0, 0, 1, 1]]    -- neuron 7: fires on x[2]+x[3] > 0

x = [0.5, -0.3, 0.8, 0.1]

h = ReLU(A @ x)
  = ReLU([0.5, -0.3, 0.8, 0.1, -0.5, 0.3, 0.2, 0.9])
  = [0.5, 0, 0.8, 0.1, 0, 0.3, 0.2, 0.9]

Active neurons: {0, 2, 3, 5, 6, 7}  (6 of 8 = 75% active, 25% sparse)
Silent neurons: {1, 4}               (x[1] < 0, x[0] > 0)
```

Composition: Pool A = neurons 0-3, Pool B = neurons 4-7:
```
A_composed = [A_poolA; A_poolB]    -- (8, 4)
B_composed = [B_poolA, B_poolB]    -- (4, 8)

MLP_composed(x) = MLP_A(x) + MLP_B(x)
```

---

## 7. Assumptions

1. **Shared backbone is prerequisite.** All domain-specific MLPs must be
   fine-tuned from the same shared backbone. If backbones differ, the MLP
   outputs operate in different representation spaces.

2. **The sum of independently-trained MLPs is useful.** This is the
   strongest assumption and the one most challenged by experiments.
   The composed output is mathematically exact (sum of pools) but may be
   out-of-distribution for downstream layers trained on single-pool output.
   EXPERIMENTAL RESULT: +5.0% zero-shot degradation at micro scale.

3. **Natural sparsity provides sufficient selectivity.** At ~50% sparsity,
   half the neurons fire for any input. For composition to be selective,
   different domains must activate different neuron subsets. If both domains
   activate the same neurons, composition adds redundancy not diversity.

4. **L1 + balance losses converge at scale.** Untested. At micro scale,
   the adaptive L1 mechanism is effectively inert (~50% natural sparsity
   matches the revised target of 50%, so no adjustment occurs). Whether
   this mechanism can push sparsity beyond 50% at larger scale is unknown.
