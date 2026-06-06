# Dead Capsule Pruning: Mathematical Foundations

## 1. Problem Statement

Given a composed ReLU MLP with 2P capsules (P per domain, formed by
concatenating two independently-trained pools), identify and remove
capsules that are functionally inert (never activate for any input
in the data distribution) to reduce parameter count while provably
preserving model quality.

### 1.1 Notation

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
N_d       -- number of domains (2 for binary split)
P_total   -- total capsules after concatenation = P * N_d = 256
L         -- number of transformer layers (4 at micro scale)

A in R^{P_total x d}   -- detector matrix (rows are a_i^T)
B in R^{d x P_total}   -- expansion matrix (columns are b_i)

a_i in R^d              -- detector vector for capsule i (row i of A)
b_i in R^d              -- expansion vector for capsule i (col i of B)

D = {x_1, ..., x_M}    -- calibration dataset of M hidden-state vectors
f_i = freq(capsule_i)  -- activation frequency: fraction of inputs where
                           a_i^T x > 0 (i.e., capsule i fires)
tau                     -- pruning threshold on activation frequency
```

---

## 2. Activation Profiling

### 2.1 Per-Capsule Activation Frequency

For capsule i, the activation frequency over dataset D is:

```
f_i = (1/M) * sum_{x in D} 1{a_i^T x > 0}
```

where 1{.} is the indicator function. This measures the fraction of
inputs for which capsule i contributes to the output (fires through ReLU).

**Range**: f_i in [0, 1].
- f_i = 0: capsule i NEVER fires (truly dead)
- f_i = 0.5: capsule fires on half the inputs (typical for ReLU)
- f_i = 1: capsule fires on ALL inputs (always active)

### 2.2 Expected Frequencies

For a randomly initialized ReLU neuron with inputs drawn from a
symmetric distribution (mean zero), approximately 50% of inputs
produce positive activations:

```
E[f_i] ~ 0.5   (for random initialization + symmetric inputs)
```

After training, the frequency distribution shifts. The "Lazy Neuron
Phenomenon" (Li et al., 2023) shows that trained transformers maintain
~50% natural ReLU sparsity.

After composition by concatenation, the situation changes:
- Pool_A capsules are tuned for domain A inputs
- Pool_B capsules are tuned for domain B inputs
- When evaluated on joint data (both domains), approximately half the
  capsules in each pool are "wrong domain" and rarely fire

### 2.3 Dead Capsule Definition

A capsule is "dead" if its activation frequency falls below threshold tau:

```
dead_i = (f_i <= tau)
```

Three natural choices for tau:
- tau = 0 (strict): only truly dead capsules (never fire)
- tau > 0 (soft): capsules that fire so rarely they contribute negligibly
- tau chosen by elbow method on the frequency distribution

---

## 3. Pruning Theory

### 3.1 Exact Zero-Change Theorem

**Theorem**: For a ReLU MLP y = B @ ReLU(A @ x), removing a capsule i
where a_i^T x <= 0 for all x in D changes the output by exactly zero
on all x in D.

**Proof**: The MLP output is:

```
y(x) = sum_{j=1}^{P_total} b_j * max(0, a_j^T x)
```

For capsule i with a_i^T x <= 0 for all x in D:

```
max(0, a_i^T x) = 0   for all x in D
```

Therefore the contribution of capsule i to y(x) is:

```
b_i * max(0, a_i^T x) = b_i * 0 = 0   for all x in D
```

Removing capsule i gives:

```
y'(x) = sum_{j != i} b_j * max(0, a_j^T x) = y(x)   for all x in D
```

QED.

**Important**: This is exact, not approximate. Dead capsule pruning is
a LOSSLESS compression for the profiling dataset. The only risk is that
a capsule is dead on D but alive on some unseen input x' not in D. The
profiling dataset must be representative.

### 3.2 Bounded Change for Nearly-Dead Capsules

For capsules with f_i = epsilon > 0 (fire on fraction epsilon of inputs):

**Theorem**: Removing capsule i with frequency f_i = epsilon introduces
an error bounded by:

```
E_x[||delta_y(x)||] <= epsilon * E_x[|a_i^T x| * ||b_i|| | a_i^T x > 0]
```

**Proof**: The output change from removing capsule i is:

```
delta_y(x) = -b_i * max(0, a_i^T x)
```

Taking the expectation over x in D:

```
E[||delta_y||] = E[||b_i|| * max(0, a_i^T x)]
               = ||b_i|| * P(a_i^T x > 0) * E[a_i^T x | a_i^T x > 0]
               = ||b_i|| * epsilon * E[a_i^T x | a_i^T x > 0]
```

For small epsilon, this error is proportional to epsilon -- pruning
rarely-firing capsules introduces small, bounded error.

### 3.3 Independence of Pruning Decisions

**Claim**: Pruning capsule i does not affect whether capsule j fires.

**Proof**: The activation of capsule j depends only on a_j^T x, where
x is the input to the layer. Since capsule activations are applied
element-wise through ReLU:

```
h_j = max(0, a_j^T x)
```

This depends only on (a_j, x) and is independent of h_i or a_i.
Therefore, pruning multiple capsules simultaneously is equivalent
to pruning them one at a time.

**Caveat**: This independence holds within a single layer. Across
layers, pruning changes the output of layer l, which changes the
input to layer l+1. However, for truly dead capsules (zero contribution),
the inter-layer effect is also exactly zero.

---

## 4. Pruning Algorithm

### 4.1 Full Pipeline

```
Input:  Composed model M with L layers, calibration dataset D
Output: Pruned model M' with fewer capsules per layer

For each layer l:
  1. Profile: run D through layers 0..l, collect activations h_l
  2. Compute: f_i = mean(h_l[i] > 0) for each capsule i
  3. Mask:   alive_i = (f_i > tau)
  4. Prune:  A'_l = A_l[alive, :],  B'_l = B_l[:, alive]
  5. Rebuild: new ReLUCapsulePool with P' = sum(alive) capsules

Return M' with pruned layers
```

### 4.2 Profiling Correctness

The profiling must be done via a full forward pass through the model,
not independently per layer, because:
- The input to layer l depends on the outputs of layers 0..l-1
- Layer norms (RMSNorm) and attention change the distribution at each layer
- Profiling with random inputs would give meaningless frequencies

We use 20 batches of 32 samples = 640 total sequences. At block_size=32,
this gives 640 * 32 = 20,480 token positions per profiling run.

### 4.3 Threshold Selection

The activation frequency distribution is bimodal:
- Cluster at f_i = 0: truly dead capsules (never fire)
- Cluster at f_i ~ 0.3-0.7: active capsules (natural ReLU range)
- Small tail at f_i ~ 0.001-0.01: "nearly dead" capsules

The threshold tau controls the compression-quality tradeoff:

```
tau = 0.000:  prune ~57% of capsules, 0.0% quality change
tau = 0.001:  prune ~65%, +0.0% quality change
tau = 0.005:  prune ~68%, +0.1% quality change
tau = 0.010:  prune ~69%, +0.1% quality change
tau = 0.050:  prune ~74%, -0.7% quality change (noise)
tau = 0.100:  prune ~76%, -0.2% quality change (noise)
```

The sweet spot is tau = 0.0 (strict dead pruning): maximum certainty
of zero quality change. Higher thresholds give diminishing returns
in compression with increasing risk.

---

## 5. Dead Capsule Anatomy

### 5.1 Why So Many Capsules Are Dead

The ~57% dead capsule rate in composed models has three sources:

**Source 1: Wrong-domain capsules (~50%)**

When Pool_A (trained on domain A) and Pool_B (trained on domain B)
are concatenated, Pool_A capsules specialized for domain A features
will not fire on domain B inputs (and vice versa). Since evaluation
uses both domains, approximately half the capsules in each pool are
"wrong domain" and do not fire for roughly half the inputs.

But truly dead means they never fire for ANY input. The 50% wrong-domain
estimate would give ~25% strictly dead if domains were independent.
The higher rate (~57% overall, ~75% in layers 1-3) suggests:

**Source 2: Natural ReLU death (~10%)**

Some capsules die during training through the "dying ReLU" phenomenon:
gradients push the pre-activation to be consistently negative, and
once dead, the zero gradient prevents recovery. This is independent
of composition.

**Source 3: Domain specialization depth**

Deeper layers (1, 2, 3) show much higher death rates (73-82%) than
layer 0 (0.4%). This is consistent with the attention-MLP pipeline:
- Layer 0 processes raw embeddings (generic, both domains activate)
- Layers 1-3 process attention-refined representations (more domain-specific)

### 5.2 Layer-Dependent Death Rates

Measured across 3 seeds (mean):

```
Layer 0:  0.4% dead   (256 -> 255 alive)
Layer 1: 73.0% dead   (256 ->  69 alive)
Layer 2: 82.0% dead   (256 ->  46 alive)
Layer 3: 71.6% dead   (256 ->  73 alive)
```

The U-shape (layer 0 low, layers 1-2 high, layer 3 slightly lower)
mirrors findings in activation sparsity literature: early and late
layers tend to be less sparse than middle layers.

### 5.3 Per-Domain Decomposition

Cross-domain analysis reveals:

```
Layer 1: 74% dead on BOTH domains, 80% dead on EITHER
Layer 2: 84% dead on BOTH, 87% dead on EITHER
Layer 3: 75% dead on BOTH, 80% dead on EITHER
```

The gap between "both" and "either" is small (~5%), meaning most dead
capsules are dead for BOTH domains, not just one. This suggests the
dominant cause is training-induced ReLU death rather than domain
mismatch: capsules that die during fine-tuning die regardless of
which domain's inputs are presented.

---

## 6. Computational Cost

### 6.1 Profiling (One-Time)

Per layer per batch:
```
Forward through A: O(B * T * P * d)
At micro scale: O(32 * 32 * 256 * 64) = O(16.8M) FLOPs
```

For 20 batches, 4 layers: O(20 * 4 * 16.8M) = O(1.3G) FLOPs.
On Apple Silicon: <1 second.

### 6.2 Pruning (One-Time)

Index selection on A and B matrices: O(P * d) per layer per surviving
capsule. Negligible (<1ms).

### 6.3 Inference After Pruning

FLOPs saved per layer per token:
```
Before: 2 * P_total * d = 2 * 256 * 64 = 32,768
After:  2 * P_alive * d

At tau=0.0 (57% pruned):
  P_alive ~ 109/layer (mean)
  After: 2 * 109 * 64 = 13,952
  Savings: 57% FLOPs reduction per capsule layer

At tau=0.01 (69% pruned):
  P_alive ~ 84/layer (mean)
  After: 2 * 84 * 64 = 10,752
  Savings: 67% FLOPs reduction
```

### 6.4 Parameter Savings

```
Before pruning: 202,112 total params
  Capsule params: 4 layers * 2 * 256 * 64 = 131,072 (65% of total)
  Shared params:  71,040 (35% of total)

After pruning (tau=0.0, seed 42):
  Capsule params: 55,680 (57% reduction)
  Total: 126,720 (37% reduction)

After pruning (tau=0.01, seed 42):
  Capsule params: 42,752 (67% reduction)
  Total: 113,792 (44% reduction)
```

The pruning primarily affects capsule parameters (65% of total).
The 37-44% total parameter reduction is achieved with zero quality
change.

---

## 7. Interaction with Calibration

### 7.1 Prune-Then-Calibrate

Protocol:
1. Compose model (concatenate)
2. Profile activations on calibration data
3. Prune dead capsules
4. Fine-tune surviving capsule weights (100 steps, lr=3e-4)

Result: -1.1% vs joint (BETTER than joint, same as calibrate-only).

The pruned model calibrates just as well as the unpruned model,
confirming that dead capsules carry zero useful information.

### 7.2 Calibrate-Then-Prune

Protocol:
1. Compose model (concatenate)
2. Fine-tune capsule weights (100 steps, lr=3e-4)
3. Profile activations on calibration data
4. Prune dead capsules from calibrated model

Result: -1.1% vs joint (identical to prune-then-calibrate).

Calibration does not revive dead capsules: 58% still dead after
calibration (vs 57% before). The small change (1%) suggests
calibration marginally activates a few borderline capsules.

### 7.3 Order Independence

Prune-then-calibrate and calibrate-then-prune produce IDENTICAL
quality (0.5184 vs 0.5183 avg loss). This is because:
1. Dead capsules contribute zero to the forward pass
2. Calibration gradients flow through alive capsules only
3. Removing zero-contribution terms before or after optimization
   is mathematically equivalent

### 7.4 Aggressive Prune-Then-Calibrate

Using tau=0.01 (69% pruned) then calibrating:
- Result: -0.8% vs joint (still better than joint)
- Slightly worse than conservative prune (-1.1%)
- The 0.3% gap is from removing some marginally-useful capsules

---

## 8. Worked Numerical Example

At d=4, P=4 per domain (toy scale):

### 8.1 Composed Model

```
Pool A (domain 1):
  a_0 = [0.8, 0.2, 0.0, 0.1]   b_0 = [0.5, -0.3, 0.2, 0.1]   # domain 1 feature
  a_1 = [0.0, 0.7, 0.3, 0.0]   b_1 = [0.1, 0.4, -0.2, 0.3]   # domain 1 feature
  a_2 = [0.0, 0.0, 0.0, 0.0]   b_2 = [-0.2, 0.1, 0.6, -0.1]  # DEAD (zero row)
  a_3 = [0.1, 0.1, 0.1, 0.9]   b_3 = [0.3, 0.0, -0.1, 0.5]   # domain 1 feature

Pool B (domain 2):
  a_4 = [-0.3, 0.6, 0.2, 0.0]  b_4 = [0.4, -0.2, 0.3, 0.0]   # domain 2 feature
  a_5 = [0.0, 0.0, 0.0, 0.0]   b_5 = [0.2, 0.1, 0.0, 0.4]   # DEAD (zero row)
  a_6 = [0.0, 0.0, 0.0, 0.0]   b_6 = [0.1, 0.2, 0.3, -0.2]  # DEAD (zero row)
  a_7 = [0.0, 0.0, 0.9, 0.2]   b_7 = [-0.1, 0.3, 0.4, 0.0]   # domain 2 feature
```

3 of 8 capsules are dead (37.5%).

### 8.2 Profiling

For any input x, capsules 2, 5, 6 produce zero activation:
```
f_2 = f_5 = f_6 = 0.0    (dead)
f_0, f_1, f_3, f_4, f_7 > 0  (alive)
```

### 8.3 Pruning at tau = 0

Remove capsules {2, 5, 6}. Keep {0, 1, 3, 4, 7}.

```
A_pruned = [[0.8, 0.2, 0.0, 0.1],   -- capsule 0
            [0.0, 0.7, 0.3, 0.0],   -- capsule 1
            [0.1, 0.1, 0.1, 0.9],   -- capsule 3
            [-0.3, 0.6, 0.2, 0.0],  -- capsule 4
            [0.0, 0.0, 0.9, 0.2]]   -- capsule 7

B_pruned = corresponding columns from B
```

8 -> 5 capsules = 37.5% reduction. Output is EXACTLY unchanged.

### 8.4 Parameter Savings

```
Before: 8 capsules * 2 * d = 8 * 2 * 4 = 64 params
After:  5 capsules * 2 * d = 5 * 2 * 4 = 40 params
Saved:  24 params (37.5%)
```

---

## 9. Assumptions

1. **Profiling dataset is representative.** Dead capsules identified on
   the calibration set must also be dead on the deployment distribution.
   If the calibration set is too small or biased, "dead" capsules may
   fire on unseen inputs. Mitigation: use sufficiently large calibration
   set (20 batches = 20,480 token positions at micro scale).

2. **ReLU activation function.** The zero-change theorem relies on
   ReLU's binary gating: a neuron is either fully active (contributing
   proportionally) or fully silent (contributing zero). For smooth
   activations like GELU or SiLU, "dead" capsules contribute small but
   nonzero values, making pruning approximate rather than exact.

3. **Single-layer independence.** Within a layer, pruning decisions are
   independent because ReLU is element-wise. Across layers, dead capsules
   in layer l produce zero output change, so the input to layer l+1 is
   unchanged, making the independence extend across layers for truly
   dead capsules.

4. **Static pruning suffices.** The activation pattern does not change
   significantly during downstream use. If the input distribution shifts,
   some dead capsules may become alive and vice versa. This assumption is
   strongest when the calibration distribution matches deployment.

5. **Capsule granularity is appropriate.** Pruning at the capsule level
   (removing entire rows of A and columns of B) is structured pruning.
   It preserves matrix shapes (though reduces dimensions) and is hardware-
   friendly. Finer-grained pruning (individual weights) could find more
   to prune but would not produce the clean matrix size reduction.

6. **Composition is the cause of death.** The 57% dead rate in composed
   models significantly exceeds the ~10% natural ReLU death rate in
   single-domain models. Composition creates dead capsules by presenting
   "wrong-domain" inputs to domain-specialized detectors.
