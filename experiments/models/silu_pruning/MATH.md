# SiLU Pruning: Mathematical Foundations

## 1. Problem Statement

Given a capsule MLP using SiLU activation: y = B @ SiLU(A @ x), determine
whether magnitude-threshold pruning can remove capsules with near-zero
activations while bounding quality degradation.

### 1.1 Notation

```
d         -- embedding dimension (64 at micro scale)
P         -- number of capsules per pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)

A in R^{P x d}     -- detector matrix (rows are a_i^T)
B in R^{d x P}     -- expansion matrix (columns are b_i)

a_i in R^d          -- detector vector for capsule i (row i of A)
b_i in R^d          -- expansion vector for capsule i (col i of B)

D = {x_1, ..., x_M} -- calibration dataset of M hidden-state vectors
SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))

mu_i = mean_abs(capsule_i) = (1/M) * sum_{x in D} |SiLU(a_i^T x)|
tau   -- pruning threshold on mean absolute activation
```

---

## 2. SiLU vs ReLU: Activation Properties

### 2.1 ReLU Properties (Reference)

```
ReLU(z) = max(0, z)
```

- Binary gating: output is exactly 0 for z <= 0
- Dead neurons: if a_i^T x <= 0 for all x in D, capsule i contributes
  EXACTLY zero to output. Pruning is LOSSLESS.
- Natural sparsity: ~50% of activations are zero for random inputs
- Expected distribution: bimodal {0, positive values}

### 2.2 SiLU Properties

```
SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))
```

Key properties:
- **Never exactly zero** for z != 0: SiLU(z) = 0 iff z = 0
- **Minimum**: SiLU(z_min) ~ -0.2784 at z_min ~ -1.278
- **Near-zero region**: |SiLU(z)| < epsilon for z in [-C(epsilon), C(epsilon)]
  where C(epsilon) depends on epsilon
- **Asymptotic**: SiLU(z) -> z for z >> 0, SiLU(z) -> 0 for z << 0
- **Smooth**: infinitely differentiable, no hard transitions

### 2.3 The Fundamental Difference

For ReLU: P(activation = 0) > 0 (substantial mass at zero)
For SiLU: P(activation = 0) = 0 (zero probability of exact zero)

This means:
- ReLU dead neuron count: well-defined, binary, stable
- SiLU "dead" neuron count: requires arbitrary threshold, continuous, ambiguous

The question becomes: does the SiLU activation distribution have a
low-magnitude tail that can be pruned, analogous to ReLU's dead neurons?

---

## 3. Magnitude-Threshold Pruning Theory

### 3.1 Error Bound for Pruning Capsule i

**Theorem**: Removing capsule i from a SiLU MLP y = B @ SiLU(A @ x)
introduces an output error bounded by:

```
||delta_y(x)|| = ||b_i * SiLU(a_i^T x)|| = ||b_i|| * |SiLU(a_i^T x)|
```

Taking the expectation over calibration set D:

```
E_x[||delta_y||] = ||b_i|| * mu_i
```

where mu_i = (1/M) * sum_x |SiLU(a_i^T x)| is the mean absolute activation.

**Proof**: The MLP output is:

```
y(x) = sum_{j=1}^{P} b_j * SiLU(a_j^T x)
```

Removing capsule i:

```
y'(x) = sum_{j != i} b_j * SiLU(a_j^T x)
delta_y(x) = y(x) - y'(x) = b_i * SiLU(a_i^T x)
||delta_y(x)|| <= ||b_i|| * |SiLU(a_i^T x)|
```

Taking expectation:

```
E[||delta_y||] <= ||b_i|| * E[|SiLU(a_i^T x)|] = ||b_i|| * mu_i
```

QED.

### 3.2 Aggregate Error from Multiple Pruning

For pruning a set S of capsules simultaneously:

```
||delta_y(x)|| = ||sum_{i in S} b_i * SiLU(a_i^T x)||
              <= sum_{i in S} ||b_i|| * |SiLU(a_i^T x)|
```

Expected aggregate error:

```
E[||delta_y||] <= sum_{i in S} ||b_i|| * mu_i
```

If all pruned capsules have mu_i <= tau:

```
E[||delta_y||] <= tau * sum_{i in S} ||b_i||
```

### 3.3 Comparison with ReLU

For ReLU, when mu_i = 0 (truly dead):
- Error = 0 (exact, lossless)

For SiLU, the best achievable mu_i > 0:
- Error > 0 (always approximate)
- Error is bounded by tau * ||b_i||

The critical question: how small is the smallest mu_i in practice?

---

## 4. Empirical Activation Distribution

### 4.1 Measured Distribution (d=64, P=128, 3 seeds)

After 300 steps of training on single domain:

```
                    min(mu_i)    median(mu_i)    max(mu_i)
SiLU Layer 0:       0.069-0.089    0.089-0.124     0.175-0.203
SiLU Layer 1:       0.060-0.098    0.084-0.128     0.145-0.209
SiLU Layer 2:       0.046-0.088    0.083-0.113     0.153-0.165
SiLU Layer 3:       0.066-0.097    0.091-0.127     0.146-0.194

ReLU Layer 0:       0.000          ~0.5            ~0.8
ReLU Layer 1:       0.000          ~0.1            ~0.3
ReLU Layer 2:       0.000          ~0.05           ~0.2
ReLU Layer 3:       0.000          ~0.04           ~0.1
```

### 4.2 Key Observations

1. **No capsule has mu_i < 0.046 for SiLU** (across all seeds and layers).
   This is ~46x above tau=0.001 and ~4.6x above tau=0.01.

2. **ReLU has substantial mass at mu_i = 0** (17.6% of capsules at f=0
   in single-domain, 57% in composed models).

3. **SiLU distribution is unimodal**: no dead/alive bimodality. All
   capsules have comparable activation magnitudes (range ~3-4x from
   min to max within each layer).

4. **Layer independence**: all layers show similar distributions for SiLU.
   ReLU shows dramatic layer dependence (layer 0: 0% dead, layers 1-3:
   15-38% dead).

### 4.3 The Floor Effect

The minimum mean_abs for SiLU is bounded from below by the intrinsic
nonlinearity. For a capsule with random-like detector a_i:

```
mu_i = E[|SiLU(a_i^T x)|]
```

Since SiLU(z) ~ z for z >> 0 and SiLU(z) ~ 0 for z << 0, the mean
absolute activation depends on the variance of a_i^T x. For normalized
inputs and random a_i:

```
a_i^T x ~ N(0, ||a_i||^2 * Var(x))
```

The mean of |SiLU(Z)| for Z ~ N(0, sigma^2) is bounded below by:

```
E[|SiLU(Z)|] >= E[Z * sigmoid(Z) * 1{Z > 0}]
             >= E[Z * 0.5 * 1{Z > 0}]     (sigmoid > 0.5 for z > 0)
             = 0.5 * E[Z | Z > 0] * P(Z > 0)
             = 0.5 * sigma * sqrt(2/pi) * 0.5
             = sigma / (2 * sqrt(2*pi))
             ~ 0.20 * sigma
```

For sigma ~ 0.5 (typical at d=64), this gives mu_i >= ~0.10, consistent
with the observed floor of 0.046-0.098 (lower due to SiLU's negative
region and non-Gaussian input distribution).

---

## 5. Threshold Sweep Results

### 5.1 Pruning Yield vs Threshold (3-seed mean)

```
tau=0.001:  0.0% pruned,  delta=+0.00%
tau=0.005:  0.0% pruned,  delta=+0.00%
tau=0.010:  0.0% pruned,  delta=+0.00%
tau=0.050:  0.1% pruned,  delta=-0.01%
tau=0.100: 32.0% pruned,  delta=+1.01%
```

### 5.2 Interpretation

There is a **threshold gap** between tau=0.05 (0.1% pruned) and
tau=0.10 (32% pruned). The activation distribution has a floor at
~0.05-0.09, and the threshold must exceed this floor to prune anything.

This means:
- **Safe thresholds (tau <= 0.05)**: prune essentially nothing (0-0.1%)
- **Aggressive threshold (tau = 0.10)**: prune 32% but cuts into
  functional capsules, causing +1.01% degradation

There is no "sweet spot" analogous to ReLU's tau=0 (57% pruned, 0%
degradation). The SiLU activation floor prevents lossless pruning.

### 5.3 Comparison with ReLU Dead Pruning

```
Method                    | % Pruned | Quality Delta
ReLU tau=0 (single-domain)| 17.6%    | -0.00%
ReLU tau=0 (composed)     | 56.8%    | -0.00%
SiLU tau=0.01             | 0.0%     | +0.00%
SiLU tau=0.10             | 32.0%    | +1.01%
```

ReLU pruning is strictly superior: higher compression at zero quality cost.
SiLU pruning is a quality-compression tradeoff with no free compression.

---

## 6. Computational Cost

### 6.1 Profiling

Per layer per batch:
```
Forward through A: O(B * T * P * d)
SiLU computation:  O(B * T * P) (element-wise)
Magnitude accumulation: O(B * T * P) (element-wise abs + sum)
```

Total: O(n_batches * L * B * T * P * d)
At micro scale: O(20 * 4 * 32 * 32 * 128 * 64) = O(3.4G) FLOPs
On Apple Silicon: <2 seconds.

### 6.2 Threshold Selection

Unlike ReLU (tau=0 is the natural choice), SiLU pruning requires
threshold selection. Two approaches:

1. **Fixed quality budget**: Set maximum allowable delta_quality,
   binary search for the threshold that achieves it.

2. **Distribution-based**: Set tau at the N-th percentile of the
   mu_i distribution (e.g., prune the bottom 10% by magnitude).

Neither approach avoids the fundamental problem: SiLU's activation
floor means small thresholds prune nothing and large thresholds
prune functional capsules.

---

## 7. Worked Numerical Example

At d=4, P=4 (toy scale):

### 7.1 SiLU Activations

```
For input x = [0.5, -0.3, 0.8, 0.1]:

Capsule 0: a_0 = [0.2, 0.1, 0.3, 0.1]
  pre = a_0^T x = 0.38
  SiLU(0.38) = 0.38 * sigmoid(0.38) = 0.38 * 0.594 = 0.226

Capsule 1: a_1 = [-0.1, 0.4, -0.2, 0.3]
  pre = a_1^T x = -0.27
  SiLU(-0.27) = -0.27 * sigmoid(-0.27) = -0.27 * 0.433 = -0.117

Capsule 2: a_2 = [0.01, 0.01, 0.01, -0.01]  (nearly zero detector)
  pre = a_2^T x = 0.009
  SiLU(0.009) = 0.009 * sigmoid(0.009) = 0.009 * 0.502 = 0.0045

Capsule 3: a_3 = [0.3, -0.2, 0.5, 0.0]
  pre = a_3^T x = 0.61
  SiLU(0.61) = 0.61 * sigmoid(0.61) = 0.61 * 0.648 = 0.395
```

### 7.2 Pruning Analysis

```
mu_0 ~ 0.226  (active)
mu_1 ~ 0.117  (active)
mu_2 ~ 0.005  (near-zero — candidate for pruning)
mu_3 ~ 0.395  (active)
```

At tau=0.01: only capsule 2 would be pruned (mu_2 = 0.005 < 0.01).
Error: ||b_2|| * 0.005 — bounded and small.

But in practice, with trained detectors at d=64, the minimum mu_i
is ~0.046, not 0.005. The "nearly zero detector" scenario above does
not occur after training — gradient-based optimization pushes all
detectors to have non-negligible alignment with input features.

---

## 8. Assumptions

1. **Mean absolute activation is an adequate pruning criterion.** The
   mean may hide a capsule that is usually inactive but fires strongly
   for rare inputs. The max_abs metric addresses this but finds even
   fewer pruning candidates.

2. **The activation floor is a property of SiLU, not of micro scale.**
   This assumption needs macro validation. At d=896 with diverse inputs,
   the activation floor may be different (higher diversity could push
   some detectors further from any input pattern).

3. **Single-domain training is representative.** The experiment trains
   on one domain. Composed models (concatenation of two domains) might
   create more pruning opportunities as cross-domain capsules have
   weaker activations. However, the SiLU floor effect would still
   prevent exact-zero pruning.

4. **The error bound is tight.** The triangle inequality bound
   (E[||delta_y||] <= tau * sum||b_i||) may be loose if pruned
   capsules' contributions partially cancel. Empirical delta (+1.01%
   at 32% pruning) is consistent with a moderate bound.

5. **Profiling dataset is representative.** Same as ReLU pruning:
   the calibration set must cover the deployment distribution. With
   20 batches x 32 samples = 20,480 token positions, this is
   sufficient at micro scale.

6. **SiLU's smoothness makes gradient-based training explore more of
   the activation space.** Unlike ReLU where neurons can die (zero
   gradient), SiLU always provides gradient signal, keeping all
   capsules "alive" throughout training. This is both SiLU's strength
   (no dead neurons, no wasted capacity) and its limitation for
   pruning (no free compression).
