# Huffman Macro Routing Skew: Mathematical Foundations

## 1. Problem Statement

Given a production MoE with L experts and an empirical expert utilization
distribution f = (f_1, ..., f_L), determine whether Huffman tree routing
provides at least 5% expected routing depth reduction compared to a balanced
binary tree.

This requires two conditions:
1. The utilization distribution f is sufficiently non-uniform (H(f) < 0.95 * log2(L))
2. The Huffman expected depth is at least 5% less than the balanced depth

---

## 2. Notation

```
L          = number of leaf experts (e.g., 8 for Mixtral, 256 for DeepSeek-V3, 512 for Qwen3)
D_bal      = ceil(log2(L)), depth of balanced binary tree
f_l        = utilization frequency of expert l, sum_l f_l = 1
H(f)       = -sum_l f_l * log2(f_l), Shannon entropy (bits)
H_max      = log2(L), maximum entropy (uniform distribution)
H_norm     = H(f) / H_max, normalized entropy in [0, 1]
E[d]       = sum_l f_l * d_l, expected routing depth under Huffman
R          = (D_bal - E[d]) / D_bal, fractional depth reduction
G(f)       = Gini coefficient of distribution f
alpha      = Zipf exponent (f_i proportional to 1/(i+1)^alpha)
w          = uniform mixture weight (balance loss strength)
```

---

## 3. Expert Utilization Distribution Model

### 3.1 Mixture Model

Real MoE routers operate under two competing forces:
- **Natural specialization**: tokens prefer certain experts, creating Zipf-like skew
- **Balance loss/bias**: auxiliary loss or per-expert bias pushes toward uniform

We model the equilibrium as a mixture:

```
f_i = w * (1/L) + (1 - w) * Z_i(alpha)
```

where Z_i(alpha) = (1/(i+1)^alpha) / sum_j (1/(j+1)^alpha) is the Zipf distribution
and w in [0, 1] controls balance loss strength.

### 3.2 Parameter Interpretation

| Parameter | Physical meaning |
|-----------|-----------------|
| w = 0.0   | No balance loss (pure specialization) |
| w = 0.1   | DeepSeek-V3 style (auxiliary-loss-free, soft bias) |
| w = 0.3   | Moderate balance loss |
| w = 0.5   | Standard balance loss (Switch Transformer) |
| w = 0.7   | Strong balance loss with capacity factor |
| alpha = 0.3 | Mild specialization |
| alpha = 0.5 | Moderate specialization |
| alpha = 1.0 | Strong specialization (Zipf's law) |
| alpha = 1.5 | Very strong specialization |

### 3.3 Entropy of the Mixture

For the mixture distribution:

```
H(w, alpha, L) = -sum_{i=1}^L f_i * log2(f_i)
```

where f_i = w/L + (1-w) * Z_i(alpha).

**Property**: H is monotonically increasing in w (more uniform = higher entropy)
and monotonically decreasing in alpha (more skew = lower entropy).

**Boundary condition**: At w = 1, H = log2(L) = H_max (perfectly uniform).
At w = 0, H = H(Zipf(alpha)) which depends on alpha and L.

---

## 4. Huffman Depth Reduction Analysis

### 4.1 Shannon Bound

For any binary prefix code (including Huffman):

```
H(f) <= E[d] < H(f) + 1
```

The depth reduction is therefore bounded:

```
R = (D_bal - E[d]) / D_bal
R_max = (D_bal - H(f)) / D_bal       (upper bound on reduction)
R_min = (D_bal - H(f) - 1) / D_bal   (lower bound on reduction)
```

### 4.2 Kill Criterion Derivation

Kill criterion 1: H_norm > 0.95
  => H(f) > 0.95 * log2(L)
  => D_bal - H(f) < 0.05 * log2(L)
  => R_max < 0.05 * log2(L) / D_bal

For L that is a power of 2, D_bal = log2(L), so:
  => R_max < 0.05 = 5%

This means near-uniform distributions (H_norm > 0.95) can NEVER provide
more than ~5% Huffman reduction. The kill criteria are mathematically linked.

Kill criterion 2: R < 0.05
  => E[d] > 0.95 * D_bal

### 4.3 Critical Skew Boundary

Combining the mixture model with the kill criteria, we can find the boundary
in (alpha, w) space where Huffman becomes useful.

Empirically (from the sensitivity sweep), the boundary is approximately:

```
For L >= 64:   alpha >= 0.7 with w <= 0.1   (heavy skew, weak balance)
               alpha >= 1.0 with w <= 0.5   (strong skew, moderate balance)

For L = 8:     alpha >= 1.0 with w <= 0.3   (need stronger skew at small L)
```

The boundary can be approximated as:

```
alpha_critical(w) approximately = 0.6 + 0.4 * w    (for L >= 64)
```

meaning that stronger balance loss (higher w) requires stronger natural
specialization (higher alpha) to maintain Huffman benefit.

---

## 5. Gradient Flow Through Deep Huffman Paths

### 5.1 Sigmoid Chain Gradient

Each gate in the Huffman tree computes g_i(x) = sigmoid(w_i^T x + b_i).

The gradient through a chain of D gates is:

```
d(prod gates)/dz = prod_{i=1}^{D} [g_i * (1 - g_i)]
```

Since sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)), and max(p*(1-p)) = 0.25
at p = 0.5:

```
|gradient_chain| <= 0.25^D
```

### 5.2 Critical Depth

At "sharp" gates (p ~ 0.9), the gradient per gate is p*(1-p) = 0.09.
The gradient magnitude after D gates is 0.09^D.

| D  | Gradient magnitude | Status |
|----|-------------------|--------|
| 3  | 7.3e-4 | Healthy |
| 6  | 5.3e-7 | Weak |
| 9  | 3.9e-10 | Vanished |
| 12 | 2.8e-13 | Dead |

**Critical depth for sharp gates**: D_crit approximately = 6 (gradient < 1e-6 at D=5.7).

### 5.3 Huffman Max Depth at Zipf(1.0)

| L   | D_bal | D_max(Huffman) | Gradient(sharp) |
|-----|-------|----------------|-----------------|
| 8   | 3     | 4              | 6.6e-5 |
| 16  | 4     | 6              | 5.3e-7 |
| 32  | 5     | 7              | 4.8e-8 |
| 64  | 6     | 8              | 4.3e-9 |
| 128 | 7     | 9              | 3.9e-10 |
| 256 | 8     | 11             | 3.1e-12 |
| 512 | 9     | 12             | 2.8e-13 |

**Conclusion**: Gradient vanishing is a practical concern at L >= 128.
The deepest Huffman paths (rare experts) receive near-zero gradients.
However, these are precisely the RARELY USED experts, so the training
impact is weighted by their low utilization frequency.

**Mitigation strategies**:
1. Depth-scaled learning rate: lr_leaf = lr_base * (D_max / depth_leaf)
2. Tree depth cap: limit Huffman max depth to D_bal + 2
3. The rare experts at deep paths need fewer gradient updates precisely
   because they handle fewer tokens

---

## 6. Worked Example: DeepSeek-V3 (L=256, Moderate Skew)

### Setup
```
L = 256, D_bal = ceil(log2(256)) = 8
Model: mixture_zipf(alpha=0.6, w=0.3)  [moderate skew, weak balance]
```

### Computation
```
f_i = 0.3 * (1/256) + 0.7 * Z_i(0.6)

Zipf(0.6): f_1 = 0.0086, f_2 = 0.0057, ..., f_256 = 0.0018
Mixture:   f_1 = 0.3*0.0039 + 0.7*0.0086 = 0.0072
           f_256 = 0.3*0.0039 + 0.7*0.0018 = 0.0024

H(f) = 7.75 bits
H_max = 8.0 bits
H_norm = 7.75/8.0 = 0.968

Huffman tree:
  E[d] = 7.78
  Max depth = 9

R = (8 - 7.78) / 8 = 2.7%
```

### Kill Assessment
- H_norm = 0.968 > 0.95  => KILLED by near-uniform criterion
- R = 2.7% < 5%          => KILLED by insufficient reduction

This moderate skew scenario does NOT provide enough benefit.

### What Would Survive
```
Same L=256 with alpha=1.0, w=0.1 (heavy skew, weak balance):
H_norm = 0.813
R = 18.3%
E[d] = 6.53 vs D_bal = 8

This saves 1.47 gate evaluations per token on average.
At D_bal = 8 gates per path, this is significant.
```

---

## 7. Key Assumptions

1. **Mixture model approximates reality.** The Zipf+uniform mixture is a
   first-order approximation. Real distributions may have multi-modal structure
   (e.g., domain clusters) that could produce even more skew.

2. **Frequency stationarity.** The Huffman tree is built once. Distribution
   shifts (different input domains at inference time) may change the optimal
   tree. The splay-tree experiment (KILLED at micro scale) was one approach;
   periodic rebuilding is another.

3. **Top-k routing induces correlation.** With top-k, the utilization of
   expert i depends on all other experts' scores. This creates complex
   dependencies not captured by independent Zipf modeling.

4. **Balance loss strength is unknown.** We model w as a parameter, but the
   actual effective balance strength in production systems depends on the
   specific auxiliary loss coefficient, training dynamics, and expert capacity.
   The key empirical question is: what is the actual H_norm of a trained model?
