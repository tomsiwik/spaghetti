# Activation-Based Behavioral Deduplication: Mathematical Foundations

## 1. Problem Statement

Given a composed ReLU MLP with 2P capsules (P per domain), identify
functionally redundant capsule pairs using activation-based metrics
rather than weight-space cosine similarity. Weight-cosine (Exp 8)
found only 1.9% redundancy. The question is whether behavioral
analysis discovers additional functional redundancy invisible
in weight space.

### 1.1 Notation

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
N_d       -- number of domains (2 for binary split)
P_total   -- total capsules after concatenation = P * N_d = 256
L         -- number of transformer layers (4 at micro scale)
N         -- number of input positions profiled

A in R^{P_total x d}   -- detector matrix (rows are a_i^T)
B in R^{d x P_total}   -- expansion matrix (columns are b_i)

a_i in R^d              -- detector vector for capsule i
b_i in R^d              -- expansion vector for capsule i
h_i(x) = ReLU(a_i^T x) -- activation of capsule i on input x (scalar)

f_i in {0,1}^N          -- binary fire vector: f_i[n] = 1 iff h_i(x_n) > 0
c_i = |{n : f_i[n] = 1}| -- fire count for capsule i

J(i,j)                  -- co-activation Jaccard similarity
rho(i,j)                -- output contribution correlation
cos_b(i,j)              -- cosine similarity of expansion vectors b_i, b_j
```

---

## 2. Behavioral Similarity Metrics

### 2.1 Co-Activation Jaccard Similarity

For capsules i and j with binary fire vectors f_i, f_j over N positions:

```
J(i,j) = |f_i AND f_j| / |f_i OR f_j|
        = (sum_n f_i[n] * f_j[n]) / (c_i + c_j - sum_n f_i[n] * f_j[n])
```

Properties:
- J(i,j) in [0, 1]
- J(i,j) = 1 iff capsules fire on identical input sets
- J(i,j) = 0 iff capsules never co-fire
- Undefined (set to 0) if both are dead (c_i = c_j = 0)

**Matrix computation**: The co-fire count matrix is:

```
C = F^T F       where F in R^{N x P_total} is the binary fire matrix
C[i,j] = sum_n f_i[n] * f_j[n] = |fire_i AND fire_j|
```

And the Jaccard matrix:

```
J[i,j] = C[i,j] / (c_i + c_j - C[i,j])
```

This is O(N * P^2) to compute via the matrix product.

### 2.2 Why Jaccard Diverges from Weight Cosine

Two capsules i, j with low weight cosine cos(a_i, a_j) can have
high Jaccard J(i,j) when:

1. **Low-rank data distribution**: If inputs x live in a k-dimensional
   subspace (k << d), the effective decision boundaries of a_i and a_j
   may be identical when projected onto this subspace, even though
   a_i and a_j differ in the full d-dimensional space.

2. **ReLU many-to-one mapping**: ReLU(a_i^T x) > 0 defines a half-space
   in R^d. Two half-spaces defined by different normals can have
   identical intersection with a data manifold.

**Formal statement**: Let M be the data manifold. Define the
effective activation region:

```
R_i = {x in M : a_i^T x > 0}
```

Weight cosine measures angle between a_i and a_j in R^d.
Jaccard measures overlap between R_i and R_j relative to M.
These are equal only when M = R^d (isotropic data).

For concentrated data distributions (as in transformers, where
hidden states occupy a low-rank subspace):

```
cos(a_i, a_j) << 1   but   J(i,j) -> 1
```

is possible whenever the directions that distinguish a_i from a_j
are orthogonal to M.

### 2.3 Output Contribution Correlation

For each position n, capsule i's output contribution is:

```
o_i(x_n) = h_i(x_n) * b_i     (a d-dimensional vector)
```

The pairwise output dot product summed over positions:

```
D[i,j] = sum_n dot(o_i(x_n), o_j(x_n))
        = sum_n h_i(x_n) * h_j(x_n) * (b_i . b_j)
        = (sum_n h_i(x_n) * h_j(x_n)) * (b_i . b_j)
        = H[i,j] * B_dot[i,j]
```

where:
```
H[i,j] = sum_n h_i(x_n) * h_j(x_n)    -- activation magnitude co-occurrence
B_dot[i,j] = b_i . b_j                  -- expansion vector alignment
```

The normalized output correlation:

```
rho(i,j) = D[i,j] / sqrt(D[i,i] * D[j,j])
```

This captures functional redundancy: rho(i,j) near 1 means both capsules
produce nearly identical output contributions across the dataset, regardless
of their weight representations.

### 2.4 Combined Redundancy Criterion

A pair (i,j) is **behaviorally redundant** if:

```
J(i,j) > tau_J           (fire on similar inputs)
AND
rho(i,j) > tau_rho        (produce similar outputs)
AND
c_i > 0 AND c_j > 0      (both alive)
```

Default thresholds: tau_J = 0.7, tau_rho = 0.3.

The choice of tau_rho significantly affects results. Empirically:
- tau_rho = 0.3: 19.3% of capsules in behavioral-only redundant pairs
- tau_rho = 0.5: 10.8% (moderate correlation, still above kill threshold)
- tau_rho = 0.7: 1.4% (strong correlation, below kill threshold)

This shows that co-firing (high Jaccard) is common in Layer 0, but
truly correlated output contributions (high rho) are much rarer. The
gap between Jaccard overlap and output correlation reflects the fact
that co-firing capsules may produce outputs in different directions
(different b_i vectors), even though they activate on the same inputs.

The Jaccard criterion alone is necessary but not sufficient: two capsules
may co-fire but produce opposite outputs (if b_i . b_j < 0). The output
correlation criterion ensures they are functionally redundant, not just
co-activated.

---

## 3. Merging Strategy

We reuse the capsule_dedup merging rule (a-average, b-sum) since it
preserves the additive output that downstream layers expect:

```
a_merged = (a_i + a_j) / 2
b_merged = b_i + b_j
```

**Justification under behavioral redundancy**: When J(i,j) is high,
capsules i and j fire on nearly identical inputs. For any input x
where both fire:

```
output_unmerged = b_i * h_i(x) + b_j * h_j(x)
               ~ (b_i + b_j) * h_avg(x)     [since h_i(x) ~ h_j(x) when co-firing]
               = b_merged * h_merged(x)
```

The approximation quality is:

```
||output_unmerged - output_merged||
  = ||(b_i * h_i + b_j * h_j) - (b_i + b_j) * h_merged||
  = ||b_i * (h_i - h_merged) + b_j * (h_j - h_merged)||
  <= ||b_i|| * |h_i - h_merged| + ||b_j|| * |h_j - h_merged|
```

When J(i,j) is high, h_i and h_j are similar for most inputs,
so h_merged = ReLU(a_merged^T x) is a good approximation.

---

## 4. Computational Cost

### 4.1 Profiling

Per layer, per batch of size B*T:
```
Fire mask:    O(B * T * P)          -- one ReLU evaluation
Co-fire:      O(B * T * P^2)       -- F^T @ F matrix product
H matrix:     O(B * T * P^2)       -- h^T @ h matrix product
```

For n_batches = 20, B = 32, T = 32, P = 256:
```
Per layer: 20 * 1024 * 256^2 = 1.34B operations (for co-fire)
Total (4 layers): 5.4B operations
```

This is ~100x more than weight-cosine (which is O(P^2 * d) = 4.2M).
Still negligible in wall time (~1 second on Apple Silicon).

### 4.2 Comparison with Weight-Cosine

```
Weight-cosine: O(P^2 * d) per layer, no data needed
Behavioral:    O(N * P^2) per layer, requires profiling data

At micro: P=256, d=64, N=20*1024=20480
  Weight-cosine: 4.2M ops/layer
  Behavioral:    1.3B ops/layer (300x more)
```

The behavioral approach trades compute for information: it accesses
the actual data distribution, which weight-cosine cannot.

---

## 5. Expected Behavioral Redundancy

### 5.1 Layer 0 Analysis

Layer 0 receives direct embeddings (close to input space). At micro
scale, the two domains (a-m vs n-z names) share most character
distributions. Layer 0 capsules detect low-level character patterns
that are common across domains.

**Expected**: High co-activation Jaccard in Layer 0 because:
- Inputs are high-dimensional embeddings with shared character statistics
- Layer 0 capsules learn generic character n-gram detectors
- Same characters trigger same detectors regardless of domain

### 5.2 Deeper Layers

Layers 1-3 develop domain-specific representations. By layer 3,
capsules encode domain-specific sequence patterns. Cross-pool
co-activation should decrease with depth.

**Expected**: Low Jaccard in layers 1-3 because domain specialization
causes capsules from different pools to fire on different inputs.

### 5.3 Dead Capsule Interaction

At micro scale, ~60% of capsules are dead (never fire). Dead capsules
have undefined Jaccard (0/0, set to 0). The behavioral analysis
operates only on the ~40% alive capsules, giving a smaller but
more meaningful comparison space.

---

## 6. Worked Numerical Example

At d=4, P=4 per domain (toy scale):

### 6.1 Setup

```
Pool A capsules 0-3:
  a_0 = [0.8, 0.2, 0.1, 0.0]    b_0 = [0.5, -0.3, 0.1, 0.2]
  a_1 = [0.0, 0.9, 0.1, 0.0]    b_1 = [0.1, 0.4, -0.1, 0.3]
  a_2 = [-0.3, 0.1, 0.7, 0.2]   b_2 = [-0.2, 0.1, 0.5, -0.1]
  a_3 = [0.1, -0.1, 0.2, 0.8]   b_3 = [0.3, 0.0, -0.1, 0.4]

Pool B capsules 4-7:
  a_4 = [0.2, 0.8, -0.1, 0.3]   b_4 = [0.4, -0.1, 0.2, 0.1]
  a_5 = [0.6, 0.0, 0.5, 0.1]    b_5 = [0.3, -0.2, 0.3, 0.0]
  a_6 = [0.0, 0.1, 0.0, 0.9]    b_6 = [0.2, 0.1, -0.2, 0.5]
  a_7 = [-0.5, 0.3, 0.6, 0.0]   b_7 = [-0.1, 0.2, 0.4, -0.2]
```

### 6.2 Weight Cosine

```
cos(a_0, a_5) = (0.8*0.6 + 0.2*0 + 0.1*0.5 + 0*0.1) / (0.83 * 0.79)
              = 0.53 / 0.66 = 0.80  (below tau=0.95, NOT weight-redundant)

cos(a_3, a_6) = (0.1*0 + (-0.1)*0.1 + 0.2*0 + 0.8*0.9) / (0.84 * 0.91)
              = 0.71 / 0.76 = 0.93  (close but below tau=0.95)
```

### 6.3 Behavioral Analysis

Suppose data distribution concentrates on x ~ [alpha, alpha, beta, 0]
(first two coordinates correlated, third varies, fourth near zero).

For capsules 0 and 5:
```
a_0^T x = 0.8*alpha + 0.2*alpha + 0.1*beta = 1.0*alpha + 0.1*beta
a_5^T x = 0.6*alpha + 0.0*alpha + 0.5*beta = 0.6*alpha + 0.5*beta
```

Both fire whenever alpha > 0 and beta > 0 (most inputs).
Despite cos(a_0, a_5) = 0.80 < 0.95, they have high Jaccard
because the data rarely probes the directions where they disagree.

For capsules 3 and 6:
```
a_3^T x = 0.1*alpha - 0.1*alpha + 0.2*beta + 0 = 0.2*beta
a_6^T x = 0 + 0.1*alpha + 0 + 0 = 0.1*alpha
```

Different activation patterns: 3 fires when beta > 0, 6 fires when
alpha > 0. Despite cos(a_3, a_6) = 0.93, their Jaccard may be low
if alpha and beta are uncorrelated.

This demonstrates how behavioral analysis captures data-dependent
redundancy that weight-cosine misses (pair 0,5) and how weight-cosine
can overestimate redundancy (pair 3,6).

---

## 7. Assumptions

1. **Profiling data is representative**: The 20 batches x 32 samples
   must sample the data distribution adequately. Exp 12 validated
   that 20 batches is sufficient for binary dead/alive classification.
   Jaccard requires finer-grained statistics.

2. **Behavioral redundancy implies safe merging**: Two capsules that
   co-fire and produce correlated outputs can be merged using the
   a-average/b-sum rule. This assumes the merged detector preserves
   the activation region to adequate precision.

3. **Layer independence**: We compute behavioral similarity independently
   per layer. Cross-layer interactions (a dead capsule in layer 2 that
   receives no activation from a dead capsule in layer 1) are not
   modeled.

4. **Threshold sensitivity**: The Jaccard threshold (0.7) and output
   correlation threshold (0.3) are hyperparameters. The experiment
   sweeps multiple values to assess sensitivity.

5. **Rank-1 capsule structure**: The output contribution factorizes as
   h_i(x) * b_i. For multi-layer experts, the output contribution
   would not factor this way, requiring more expensive metrics.
