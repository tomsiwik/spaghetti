# Capsule Deduplication: Mathematical Foundations

## 1. Problem Statement

Given a composed ReLU MLP with 2P capsules (P per domain, formed by
concatenating two independently-trained pools), identify and merge
capsules with near-identical detector vectors to reduce parameter
count while preserving model quality.

### 1.1 Notation

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
N_d       -- number of domains (2 for binary split)
P_total   -- total capsules after concatenation = P * N_d
L         -- number of transformer layers (4 at micro scale)

A in R^{P_total x d}   -- detector matrix (rows are a_i^T)
B in R^{d x P_total}   -- expansion matrix (columns are b_i)

a_i in R^d              -- detector vector for capsule i (row i of A)
b_i in R^d              -- expansion vector for capsule i (col i of B)

tau                     -- cosine similarity threshold (sweep: 0.90, 0.95, 0.99)
```

---

## 2. Redundancy Detection

### 2.1 Cosine Similarity of Detector Vectors

For capsules i and j, their cosine similarity is:

```
cos(a_i, a_j) = (a_i^T a_j) / (||a_i|| * ||a_j||)
```

**Claim**: For rank-1 capsules with ReLU gating, cos(a_i, a_j) > tau
implies near-identical activation patterns.

**Proof sketch**: Capsule i fires iff a_i^T x > 0. The set of inputs
that fire capsule i is the half-space H_i = {x : a_i^T x > 0}.
If cos(a_i, a_j) = 1, then a_j = c * a_i for some c > 0, so
H_i = H_j exactly. For cos(a_i, a_j) = tau close to 1, the angle
theta = arccos(tau) between a_i and a_j is small:

```
tau = 0.90  =>  theta = 25.8 degrees
tau = 0.95  =>  theta = 18.2 degrees
tau = 0.99  =>  theta =  8.1 degrees
```

The fraction of the input distribution where i and j disagree
(one fires, the other does not) is proportional to theta / 180.
At tau = 0.95, disagreement occurs on at most 10% of inputs (for
isotropically distributed inputs).

**Important caveat**: This assumes inputs are distributed across all
directions. In practice, transformer hidden states concentrate in a
low-dimensional subspace, so the effective disagreement may be much
lower (or higher, if the subspace sits near the decision boundary).

### 2.2 Why Cosine Suffices for Rank-1 (But Not Higher Rank)

For rank-1 capsules, the detector vector a_i IS the entire routing
mechanism. The routing decision ReLU(a_i^T x) depends only on the
direction and magnitude of a_i relative to x.

For multi-layer experts (as in standard MoE), the routing behavior
depends on the full nonlinear function, not just the first-layer
weights. Metrics like CKA (Centered Kernel Alignment) or activation
correlation are needed. But for rank-1 capsules, the a_i vector IS
both the weight and the routing key, so cosine similarity is exact.

### 2.3 Cross-Pool vs Within-Pool Redundancy

We compute the full P_total x P_total pairwise cosine similarity
matrix but distinguish:

```
S_cross[i,j] = cos(a_i, a_j)  where i in Pool_A, j in Pool_B
S_within_A[i,j] = cos(a_i, a_j)  where both i, j in Pool_A
S_within_B[i,j] = cos(a_i, a_j)  where both i, j in Pool_B
```

Cross-pool redundancy reveals shared knowledge (both domains learned
the same detector). Within-pool redundancy reveals internal over-
parameterization. Both types can be deduplicated.

---

## 3. Merging Strategy

### 3.1 The a-Average, b-Sum Rule

Given capsules i and j identified as redundant (cos(a_i, a_j) > tau):

```
a_merged = (a_i + a_j) / 2         -- average detector
b_merged = b_i + b_j               -- sum expansion
```

**Justification for averaging a**: The merged detector should fire on
the same inputs as both originals. Since a_i and a_j point in nearly
the same direction (cos > tau), their average points in a direction
between them, preserving the activation region.

```
||a_merged - a_i|| = ||(a_j - a_i) / 2||

For cos(a_i, a_j) = tau, ||a_i|| = ||a_j|| = 1 (unit vectors):
  ||a_j - a_i||^2 = 2 - 2*tau
  ||a_merged - a_i|| = sqrt((2 - 2*tau) / 4) = sqrt((1 - tau) / 2)

At tau = 0.95: ||a_merged - a_i|| = 0.158 (small deviation)
At tau = 0.99: ||a_merged - a_i|| = 0.071 (very small)
```

**Justification for summing b**: Before merging, the unmerged model
computes (for inputs where both fire):

```
output_unmerged = b_i * (a_i^T x) + b_j * (a_j^T x)
               ~ b_i * (a^T x) + b_j * (a^T x)    [since a_i ~ a_j]
               = (b_i + b_j) * (a^T x)
```

So b_merged = b_i + b_j preserves the full-strength additive output
that downstream layers expect. If we averaged b instead, the merged
capsule would produce half the expected output magnitude, violating
the downstream distribution.

### 3.2 Handling Magnitude Differences

The a vectors may have different norms. For merging:

```
a_merged = (a_i / ||a_i|| + a_j / ||a_j||) / 2 * (||a_i|| + ||a_j||) / 2
```

This normalizes directions before averaging, then restores the average
magnitude. In practice, we simply average the raw vectors since the
norms tend to be similar after identical training procedures.

### 3.3 Multi-Way Merging

If capsules {i, j, k} form a cluster (all pairwise cos > tau):

```
a_merged = mean(a_i, a_j, a_k)
b_merged = b_i + b_j + b_k
```

The b-sum rule extends naturally: the unmerged model would sum all
three contributions, so the merged capsule must produce the same
total output.

---

## 4. Deduplication Algorithm

### 4.1 Greedy Pairwise Merging

For each layer l:

```
1. Compute S = cosine_similarity(A_l)   -- (P_total, P_total) matrix
2. Find all pairs (i,j) with S[i,j] > tau, i != j
3. Sort pairs by S[i,j] descending (merge most similar first)
4. For each pair (i,j) in sorted order:
   a. If both i and j are still unmerged:
      - Create merged capsule: a_new = avg(a_i, a_j), b_new = b_i + b_j
      - Mark i and j as merged
5. Collect: unmerged capsules + new merged capsules
6. Rebuild A_l and B_l from collected capsules
```

This is a greedy matching algorithm. It does not find the globally
optimal matching but is O(P_total^2) per layer, which is negligible.

### 4.2 Cluster-Based Merging (Alternative)

Instead of pairwise greedy, cluster capsules by cosine similarity:

```
1. Build graph: node = capsule, edge if cos > tau
2. Find connected components
3. For each component of size k:
   a_merged = mean(a_i for i in component)
   b_merged = sum(b_i for i in component)
```

This handles multi-way redundancy (3+ capsules all similar) in one
pass. We implement this as the primary algorithm.

---

## 5. Computational Cost

### 5.1 Deduplication (One-Time)

Per layer:
```
Cosine similarity matrix: O(P_total^2 * d)
At micro scale: O(256^2 * 64) = O(4.2M) operations
```

For L=4 layers: O(16.8M) operations total. Negligible (<1ms on any hardware).

### 5.2 Inference After Deduplication

If k capsules are removed per layer:
```
FLOPs saved per layer per token: 2 * k * d
At micro scale (k ~ 50, d = 64): 6,400 FLOPs/layer/token saved

Before: 2 * 256 * 64 = 32,768 FLOPs/layer/token
After:  2 * 206 * 64 = 26,368 FLOPs/layer/token
Savings: ~20%
```

### 5.3 Parameter Savings

Per layer:
```
Params removed: k * d (from A) + d * k (from B) = 2 * k * d
At micro scale (k ~ 50, d = 64): 6,400 params/layer

Total for L=4: 25,600 params saved
Original composed model: ~202K params (shared) + 4 * 32,768 = 131K capsule params
Savings: 25,600 / 131,072 = ~20% of capsule params
```

---

## 6. Expected Redundancy Estimate

The Procrustes experiment (Exp 3) found 54% of fine-tuning knowledge
is shared between the two domains. If this manifests as redundant
capsules:

```
Expected redundant pairs: 0.54 * P = 0.54 * 128 = ~69 capsules/domain
(at cos > 0.95 threshold -- actual count will be lower)
```

For N_d=2 domains, P=128/domain:
```
Cross-pool pairs to check: 128 * 128 = 16,384
Expected matches at cos > 0.95: unknown (experiment will measure)

Optimistic: 50% match rate -> 64 capsules merged -> 25% reduction
Pessimistic: 5% match rate -> 6 capsules merged -> 2.5% reduction
```

The key unknown is whether "54% shared knowledge" (measured by
Procrustes alignment of weight deltas) corresponds to 54% shared
capsules (measured by cosine similarity of individual a_i vectors).

---

## 7. Worked Numerical Example

At d=4, P=4 per domain (toy scale):

### 7.1 Two Domain Pools

```
Pool A (capsules 0-3):
  a_0 = [0.8, 0.2, 0.0, 0.1]   b_0 = [0.5, -0.3, 0.2, 0.1]
  a_1 = [0.0, 0.7, 0.3, 0.0]   b_1 = [0.1, 0.4, -0.2, 0.3]
  a_2 = [0.3, 0.0, 0.8, 0.1]   b_2 = [-0.2, 0.1, 0.6, -0.1]
  a_3 = [0.1, 0.1, 0.1, 0.9]   b_3 = [0.3, 0.0, -0.1, 0.5]

Pool B (capsules 4-7):
  a_4 = [0.79, 0.22, 0.01, 0.09]  b_4 = [0.4, -0.2, 0.3, 0.0]   # similar to a_0
  a_5 = [0.0, 0.1, 0.0, 0.95]     b_5 = [0.2, 0.1, 0.0, 0.4]    # similar to a_3
  a_6 = [-0.5, 0.6, 0.3, 0.0]     b_6 = [0.1, 0.2, 0.3, -0.2]   # unique
  a_7 = [0.0, 0.0, 0.9, 0.2]      b_7 = [-0.1, 0.3, 0.4, 0.0]   # somewhat similar to a_2
```

### 7.2 Cosine Similarity Matrix (Cross-Pool Only)

```
         a_4    a_5    a_6    a_7
a_0    0.999  0.174 -0.310  0.090    <- a_0 ~ a_4 (cos=0.999)
a_1    0.258  0.085  0.488  0.316
a_2    0.335  0.194  0.040  0.967    <- a_2 ~ a_7 (cos=0.967)
a_3    0.202  0.975 -0.035  0.290    <- a_3 ~ a_5 (cos=0.975)
```

### 7.3 Merging at tau = 0.95

Three pairs exceed threshold:
1. (0, 4): cos = 0.999  -> merge
2. (3, 5): cos = 0.975  -> merge
3. (2, 7): cos = 0.967  -> merge

After merging:
```
Merged capsule M0 (from 0+4):
  a_M0 = (a_0 + a_4) / 2 = [0.795, 0.210, 0.005, 0.095]
  b_M0 = b_0 + b_4       = [0.9, -0.5, 0.5, 0.1]

Merged capsule M3 (from 3+5):
  a_M3 = (a_3 + a_5) / 2 = [0.05, 0.10, 0.05, 0.925]
  b_M3 = b_3 + b_5       = [0.5, 0.1, -0.1, 0.9]

Merged capsule M2 (from 2+7):
  a_M2 = (a_2 + a_7) / 2 = [0.15, 0.00, 0.85, 0.15]
  b_M2 = b_2 + b_7       = [-0.3, 0.4, 1.0, -0.1]
```

Surviving capsules: {1, 6, M0, M2, M3} = 5 capsules (from 8)
**Reduction: 8 -> 5 = 37.5%**

### 7.4 Quality Check

For input x = [0.5, -0.3, 0.8, 0.1]:

Before merging (all 8 capsules):
```
h = ReLU(A @ x) = ReLU([0.36, -0.03, 0.78, 0.14, 0.355, 0.065, -0.41, 0.74])
                = [0.36, 0, 0.78, 0.14, 0.355, 0.065, 0, 0.74]
output = sum of b_i * h_i for active capsules
```

After merging (5 capsules):
```
h_merged = ReLU(A_merged @ x)
For M0: a_M0^T x = 0.795*0.5 + 0.210*(-0.3) + 0.005*0.8 + 0.095*0.1 = 0.35
For 1:  a_1^T x = -0.03 -> ReLU = 0 (same as before)
For M2: a_M2^T x = 0.15*0.5 + 0*(-0.3) + 0.85*0.8 + 0.15*0.1 = 0.77
For 6:  a_6^T x = -0.41 -> ReLU = 0 (same as before)
For M3: a_M3^T x = 0.05*0.5 + 0.10*(-0.3) + 0.05*0.8 + 0.925*0.1 = 0.1275

output_merged = b_M0 * 0.35 + b_M2 * 0.77 + b_M3 * 0.1275
```

The merged output approximates the unmerged output because:
- b_M0 * h_M0 ~ b_0 * h_0 + b_4 * h_4 (since h_0 ~ h_4 ~ h_M0)
- The approximation error scales with (1 - tau)

---

## 8. Assumptions

1. **Rank-1 capsule structure**: Cosine similarity of a_i vectors is
   sufficient for redundancy detection ONLY because each capsule is a
   single linear projection + ReLU gate. For multi-layer experts,
   deeper metrics (CKA, activation correlation) would be needed.

2. **Shared pretraining produces alignable representations**: Domain
   pools fine-tuned from the same base develop a_i vectors in the same
   representation space. If pools were trained from different random
   initializations, cosine similarity would be meaningless.

3. **Greedy matching is adequate**: The greedy pairwise algorithm does
   not find the globally optimal matching. With P=128, the difference
   between greedy and optimal is likely negligible.

4. **b-sum preserves downstream compatibility**: The summed b vectors
   produce outputs with the same magnitude as the sum of individual
   contributions. This assumes the unmerged model's output distribution
   is dominated by the additive contributions of matched capsules.

5. **Redundancy is detectable at weight level**: Two capsules may
   compute similar functions on the data distribution but have
   dissimilar weight vectors (e.g., if the data occupies a low-rank
   subspace). Cosine similarity in weight space may underestimate
   functional redundancy. Conversely, capsules with similar weights
   may differ on rare inputs near the ReLU boundary.

6. **The 54% shared knowledge finding transfers to capsule-level
   redundancy**: The Procrustes experiment measured shared knowledge
   as the fraction of weight delta norm that is common. This may or
   may not correspond to 54% of individual capsules being similar.
