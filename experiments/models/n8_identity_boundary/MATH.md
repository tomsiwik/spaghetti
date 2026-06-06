# N=8 Identity Boundary: Mathematical Foundations

## 1. Problem Statement

The N=5 identity scaling experiment (n5_identity_scaling) established that
capsule death identity degrades approximately linearly with N, at a rate
of ~0.026 Jaccard per additional domain. Extrapolating, this predicted
a safe limit of ~N=8 (Jaccard ~ 0.71, near the 0.70 threshold).

This experiment tests that prediction by measuring identity preservation
at N=8 using an octonary domain split. Additionally, it measures at all
intermediate N values (2 through 8) to determine whether degradation is
truly linear or exhibits a phase transition or sublinear saturation.

---

## 2. Notation

All notation follows capsule_identity/MATH.md and n5_identity_scaling/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
N         -- number of composed domains (8 in this experiment)
P_total   -- total capsules per single-domain model = P * L = 512
P_comp    -- total capsules in composed model = N * P * L = 4096

a_i in R^d   -- detector vector for capsule i (row i of matrix A)
b_i in R^d   -- output vector for capsule i (column i of matrix B)

D^{single}_k = {(l, c) : f_{l,c}^{single_k} = 0}
    -- dead set of domain k's model on own-domain data

D^{comp} = {(l, c) : f_{l,c}^{comp} = 0}
    -- dead set of N-domain composed model on joint data

D^{comp}_k = {(l, c - kP) : (l, c) in D^{comp}, kP <= c < (k+1)P}
    -- domain k's slice of the composed dead set (re-indexed to [0,P))
```

---

## 3. Perturbation Scaling Revisited

### 3.1 The N=5 Linear Model

The N=5 experiment measured the combined Jaccard trajectory:

```
J_quintary(N=2) = 0.871
J_quintary(N=5) = 0.792
rate = (0.871 - 0.792) / 3 = 0.026/domain
```

This predicted J(N=8) = 0.792 - 3 * 0.026 = 0.714 (near the 0.70 threshold).

### 3.2 Why Linear Might Overestimate Degradation

The linear model assumes each additional domain contributes an independent,
additive perturbation to the hidden state:

```
||delta_l|| = sum_{j != k} ||B_j * ReLU(A_j * h)||
```

The worst-case bound is (N-1) * max_j ||B_j * ReLU(A_j * h)||, which is
linear in N. However, real perturbations from uncorrelated domains can
exhibit partial cancellation in the sum:

```
||sum_{j != k} B_j * ReLU(A_j * h)||^2
  = sum_j ||B_j * ReLU(A_j * h)||^2 + 2 * sum_{i<j} <B_i ReLU(A_i h), B_j ReLU(A_j h)>
```

If the cross-terms are approximately zero (uncorrelated outputs), then:

```
||delta_l|| ~ sqrt((N-1)) * sigma_j
```

where sigma_j is the typical per-domain residual norm. This gives
sqrt(N)-scaling rather than linear scaling.

### 3.3 Implication for Jaccard Degradation

Under the sqrt-model, the expected number of flipped capsules scales as:

```
E[n_flipped] ~ alpha * P * L * sqrt(N-1) * p_flip
```

This predicts:
- Degradation rate should DECREASE as N grows (marginal perturbation shrinks)
- The rate(N1->N2) / rate(N2->N3) ratio should be < 1.0

### 3.4 Empirical Scaling

From the N=8 experiment:

```
J(2) = 0.894, J(5) = 0.834, J(8) = 0.800

rate(2->5) = (0.894 - 0.834) / 3 = 0.0199/domain
rate(5->8) = (0.834 - 0.800) / 3 = 0.0115/domain

ratio = rate(5->8) / rate(2->5) = 0.58
```

The ratio 0.58 < 1.0 confirms sublinear degradation. This is consistent
with the sqrt-model: sqrt(7)/sqrt(4) = 1.32, while the linear model
predicts a constant rate.

Linear regression across N=2..8 gives:

```
J = 0.910 - 0.0141 * N   (RMSE = 0.0081)
```

The overall rate of 0.014/domain is approximately half the N=5 experiment's
rate of 0.026/domain. The difference comes from:
1. Sublinear scaling (sqrt effect)
2. Different domain splits (octonary vs quintary)
3. Fewer capsules per domain relative to total (P/NP = 1/8 vs 1/5)

---

## 4. Set Similarity Metrics

Same as n5_identity_scaling/MATH.md (Jaccard, overlap coefficient, combined
Jaccard via union of shifted single-domain sets).

### 4.1 Overlap Coefficient Stability

The overlap coefficient remained remarkably stable across N:

```
OC(N=2) = 0.970
OC(N=8) = 0.966
```

This means ~97% of capsules that are dead in single-domain models remain
dead after 8-way composition. The Jaccard decline (0.894 -> 0.800) is
entirely from composition KILLING additional capsules, not from REVIVING
existing dead ones. The asymmetry (kill >> revive) persists.

---

## 5. Null Model (Independent Death at N=8)

If death were independent between single-domain and composed settings:

```
p_single ~ 0.501    (mean across 8 domains, 3 seeds)
p_composed ~ 0.585  (mean at N=8)

E[J_null] = (p_single * p_composed) / (p_single + p_composed - p_single * p_composed)
          = (0.501 * 0.585) / (0.501 + 0.585 - 0.501 * 0.585)
          = 0.293 / 0.793
          = 0.370
```

Our measured J = 0.800 >> 0.370, confirming death identity remains
highly non-independent even at N=8. The same capsules deterministically
die in both single-domain and composed settings.

---

## 6. Domain Size Effect on Per-Domain Jaccard

Per-domain Jaccard shows correlation with domain size and single-domain
death rate:

```
Domain  | Names  | Death% | J (N=8 mean)
--------|--------|--------|-------------
a-c     | 7,258  | 54.8%  | 0.843
j-l     | 6,957  | 54.9%  | 0.841
m-o     | 4,078  | 51.8%  | 0.799
d-f     | 3,638  | 51.4%  | 0.821
s-u     | 3,441  | 52.1%  | 0.816
v-z     | 2,281  | 47.7%  | 0.770
p-r     | 2,246  | 43.8%  | 0.750
g-i     | 2,134  | 44.5%  | 0.747
```

Smaller domains and lower death rates correlate with lower Jaccard.
Domains with fewer dead capsules have proportionally more borderline
capsules (those near the death/alive boundary), making them more
susceptible to perturbation-induced identity changes.

---

## 7. Experimental Design

### 7.1 Protocol

1. Pretrain base model on ALL data (300 steps, d=64)
2. Fine-tune only MLP weights per domain (attention frozen, 200 steps),
   8 domains: a-c, d-f, g-i, j-l, m-o, p-r, s-u, v-z
3. Profile each single-domain model on own-domain validation data
   (20 batches x 32 samples)
4. Compose at N=2, 3, 4, 5, 6, 7, 8 by concatenating weight matrices
5. Profile each composed model on joint validation data
6. Profile N=8 composed model on each domain separately
7. Compute Jaccard, overlap coefficient, decomposition for all N values
8. Repeat for 3 seeds (42, 123, 7)

### 7.2 Domain Size Distribution

```
a-c:  7,258 names (22.7%)
d-f:  3,638 names (11.4%)
g-i:  2,134 names ( 6.7%)
j-l:  6,957 names (21.7%)
m-o:  4,078 names (12.7%)
p-r:  2,246 names ( 7.0%)
s-u:  3,441 names (10.7%)
v-z:  2,281 names ( 7.1%)
```

More unequal than the quintary split (7.1x ratio between largest and
smallest domains vs 4.4x for quintary).

---

## 8. Kill Criteria

1. Combined Jaccard at N=8 < 0.70
2. Per-domain minimum Jaccard < 0.50 for any domain-seed combination

---

## 9. Worked Numerical Example

At d=4, P=4, L=1, N=4 (4 capsules per domain, 16 total composed):

### Single-domain dead sets
```
Domain 0: D^{single}_0 = {0, 1}       (50% dead)
Domain 1: D^{single}_1 = {2, 3}       (50% dead)
Domain 2: D^{single}_2 = {0, 3}       (50% dead)
Domain 3: D^{single}_3 = {1, 2}       (50% dead)
```

### Composed model (4 domains, 16 capsules)
```
Domain 0 at [0..3]:   dead = {0, 1, 3}     (capsule 3 killed)
Domain 1 at [4..7]:   dead = {6, 7}         = {2, 3} re-indexed
Domain 2 at [8..11]:  dead = {8, 11}        = {0, 3} re-indexed
Domain 3 at [12..15]: dead = {13, 14}       = {1, 2} re-indexed
D^{comp} = {0, 1, 3, 6, 7, 8, 11, 13, 14}
```

### Metrics
```
Domain 0: J({0,1}, {0,1,3}) = 2/3 = 0.667
Domain 1: J({2,3}, {2,3})   = 1.0
Domain 2: J({0,3}, {0,3})   = 1.0
Domain 3: J({1,2}, {1,2})   = 1.0

D_union_single = {0, 1, 6, 7, 8, 11, 13, 14}
J_combined = |{0,1,6,7,8,11,13,14} & {0,1,3,6,7,8,11,13,14}| / |{0,1,3,6,7,8,11,13,14}|
           = 8 / 9 = 0.889
```

Only 1 of 8 single-domain dead capsules changed status. The new kill
(capsule 3 in domain 0) came from composition perturbation.

---

## 10. Assumptions

1. **Same as n5_identity_scaling assumptions 1-4.** Activation frequency is
   deterministic, capsule identity is preserved across composition, shared
   attention is identical, binary dead/alive threshold at f=0.

2. **Octonary split produces sufficiently distinct domains.** All 8 domains
   are character-level name generation, split by first letter. The domains
   are more similar to each other than real-world domains. This is likely
   a best-case for identity preservation.

3. **Domain order does not affect N-sweep.** The N=2 composition uses
   {a-c, d-f}. A different pair might give different Jaccard.

4. **Sublinear scaling of perturbation.** Evidence supports sqrt-like
   scaling due to partial cancellation of uncorrelated domain residuals.
   The linear model from N=5 overestimated degradation by ~0.086 at N=8.
