# N=5 Identity Scaling: Mathematical Foundations

## 1. Problem Statement

Exp 16 (capsule_identity) established that the per-capsule dead/alive identity
is preserved across N=2 composition (Jaccard=0.895, well above the 0.50 kill
threshold). The adversarial review flagged that the perturbation from composition
grows linearly with N: each additional domain adds its capsule residuals to the
hidden state. This experiment measures whether identity preservation degrades
below the safety threshold (Jaccard < 0.70) when scaling to N=5 domains.

---

## 2. Notation

All notation follows capsule_identity/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
N         -- number of composed domains (2 in Exp 16, 5 here)
P_total   -- total capsules per single-domain model = P * L = 512
P_comp    -- total capsules in composed model = N * P * L

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

## 3. Perturbation Scaling with N

### 3.1 Single-Domain Hidden State

In a single-domain model for domain k, the hidden state at layer l is:

```
x_l^{single} = x_{l-1} + Attn_l(x_{l-1}) + MLP_k_l(Norm(x_{l-1} + Attn_l(x_{l-1})))
```

where MLP_k_l is domain k's capsule pool at layer l:

```
MLP_k_l(h) = B_k * ReLU(A_k * h)
```

### 3.2 Composed Hidden State

In the N-domain composed model, the MLP at layer l contains all N domains'
capsule pools concatenated:

```
MLP_comp_l(h) = sum_{j=0}^{N-1} B_j * ReLU(A_j * h)
```

The hidden state seen by domain k's capsules is:

```
x_l^{comp} = x_{l-1}^{comp} + Attn_l(x_{l-1}^{comp}) + sum_{j=0}^{N-1} B_j * ReLU(A_j * Norm(z_l))
```

where z_l = x_{l-1}^{comp} + Attn_l(x_{l-1}^{comp}).

### 3.3 Perturbation from Other Domains

The perturbation to domain k's input, relative to single-domain inference, comes
from the other (N-1) domains' capsule residuals. At layer l:

```
delta_l = sum_{j != k} B_j * ReLU(A_j * Norm(z_l))
```

The perturbation magnitude:

```
||delta_l|| <= sum_{j != k} ||B_j * ReLU(A_j * Norm(z_l))||
            <= (N-1) * max_j ||B_j * ReLU(A_j * Norm(z_l))||
```

This scales linearly with (N-1). At N=2, the perturbation comes from 1 other
domain. At N=5, it comes from 4 other domains -- a 4x increase in worst-case
perturbation magnitude.

### 3.4 Effect on Capsule Death/Alive Boundary

Capsule i in domain k fires when:

```
a_i^T * Norm(z_l) > 0     (single-domain)
a_i^T * Norm(z_l + delta_l) > 0     (composed)
```

The margin for capsule i is m_i = a_i^T * Norm(z_l). Capsules with |m_i| >> ||delta_l||
are robust to composition. Capsules with |m_i| ~ ||delta_l|| are vulnerable.

As N increases, ||delta_l|| grows, and more borderline capsules cross the
death boundary. This predicts:
1. Jaccard should decrease with N (more capsules change status)
2. The decrease should be sublinear if delta_l from different domains are
   uncorrelated (partial cancellation in the sum)
3. Overlap coefficient should remain high because most dead capsules have
   large negative margins (well past the boundary)

### 3.5 Expected Jaccard Degradation

Assume the fraction of borderline capsules (|m_i| < epsilon) is constant at
some rate alpha. Each additional domain has probability p_flip of flipping
a borderline capsule's status.

For N domains composing onto domain k:
```
E[n_flipped] ~ alpha * P * L * (N-1) * p_flip
```

The expected Jaccard degradation per additional domain:
```
Delta_J per domain ~ -alpha * p_flip * P * L / |D_union|
```

This is approximately constant per additional domain, predicting a linear
decline in Jaccard with N.

---

## 4. Set Similarity Metrics

Same as capsule_identity/MATH.md (Jaccard, overlap coefficient, Dice).

### 4.1 N-Domain Combined Jaccard

```
D_union_single = Union_{k=0}^{N-1} {(l, c + kP) : (l, c) in D^{single}_k}

J_combined = J(D_union_single, D^{comp})
           = |D_union_single & D^{comp}| / |D_union_single | D^{comp}|
```

### 4.2 N-Domain Decomposition

For each domain k:
```
D^{comp}_k = {(l, c - kP) : (l, c) in D^{comp}, kP <= c < (k+1)P}

n_BB_k = |D^{single}_k & D^{comp}_k|        (preserved death)
n_SO_k = |D^{single}_k - D^{comp}_k|        (revived by composition)
n_CO_k = |D^{comp}_k - D^{single}_k|        (killed by composition)
n_AA_k = P*L - n_BB_k - n_SO_k - n_CO_k     (preserved alive)

n_BB_k + n_SO_k + n_CO_k + n_AA_k = P * L
```

---

## 5. Null Model (Independent Death at N=5)

If death were independent between single-domain and composed settings:

```
p_single ~ 0.484    (mean across 5 domains, 3 seeds)
p_composed ~ 0.576  (mean at N=5)

E[J_null] = (p_single * p_composed) / (p_single + p_composed - p_single * p_composed)
          = (0.484 * 0.576) / (0.484 + 0.576 - 0.484 * 0.576)
          = 0.279 / 0.781
          = 0.357
```

Our measured J = 0.792 >> 0.357, confirming death identity is highly
non-independent (the same capsules deterministically die in both settings).

---

## 6. Experimental Design

### 6.1 Protocol

1. Pretrain base model on ALL data (300 steps, shared attention + MLP, d=64)
2. Fine-tune only MLP weights per domain (attention frozen, 200 steps),
   5 domains: a-e, f-j, k-o, p-t, u-z
3. Profile each single-domain model on own-domain validation data
   (20 batches x 32 samples)
4. Compose at N=2, 3, 4, 5 by concatenating weight matrices
5. Profile each composed model on joint validation data
6. Profile N=5 composed model on each domain separately
7. Compute Jaccard, overlap coefficient, decomposition for all N values
8. Repeat for 3 seeds (42, 123, 7)

### 6.2 Why N-Sweep

By measuring at N=2,3,4,5, we can:
- Confirm the Exp 16 N=2 result under the quintary split
- Measure the degradation trajectory (linear? sublinear?)
- Extrapolate to estimate the N at which Jaccard would reach 0.70

### 6.3 Domain Size Distribution

The quintary split has unequal domain sizes:

```
a-e: 10,479 names (32.7%)
f-j:  4,973 names (15.5%)
k-o:  8,613 names (26.9%)
p-t:  5,609 names (17.5%)
u-z:  2,359 names ( 7.4%)
```

The u-z domain is the smallest (2,359 names), which may affect training
quality and death rates. Smaller training sets can lead to less stable
representations and potentially more borderline capsules.

---

## 7. Kill Criterion

```
Combined Jaccard at N=5 < 0.70
```

If killed: Pre-composition profiling is unreliable at N=5. Post-composition
profiling is required at higher fan-out.

If passed: Death identity remains sufficiently preserved at N=5 for
pre-composition profiling to be practical.

---

## 8. Worked Numerical Example

At d=4, P=4, L=1, N=3 (4 capsules per domain, 12 total composed):

### Single-domain dead sets
```
Domain 0: D^{single}_0 = {0, 1}      (capsules 0,1 dead)
Domain 1: D^{single}_1 = {2, 3}      (capsules 2,3 dead)
Domain 2: D^{single}_2 = {0, 3}      (capsules 0,3 dead)
```

### Composed model (3 domains, 12 capsules)
```
Domain 0 at [0..3]:  dead = {0, 1, 3}     (capsule 3 killed by composition)
Domain 1 at [4..7]:  dead = {6, 7}        = {2, 3} re-indexed
Domain 2 at [8..11]: dead = {8, 11}       = {0, 3} re-indexed

D^{comp} = {0, 1, 3, 6, 7, 8, 11}
```

### Metrics
```
Domain 0: J(D^{single}_0, D^{comp}_0) = J({0,1}, {0,1,3}) = 2/3 = 0.667
Domain 1: J(D^{single}_1, D^{comp}_1) = J({2,3}, {2,3}) = 1.0
Domain 2: J(D^{single}_2, D^{comp}_2) = J({0,3}, {0,3}) = 1.0

D_union_single = {0, 1, 6, 7, 8, 11}
J_combined = |{0, 1, 6, 7, 8, 11} & {0, 1, 3, 6, 7, 8, 11}| / |{0, 1, 3, 6, 7, 8, 11}|
           = |{0, 1, 6, 7, 8, 11}| / |{0, 1, 3, 6, 7, 8, 11}|
           = 6 / 7 = 0.857
```

---

## 9. Assumptions

1. **Same as Exp 16 assumptions 1-4.** Activation frequency is deterministic,
   capsule identity is preserved across composition, shared attention is
   identical, binary dead/alive threshold at f=0.

2. **Quintary split produces sufficiently distinct domains.** All 5 domains
   share character-level name generation, just split by first letter. The
   domains are more similar to each other than real-world domains (code vs
   prose vs math). This is a best-case for identity preservation.

3. **Domain order does not affect N-sweep.** The N=2 composition uses
   {a-e, f-j}. A different pair might give different Jaccard. We use the
   first N domains in alphabetical order for consistency.

4. **Linear scaling of perturbation.** The perturbation from (N-1) other
   domains scales linearly in the worst case. In practice, partial
   cancellation from uncorrelated domain signals may cause sublinear scaling.
