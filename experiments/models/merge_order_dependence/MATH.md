# Merge Order Dependence: Mathematical Foundations

## 1. Setup

We have N LoRA expert deltas d_1, ..., d_N in R^D, where D is the total
flattened parameter count. Classical Gram-Schmidt produces orthogonalized
deltas d_1', ..., d_N' such that d_i' . d_j' = 0 for all i != j.

The process is sequential:

  d_1' = d_1
  d_k' = d_k - sum_{i=1}^{k-1} proj(d_k, d_i')   for k = 2, ..., N

where proj(u, v) = (u . v / v . v) * v.

## 2. Order Dependence: Formal Analysis

### 2.1 What Changes with Ordering

Let sigma be a permutation of {1, ..., N}. Define d_k'^(sigma) as the
orthogonalized version of d_{sigma(k)} when processed in order sigma.

**The merged sum changes with ordering:**
  S(sigma) = sum_k d_k'^(sigma)

This is because d_k'^(sigma) = d_{sigma(k)} - projection onto span(d_{sigma(1)}', ..., d_{sigma(k-1)}')
and this projection depends on which vectors came earlier.

**The merged average changes proportionally:**
  A(sigma) = (1/N) * S(sigma)

### 2.2 What Does NOT Change

The *subspace* spanned by {d_1'^(sigma), ..., d_N'^(sigma)} is the same
for all sigma. Gram-Schmidt is a change of basis within the same column
space. What changes is how the original vectors are decomposed.

### 2.3 Bound on Order Variation

For the merged average A(sigma), the variation across orderings is bounded by
the pairwise cosine similarities of the original deltas.

**Theorem (informal):** If all pairwise cosines satisfy |cos(d_i, d_j)| < epsilon
for all i != j, then for any two orderings sigma, tau:

  ||A(sigma) - A(tau)||_2 / ||A(sigma)||_2 = O(epsilon)

**Sketch:** The projection removed from d_k is at most sum_{i<k} cos^2(d_k, d_i').
When all pairwise cosines are epsilon-small, this projection is O(N * epsilon^2)
in norm. The reallocation of this small projection across different orderings
produces variation O(epsilon) in the merged vector.

### 2.4 Numerical Verification

At d=64, N=5, max pairwise cosine = 0.034:
  - CV of merged loss = 0.029%
  - Worst/best gap = 0.094%
  - Predicted bound: O(0.034) = O(3.4%), actual is 30x below

At d=64, N=8, max pairwise cosine = 0.056:
  - CV of merged loss = 0.015%
  - Worst/best gap = 0.044%
  - Actual is even further below the bound (more experts dilute ordering effect)

### 2.5 Scaling to Production

At d=896 (Qwen 0.5B), measured cos = 0.0002. The order dependence bound:
  variation < O(0.0002) < 0.02%

At d=4096 (Qwen 7B), cos << 0.0001. Order dependence is unmeasurable.

## 3. Synthetic Stress Test: Order Dependence vs. Overlap

We created synthetic experts with controlled pairwise cosine c and measured
the cosine similarity between merged vectors from different orderings.

**Empirical relationship (N=10, D=4096):**

| c (pairwise) | min cos(merged) | variation (%) |
|:-------------|:----------------|:--------------|
| 0.01         | 0.996           | 0.38          |
| 0.05         | 0.970           | 3.0           |
| 0.10         | 0.923           | 7.7           |
| 0.20         | 0.811           | 18.9          |
| 0.30         | 0.700           | 30.0          |
| 0.50         | 0.549           | 45.1          |
| 0.70         | 0.437           | 56.3          |

The relationship is approximately linear:

  variation(%) ~ 80 * c

This makes intuitive sense: the fraction of each delta that is "reassigned"
by GS is proportional to the overlap c. Different orderings reassign to
different experts, creating variation proportional to c.

**Critical threshold for SOLE:** The 5% CV kill criterion is violated when
c > 0.06. But SOLE production cosines are c = 0.0002, which is 300x below
this threshold. Order dependence is a non-issue for SOLE.

## 4. Order-Invariant Alternatives

### 4.1 SVD Simultaneous Orthogonalization

Stack deltas into M = [d_1 | ... | d_N] in R^{D x N}, compute SVD:
  M = U S V^T

Assign each expert to a unique basis vector via Hungarian algorithm
(maximize alignment). The result is order-invariant by construction.

**Problem:** SVD assigns each expert a single basis vector, discarding
components along other basis vectors. Signal retention drops severely:

| Pairwise cos | GS min retention | SVD min retention | Ratio |
|:-------------|:-----------------|:------------------|:------|
| 0.10         | 0.968            | 0.398             | 2.4x  |
| 0.30         | 0.865            | 0.351             | 2.5x  |
| 0.50         | 0.734            | 0.296             | 2.5x  |
| 0.70         | 0.570            | 0.229             | 2.5x  |

SVD retains 2.5x less signal than GS. It trades order invariance for
massive information loss. Not recommended.

### 4.2 Symmetric Gram-Schmidt

Average of GS across K random orderings:
  d_k'^(sym) = (1/K) sum_{sigma} d_k'^(sigma)

Properties:
- Approximately order-invariant (converges as K -> infinity)
- Signal retention nearly identical to standard GS (within 1%)
- Post-GS cosines are NOT exactly zero (max 0.05 at cos=0.7)
- Very close to standard GS (cosine > 0.989 between merged vectors)
- Cost: K * O(N^2 * D) vs O(N^2 * D) for standard GS

**Verdict:** Symmetric GS is the correct order-invariant alternative IF
order invariance is needed. But it costs 50x more compute for negligible
benefit in the near-orthogonal regime.

## 5. Implications for SOLE

### 5.1 The Reviewer Attack is Neutralized

The attack: "GS is order-dependent, first expert is privileged."

The defense: At production scale (d >= 896), pairwise cosines are < 0.001.
Order dependence is < 0.1%, which is unmeasurable noise. The first expert
"retains 100% of its signal" and the last expert retains 99.98% of its signal.
There is no meaningful privilege.

### 5.2 GS is Unnecessary for SOLE

The prior experiment (gram_schmidt_composition) already showed GS adds no
benefit because deltas are already near-orthogonal. This experiment adds:
GS also creates no order-dependence problem, but only because the overlap
it orthogonalizes is negligible.

### 5.3 When Order Would Matter

Order dependence becomes significant (CV > 5%) only when pairwise cosines
exceed 0.06. This could occur:
- In very low-dimensional spaces (d < 32)
- With heavily overlapping domains (e.g., two math dialects)
- With many experts in a small subspace (N >> d^2/r^2)

In all these cases, the correct solution is NOT order-invariant orthogonalization
but rather capacity management (Grassmannian skeleton, domain clustering).

## 6. Assumptions

1. **Flattened vector space:** Delta dicts are flattened to single vectors.
   Layer-wise analysis might show different order sensitivity per layer.
2. **Average merge (1/N):** Results assume the GS average merge strategy.
   GS sum (no 1/N) would show the same CV but different absolute losses.
3. **Linear quality model:** We assume merged delta quality correlates with
   merged vector similarity. Nonlinear interactions could amplify small
   vector differences into larger quality effects. (Not observed in Phase 1.)
4. **Uniform expert importance:** If some experts are more important than
   others, ordering could matter more by affecting the important expert's
   signal retention. In practice, all experts retain > 99.8% signal.
