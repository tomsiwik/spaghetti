# Collision Scaling: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro) |
| d_ff | MLP intermediate dimension | 256 (= 4d) |
| L | Number of MLP layers | 4 |
| r | LoRA rank | 8 |
| alpha | LoRA scaling factor | 8 |
| N | Number of domain experts | varies: {5, 10, 15, 20, 30, 50} |
| K | Number of semantic clusters | ceil(N/5), min 1, max 10 |
| M_k | Domains in cluster k | ~N/K (balanced) |
| V | Vocabulary size | 32 |
| T | Context length | 16 |
| D | Delta vector dimension | L * 2 * d * d_ff = 131,072 |
| tau | Collision threshold | 0.1 (|cos| > tau is a "collision") |

## 2. Collision Rate Definition

The **collision rate** C(N) at expert count N is the fraction of unique pairs
(i, j) with i < j whose absolute cosine similarity exceeds threshold tau:

    C(N) = |{(i,j) : i<j, |cos(v_i, v_j)| > tau}| / P(N)

where P(N) = N(N-1)/2 is the total number of unique pairs.

We decompose this into within-cluster and cross-cluster components:

    C(N) = (P_w * C_w(N) + P_x * C_x(N)) / P(N)

where:
- P_w = sum_k M_k(M_k-1)/2 = total within-cluster pairs
- P_x = P(N) - P_w = total cross-cluster pairs
- C_w(N) = within-cluster collision rate
- C_x(N) = cross-cluster collision rate

## 3. Scaling Analysis

### 3.1 Pair Count Scaling

With K = ceil(N/5) clusters, each of size M ~ N/K ~ 5:

    P(N) = N(N-1)/2 ~ O(N^2)
    P_w = K * M(M-1)/2 ~ K * 10 ~ O(N)    [since K ~ N/5, M ~ 5]
    P_x = P(N) - P_w ~ O(N^2) - O(N) ~ O(N^2)

Key insight: as N grows, the proportion of within-cluster pairs shrinks:

    P_w / P(N) ~ O(N) / O(N^2) = O(1/N) -> 0

Since collisions are overwhelmingly within-cluster (empirically C_x ~ 0), the
total collision count grows as O(N) while total pairs grow as O(N^2), so:

    C(N) ~ O(N) / O(N^2) = O(1/N)

This predicts a power law with exponent beta ~ -1.

### 3.2 Expected Collision Count

Let W be the number of within-cluster collisions. In our balanced cluster design
with M ~ 5 domains per cluster:

    E[W] = K * E[collisions per cluster]
         = K * C_w_per_cluster * M(M-1)/2
         ~ (N/5) * C_w * 10
         = 2N * C_w

where C_w is the per-pair within-cluster collision probability. If C_w is
approximately constant (independent of N, since cluster structure is fixed
at ~5 domains), then:

    E[W] ~ O(N)

And the collision rate:

    C(N) = E[W] / P(N) = 2N * C_w / (N(N-1)/2) = 4 * C_w / (N-1) ~ O(1/N)

### 3.3 Power Law Model

We fit: C(N) = a * N^beta

From the O(1/N) analysis: beta should be approximately -1.

Observed: beta = -0.575 (3-seed average, R^2 = 0.922).

The deviation from -1 reflects that:
1. Within-cluster collision rate C_w is not strictly constant -- it increases
   slightly with N because at larger N, some clusters end up with >5 members
   (uneven distribution) or domain similarity accidentally increases.
2. At small N (5, 10), the cluster structure is immature (1-2 clusters),
   creating higher baseline collision rates.

### 3.4 Superlinearity Test

The hypothesis is killed if collision rate grows superlinearly, meaning:
- Power law beta > 1 (collisions grow faster than pairs), OR
- Quadratic coefficient a > 0 AND quadratic R^2 >> linear R^2

Observed:
- beta = -0.575 (strongly sublinear -- collisions DECREASE as fraction of pairs)
- The absolute collision count grows roughly linearly (~O(N)), but the RATE
  (per pair) shrinks because pairs grow as O(N^2)

## 4. Within-Cluster vs Cross-Cluster Decomposition

### 4.1 Empirical Structure

| N | K clusters | Within rate | Cross rate | Within/Total ratio |
|---|-----------|-------------|------------|-------------------|
| 5 | 1 | 3.3% | 0.0% | 100% |
| 10 | 2 | 5.0% | 0.0% | 100% |
| 15 | 3 | 4.4% | 0.0% | 100% |
| 20 | 4 | 5.8% | 0.0% | 100% |
| 30 | 6 | 10.0% | 0.0% | 100% |
| 50 | 10 | 9.7% | 0.03% | ~100% |

Cross-cluster collision rate is effectively zero. All collisions are
within-cluster, confirming the block-diagonal structure from
orthogonality_by_domain_type.

### 4.2 Within-Cluster Power Law

Within-cluster collision rate C_w(N) grows with beta = 0.50 (R^2 = 0.83).

This makes sense: as N grows, some clusters get more members, and more
members means more pairs within the high-similarity zone. But C_w is
bounded by the number of within-cluster pairs, which grows only as O(N).

## 5. Cosine Similarity Scaling

Mean pairwise cosine decreases with N:

| N | Mean |cos| | Within |cos| | Cross |cos| |
|---|-----------|-------------|-------------|
| 5 | 0.039 | 0.039 | -- |
| 10 | 0.026 | 0.044 | 0.012 |
| 15 | 0.020 | 0.050 | 0.008 |
| 20 | 0.018 | 0.051 | 0.009 |
| 30 | 0.013 | 0.056 | 0.006 |
| 50 | 0.012 | 0.056 | 0.008 |

Within-cluster cosine stabilizes around 0.05 (consistent with the 0.060
from orthogonality_by_domain_type at N=15). Cross-cluster cosine is
stable at ~0.008. The aggregate mean drops because cross-cluster pairs
dominate at large N.

## 6. Random Baseline Comparison

For random vectors in D = 131,072 dimensions:

    E[|cos(v_i, v_j)|] = sqrt(2 / (pi * D)) = 0.00220

Cross-cluster cosine (0.008) is ~3.6x above random baseline, indicating
minimal but nonzero structure from the shared base model.

Within-cluster cosine (0.056) is ~25x above random baseline, confirming
strong cluster structure in parameter space.

## 7. Extrapolation to Production Scale

At d=3584 (Qwen2.5-7B), D_FFN = 3.8 billion:

    Random E[|cos|] = 9.2e-6
    Expected within-cluster: ~100x random ~ 10^{-3}
    Expected cross-cluster: ~4x random ~ 4 * 10^{-5}

With tau = 0.1, collision rate at production scale would be essentially zero,
since even within-cluster cosine is 100x below threshold.

For N = 5000 experts with K ~ 1000 clusters (5 per cluster):
- P(5000) = 12,497,500 pairs
- Within-cluster pairs: ~10,000 (0.08% of total)
- Even if all within-cluster pairs collide: 10000/12497500 = 0.08%
- Far below 30% kill threshold

## 8. Assumptions

1. **Balanced cluster assignment.** We assign ~N/K = ~5 domains per cluster.
   Real expert libraries may have uneven distribution (50 python experts, 2
   cooking experts), which would increase within-cluster collisions for
   over-represented clusters. The architecture handles this via rank
   allocation or Gram-Schmidt projection for dense clusters.

2. **Cluster count scales with N.** We use K = ceil(N/5). If domains were
   drawn from a fixed number of clusters (e.g., K=3 regardless of N),
   within-cluster collision rate would grow as clusters become larger.
   The O(1/N) result depends on cluster count growing proportionally.

3. **Synthetic Markov chain data.** Real domain similarity has richer
   structure. The synthetic approach is conservative: it captures only
   distributional similarity, not semantic or structural similarity that
   would create stronger within-cluster effects.

4. **Micro scale (d=64).** At this scale, cosines are ~100x higher than
   d=896. The absolute collision rates reported here are upper bounds for
   production scale.

## 9. Worked Numerical Example

N=20, K=4 clusters, M=5 per cluster:

    Total pairs: 20*19/2 = 190
    Within-cluster pairs: 4 * 5*4/2 = 40
    Cross-cluster pairs: 190 - 40 = 150

    Within-cluster fraction: 40/190 = 21%
    Cross-cluster fraction: 150/190 = 79%

    Observed collision count: 2.33 (3-seed avg)
    Collision rate: 2.33/190 = 1.23%

    All collisions are within-cluster: 2.33/40 = 5.83% of within-cluster pairs
    Cross-cluster collision rate: 0/150 = 0%

At N=50, K=10 clusters:

    Total pairs: 50*49/2 = 1225
    Within-cluster pairs: 10 * 5*4/2 = 100
    Cross-cluster pairs: 1225 - 100 = 1125

    Within-cluster fraction: 100/1225 = 8.2%
    Cross-cluster fraction: 1125/1225 = 91.8%

    Observed collision count: 10 (3-seed avg)
    Collision rate: 10/1225 = 0.82%

    Within pairs collision: 10/100 = 9.67%
    Cross collision: ~0.03%
