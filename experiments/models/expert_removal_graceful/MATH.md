# Expert Removal Graceful: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 896 (Qwen 0.5B) |
| r | LoRA rank | 16 (production) |
| N | Number of expert adapters | {10, 20, 50, 100, 200} |
| k | Index of expert to remove | 0 <= k < N |
| D | Flattened delta vector dimension | d^2 = 802,816 |
| delta_i | Flattened weight delta of expert i | (D,) |
| delta_i' | GS-orthogonalized delta of expert i | (D,) |
| W_base | Base model weights | - |
| W_merged | Merged model weights | - |
| cos(i,j) | Cosine similarity between delta_i and delta_j | [-1, 1] |

## 2. The Composition Model

### 2.1 Gram-Schmidt Composition

Given N LoRA experts with flattened deltas delta_1, ..., delta_N, Gram-Schmidt
orthogonalization produces:

    delta_1' = delta_1
    delta_k' = delta_k - sum_{i<k} proj(delta_k, delta_i')    for k = 2, ..., N

where proj(u, v) = (u . v / v . v) * v.

The merged model is:

    W_merged = W_base + sum_{i=1}^{N} delta_i'

### 2.2 Key Property: Under Perfect Orthogonality

If cos(delta_i, delta_j) = 0 for all i != j, then:

    proj(delta_k, delta_i') = 0    for all i < k

Therefore delta_k' = delta_k for all k. GS is a no-op.

In this case:

    W_merged = W_base + sum_{i=1}^{N} delta_i

## 3. Expert Removal Analysis

### 3.1 Problem Statement

Given W_merged = W_base + sum delta_i', remove expert k and obtain a model
equivalent to having never added expert k.

### 3.2 Ground Truth ("Never Added" Baseline)

The ground truth is recomputing GS on the remaining N-1 experts:

    W_gt = W_base + sum_{i != k} delta_i''

where delta_i'' are the GS-orthogonalized deltas of the N-1 remaining experts
(re-indexed).

### 3.3 Naive Subtraction

The naive approach subtracts the stored orthogonalized delta:

    W_naive = W_merged - delta_k' = W_base + sum_{i != k} delta_i'

### 3.4 Error Analysis

The error of naive subtraction is:

    E = W_naive - W_gt = sum_{i != k} (delta_i' - delta_i'')

For expert i != k, the difference delta_i' - delta_i'' depends on whether
expert k contributed to the orthogonalization of expert i.

**Case 1: i < k (experts processed before k in GS order)**

    delta_i' and delta_i'' are identical: expert k had no influence on
    their orthogonalization (GS processes sequentially, earlier experts
    are unaffected by later ones).

**Case 2: i > k (experts processed after k in GS order)**

    delta_i' = delta_i - proj(delta_i, delta_k') - sum_{j<i, j!=k} proj(delta_i, delta_j')
    delta_i'' = delta_i - sum_{j<i', j!=k} proj(delta_i, delta_j'')

    The difference arises because:
    (a) In delta_i', the projection onto delta_k' was subtracted
    (b) The subsequent orthogonalized deltas delta_j' (j > k) were computed
        with delta_k' in the basis, while delta_j'' were computed without it

    The error propagates through a cascade: removing delta_k' from the
    basis changes delta_{k+1}', which changes delta_{k+2}', etc.

### 3.5 Error Bound

The error magnitude is bounded by the cosine similarities:

    ||E||_F <= sum_{i>k} ||proj(delta_i, delta_k')||

For the first expert after k (expert k+1):

    ||proj(delta_{k+1}, delta_k')|| = |cos(delta_{k+1}, delta_k')| * ||delta_{k+1}||

At SOLE production cosines (cos ~ 0.0002 at d=896):

    ||proj|| ~ 0.0002 * ||delta|| per expert

    Total error ~ 0.0002 * (N - k - 1) * ||delta||

For N=50, removing middle expert (k=25):

    Error ~ 0.0002 * 24 * ||delta|| = 0.0048 * ||delta||

    Relative error ~ 0.48% (well within 3% kill criterion)

### 3.6 Position Dependence

Removing expert at position k in the GS ordering affects all experts i > k.

- Position 0 (first): affects N-1 subsequent experts -> maximum error
- Position N-1 (last): affects 0 subsequent experts -> zero error
- Position k: affects N-k-1 subsequent experts

Error ~ cos_mean * (N - k - 1) * ||delta||_avg

This is confirmed experimentally:
- Position 0:  recon error = 24.3% (clustered cos=0.3)
- Position 10: recon error = 9.4%
- Position 15: recon error = 2.5%
- Position 19: recon error = 0.0%

### 3.7 Regime Classification

| Regime | cos range | Naive viable? | Evidence |
|--------|-----------|---------------|----------|
| SOLE production | cos ~ 0.0002 | YES (error < 0.2%) | Test 1 |
| Moderate overlap | cos ~ 0.3 | MARGINAL (error 2-10%) | Test 2 |
| High overlap | cos ~ 0.5 | NO (error 8-12%) | Test 3 |

## 4. Timing Analysis

GS recomputation has complexity O(N^2 * D):

    For each of N experts, project against N previous -> N^2 dot products
    Each dot product costs D multiplications

Measured timing (d=896, r=16, D=802,816):

| N | GS recompute (s) | Naive (ms) | Speedup |
|---|-------------------|------------|---------|
| 10 | 0.051 | 0.6 | 85x |
| 20 | 0.223 | 0.6 | 372x |
| 50 | 1.059 | 0.3 | 3530x |
| 100 | 4.363 | 0.6 | 7272x |
| 200 | 14.798 | 0.2 | 59192x |

At N=50: recompute takes 1.06s (well within 10 min kill criterion).
At N=200: recompute takes 14.8s (still fast).

Extrapolating O(N^2): N=1000 ~ 370s (6.2 min), N=2000 ~ 1480s (24.7 min).
Kill criterion of 10 min is hit around N~1600.

## 5. Key Insight

The critical finding is that naive subtraction is sufficient if and only
if cos(delta_i, delta_j) is small. In SOLE's production regime
(cos ~ 0.0002 at d=896), naive subtraction has < 0.2% reconstruction error
with 2000x speedup. This is because:

1. Near-orthogonality means GS is approximately a no-op
2. If GS barely changes the deltas, subtracting the GS delta is
   approximately equivalent to subtracting the original delta
3. The cascade error from removing a basis vector is proportional to
   the projections it absorbed, which are proportional to cosine similarity

For high-overlap scenarios (cos > 0.1), GS recomputation is necessary but
cheap (1s at N=50, 15s at N=200). The 10-min threshold is not approached
until N > 1600, far beyond current production targets.

## 6. Assumptions

1. Single linear layer simulation. Multi-layer LoRA deltas have the same
   structure when flattened -- the math generalizes directly.
2. Synthetic experts with controlled cosine. Real LoRA experts at d=896
   have cos=0.0002, which is in the near-orthogonal regime where naive
   subtraction works perfectly.
3. Reconstruction error as PPL proxy. The relative Frobenius norm error
   between weight matrices is a conservative upper bound on PPL change
   (PPL is Lipschitz in weights for bounded activations).
