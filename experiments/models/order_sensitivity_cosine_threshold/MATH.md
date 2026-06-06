# Order Sensitivity Cosine Threshold: Mathematical Foundations

## 1. Setup

We have N expert delta vectors d_1, ..., d_N in R^D. Gram-Schmidt (GS)
produces orthogonalized deltas d_1', ..., d_N'. The merged average is:

  A(sigma) = (1/N) * sum_k d_k'^(sigma)

where sigma is a permutation defining the GS processing order.

## 2. Order Sensitivity Metric

We define order sensitivity as the cosine-distance variation of A(sigma)
across random orderings:

  variation(%) = (1 - min_{sigma, tau} cos(A(sigma), A(tau))) * 100

where the minimum is taken over K random orderings (K=50 in our experiment).

The kill criteria use variation > 5% as the "practically significant" threshold.

## 3. Scaling with Cosine and N

### 3.1 Prior Result (N fixed)

The parent experiment (merge_order_dependence) established:

  variation(%) ~ slope * cos_pairwise

with slope ~ 80 at N=10 (flattened, D=4096).

### 3.2 New Result: Slope Depends on N

This experiment reveals the slope is N-dependent:

  slope(N) ~ alpha * N^beta

Fitted values:

| N  | slope | cos_5pct |
|----|-------|----------|
| 5  | 43.8  | 0.114    |
| 10 | 95.9  | 0.052    |
| 20 | 169.3 | 0.030    |
| 50 | 277.1 | 0.018    |

Fitting slope = alpha * N^beta via log-log regression:
  log(slope) = log(alpha) + beta * log(N)

  beta = 0.794, alpha = 13.84  (R2=0.971)

So: slope(N) ~ 13.8 * N^0.79

### 3.3 The 5% Threshold

The threshold cosine where variation = 5% is more precisely fit directly
from the interpolated crossing points:

  cos_5pct(N) = 0.616 * N^(-0.935)   (R2=0.9999)

| N   | Predicted | Measured  | Error |
|-----|-----------|-----------|-------|
| 5   | 0.137     | 0.137     | 0.0%  |
| 10  | 0.072     | 0.072     | 0.0%  |
| 20  | 0.037     | 0.037     | 1.4%  |
| 50  | 0.016     | 0.016     | 0.6%  |

The near-unity exponent (-0.935 ~ -1) means the threshold is approximately
inversely proportional to N: doubling the number of experts roughly halves
the safe cosine range.

Extrapolations:
  N=100:  cos_5pct = 0.0083
  N=500:  cos_5pct = 0.0019
  N=1000: cos_5pct = 0.0010
  N=5000: cos_5pct = 0.0002

### 3.4 Why Slope Grows with N

In GS, each vector d_k loses its projection onto the span of {d_1', ..., d_{k-1}'}.
The total signal lost by expert k is:

  ||d_k - d_k'||^2 = sum_{i<k} (d_k . d_i')^2 / ||d_i'||^2

For near-orthogonal experts (cos << 1), this is approximately:

  ~ sum_{i<k} cos^2(d_k, d_i) ~ (k-1) * cos^2

Summing over all experts, the total GS adjustment is O(N^2 * cos^2). Different
orderings redistribute this adjustment differently, creating variation:

  variation ~ O(N * cos)

The linear dependence on N (sub-linear N^0.80 in practice due to averaging)
explains why the slope grows with N.

## 4. Norm CV vs Cosine Variation

The experiment measures two metrics:
1. **Variation%** = (1 - min pairwise cosine of merged vectors) * 100
2. **Norm CV%** = coefficient of variation of merged vector L2 norms

Norm CV remains negligible (~0.001%) across all conditions. This means GS
preserves the total signal magnitude but changes its direction. The practical
implication: order dependence changes which knowledge is retained, not how much.

## 5. Implications for SOLE

### 5.1 The Universal Safety Criterion

Instead of a fixed cos=0.06 threshold, the correct criterion is:

  **SOLE is order-safe iff cos_pairwise < 0.616 * N^(-0.935)**

At production scale:
  d=896:  cos = 0.0002
  d=4096: cos < 0.0001

Safety margins at d=896:
  N=100:  need cos < 0.0083 (margin: 41.6x)
  N=500:  need cos < 0.0019 (margin: 9.2x)
  N=1000: need cos < 0.0010 (margin: 4.8x)
  N=5000: need cos < 0.0002 (margin: 1.1x)

SOLE remains safe up to N ~ 5000 experts at d=896 (where cos_5pct converges
to the production cosine). At d=4096 (Qwen 7B, cos < 0.0001), the threshold
doubles to N ~ 10,000.

### 5.2 The N=10 Coincidence

The parent experiment's "cos=0.06" threshold was specific to N=10. It was
correct for that N but was erroneously presented as a universal threshold.
This experiment corrects that: the threshold is N-dependent.

## 6. Assumptions

1. **Synthetic vectors with uniform cosine.** Real LoRA deltas have
   non-uniform pairwise cosines. The threshold applies to max pairwise cosine.
2. **Flattened analysis.** Per-sublayer analysis may differ (slope ~ 62 from
   layerwise experiment), but the N-dependence pattern should be similar.
3. **Average merge (1/N).** Weighted merge with unequal weights would show
   different sensitivity patterns.
4. **Fixed D=4096.** The slope may depend weakly on D, but prior experiments
   showed the relationship is primarily cos-determined, not D-determined.
