# PPL-Probe Temperature Sensitivity: Mathematical Foundations

## Problem Statement

In SOLE, PPL-probe weighting composes K selected experts by computing weights
from per-expert loss scores on a small probe set. The weight computation uses
a softmax with temperature parameter tau:

$$w_i = \frac{\exp(-s_i / \tau)}{\sum_{j=1}^{K} \exp(-s_j / \tau)}$$

where $s_i$ is the loss of expert $i$ on the probe input. Prior experiments
used $\tau = 1.0$ without justification. This experiment tests whether quality
is robust across temperature values.

## Notation

| Symbol | Shape/Type | Description |
|--------|-----------|-------------|
| d | scalar | Model dimension (d=32 in micro) |
| r | scalar | LoRA rank (r=4 in micro) |
| K | scalar | Number of selected experts (K in {2,3,5}) |
| N | scalar | Total expert pool size (N=5) |
| A_i | r x d | Expert i down-projection |
| B_i | d x r | Expert i up-projection |
| Delta_i | d x d | Expert i full delta: B_i @ A_i |
| s_i | scalar | Loss score of expert i on probe |
| tau | scalar | Softmax temperature |
| w_i | scalar | Weight for expert i |
| T | d x d | Target (oracle-optimal) composition |

## Temperature Regimes

The temperature controls the entropy of the weight distribution:

- **tau -> 0**: argmax behavior, w -> one-hot on best expert (top-1)
- **tau = 1.0**: standard softmax, moderate discrimination
- **tau -> inf**: uniform weights, w_i -> 1/K (equal-weight baseline)

### Entropy Analysis

Shannon entropy of weight vector w(tau):

$$H(\tau) = -\sum_{i=1}^K w_i(\tau) \log w_i(\tau)$$

Bounds:
- H(0) = 0 (one-hot)
- H(inf) = log(K) (uniform)
- H is monotonically increasing in tau

### Key Insight

If the loss scores s_1, ..., s_K have small variance relative to tau, then
the softmax is approximately uniform regardless of tau. Conversely, if
variance is large relative to tau, the softmax is approximately argmax.

The critical scale is: $\tau^* \sim \text{std}(s_1, ..., s_K)$

For tau >> std(s), weights are near-uniform. For tau << std(s), weights are
near-one-hot. The "useful" range of tau is within an order of magnitude of
std(s).

## Oracle Construction

The oracle-optimal composition minimizes Frobenius distance to a random
target matrix T:

$$\min_{w} \| T - \sum_i w_i \Delta_i \|_F^2$$

This is a least-squares problem in w. Let $\delta_i = \text{vec}(\Delta_i)$
and $t = \text{vec}(T)$. Then:

$$w^* = (D^T D)^{-1} D^T t$$

where $D = [\delta_1, ..., \delta_K]$ is the matrix of vectorized expert deltas.

## Quality Metric

Gap improvement measures how much PPL-probe improves over equal-weight:

$$\text{gap}(\tau) = \frac{\|T - \Delta_{equal}\|_F - \|T - \Delta_{probe}(\tau)\|_F}{\|T - \Delta_{equal}\|_F} \times 100$$

Positive = probe is better than equal-weight. The variance of this metric
across tau values determines robustness.

## Kill Criteria Formalization

- **K1**: $\text{Var}_\tau[\text{mean\_gap}(\tau)] > 25$ (i.e., >5pp std dev)
- **K2**: $|\log_2(\tau^* / 1.0)| > 1$ (optimal tau differs from 1.0 by >2x)

## Worked Example (d=4, r=2, K=2)

Two experts with losses s_1=1.2, s_2=0.8 (expert 2 is better):

| tau | w_1 | w_2 | entropy |
|-----|-----|-----|---------|
| 0.1 | 0.018 | 0.982 | 0.11 |
| 0.5 | 0.310 | 0.690 | 0.63 |
| 1.0 | 0.401 | 0.599 | 0.67 |
| 2.0 | 0.450 | 0.550 | 0.69 |
| 5.0 | 0.480 | 0.520 | 0.69 |

The weight distribution is relatively insensitive to tau for moderate score
differences. This is the hypothesis: that tau=1.0 is not special and a wide
range works comparably well.

## Computational Cost

Per seed: K * N_tau matrix multiplications (d x d) plus one SVD for oracle.
Total: 50 seeds * 3 K-values * 5 tau-values * 5 experts = 3750 compositions.
Each is O(d^2 * r) = O(32^2 * 4) = O(4096). Negligible compute.
