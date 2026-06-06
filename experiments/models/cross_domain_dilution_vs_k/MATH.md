# Cross-Domain Dilution vs Top-K: Mathematical Foundations

## 1. Setup

Given N domain experts with rank-r deltas {Delta_1, ..., Delta_N} applied to
frozen base W_base in R^{d x d}, a cross-domain query Q_{ij} requires
knowledge from domains i and j. The question is how to weight the two
relevant expert deltas when composing them.

**Notation:**
- W_base: base model weights (d x d per layer, L layers)
- Delta_k: expert k's weight delta (rank r, stored as B_k @ A_k)
- Q_t: query batch at test time, t in {1, ..., T}
- L(W, Q): NTP loss of model W on query Q
- w_k: composition weight for expert k (sum w_k = 1)

## 2. Composition Strategies

### 2.1 Equal Weight (Baseline)

  W_{eq} = W_base + (1/K) sum_{k in S} Delta_k

where S is the set of K involved experts. All experts contribute equally.

**Problem:** When one expert is relevant and the other adds noise, the
noise dilutes the useful signal. The gap vs base is:

  gap_{eq} = L(W_{eq}, Q) - L(W_base, Q)

Parent experiment measured: mean gap = -0.6% but with 24.3% worst case
(arith_reverse), demonstrating severe dilution on specific pairs.

### 2.2 Weighted Composition (General)

  W_{weighted} = W_base + sum_{k in S} w_k * Delta_k

where w_k >= 0 and sum w_k = 1 (or sum w_k = K to match equal-weight scale).

The weights w_k are computed from a relevance score r_k:

  w_k = exp(r_k / tau) / sum_j exp(r_j / tau)    (softmax with temperature tau)

### 2.3 Relevance Scoring Functions

We test 4 scoring functions, ordered by information requirements:

**2.3.1 Activation Magnitude (no labels, weight-space only)**

  r_k^{act} = sum_l || Delta_k^l @ h_query ||_2

where h_query is the mean hidden state of query tokens through the base model,
and the sum is over layers. Intuition: experts whose deltas strongly transform
the query direction are more relevant.

**2.3.2 Logit Difference (no labels, requires forward pass)**

  r_k^{logit} = || f(W_base + Delta_k, Q) - f(W_base, Q) ||_{RMS}

where f returns logits. Intuition: experts that change predictions more are
more "activated" by this query.

**2.3.3 PPL Probe (no labels at test time, requires small probe buffer)**

  r_k^{probe} = -L(W_base + Delta_k, Q_{probe})

where Q_{probe} is a small (n=10) subset of queries. This is a cheap
approximation to the full loss-based oracle.

**2.3.4 Full Loss Oracle (requires labels, upper bound)**

  r_k^{loss} = -L(W_base + Delta_k, Q)

This uses the full test set with ground truth. It is not production-viable
but establishes the upper bound on what weighted composition can achieve.

## 3. Theoretical Analysis

### 3.1 Why Equal Weight Fails

Consider K=2 experts with deltas Delta_1 (relevant) and Delta_2 (noise).
The equal-weight composed model is:

  W_{eq} = W_base + (Delta_1 + Delta_2) / 2

The output perturbation is:

  f(W_{eq}, x) - f(W_base, x) = (Delta_1(x) + Delta_2(x)) / 2

If Delta_2 is orthogonal to the task and has comparable norm to Delta_1,
then the noise contribution is:

  ||Delta_2(x)|| / ||Delta_1(x) + Delta_2(x)|| ~ 1/sqrt(2) ~ 70.7%

This 70% noise fraction explains the severe degradation on pairs where
one expert is irrelevant.

### 3.2 Optimal Weights

The optimal weights minimize the composed model's loss:

  w* = argmin_{w: sum w_k = 1} L(W_base + sum w_k Delta_k, Q)

For linear models with squared loss, the optimal weight for expert k is
proportional to:

  w_k* prop to <Delta_k, X^T Y> / ||Delta_k||^2

i.e., the correlation of the expert's delta with the data gradient.
This is what the loss-based oracle approximates.

### 3.3 Proxy Quality

The key question is whether a cheap proxy r_k can approximate w_k*. We
measure this by the Pearson correlation between proxy weights and oracle
weights across all (cross_type, seed) pairs.

If r(proxy, oracle) ~ 1, then the proxy is sufficient for production use.

## 4. Measured Results (d=32, r=4, N=5, K=2, 5 seeds)

### 4.1 Strategy Comparison

| Strategy | Mean Gap | Std | Max | Improvement vs EW |
|----------|----------|-----|-----|-------------------|
| equal_weight | -0.60% | 18.80% | +34.1% | baseline |
| activation_weighted | -0.71% | 18.66% | +32.6% | +0.11pp |
| logit_diff_weighted | +1.59% | 21.84% | +62.5% | -2.19pp |
| ppl_probe_weighted | -9.94% | 13.72% | +23.4% | +9.34pp |
| loss_weighted | -10.06% | 13.69% | +23.8% | +9.46pp |
| top1_oracle | -5.85% | 16.65% | +37.4% | +5.25pp |

### 4.2 Proxy-Oracle Correlation

| Proxy | Pearson r vs oracle | Weight std |
|-------|---------------------|------------|
| activation | 0.023 | 0.007 |
| logit_diff | -0.245 | 0.079 |
| ppl_probe (n=10) | 0.990 | 0.207 |

The ppl_probe proxy with only 10 examples achieves r=0.990 correlation
with the full-dataset oracle, and nearly identical performance
(9.34pp vs 9.46pp improvement).

### 4.3 arith_reverse Worst Case

| Strategy | Mean Gap | Max Gap |
|----------|----------|---------|
| equal_weight | +24.3% | +34.1% |
| ppl_probe | -8.5% | +6.3% |
| loss_weighted | -8.5% | +6.3% |
| top1_oracle | -7.9% | +6.8% |

Weighted composition completely resolves the arith_reverse failure mode:
from +24.3% degradation to -8.5% improvement vs base.

## 5. Complexity

| Strategy | Forward passes | Labels needed | Production-viable |
|----------|---------------|---------------|-------------------|
| equal_weight | 1 | No | Yes |
| activation_weighted | 1 | No | Yes |
| logit_diff_weighted | K+1 | No | Yes |
| ppl_probe_weighted | K+1 | No* | Yes (with probe buffer) |
| loss_weighted | K+1 | Yes | No (oracle only) |
| top1_oracle | K+1 | Yes | No (oracle only) |

*ppl_probe uses NTP loss on a probe buffer, which does not require
explicit labels -- just representative text from the query distribution.

## 6. Assumptions

1. **Two-expert composition only.** K=2 is the simplest case. Dilution
   worsens with K>2 (each expert weighted 1/K), making selective weighting
   more important at scale.

2. **Synthetic sequential cross-domain queries.** Real cross-domain queries
   require simultaneous dual-domain understanding, not sequential chaining.

3. **Rank-4 at d=32.** Capacity-constrained. At d=4096/r=16, expert
   deltas carry more distinctive information, potentially making
   weight-space relevance signals (activation, logit_diff) more
   discriminative.

4. **PPL probe requires representative queries.** The 10-example probe
   works because it samples from the same distribution as the test set.
   In production, the probe buffer would contain recent queries for the
   domain, which may be less representative.

5. **Random-initialized base model.** No language prior. Pretrained bases
   may exhibit different expert-query interaction patterns.

## 7. Worked Example

At d=32, r=4, consider arith_reverse cross-domain query "12+34=46>64":
- arithmetic expert delta has high activation on digit tokens
- reverse expert delta has high activation on letter tokens
- For this query (mostly digits + one reversal), arithmetic is more relevant

Equal weight: w_arith = w_reverse = 0.5
Loss oracle: w_arith ~ 0.7, w_reverse ~ 0.3 (arithmetic dominates)
PPL probe (n=10): w_arith ~ 0.69, w_reverse ~ 0.31 (nearly identical)

The 0.2 weight shift from reverse -> arithmetic reduces noise injection
by ~40%, explaining the gap improvement from +24.3% to -8.5%.
