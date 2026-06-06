# Cross-Domain Composition: Mathematical Foundations (Revision 2)

## 1. Setup

Given a frozen base model W_base in R^{d x d} and N domain experts producing
rank-r deltas {Delta_1, ..., Delta_N}, a cross-domain query Q_{ij} requires
knowledge from domains i and j simultaneously.

**Example:** "compute 12+34=46, then reverse: 64" requires both arithmetic
(addition) and reversal (string reversal) capabilities.

## 2. Composition Strategies

### 2.1 Multi-Expert (Oracle 2-Expert)

For a cross-domain query Q_{ij}, compose the 2 relevant experts:

  W_{multi} = W_base + (1/2)(Delta_i + Delta_j)

This requires an oracle (or learned router) that identifies which 2 experts
are relevant. This is NOT what hash-ring routing does.

### 2.2 Single-Expert (Hash-Ring Scenario)

Hash-ring routes to ONE expert. For cross-domain Q_{ij}, the model sees:

  W_{single} = W_base + Delta_k    where k = argmin_{m in {i,j}} L(W_base + Delta_m, Q_{ij})

In practice, k is determined by the hash function, not by oracle selection.
We report the best-case single expert as an upper bound on hash-ring performance.

### 2.3 Naive Merge (All N Experts)

  W_{naive} = W_base + (1/N)(sum_k Delta_k)

Each expert contributes 1/N of its signal. At N=5, each expert has 20% weight.
At N=500, each has 0.2% -- severe dilution.

## 3. Quality Gap Analysis

Let L(W, Q) be the loss of model W on query set Q.

### 3.1 Per-Type Gap

For each cross-domain type t and seed s:

  gap_{t,s} = [L(W_{strategy}, Q_t^s) - L(W_{base}, Q_t^s)] / L(W_{base}, Q_t^s) * 100%

### 3.2 Aggregate Gap (with cancellation caveat)

  gap_agg = mean_{t,s} gap_{t,s}

**Cancellation artifact:** When some gap_{t,s} are positive (strategy worse) and
others negative (strategy better), the mean can be small even if individual gaps
are large. The aggregate gap is an AVERAGE that hides per-type variance.

To report honestly:
- mean +/- stddev across all (t,s) pairs
- max gap across all (t,s) pairs
- 75th percentile gap
- count of (t,s) pairs exceeding the kill threshold

## 4. Measured Results (d=32, r=4, N=5, 10 cross-domain types, 5 seeds)

### 4.1 Multi-Expert (Oracle 2)

| Statistic | Value |
|-----------|-------|
| Mean gap | -1.0% |
| Std dev | 16.7% |
| Max gap | +49.2% |
| 75th percentile | +11.1% |
| Trials exceeding 20% | 7/50 (14%) |
| Positive gaps (worse) | 22/50 (44%) |
| Negative gaps (better) | 28/50 (56%) |
| Mean of positive only | +15.0% |
| Mean of negative only | -13.6% |

The small aggregate mean (-1.0%) results from partial cancellation of positive
and negative gaps across different cross-domain types.

### 4.2 Single-Expert (Hash-Ring)

| Statistic | Value |
|-----------|-------|
| Mean gap | -7.0% |
| Std dev | 15.2% |
| Max gap | +31.2% |
| 75th percentile | +0.8% |
| Trials exceeding 20% | 3/50 (6%) |

Surprisingly, single-expert performs better on average than multi-expert. This
occurs because the second expert can add noise rather than complementary signal,
particularly when cross-domain queries are sequential (first half uses domain A,
second half uses domain B -- so domain A expert alone covers the hard part).

### 4.3 Routing (K2)

| Statistic | Value |
|-----------|-------|
| Error rate | 8.0% (4/50) |
| Random baseline | 40% (2/5 domains involved) |
| All errors on | reverse_parity |

The 8.0% error rate with 50 trials gives a 95% CI of approximately [2.2%, 19.2%]
(exact binomial). This is well below the 20% threshold but the confidence interval
is wide due to limited sample size.

## 5. Per-Type Breakdown (Multi-Expert)

| Type | Mean Gap | Std | Interpretation |
|------|----------|-----|----------------|
| arith_reverse | +21.5% | 15.1% | FAILS per-type, high variance |
| arith_sort | +10.5% | 8.2% | marginal |
| arith_repeat | -8.4% | 4.1% | composition helps |
| arith_parity | +0.9% | 7.1% | neutral |
| reverse_repeat | -15.8% | 5.1% | composition helps strongly |
| reverse_sort | -14.1% | 5.8% | composition helps strongly |
| reverse_parity | -18.2% | 7.0% | composition helps strongly |
| repeat_sort | +17.4% | 6.4% | marginal, near threshold |
| repeat_parity | +11.8% | 11.8% | marginal, high variance |
| sort_parity | -15.9% | 3.6% | composition helps strongly |

Pattern: cross-domain types involving the reverse expert tend to benefit from
composition (reverse_repeat, reverse_sort, reverse_parity all negative gaps).
Types involving arithmetic + another domain tend to suffer (arith_reverse, arith_sort
positive gaps). This may reflect that the arithmetic expert has the most distinct
token distribution (digits + operators vs. letters).

## 6. Assumptions

1. **Synthetic cross-domain queries:** Constructed by chaining domain operations
   sequentially. Real cross-domain queries involve simultaneous understanding.

2. **Oracle routing for multi-expert:** We assume the router correctly identifies
   which 2 experts are relevant. Hash-ring routing selects one expert, not two.

3. **Best-case single expert:** The K1_single metric uses the best of the two
   involved experts as an upper bound. Actual hash-ring routing may select
   a non-involved expert (captured by K2 error rate).

4. **Low-rank at micro scale:** Rank-4 at d=32 is capacity-constrained.
   At d=4096/r=16, signal retention is 95%+.

5. **Untrained base model:** Random weights, no language prior.

## 7. Complexity

| Operation | Single expert | Multi-expert (k=2) |
|-----------|---------------|-------------------|
| Inference | O(d^2) | O(d^2) (pre-merged) |
| Routing | O(log N) hash | O(k * routing_cost) |
| Merge | 0 | O(k * d * r) per-query or O(d^2) pre-merge |

Multi-expert adds routing complexity but no inference overhead if pre-merged.
