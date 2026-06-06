# PPL-Probe Weighting at K=3+ Expert Composition

## Problem Statement

At K=2 expert composition, PPL-probe weighting achieves r=0.990 correlation with
the full-dataset oracle and +9.34pp improvement over equal-weight baselines.
Does this scale to K=3 and beyond, or does the discrimination task become too
hard when the probe must rank 3+ experts?

## Notation

| Symbol | Shape / Type | Definition |
|--------|-------------|------------|
| K | scalar | Number of experts composed simultaneously |
| N | scalar | Total expert pool (N=5 in our setup) |
| Delta_i | weight-shaped | Low-rank weight delta for expert i, rank r |
| w_i | scalar in [0,1] | Composition weight for expert i, sum(w_i)=1 |
| L_i | scalar | Loss of expert i on probe buffer |
| n | scalar | Probe buffer size (n=10 examples) |
| d | scalar | Model dimension (d=32 at micro scale) |
| r | scalar | Rank per expert (r=4) |

## Composition Strategies

### Equal Weight (baseline)
For K experts {i_1, ..., i_K}:

    W_composed = W_base + (1/K) * sum_{j=1}^{K} Delta_{i_j}

As K grows, each expert's contribution shrinks as 1/K. This is the **dilution
problem**: irrelevant experts contribute equal noise as relevant ones.

### PPL-Probe Weighted
For each expert i_j in the K-tuple, evaluate loss on a small probe buffer
of n examples from the test distribution:

    score_{i_j} = -L_{i_j}(probe)    (negative loss = higher relevance)
    w_{i_j} = softmax(score / tau)    (tau=1.0)
    W_composed = W_base + sum_{j=1}^{K} w_{i_j} * Delta_{i_j}

### Loss Weighted (Oracle)
Same as PPL-probe but uses the FULL test set instead of n=10 probe:

    score_{i_j} = -L_{i_j}(full_test)
    w_{i_j} = softmax(score / tau)

### Top-1 Oracle
Select only the single best expert from the K-tuple:

    i* = argmin_{i_j} L_{i_j}(full_test)
    W_composed = W_base + Delta_{i*}

## Scaling Analysis

### Discrimination Difficulty
At K=2, the probe must distinguish between 2 experts (binary ranking).
At K=3, it must produce a 3-way ranking. The number of possible orderings
grows as K!, but the probe's effective information stays constant at n
examples. The key question is whether n=10 provides sufficient signal for
3-way discrimination.

### Expected Dilution
Equal-weight dilution worsens as K increases because the probability that
all K experts are equally relevant decreases. For a query touching domains
{A, B}, a K=3 composition that includes irrelevant expert C assigns weight
1/3 to C (vs 0 ideally). The waste fraction is (K-K_relevant)/K.

For K=3 with 2 relevant domains: waste = 1/3 = 33%
For K=5 with 2 relevant domains: waste = 3/5 = 60%

This means the OPPORTUNITY for probe-weighting grows with K, but so does the
difficulty of getting the weights right.

### Information-Theoretic Bound
The probe buffer provides H_probe = n * H_per_example bits of information
about expert relevance. To discriminate K experts, we need at least
log2(K!) bits for a full ranking. For K=3: log2(6) = 2.58 bits.
For K=5: log2(120) = 6.91 bits.

With n=10 examples of character-level data (~5-15 tokens each), the probe
should provide >> 7 bits of discriminative signal, so information-theoretic
limits should not bind.

## Combinatorics

From N=5 domains:
- K=2: C(5,2) = 10 pairs (parent experiment)
- K=3: C(5,3) = 10 triples
- K=5: C(5,5) = 1 quintuple (all experts)

For K=3 cross-domain test data, we use K=2 cross-domain generators
from the parent experiment. Each K=2 cross-domain type involves 2 specific
domains. For a K=3 triple, the cross-domain queries are the same as the
K=2 pair that the triple contains. We evaluate all 3 possible K=2 subsets
of a K=3 triple and use the one that best matches the query distribution.

## Kill Criteria

- **K1**: PPL-probe vs oracle correlation drops below r=0.8 at K=3
  - Rationale: r=0.990 at K=2 means near-perfect tracking. A drop below 0.8
    would indicate the probe cannot discriminate 3-way rankings.
- **K2**: K=3 probe-weighted is worse than K=2 probe-weighted
  - Rationale: if adding a 3rd expert with smart weighting cannot match
    2-expert composition, the approach does not scale.

## Worked Example (K=3, d=32, r=4)

Consider triple (arithmetic, reverse, sort) evaluated on arith_reverse queries:
- Probe evaluates 10 arith_reverse examples on each single expert
- Expected: arithmetic expert has low loss, reverse expert has medium loss,
  sort expert has high loss
- Softmax weights might be: w_arith=0.6, w_reverse=0.3, w_sort=0.1
- Equal weight would be: w_arith=0.33, w_reverse=0.33, w_sort=0.33
- Probe concentrates weight on relevant experts, suppressing sort noise

The improvement over equal-weight should be LARGER at K=3 than K=2 because
there is more noise to suppress (1 irrelevant expert out of 3 vs 0 out of 2
for a 2-domain query).
