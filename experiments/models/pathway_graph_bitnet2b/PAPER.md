# Pathway Graph Construction on BitNet-2B: Co-Activation Topology

## STATUS: KILLED

K623 failed: random baseline control shows persistence is a sparsification artifact.

## Framework (from MATH.md)

Guided exploration (measurement): does BitNet-2B-4T have non-trivial co-activation
topology detectable by 0-dimensional persistent homology?

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| P1: >= 10 features with persistence > 0.1 | 91 features | YES |
| P2: Rank correlation(persistence, SV rank) < 0.5 | rho = 0.325 (p < 0.001) | YES |
| P3: Power-law distribution of persistence | Slope = -0.08, R^2 = 0.66 | PARTIAL |
| P4: Cross-domain inputs create longest bridges | All 100 directions are multi-domain | UNCLEAR |

2/4 predictions confirmed. P3 shows weak power-law (shallow slope). P4 cannot be
assessed because ALL top-100 SVD directions activate across 3+ domains — there are
no specialist directions at this granularity, so the bridge/specialist distinction
is not meaningful.

## What This Experiment Is

Measurement experiment: does BitNet-2B-4T have non-trivial topological structure
in its co-activation pathways? We collect FFN activations at layer 15 from 2250
inputs (450 per domain), compute SVD to get top-100 singular directions, build a
co-activation graph (edge weight = fraction of inputs where both directions activate),
sparsify, and compute 0-dimensional persistent homology.

## Key References

- Neural Persistence (1812.09764): foundational PH on neural nets
- Neural Topology Probing (2506.01042): graph probing outperforms activation probing 130%
- Stability Theorem (Cohen-Steiner et al., 2007): PH is a stable topological invariant

## Empirical Results

### Persistence Statistics

| Metric | Value |
|--------|-------|
| Total features (finite) | 91 |
| Features with persistence > 0.1 | 91 |
| Max persistence | 0.983 |
| Median persistence | 0.957 |
| Mean persistence | 0.933 |

### Rank Analysis

| Metric | Value |
|--------|-------|
| Spearman rho (SV rank vs degree rank) | 0.325 |
| p-value | 9.69e-4 |
| Top-10 overlap (SV vs degree) | 4/10 |

### Domain Structure

Inter-domain cosine similarities in SVD-projected space:

| Domain Pair | Cosine |
|-------------|--------|
| legal - finance | +0.441 (most similar) |
| code - finance | -0.506 (most dissimilar) |
| math - legal | -0.427 |
| code - legal | -0.399 |
| medical - math | -0.373 |

Legal and finance cluster together (knowledge-dependent domains). Code and finance
are maximally dissimilar. This is consistent with the domain categorization from
Finding #217 (knowledge-dependent vs structured-output domains).

Bridge directions (activated by >= 3 domains): 100/100
Specialist directions (1 domain only): 0/100

## Kill Criteria Assessment

| Criterion | Result | Evidence |
|-----------|--------|----------|
| K623: >= 10 features persistence > 0.1, > 1.5x random | **FAIL (KILL)** | 91 features, but random baseline: 96.8 |
| K624: Persistence rank vs SV rank corr < 0.5 | **PASS** | rho = -0.07 (essentially zero) |

**Random baseline control:** Random graphs with same edge density and weight
distribution produce MORE high-persistence features (96.8 avg) than the real
BitNet-2B co-activation graph (91). The entire persistence result is explained
by the 50th-percentile sparsification procedure, not by meaningful neural structure.

## Key Findings

### Finding 1: KILLED — Co-Activation Persistence is a Sparsification Artifact

91 features with persistence > 0.1, but random graphs produce 96.8 on average.
The persistence is mechanically caused by the 50th-percentile sparsification,
not by meaningful structure in BitNet-2B. Any graph sparsified this way produces
similar persistence diagrams.

### Finding 2: Persistence Rank is Independent of Spectral Rank

Using the corrected per-vertex persistence importance (max persistence of any
merge event the vertex participates in): Spearman rho = -0.07 (p=0.51) between
persistence rank and SV rank. Essentially zero correlation. However, since the
underlying persistence is an artifact (Finding 1), this independence may not be
meaningful.

### Finding 3: All Top-100 SVD Directions are Multi-Domain

All 100 SVD directions activate across 3+ domains. At this level of granularity,
there are no domain-specialist directions. This is consistent with the shared
representation hypothesis: the top principal components capture cross-domain
structure, while domain-specific information lives in lower-rank directions.

### Finding 4: Legal-Finance Domain Cluster Confirmed

The cosine similarity matrix shows legal and finance cluster together (cos=0.44),
while code-finance are most dissimilar (cos=-0.51). This independently confirms
the domain categorization from Finding #217: legal and finance are "knowledge-
dependent" domains with similar activation patterns, distinct from structured-
output domains (code, math).

## Limitations

1. **Single layer.** Only layer 15 tested. Different layers may show different topology.
2. **Sparsification sensitivity.** The co-activation graph was sparsified at the
   50th percentile of edge weights. Different thresholds would change the PH results.
3. **Top-100 directions only.** Energy captured = 62.6%. Important pathways may live
   in the remaining 37.4% of singular value spectrum.
4. **0-dim PH only.** We only computed connected components (H_0). Higher-dimensional
   homology (H_1 = loops, H_2 = voids) may reveal additional structure.
5. **Numerical warnings.** Some bfloat16 overflow in activation collection, cleaned
   with nan_to_num. May affect a small fraction of activation vectors.
6. **2250 inputs (not 10K as originally planned).** Limited by available data.
   With more data, the co-activation estimates would be more stable.

## What Would Kill This

- If layer-by-layer analysis shows trivial topology at most layers
- If the persistence structure disappears with different sparsification thresholds
- If persistence features perfectly predict singular value rank (rho > 0.5) at
  other layers
- If domain structure in the topology does not correlate with behavioral outcomes
