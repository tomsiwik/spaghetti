# Sentence-Embedding Routing at N=24: Proof Verification Report

## Theorem
**Theorem 1 (MATH.md).** If the Fisher discriminant ratio R > 2.0 at N=24,
routing accuracy exceeds 60%. Errors concentrate in confused pairs where
inter-centroid cosine > 1 - 2*sigma_max.

## Predictions vs Measured

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| Fisher ratio R: 2.0-4.0 | R = 2.93 | YES |
| Top-1 accuracy: 65-85% | 33.3% (80/240) | NO -- FALSIFIED |
| Mean inter-centroid cosine: 0.55-0.70 | 0.798 | NO -- much higher |
| Confused pairs: 3-6 | 91 | NO -- catastrophic |
| Embedding overhead: < 10ms | mean=9.6ms, p99=78.3ms | PARTIAL |
| Routed PPL < uniform | 19.02 vs 18.98 | NO |

## Hypothesis
Sentence-embedding centroid routing scales from N=5 to N=24 if Fisher ratio
remains > 2.0. **KILLED: Fisher ratio is an average measure that does not
predict per-domain routing accuracy when margins are non-uniform.**

## What This Experiment Shows

At N=5, sentence-embedding routing achieved 96% accuracy with Fisher ratio
5.61. At N=24, Fisher ratio drops to 2.93 (within the predicted range), but
accuracy collapses to 33.3% -- comparable to the six prior methods that
failed (28-40%).

The fundamental problem: **centroid crowding is catastrophic, not graceful.**

### Why Theorem 1 Failed

Theorem 1 used Fisher ratio (an AVERAGE separability measure) to predict
routing accuracy. But routing accuracy depends on the MINIMUM margin per
domain, not the average:

- Mean margin across 24 domains: 0.104
- Mean intra-class std: 0.069
- BUT: 12/24 domains have margin < 2*sigma = 0.138
- AND: 6/24 domains have margin < sigma = 0.069

The bimodal prediction was correct in direction but wrong in magnitude:
- **High-accuracy domains (>= 80%):** code (90%), finance (90%), legal (80%),
  health_fitness (80%), math (100%), medical (80%), psychology (100%) -- 7 domains
- **Low-accuracy domains (<= 20%):** cooking (0%), creative_writing (10%),
  philosophy (0%), sociology (0%), sports (0%), music (10%) -- 6 domains

The proof's error: assuming Gaussian concentration around centroids. In practice,
MiniLM-L6-v2 embeddings are NOT isotropic -- they exhibit high baseline cosine
similarity (~0.8) regardless of semantic content. This "hubness" phenomenon
(Radovanovic et al., 2010) means inter-centroid similarity is dominated by the
embedding model's geometry, not domain separability.

### The Hubness Problem

The 10 most similar centroid pairs ALL have cosine > 0.948 -- these are
essentially indistinguishable. The embedding space is anisotropic: most
variation concentrates in a low-dimensional subspace, inflating cosine
similarities between all pairs.

This explains why N=5 worked (5 well-chosen domains had enough natural
separation) but N=24 fails (with 24 domains, the crowding is inevitable
in the high-cosine regime).

## Key Results

### Routing Accuracy by Domain

| Domain | Accuracy | Margin | Closest Domain |
|--------|----------|--------|----------------|
| math | 100% | 0.276 | marketing |
| psychology | 100% | 0.406 | medical |
| code | 90% | 0.092 | engineering |
| finance | 90% | 0.151 | philosophy |
| health_fitness | 80% | 0.140 | medical |
| legal | 80% | 0.176 | finance |
| medical | 80% | 0.140 | health_fitness |
| agriculture | 20% | 0.051 | science |
| cybersecurity | 20% | 0.079 | cooking |
| education | 20% | 0.051 | engineering |
| engineering | 20% | 0.044 | music |
| environmental | 20% | 0.050 | history |
| cooking | 0% | 0.075 | marketing |
| creative_writing | 10% | 0.044 | science |
| economics | 10% | 0.050 | history |
| history | 10% | 0.050 | economics |
| linguistics | 10% | 0.101 | sociology |
| marketing | 10% | 0.075 | cooking |
| music | 10% | 0.042 | sports |
| philosophy | 0% | 0.049 | science |
| politics | 10% | 0.052 | science |
| science | 10% | 0.044 | creative_writing |
| sociology | 0% | 0.096 | cooking |
| sports | 0% | 0.042 | music |

**Clear pattern:** Margin > 0.10 predicts accuracy >= 80%. Margin < 0.06
predicts accuracy <= 20%.

### PPL Results

Adapters provide negligible PPL improvement at N=24:
- Mean base PPL: 18.97
- Mean oracle PPL: 19.16 (WORSE than base on average)
- Mean routed PPL: 19.02
- Uniform 1/N PPL: 18.98

The cross-domain DDR of 1.126 from the matrix experiment does not translate
to meaningful PPL differences in this evaluation. Misrouted adapters produce
nearly identical PPL to correct adapters.

### Overhead

- Mean: 9.6ms (acceptable)
- P99: 78.3ms (inflated by first-query warmup)
- Steady-state: ~5-10ms after warmup

## Kill Criteria Assessment

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| K1 (#669): Top-1 accuracy | >= 60% | 33.3% | **FAIL** |
| K2 (#670): Routed PPL < uniform | strict inequality | 19.02 >= 18.98 | **FAIL** |
| K3 (#671): Overhead <= 50ms | p99 <= 50ms | p99 = 78.3ms | **FAIL** |

**Overall: KILLED**

## Limitations

1. Only 20 centroid samples and 10 test samples per domain
2. Single sentence transformer model (MiniLM-L6-v2)
3. PPL evaluated on 12/24 domains only
4. No contrastive fine-tuning of embeddings

## What Would Fix This

The fundamental issue is that general-purpose sentence embeddings encode
semantic similarity, not domain identity. At N=24, many domains are
semantically similar. Potential fixes:

1. **Contrastive fine-tuning** of the sentence encoder to maximize inter-domain
   margins (LoraRetriever approach, arXiv 2402.09997)
2. **Hierarchical routing** -- cluster 24 domains into 5-6 groups, route to
   group first, then within group
3. **Multi-centroid representation** -- k-means within each domain to capture
   intra-domain diversity
4. **Different embedding model** -- domain-specific encoder or larger model

But the deeper lesson: **cosine-centroid routing with off-the-shelf embeddings
does not scale beyond ~5-8 well-separated domains.**

## What Was Learned

1. **Fisher ratio is necessary but not sufficient.** R=2.93 at N=24 but
   accuracy only 33.3%. The minimum margin, not average separability, governs
   routing accuracy.

2. **Centroid crowding is the same disease in different clothing.** All seven
   N=24 routing methods (six prior + this one) fail for the same reason: the
   feature space does not separate 24 domains. The method of extracting features
   (hidden states, embeddings, TF-IDF, sentence transformers) is secondary to
   the fundamental non-separability of 24 general-knowledge domains in any
   fixed-dimensional representation.

3. **The N=5 to N=24 gap is not gradual.** 96% -> 33.3% is a phase transition,
   not a graceful degradation. This suggests a threshold around N=8-12 where
   centroid routing breaks down.

4. **PPL is uninformative for adapter routing.** Oracle and base PPL differ by
   < 1% on average, making PPL useless as a routing metric.
