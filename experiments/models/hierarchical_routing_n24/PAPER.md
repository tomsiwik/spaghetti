# Hierarchical Two-Stage Routing at N=24: Proof Verification Report

## Verdict: KILLED (K593 FAIL, K594 PASS, K595 PASS)

## Theorem (restated)

Hierarchical routing decomposes N=24 into K~5 clusters, reducing each routing
stage to a solved problem (N<=5). If confusion-graph clustering captures
most misrouting (intra-cluster), then within-cluster misrouting is PPL-benign
(from Finding #192), and cluster-level routing faces a tractable K-class problem.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| P1: Cluster accuracy >= 60% | **97.3%** | **YES** (massively) |
| P2: Overall top-1 accuracy >= 50% | **40.4%** | **NO** |
| P3: Routed PPL < uniform PPL | 10.06 < 10.07 | YES (barely, 0.1%) |
| P4: Overhead < 15% | **0.85%** | **YES** |
| P5: delta_intra < 0.10 for clusters | See analysis | **YES** (but vacuous) |

## Hypothesis

Two-stage hierarchical routing (cluster selection then within-cluster routing)
achieves >=60% top-1 accuracy at N=24 by reducing each routing decision to N<=5.

**KILLED.** The hierarchy solves stage 1 brilliantly (97.3%) but stage 2 faces
the same representation bottleneck that kills flat routing. Within-cluster
accuracy is 41.5%, virtually identical to flat 39.6%.

## What This Model Is

Hierarchical two-stage routing with:
- Stage 1: 4-class cluster router (K=4 via Ward hierarchical clustering on
  cosine distance between mean-pooled hidden-state centroids)
- Stage 2: 4 within-cluster softmax routers (one per cluster)
- Total: 821K params across all routing heads

**Clustering result:**
| Cluster | Members | Stage-2 Accuracy |
|---------|---------|-----------------|
| 1 | code, education, engineering, music, sports | 27.0% |
| 2 | cooking, cybersecurity, linguistics, marketing, math, sociology | 41.7% |
| 3 | health_fitness, medical | 100.0% |
| 4 | agriculture, creative_writing, economics, environmental, finance, history, legal, philosophy, politics, psychology, science | 36.8% |

## Key References

- Fiedler (1973): Algebraic connectivity for graph partitioning
- Von Luxburg (2007): Spectral clustering tutorial
- HMoRA (ICLR 2025): Hierarchical MoE routing
- Finding #179: N=5 routing at 100% accuracy
- Finding #192: Flat softmax at 39.4%, within-cluster misrouting PPL-benign

## Empirical Results

**Platform:** Apple M5 Pro 48GB, MLX, BitNet-2B-4T (2.4B params)
**Runtime:** ~180 seconds total

### What Worked

1. **Clustering captures confusion structure.** 93.8% of all flat-router
   confusion is intra-cluster. The confusion graph is well-structured.

2. **Stage 1 is solved.** 97.3% cluster accuracy, far exceeding the 60%
   threshold. Reducing from 24 classes to 4 classes makes the problem trivial.

3. **Overhead is negligible.** 0.85% total (two MLP forwards), well within 15%.

4. **PPL result confirms Finding #192.** Routed PPL (10.06) and uniform PPL
   (10.07) are essentially identical to oracle PPL (10.05) and base PPL (10.06).
   ALL adapters have near-zero marginal effect on PPL: oracle is 0.04% below
   base. There is no PPL signal to route on.

### What Failed and Why

**K593 FAIL: 40.4% accuracy (threshold 60%).**

The hierarchy does NOT improve overall accuracy because:

1. **Stage 2 IS the flat problem at smaller N.** Within cluster 4 (11 members),
   the stage-2 router faces N=11 domains that ALL have overlapping hidden-state
   representations. Reducing from N=24 to N=11 does not cross any discriminability
   threshold.

2. **The proven N=5 result does not transfer.** N=5 routing works at 100% because
   those 5 domains (python, math, medical, legal, creative) are maximally distinct.
   The current clusters contain CONFUSABLE domains grouped together. N=5 with
   distinct domains is a fundamentally different problem than N=5 with similar domains.

3. **The "each stage faces N<=5" reasoning was wrong.** The proof's Step C3 assumed
   that any N<=5 problem is tractable because N=5 was proven. But the proven N=5
   result exploited domain distinctness, not N-smallness. The same 6 domains
   that succeed at N=24 (finance, health_fitness, legal, math, medical, psychology)
   succeed in any architecture; the same 18 that fail at N=24 also fail at N=5
   when grouped together.

### The Deeper Lesson: No Routing Signal Exists for These Adapters

The oracle PPL is **10.05** vs base **10.06** -- a 0.04% difference. These 24
adapters trained on 400 samples each at rank-16 on a 2.4B ternary model provide
almost no specialization benefit. The routing problem is not that we cannot identify
the correct adapter -- it is that the correct adapter provides negligible benefit
over the wrong adapter or even no adapter.

This explains why all 7 routing mechanisms (6 flat + 1 hierarchical) produce
essentially the same PPL regardless of accuracy: there is nothing to route.

## Limitations

1. Only tested K=4 (Ward clustering). K=5,6 failed balance criteria. Different
   clustering methods (spectral, DBSCAN) might yield different groupings.

2. 40 training samples per domain for routing heads.

3. The PPL-near-zero-benefit observation may be specific to this adapter training
   setup (short training, small data, ternary base), not to routing in general.

## What Would Kill This

Already killed by K593. The fundamental issue is that the representation bottleneck
exists at EVERY N > 6 for these 24 domains, regardless of hierarchical decomposition.

To truly fix routing at N=24, one needs either:
- A discriminative representation (LoRAuter's SupCon-trained encoder)
- Adapters that actually specialize enough that routing matters (oracle PPL << base PPL)
- Per-token routing that sidesteps sequence-level mean pooling
