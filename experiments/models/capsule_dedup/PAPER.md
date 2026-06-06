# Capsule Deduplication: Research Digest

## Hypothesis

After composing domain-specific ReLU MLPs by weight concatenation,
independently-trained capsule pools develop redundant detectors that can
be identified via cosine similarity of a_i vectors (cos > 0.95) and
merged using a-average / b-sum to reduce parameter count without
quality loss.

**Falsifiable**: If fewer than 10% of cross-pool capsule pairs have
cos > 0.95, there is insufficient redundancy to deduplicate. If
deduplication degrades quality by more than 5% vs unmerged concatenation,
the merging rule destroys useful information.

**Result: INSUFFICIENT REDUNDANCY.** Only 1.9% of capsules are
redundant at cos > 0.95 (0.1% of cross-pool pairs exceed the threshold).
The mean cross-pool cosine similarity is 0.296 -- capsules from
different domains are nearly orthogonal, not redundant. The 54% shared
knowledge found by Procrustes (Exp 3) does NOT manifest as shared
capsule detectors. However, a much larger form of waste was discovered:
~60% of capsules in the composed model are dead (never fire for any
input). Dead capsule pruning, not deduplication, is the real
opportunity for parameter reduction.

---

## What This Experiment Tests

Whether cosine-similarity-based deduplication of redundant capsules
after composition preserves quality while reducing parameter count.

Protocol:
1. Pretrain base model on ALL data (shared attention + embeddings)
2. Fine-tune only MLP weights per domain (attention frozen)
3. Compose by concatenating A and B weight matrices from both domains
4. Deduplicate: find capsules with cos(a_i, a_j) > threshold, merge
5. Evaluate: quality impact and parameter savings

Sweep: threshold in {0.90, 0.95, 0.99}, cross-pool-only vs all-pairs.

Controls:
- Joint training (upper bound)
- Unmerged concatenation (+4.3% zero-shot, from loudness_fix)
- Weight averaging (+1.5%, best zero-shot method)

---

## Lineage in the Arena

```
gpt  ->  moe  ->  capsule_moe  ->  relu_router  ->  capsule_dedup
                                    (composition     (post-hoc
                                     by concat)       deduplication)
```

---

## Key References

**BuddyMoE (2024)**: Co-activation profiling for expert merging.
Uses behavioral (activation-based) similarity rather than weight
similarity. Our approach uses weight-space similarity, which is
exact for rank-1 capsules but may underestimate functional redundancy
for higher-rank experts.

**SERE (2024)**: Similarity-based expert re-routing using Frobenius
norm of weight differences. Related approach but for multi-layer
experts where weight similarity is a weaker proxy.

**Sub-MoE (2024)**: Adaptive expert clustering with SVD subspace
merging. Demonstrates successful expert deduplication at scale, but
uses activation-based clustering, not weight-space cosine similarity.

**PuzzleMoE (2024)**: Dual-mask element-wise merging. Uses cos > 0.95
threshold for safe merging of similar expert components, consistent
with our threshold choice.

**Procrustes Decomposition (Exp 3, this project)**: Found 54% of
fine-tuning knowledge is shared across domains via Procrustes
alignment of weight deltas. Failed because ReLU applied separately
to decomposed components loses information (+5.7% vs joint).

---

## Empirical Results

### 3-Seed Aggregate (seeds 42, 123, 7)

| Method | Avg Val Loss | Std | vs Joint | vs Concat |
|--------|-------------|-----|----------|-----------|
| joint (baseline) | 0.5269 | 0.0100 | -- | -7.0% |
| concat_zero_shot | 0.5663 | 0.0330 | +7.5% | -- |
| **weight_avg** | **0.5330** | **0.0108** | **+1.1%** | **-5.9%** |
| dedup cross tau=0.90 | 0.5663 | 0.0330 | +7.5% | +0.0% |
| dedup cross tau=0.95 | 0.5663 | 0.0330 | +7.5% | -0.0% |
| dedup cross tau=0.99 | 0.5663 | 0.0330 | +7.5% | -0.0% |
| dedup all tau=0.90 | 0.5663 | 0.0330 | +7.5% | +0.0% |
| dedup all tau=0.95 | 0.5663 | 0.0330 | +7.5% | -0.0% |
| dedup all tau=0.99 | 0.5663 | 0.0330 | +7.5% | -0.0% |
| dedup + calibration | 0.5197 | 0.0099 | -1.4% | -8.2% |

### Redundancy Statistics (3-seed mean, per layer)

| Threshold | Mode | Total Removed | Per-Layer [0,1,2,3] | Pct Removed |
|-----------|------|---------------|---------------------|-------------|
| tau=0.90 | cross-pool | 29/1024 | [2, 8, 13, 6] | 2.9% |
| tau=0.95 | cross-pool | 20/1024 | [0, 5, 11, 4] | 1.9% |
| tau=0.99 | cross-pool | 17/1024 | [0, 3, 11, 3] | 1.6% |

Cross-pool and all-pairs modes produce identical results, confirming
that within-pool redundancy is zero at these thresholds. All redundancy
is cross-pool.

### Cross-Pool Cosine Similarity Distribution

| Metric | Value |
|--------|-------|
| Mean cross-pool cos | 0.296 |
| Max cross-pool cos (layer avg) | ~0.98 |
| Pairs with cos > 0.95 | ~0.1% |
| Pairs with cos > 0.90 | ~0.1% |

### Dead Capsule Analysis

| Method | Total Dead/Total | Dead Percentage |
|--------|-----------------|-----------------|
| dedup tau=0.90 | 595/994 | 59.9% |
| dedup tau=0.95 | 602/1005 | 59.9% |
| dedup tau=0.99 | 605/1008 | 60.1% |

**~60% of capsules in the composed model never fire for any input.**
This is the most significant finding of the experiment.

### Parameter Counts (seed 42)

| Method | Total Params | Capsule Params Saved |
|--------|-------------|---------------------|
| joint (P=256) | 202,112 | -- |
| concat (P=256) | 202,112 | -- |
| weight_avg (P=128) | 136,576 | 32% less total |
| dedup tau=0.90 | 196,736 | 2.7% |
| dedup tau=0.95 | 198,912 | 1.6% |
| dedup tau=0.99 | 200,064 | 1.0% |

---

## Kill Threshold Analysis

| Criterion | Value | Target | Kill | Result |
|-----------|-------|--------|------|--------|
| Dedup vs concat quality | +0.0% | <5% | >5% | **PASS** (trivially) |
| Dedup vs weight_avg | +6.3% worse | Must beat | Must beat | **KILL** |
| Dead capsules after dedup | 59.9% | <20% | >20% | **KILL** |
| Redundancy at tau=0.95 | 1.9% removed | >2.5% | <2.5% | **KILL** |
| vs joint | +7.5% | <5% | >10% | **WARN** |

**Three of four kill criteria triggered.** Deduplication at micro scale
is not viable: too few capsules are redundant, too many are dead, and
the result is worse than weight averaging.

---

## Why There Is So Little Redundancy

The hypothesis assumed that the 54% shared knowledge (from Procrustes
Exp 3) would manifest as 54% shared capsule detectors. This is wrong.

**The 54% shared knowledge is distributed, not concentrated.** Procrustes
measures shared knowledge as the overlap between weight DELTAS from the
base model. This overlap is spread across all 128 capsules as small
adjustments to each detector, not as a subset of 69 identical detectors.

In concrete terms:
- Procrustes says: "54% of the total weight change is in the same direction
  for both domains"
- This experiment says: "only 1.9% of individual capsule detectors point
  in the same direction across domains"

These are measuring different things. Shared knowledge at the matrix level
does not imply shared knowledge at the individual neuron level.

**Mean cross-pool cosine of 0.296** indicates that capsules from different
domains are nearly orthogonal. This is consistent with the N=5 scaling
experiment (Exp 4) which found mean pairwise cosine of 0.112 between
domain SUBSPACES. Individual capsules within orthogonal subspaces are
also approximately orthogonal.

---

## The Real Finding: Dead Capsule Waste

The most important discovery is that **60% of capsules in the composed
model are dead** -- they never activate for any input in the evaluation
set. This is a much larger source of waste than redundancy.

Sources of dead capsules:
1. **Natural ReLU sparsity (~50%)**: Half of neurons are inactive for
   any given input. But "dead" here means never active for ANY input.
2. **Domain specialization**: Capsules fine-tuned for domain A may have
   detector vectors that never trigger on domain B inputs (and vice versa).
3. **Composition artifact**: The concatenated model has 256 capsules but
   each input only needs ~128 (one domain's worth). The "other domain's"
   capsules are systematically inactive.

This suggests an alternative approach: instead of deduplicating redundant
capsules (of which there are few), **prune dead capsules** (of which
there are many). A simple pruning strategy -- run evaluation data
through the model, remove capsules that never fire -- could achieve
~60% parameter reduction with zero quality impact.

---

## Micro-Scale Limitations

1. **Similar domains**: a-m vs n-z names share character distributions.
   With truly different domains (Python vs JavaScript), capsule detectors
   may be more diverse (less redundancy) OR more specialized (more dead
   capsules).

2. **Small P**: With only 128 capsules per domain, each capsule must
   cover a wide input region. At larger P (thousands of capsules), more
   capsules may specialize to narrow, overlapping regions, increasing
   redundancy.

3. **Short training**: 200-step fine-tuning may not be enough for
   capsules to converge to their final directions. Longer training could
   increase or decrease similarity.

4. **Two domains only**: With N=5 domains (Exp 4 configuration),
   cross-domain redundancy may accumulate. Each pair of domains may
   share a few capsules, and the aggregate across all pairs may be
   significant.

5. **Dead capsule count is measured on limited data**: The 60% dead
   figure is based on 10 evaluation batches. Some "dead" capsules may
   fire on rare inputs not in the sample.

---

## What Would Kill This

### At Micro Scale (tested)

- **Insufficient redundancy**: CONFIRMED. Only 1.9% of capsules are
  redundant at cos > 0.95. The 54% shared knowledge does not manifest
  as shared capsule detectors.

- **Dedup worse than weight averaging**: CONFIRMED. Dedup produces
  the same quality as concatenation (+7.5% vs joint), which is much
  worse than weight averaging (+1.1%).

- **Massive capsule death**: CONFIRMED. 60% dead capsules is far above
  the 20% kill threshold, indicating fundamental waste in the composed
  model.

### At Macro Scale (untested)

- **More redundancy at scale**: With larger P and more diverse domains,
  capsule redundancy may increase. The experiment should be repeated at
  P=1024+ with truly different domains.

- **Dead capsule pruning**: If dead capsules can be pruned without
  quality loss, this would be a much more impactful compression than
  deduplication. Worth testing independently.

- **Activation-based similarity**: Cosine similarity in weight space
  may miss functional redundancy. BuddyMoE-style co-activation
  profiling might find more to deduplicate.

---

## The Key Insight: Shared Knowledge Is Distributed, Not Concentrated

The Procrustes experiment (Exp 3) and this experiment measure "shared
knowledge" in fundamentally different ways:

| Metric | Procrustes (Exp 3) | Cosine Dedup (This) |
|--------|-------------------|---------------------|
| What is compared | Full weight delta matrices | Individual capsule vectors |
| Granularity | Matrix-level | Neuron-level |
| Shared fraction found | 54% | 1.9% |
| Interpretation | 54% of total weight change is shared | 1.9% of neurons are shared |

The reconciliation: shared knowledge is **distributed across many neurons
as small perturbations**, not concentrated in a few neurons as large
changes. This is why Procrustes can detect it (it measures the aggregate)
but capsule deduplication cannot (it measures individuals).

**Implication for the project**: Bottom-up deduplication (find identical
capsules) is the wrong approach for the same reason top-down decomposition
failed (Procrustes + ReLU breaks). The shared knowledge is structurally
entangled in the weight matrices, not separable into individual units.

The composition protocol should accept this entanglement:
1. **Weight averaging** (+1.1%) implicitly merges the distributed shared
   knowledge by averaging all neurons. This works BECAUSE shared knowledge
   is distributed.
2. **Concatenation + calibration** (-1.4%) works by letting the calibration
   step adapt to the composed distribution. The redundancy is resolved
   by the optimization, not by pruning.
3. **Dead capsule pruning** (untested) may be the best post-hoc compression:
   remove the 60% of capsules that are truly unused, not the 1.9% that
   happen to be similar.

---

## Implications for the Project

1. **Capsule deduplication is not viable at micro scale**: Only 1.9%
   redundancy at cos > 0.95. Not worth implementing as a standard
   post-composition step.

2. **Dead capsule pruning is the real opportunity**: 60% of capsules
   never fire. A simple activation-based pruning pass could halve the
   parameter count of composed models.

3. **The 54% shared knowledge is real but not separable**: It exists
   (Procrustes confirms) but cannot be isolated at the capsule level.
   Weight averaging remains the best way to exploit it for zero-shot
   composition.

4. **Cosine similarity of a_i vectors works as a redundancy detector**:
   The mechanism is correct and validated. At micro scale with similar
   domains, there is simply very little to detect.

5. **VISION.md item 9 is resolved**: Capsule deduplication via cosine
   similarity has been tested. The answer is that redundancy at the
   individual capsule level is insufficient for meaningful compression.
   The "shared knowledge" hypothesis is correct at the matrix level but
   does not decompose to the neuron level.
