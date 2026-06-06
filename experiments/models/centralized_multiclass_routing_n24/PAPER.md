# Centralized Multi-Class Routing N=24: Proof Verification Report

## Theorem (from MATH.md)

Softmax normalization over a single multi-class routing head eliminates the
false-positive cascade and loudest-voice failure modes that killed binary heads
at N=24. With K=24 classes in d=2560 dimensions, Cover's theorem guarantees
linear separability. The proper scoring rule property of cross-entropy ensures
convergence to the true conditional distribution.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| Top-1 accuracy >70% | 39.4% | NO |
| Top-2 accuracy >85% | 53.1% | NO |
| Router params ~165K (12x fewer than binary) | 165,464 (11.9x fewer) | YES |
| Overhead <2% of base forward | 0.29% | YES |
| Routed PPL < uniform PPL | Routed 10.107 > uniform 10.080 | NO |
| FPR cascade eliminated by softmax | Eliminated (structurally) | YES |
| Accuracy >> binary heads (39.6%) | 39.4% (same) | NO |

## Verdict: KILLED

K587 FAIL: Top-1 routing accuracy 39.4% (threshold 60%)
K588 FAIL: Routed PPL 10.107 worse than uniform 10.080
K589 PASS: Overhead 0.29% (threshold 15%)

## Hypothesis

A single multi-class softmax routing head will achieve >60% top-1 accuracy
and beat uniform averaging at N=24, because softmax normalization forces
competition between classes and eliminates the false-positive cascade that
killed binary heads.

**Result: KILLED.** The multi-class router achieves 39.4% accuracy, essentially
identical to binary heads (39.6%). Softmax normalization is NOT sufficient.

## What This Experiment Reveals

### The Critical Finding: FPR Cascade Was Not the Disease

The MATH.md identified two failure modes from prior experiments:
1. Loudest-voice (killed energy gap routing)
2. FPR cascade (hypothesized as cause of binary heads failure)

This experiment proves that **eliminating both failure modes does NOT improve
routing accuracy**. The multi-class softmax head is structurally immune to
FPR cascade (Theorem 1 verified -- there is exactly one probability distribution
per input), yet achieves the same 39.4% accuracy.

This means the FPR cascade diagnosis was WRONG. It was a symptom, not the
disease. The actual disease is elsewhere.

### The Real Disease: Weak Domain Signal in Mean-Pooled Hidden States

The pattern of which domains are routable vs not is nearly identical across
both methods:

| Domain | Binary Heads | Multi-Class | Domain is... |
|--------|-------------|-------------|--------------|
| finance | 95% | 100% | Distinctive vocabulary |
| health_fitness | 100% | 100% | Distinctive vocabulary |
| legal | 100% | 95% | Distinctive vocabulary |
| math | 100% | 100% | Distinctive vocabulary |
| medical | 100% | 100% | Distinctive vocabulary |
| psychology | 100% | 100% | Distinctive vocabulary |
| sociology | 65% | 70% | Semi-distinctive |
| code | 65% | 60% | Semi-distinctive |
| agriculture | 50% | 40% | Generic |
| cooking | 5% | 15% | Generic |
| creative_writing | 20% | 10% | Generic |
| cybersecurity | 15% | 10% | Generic |
| economics | 0% | 0% | Generic |
| education | 15% | 15% | Generic |
| engineering | 25% | 10% | Generic |
| environmental | 10% | 10% | Generic |
| history | 5% | 10% | Generic |
| music | 5% | 35% | Generic |
| philosophy | 5% | 5% | Generic |
| politics | 0% | 10% | Generic |
| science | 10% | 5% | Generic |
| sports | 25% | 15% | Generic |

The same ~6 domains achieve near-perfect routing regardless of method.
The same ~14 domains are nearly random regardless of method. This is not
a routing architecture problem -- it is a **representation problem**.

### Why Accuracy is ~40% for Both Methods

With 24 domains where ~6 are perfectly routable and ~18 are near-random:
- 6 domains * ~95% accuracy = ~5.7 correct per 6
- 18 domains * ~15% accuracy = ~2.7 correct per 18
- Total: (5.7 + 2.7) / 24 = ~35% -- close to the measured 39.4%

This arithmetic is method-independent because the signal is in the hidden
states, not in the routing architecture.

### What the Proof Got Right and Wrong

**Right:**
- Softmax DOES eliminate FPR cascade (Theorem 1 verified)
- Softmax DOES eliminate loudest-voice (shift invariance verified)
- Overhead IS minimal (0.29% vs predicted <2%)
- Parameter efficiency IS 12x better (165K vs 1.97M)
- The router IS a proper scoring rule (loss converged to 0.92)

**Wrong:**
- The implicit assumption that FPR cascade was the bottleneck
- The prediction that eliminating FPR cascade would improve accuracy
- The claim that VC dimension guarantees are practically relevant
  (they guarantee separability EXISTS, not that it's LEARNABLE from
  40 samples per class with mean-pooled features)

### Deeper Analysis: The VC Dimension Red Herring

The proof cited Cover's theorem: d=2560 >> N=24 guarantees separability.
This is mathematically correct but practically irrelevant. The issue is not
whether a separating hyperplane EXISTS in R^2560, but whether:

1. Mean pooling preserves the class-discriminative information
2. 40 training samples per class are sufficient to FIND the boundary
3. The classes are actually distinct in the model's representation

The binary head experiment already proved domain signal exists (87.2% per-head
accuracy). But per-head accuracy measures 1-vs-rest binary classification,
where the "rest" class is 23x larger and easier to characterize. Multi-class
discrimination requires distinguishing between ALL 24 classes simultaneously,
which is fundamentally harder when ~14 classes map to overlapping regions of
hidden state space.

## Key References

- Switch Transformer (Fedus et al., arxiv 2101.03961): softmax routing for MoE
- Gneiting & Raftery 2007: proper scoring rules
- Cover 1965: function counting theorem
- Prior finding #179: binary heads 100% at N=5, 39.6% at N=24
- Prior finding #185: energy gap 88% at N=5, 8.3% at N=24

## Empirical Results

| Metric | Value |
|--------|-------|
| Top-1 accuracy (eval) | 39.4% |
| Top-2 accuracy (eval) | 53.1% |
| Router params | 165,464 |
| Router train time | 1.8s |
| Router overhead | 0.29% |
| Avg routed PPL | 10.107 |
| Avg uniform PPL | 10.080 |
| Avg base PPL | 10.057 |
| Avg individual PPL | 10.119 |
| Total experiment time | 244.7s |

## Limitations

1. **Small training set:** 40 samples per domain may be insufficient for
   multi-class learning. However, binary heads used the same data and the
   same ~40% accuracy resulted, suggesting data size is not the bottleneck.

2. **Mean pooling:** Aggregating all token hidden states via mean may destroy
   domain-discriminative information present in specific tokens. Future work
   could try CLS-token-like extraction or attention pooling.

3. **Single hidden layer:** The router uses d->64->K. A deeper or wider
   architecture might extract more discriminative features. However, the
   binary heads used per-class nonlinear boundaries and achieved the same
   accuracy, suggesting capacity is not the bottleneck either.

4. **Adapter quality:** Individual adapter PPL (10.119) is slightly WORSE
   than base PPL (10.057) on average. The adapters themselves may not
   provide meaningful specialization, making routing moot.

## What Would Kill This (Already Killed)

K587 FAIL: 39.4% < 60%.
K588 FAIL: Routed PPL 10.107 > uniform 10.080.

## Structural Insight for Future Work

The routing problem at N=24 is NOT an architecture problem. It is a
**representation quality problem.** No routing head architecture (binary,
softmax, energy-based) will achieve >60% accuracy if the hidden state
representation does not separate 24 domains. The path forward requires
either:

1. **Better features:** Use per-token routing (not mean-pooled), use
   intermediate layer hidden states, or train a dedicated encoder.
2. **Fewer effective classes:** Cluster the 24 domains into 6-8 macro-groups
   that ARE separable, then route at the group level.
3. **Rethink the routing target:** Route based on adapter OUTPUT quality
   (which adapter reduces loss most) rather than input similarity.
4. **Accept the PPL reality:** Individual adapters average 10.119 PPL vs
   base 10.057 PPL. The adapters barely specialize. Routing cannot extract
   signal that the adapters do not provide.
