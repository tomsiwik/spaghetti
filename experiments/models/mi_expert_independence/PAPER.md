# MI Expert Independence: Research Digest

## Hypothesis

Mutual information (MI) between expert outputs predicts composition quality
better than cosine similarity, capturing nonlinear dependencies that cosine
misses (r-squared improvement >= 0.1).

**Result: MARGINAL PASS on predictive power (best +0.134 at layer 3),
PARTIAL KILL on cost (MI-PCA 287x, MI-act 96x).**

---

## What This Experiment Tests

Whether MI-based independence metrics are better predictors of MoE
composition quality than cosine similarity. This is a diagnostic
experiment -- the model architecture is unchanged (CapsuleMoEGPT).

Three independence metrics are compared:
1. **Cosine similarity** (baseline): Angular similarity between flattened
   group output vectors. Captures linear relationships only.
2. **MI-activation** (1D KSG): MI between mean activation scalars of each
   group pair. Uses KSG estimator on 1D inputs -- reliable at N=640.
3. **MI-PCA** (multi-D KSG): MI between PCA-reduced (d=4) group outputs.
   Higher-dimensional KSG -- less reliable, more expensive.

Protocol:
1. For each of 3 seeds, train 4 CapsuleMoEGPT variants:
   - learned top-k=1, 2, 4 routing
   - uniform routing (1/G weights)
2. Profile group outputs and activations on 640 calibration samples
3. Compute all three metrics per layer per model
4. Correlate each metric with validation loss across all 12 runs
5. Compare r-squared values

---

## Lineage in the Arena

```
gpt -> capsule_moe -> mi_expert_independence
                       (diagnostic: MI vs cosine independence metrics)
```

---

## Key References

**Kraskov, Stogbauer, Grassberger (2004)**: "Estimating mutual information."
Physical Review E. The KSG k-nearest-neighbor MI estimator used here.
Algorithm 1 implemented with scipy digamma + KDTree.

**Belghazi et al. (2018)**: "MINE: Mutual Information Neural Estimation."
ICML. Neural network-based MI estimator. Considered but rejected for this
experiment: training a separate network per group pair is overkill when
1D KSG is reliable.

**Behavioral Dedup (this project)**: Found co-activation Jaccard J=0.527
at Layer 0 between domains. MI-activation extends this by measuring
nonlinear dependence strength, not just co-occurrence.

---

## Empirical Results

### Per-Layer Metric Comparison (3-seed, 4 configs = 12 data points)

| Layer | r^2(cosine) | r^2(MI-act) | r^2(MI-PCA) | MI-act improvement |
|-------|-------------|-------------|-------------|---------------------|
| 0     | 0.001       | 0.133       | 0.129       | +0.132              |
| 1     | 0.172       | 0.112       | 0.043       | -0.060              |
| 2     | 0.074       | 0.124       | 0.008       | +0.051              |
| 3     | 0.031       | 0.165       | 0.058       | +0.134              |

### Metric Summary Statistics (Layer 0, 12 runs)

| Metric | Mean | Std | Range |
|--------|------|-----|-------|
| |cosine| | 0.625 | 0.234 | 0.212 - 0.920 |
| MI-act (nats) | 4.14 | 1.28 | 1.68 - 5.82 |
| MI-PCA (nats) | 4.72 | 1.11 | 1.91 - 6.05 |
| val_loss | 0.512 | 0.004 | 0.505 - 0.519 |

### Full Results Table

| Config | Seed | Val Loss | |Cos| L0 | MI-act L0 | MI-PCA L0 |
|--------|------|----------|----------|-----------|-----------|
| learned_k2 | 42 | 0.5112 | 0.212 | 4.251 | 5.364 |
| learned_k1 | 42 | 0.5079 | 0.358 | 3.523 | 4.720 |
| learned_k4 | 42 | 0.5048 | 0.920 | 4.673 | 5.159 |
| uniform | 42 | 0.5063 | 0.860 | 5.820 | 6.055 |
| learned_k2 | 123 | 0.5160 | 0.583 | 3.639 | 4.277 |
| learned_k1 | 123 | 0.5143 | 0.382 | 4.613 | 5.194 |
| learned_k4 | 123 | 0.5194 | 0.815 | 1.844 | 3.158 |
| uniform | 123 | 0.5125 | 0.782 | 5.881 | 5.830 |
| learned_k2 | 7 | 0.5117 | 0.482 | 4.650 | 5.003 |
| learned_k1 | 7 | 0.5114 | 0.454 | 1.683 | 1.913 |
| learned_k4 | 7 | 0.5161 | 0.735 | 5.207 | 5.269 |
| uniform | 7 | 0.5132 | 0.920 | 3.955 | 4.691 |

### Computational Cost (Layer 0, 12-run mean)

| Metric | Time (s) | Ratio vs Cosine |
|--------|----------|-----------------|
| Cosine | 0.007 | 1.0x |
| MI-act | 0.639 | 96x |
| MI-PCA | 1.912 | 288x |

---

## Kill Threshold Analysis

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Best MI r^2 improvement | +0.134 (L3) | >= 0.1 | **PASS** (marginal) |
| MI-act cost ratio | 96x | < 100x | **PASS** (marginal) |
| MI-PCA cost ratio | 288x | < 100x | **KILL** |

**Overall verdict: MARGINAL PASS for MI-activation, KILL for MI-PCA.**

MI-activation passes both kill criteria, but marginally:
- r^2 improvement of +0.134 at layer 3, just over the 0.1 threshold
- Cost ratio of 96x, just under the 100x threshold
- Layer 1 shows NEGATIVE improvement (-0.06), so the finding is inconsistent

---

## The Real Finding: Both Metrics Have Low Predictive Power

The most important observation is that ALL metrics have low r-squared
values. The best is cosine at layer 1 (r^2=0.172) and MI-act at layer 3
(r^2=0.165). Neither metric is a strong predictor of composition quality.

**Root cause: The quality range is too narrow.** Val loss spans only 0.505
to 0.519 (2.7% relative range) across all routing configurations and seeds.
The different routing strategies (k=1, k=2, k=4, uniform) all produce
similar quality at micro scale, because:
1. With only G=4 groups, even top-1 routing doesn't lose much capacity
2. The toy character-level task doesn't require specialized routing
3. 500 training steps converge all configs to similar loss

With a wider quality range, the correlation test would have more statistical
power. But at micro scale, this range is what we get.

### Why MI Sometimes Beats Cosine

At Layer 0, cosine r^2 is near zero (0.001) while MI-act r^2 is 0.133.
This is because Layer 0 groups have high cosine similarity (~0.625 mean)
with low variance across configs -- the metric saturates. MI provides more
dynamic range because it measures dependence magnitude rather than direction.

At Layer 1, cosine r^2 (0.172) beats MI-act (0.112). Here cosine has more
variance and the relationship with quality is approximately linear, where
cosine excels.

### MI Detects Nonlinear Structure

MI values are consistently high (0.9 - 5.8 nats) across all layers, even
where cosine is moderate. This confirms the hypothesis that nonlinear
dependencies exist between expert outputs. However, these dependencies do
not predict composition quality much better than cosine.

The nonlinear structure exists but is not the bottleneck for composition.
The shared character-level alphabet creates strong statistical coupling
between groups regardless of routing strategy.

---

## Micro-Scale Limitations

1. **Narrow quality range.** With only 2.7% relative variation in val_loss,
   any r^2 value is statistically fragile. A wider range (e.g., composition
   vs single-domain, N=2 vs N=8 experts) would be more informative.

2. **Small G.** With G=4 groups, only 6 pairwise comparisons per layer.
   At macro scale with G=64 or G=256, MI patterns might reveal more.

3. **Mean activation reduces to 1D.** This discards per-capsule interaction
   structure. A richer MI measure (e.g., on top-k activation indices) might
   capture more useful information.

4. **Same training data for all configs.** Each config is trained from
   scratch on the same data. In the composition protocol, experts are
   trained on different domains and composed. The independence structure
   under composition might differ from single-domain training.

5. **KSG cost is implementation-dependent.** The 96x cost ratio reflects
   Python scipy KDTree. A C++ or batched GPU implementation would be
   significantly faster. The 100x threshold may be too conservative for
   a diagnostic metric that is computed once post-training.

---

## What Would Kill This

### At Micro Scale (tested)

- **MI improvement < 0.1 at all layers**: NOT triggered. Layer 3 shows
  +0.134 and Layer 0 shows +0.132. But Layer 1 shows -0.060, so the
  finding is layer-dependent and inconsistent.

- **MI cost > 100x cosine**: TRIGGERED for MI-PCA (288x). NOT triggered
  for MI-act (96x), but margin is thin.

### At Macro Scale (untested)

- **MI provides no advantage with wider quality range**: The narrow range
  (2.7%) makes this test weak. If MI still shows no improvement with
  val_loss varying 10-50% (e.g., N=2 vs N=8 experts), the hypothesis
  is definitively killed.

- **MI cost scales worse**: KSG is O(N log N) per pair, cosine is O(N*d).
  At N=100K calibration samples, the cost ratio could worsen significantly.

- **Nonlinear dependencies are routing-invariant**: If MI between expert
  outputs doesn't change with routing quality, it measures dataset
  structure rather than composition quality.

---

## Implications for the Project

1. **MI-activation is a marginally useful diagnostic** that captures
   different information than cosine at some layers. It passes both
   kill criteria, but barely.

2. **MI-PCA is killed on cost.** The 288x ratio exceeds the 100x
   threshold. The additional dimensionality over MI-act provides no
   predictive benefit.

3. **Neither metric is a strong predictor of composition quality** at
   micro scale. This is a limitation of the narrow quality range, not
   necessarily of the metrics.

4. **Nonlinear dependencies exist** between expert outputs at all layers.
   MI values of 0.9-5.8 nats confirm substantial statistical coupling.
   This coupling comes from shared input statistics (same character
   alphabet), consistent with the Layer 0 co-activation finding from
   behavioral_dedup.

5. **Recommendation**: MI-activation is not worth adopting as a primary
   diagnostic metric at micro scale. Cosine similarity is adequate for
   the narrow quality range we observe. At macro scale with diverse
   domains and wider quality variation, MI might become more informative.
   Defer to macro validation before investing in MI-based routing decisions.
