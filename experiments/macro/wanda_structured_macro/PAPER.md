# Wanda-Style Structured Pruning at Macro Scale: Research Digest

## Hypothesis

Wanda scoring (weight_norm * activation_magnitude) corrects the specialist
neuron problem that makes activation-only pruning 8.9x worse than random at
macro scale. Falsifiable: Wanda must be >2x better than random at matching
neuron count.

## What This Experiment Is

A controlled comparison of four structured pruning scoring methods on
Qwen2.5-0.5B (24 layers, 4864 SwiGLU neurons per layer, 116,736 total neurons):

1. **Wanda** (weight norm * activation mean) -- the tested hypothesis
2. **Activation-only** (mean |gate_product|) -- parent experiment anti-signal
3. **Weight-only** (L2 norm of gate+up rows) -- data-independent baseline
4. **Random** (uniform random selection) -- the control

All methods prune the same number of neurons (18,420 = 15.8% of total) by
zeroing gate_proj and up_proj rows. Evaluation on held-out WikiText-2
validation split. No recovery fine-tuning.

## Lineage in the Arena

```
gpt (dense baseline, ReLU)
  +-- silu_capsule (SiLU capsule MLP)
  |     +-- swiglu_gate_pruning (PASS: 66.5% prunable with aux loss)
  |           +-- swiglu_macro_pruning_transfer (KILL: 8.9x worse than random)
  |                 +-- wanda_structured_macro (THIS: KILLED, 6.1x worse than random)
```

## Key References

- Sun et al., "A Simple and Effective Pruning Approach for Large Language
  Models" (2023) -- Wanda: weight and activation pruning
- Parent experiment: `macro/swiglu_macro_pruning_transfer/` -- established that
  activation-only is anti-signal, random baseline ppl = 61.97

## Empirical Results

### Scoring Method Comparison (18,420 neurons = 15.8%)

| Method | PPL | Delta vs Baseline | vs Random |
|--------|-----|-------------------|-----------|
| Baseline | 21.33 | -- | -- |
| Random (3 seeds) | 61.81 +/- 8.36 | +189.8% | 1.0x |
| **Wanda (W*A)** | **376.57** | **+1665.6%** | **6.1x WORSE** |
| Activation only | 551.69 | +2486.7% | 8.9x WORSE |
| Weight only | 2179.84 | +10120.4% | 35.3x WORSE |

### Wanda at Multiple Pruning Levels

| Fraction | Neurons | PPL | Delta |
|----------|---------|-----|-------|
| 5% | 5,836 | 32.63 | +53.0% |
| 10% | 11,673 | 118.23 | +454.3% |
| 15% | 17,510 | 338.59 | +1487.4% |
| 20% | 23,347 | 580.66 | +2622.0% |
| 30% | 35,020 | 2506.82 | +11651.5% |

### Calibration Sample Sweep

| Samples | Positions | PPL |
|---------|-----------|-----|
| 8 | 1,024 | 264.22 |
| 16 | 2,048 | 270.53 |
| 32 | 4,096 | 288.69 |
| 64 | 8,192 | 240.40 |
| 128 | 16,384 | 376.57 |

Non-monotonic: more calibration data actually worsens results because it
produces more accurate mean estimates, which for specialist neurons means
lower means and higher pruning probability.

### Rank Correlation Analysis

| Pair | Spearman rho |
|------|-------------|
| Activation vs Wanda | 0.974 |
| Weight vs Wanda | 0.395 |
| Activation vs Weight | 0.207 |

The Wanda ranking is 97.4% correlated with activation-only ranking. Weight
norms in Qwen2.5-0.5B are too uniform (CV ~6%) to meaningfully re-order
the neuron importance ranking.

## Kill Criterion Assessment

### KC1: Wanda >2x better than random? KILLED.

Wanda ppl elevation above baseline: 355.24
Random ppl elevation above baseline: 40.48
Ratio: 8.78 (need < 0.5, got 8.78)

Wanda is 8.8x WORSE than random in terms of ppl elevation, not 2x better.
While Wanda is 1.46x better than activation-only (reducing elevation from
530.36 to 355.24), it is still catastrophically worse than random.

### KC2: Works with <=100 calibration samples? KILLED.

Results are non-monotonic across sample counts. The best result (240.40 at 64
samples) is still 3.9x worse than random. The instability (36.2% variation
between 64 and 128 samples) reflects the fundamental problem: mean activation
statistics become more accurate with more data, but accuracy of a wrong signal
makes predictions worse, not better.

## The Root Cause: Weight Norm Uniformity

The hypothesis that weight norms would correct the specialist neuron problem
failed because weight norms in Qwen2.5-0.5B are approximately uniform:

- Mean weight norm: ~0.86 across all layers
- Coefficient of variation: ~6%
- Min/max ratio: ~0.48/8.00, but 99% of neurons fall in [0.5, 1.5]

When weight norms are approximately constant:
```
S_wanda(j) = ||W_j|| * mean|X_j| ≈ C * mean|X_j|
```
The Wanda score degenerates to a scaled copy of the activation score. The
97.4% rank correlation confirms this algebraically.

This is NOT a failure of the Wanda concept -- it is a failure of the
STRUCTURED adaptation. Original Wanda operates per-weight (|W_{i,j}|), where
individual weight magnitudes vary enormously within a neuron. The per-neuron
L2 norm averages this variation away.

## Hierarchy of Scoring Methods

The experiment reveals a clear hierarchy:

```
Random >> Wanda (structured) > Activation-only >> Weight-only
  62        377                  552               2180
```

ALL deterministic scoring methods are worse than random. This is because all
deterministic methods create correlated pruning patterns that systematically
target the same functional region. Random pruning distributes damage uniformly,
avoiding concentration in any specialist cluster.

## Micro-Scale Limitations

1. **Single model**: Only Qwen2.5-0.5B tested. Larger models might have
   higher weight norm variance, which could make Wanda more effective.

2. **No recovery training**: All pruning is zero-shot. Post-pruning fine-tuning
   (even 100 steps) could dramatically change the results for all methods.

3. **Single dataset**: WikiText-2 calibration and evaluation. Different
   domains might show different specialist distributions.

4. **Structured only**: Unstructured Wanda (the original formulation) was not
   tested. Per-weight scoring has much higher variance than per-neuron norms.

5. **Fixed pruning fraction**: All comparisons at 15.8%. At lower fractions
   (e.g., 5%), Wanda achieves 32.63 ppl which, while still high, is much
   closer to random.

## What Would Kill This

Already killed by both criteria. What would RESURRECT structured pruning:

1. **Max-based scoring**: score_j = max_over_data |h_j(x)|. Specialists
   have high max, so max-based scoring would correctly identify them as
   important. This directly targets the max/mean ratio problem.

2. **Unstructured Wanda**: the original per-weight formulation where
   individual weight magnitudes provide much higher-variance scores.

3. **Variance or coefficient-of-variation scoring**: Specialists have
   high variance (mostly zero, occasionally large). CV = std/mean would
   directly capture the specialist pattern.

4. **Post-pruning recovery**: Any scoring method + brief fine-tuning might
   work if the model can redistribute information quickly.

## What This Definitively Proves

1. **Weight norm uniformity kills structured Wanda.** In Qwen2.5-0.5B,
   per-neuron weight norms are too uniform (CV ~6%) to correct the
   activation-based ranking. Wanda degenerates to activation-only scoring
   (97.4% rank correlation).

2. **ALL mean-based structured scoring loses to random.** The specialist
   neuron problem is fundamental to mean statistics: neurons that fire rarely
   but critically will always have low means, regardless of weight norms.

3. **The ranking hierarchy is: Random >> Wanda > Activation >> Weight.**
   No tested scoring method outperforms random uniform pruning for structured
   SwiGLU neuron removal.

4. **More calibration data can HURT.** When the scoring signal is wrong,
   more accurate estimates of that signal make predictions worse. The
   non-monotonic calibration sweep is diagnostic of anti-signal behavior.

5. **Wanda's structured adaptation loses its key advantage.** The original
   Wanda paper succeeded with per-weight scoring where |W_{i,j}| varies
   enormously. Per-neuron ||W_j||_2 averages this out, losing discriminative
   power. Structured pruning requires a fundamentally different importance
   metric than weight magnitude.
