# Activation Frequency Pruning: Research Digest

## Hypothesis

Activation frequency (firing rate) provides a better pruning signal than mean
magnitude at macro scale, because high-frequency "always-on" neurons are
redundant generalists whose removal causes graceful rather than catastrophic
degradation.

**Falsifiable.** Kill criteria:
1. Frequency-based pruning is not >2x better than random at 5% pruning fraction
2. Frequency signal correlates >0.8 (Spearman) with mean magnitude (redundant)

## What This Experiment Is

A macro-scale profiling and pruning experiment on Qwen2.5-0.5B (24 layers, 4864
SwiGLU neurons/layer, 116,736 total). Profiles activation FREQUENCY -- the fraction
of calibration positions where each neuron fires above a threshold epsilon -- and
tests whether this provides an independent or better pruning signal than the mean
magnitude signal from the parent experiment.

Builds directly on the parent experiment (swiglu_macro_pruning_transfer) which
showed mean magnitude is 8.9x worse than random pruning.

## Lineage in the Arena

```
gpt (dense baseline, ReLU)
  +-- silu_capsule (SiLU capsule MLP)
  |     +-- swiglu_gate_pruning (micro: 66.5% prunable at +1.22%)
  |           +-- swiglu_macro_pruning_transfer (macro: mean magnitude = anti-signal)
  |                 +-- activation_frequency_pruning (THIS: frequency also killed)
```

## Key References

- Parent experiment: `macro/swiglu_macro_pruning_transfer/` (mean magnitude anti-signal)
- ReDo (2024): Activation-based dead neuron detection and reinitialization
- PowerInfer (2023): Hot/cold neuron classification by activation frequency for
  GPU/CPU offloading (used frequency, but for inference scheduling not pruning)
- Wanda (Sun et al., 2023): Weight * activation magnitude for unstructured pruning

## Empirical Results

### Data Provenance

Same as parent experiment:

| Property | Calibration | Evaluation |
|----------|------------|------------|
| Dataset | WikiText-2-raw-v1 | WikiText-2-raw-v1 |
| Split | test | validation |
| Positions | 16,384 | 8,192 |

### Kill Criterion 2: Correlation Analysis

Spearman rank correlation between firing frequency f_j(eps) and mean magnitude
mu_j, computed across all 116,736 neurons:

| Epsilon | Spearman rho | |rho| | KC2 Status |
|---------|-------------|-------|------------|
| 0.001 | 0.832 | 0.832 | KILL |
| 0.005 | 0.864 | 0.864 | KILL |
| 0.010 | 0.883 | 0.883 | KILL |
| 0.020 | 0.909 | 0.909 | KILL |
| 0.050 | 0.952 | 0.952 | KILL |
| 0.100 | 0.984 | 0.984 | KILL |
| 0.200 | 0.972 | 0.972 | KILL |

**KC2 VERDICT: KILLED at ALL epsilon values.** Frequency and magnitude are strongly
monotonically related (rho > 0.83 everywhere). Neurons that fire often also have
high mean magnitude. The two signals contain essentially the same rank-ordering
information. Frequency is NOT an independent signal.

### Frequency Distribution Summary

At eps=0.01 (below median gate product of 0.078):

```
Min=0.003  P5=0.702  P25=0.807  Median=0.870  P75=0.916  P95=0.962  Max=1.000
Always-on (>99%): 1,614 (1.4%)
Never-fire (<1%):     1 (0.0%)
```

Nearly all neurons fire on >70% of positions at eps=0.01. There is very little
dynamic range in frequency at this threshold -- most neurons are "usually-on",
not bimodally split into always-on vs specialist.

### Kill Criterion 1: Pruning Quality

Baseline perplexity: 21.31

| Method | 1% | 2% | 5% | 10% | 15% |
|--------|-----|-----|------|-------|-------|
| **Freq high-first** | 107.5 | 185.8 | 1,836.7 | 30,659.8 | 178,883.2 |
| **Freq low-first** | 31.0 | 89.7 | 1,485.0 | 3,730.9 | 5,509.2 |
| **Mean mag low-first** | 25.9 | 28.4 | 44.8 | 142.1 | 468.5 |
| **Random (3-seed)** | -- | -- | 27.8 | 36.1 | 50.4 |

At 5% pruning:

| Method | PPL | Delta vs Baseline | vs Random |
|--------|-----|-------------------|-----------|
| Frequency high-first | 1,836.7 | +8,520% | 66x WORSE |
| Frequency low-first | 1,485.0 | +6,869% | 53x WORSE |
| Mean magnitude low-first | 44.8 | +110% | 1.6x WORSE |
| Random | 27.8 | +31% | baseline |

**KC1 VERDICT: KILLED.** Frequency high-first is 66x worse than random (need >2x
better). Even frequency low-first is 53x worse. Both frequency directions are
catastrophically worse than random AND worse than the parent's mean magnitude signal.

### Why Frequency High-First is the Worst Signal

The pruned neurons at 5% (high-first) have average mean magnitude of **0.48** --
nearly 5x the median neuron magnitude of 0.078. High-frequency neurons are also
the highest-magnitude neurons. Pruning them removes the model's most active and
important computation. This is strictly worse than any other signal tested.

| Direction | Avg Freq of Pruned | Avg Mean Mag of Pruned |
|-----------|-------------------|----------------------|
| High-first (1%) | 0.997 | 1.129 |
| High-first (5%) | 0.981 | 0.481 |
| Low-first (1%) | 0.560 | 0.033 |
| Low-first (5%) | 0.641 | 0.036 |

High-frequency neurons are the backbone of the model. Low-frequency neurons
are low-magnitude (specialists). The hypothesis that always-on neurons are
redundant is falsified: they are essential.

## The Core Finding

**Activation frequency is redundant with mean magnitude.** At macro scale in
Qwen2.5-0.5B, the rank ordering of neurons by "how often they fire" is nearly
identical to the ordering by "how strongly they fire on average" (Spearman
rho = 0.88 at eps=0.01, up to 0.98 at eps=0.1).

This means:
1. There are very few "specialist" neurons in the true sense (fire rarely but
   strongly). Most neurons that fire strongly do so consistently.
2. There are very few "always-on but weak" neurons. Neurons that fire often
   also tend to have moderate-to-high magnitude.
3. The activation distribution of individual neurons is NOT bimodal at the
   position level -- most neurons fire at roughly consistent rates.

The four-way ranking of signals from best to worst:

```
Random > Mean-mag-low-first >> Freq-low-first > Freq-high-first
```

This confirms and extends the parent finding: at macro scale without auxiliary
loss, ALL activation-based structured pruning signals fail. The issue is
fundamental to zero-shot structured pruning, not specific to the scoring signal.

## Comparison with Parent Experiment

| Finding | Parent (Mean Magnitude) | This (Frequency) |
|---------|------------------------|-------------------|
| Signal vs random at 5% | 8.9x worse | 66x worse (high-first) |
| Best activation signal | Low mean (44.8 ppl) | Low mean still best |
| Independent information? | N/A | No (rho > 0.83) |
| Specialist neurons? | Yes (low mean = specialist) | Confirmed (low freq = low mag) |

The frequency experiment adds strong evidence that the specialist/generalist
distinction is NOT the axis along which safe pruning operates. Both firing
frequency and mean magnitude track the same underlying property: how much
total computational work a neuron does. No activation statistic successfully
identifies "safely prunable" neurons.

## Micro-Scale Limitations

1. **Single model**: Qwen2.5-0.5B only. Larger or MoE models may differ.
2. **Single dataset**: WikiText-2 calibration. Code/math-heavy data might
   reveal more specialist neurons.
3. **Zero-shot only**: Recovery training not tested. The signal might be
   useful for pruning-then-fine-tuning approaches.
4. **SwiGLU only**: ReLU-based models may have more dead neurons with
   different frequency/magnitude relationships.

## What Would Kill This

Both kill criteria are triggered. The hypothesis is dead.

**What was learned:**
- Frequency is NOT independent of magnitude (rho > 0.83 at all epsilons)
- High-frequency neurons are the model's backbone, not redundant
- At macro scale, activation statistics (magnitude, frequency, max) all encode
  similar information about neuron importance
- Zero-shot structured pruning at macro scale cannot be saved by a better
  activation-based scoring function

**What this means for the research program:**
- The next signal to test is NOT another activation statistic, but a combined
  weight+activation score (Wanda-style: score = ||W_j|| * ||X_j||)
- Or: abandon zero-shot pruning entirely and test pruning + recovery training
- The fundamental limitation is that production models without auxiliary loss
  have no redundant structured components at the neuron level
