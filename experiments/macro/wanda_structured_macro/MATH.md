# Wanda-Style Structured Pruning at Macro Scale: Mathematical Foundations

## 1. Problem Statement

Given a pretrained SwiGLU-based language model (Qwen2.5-0.5B) where activation-only
pruning is 8.9x WORSE than random, test whether incorporating weight norms
(Wanda-style scoring) corrects the specialist neuron problem.

### 1.1 Notation

```
L         -- number of transformer layers (24 in Qwen2.5-0.5B)
d         -- embedding dimension (896)
d_ff      -- intermediate MLP dimension (4864)

W_gate^l in R^{d_ff x d}  -- gate projection (SiLU-activated) at layer l
W_up^l   in R^{d_ff x d}  -- up projection (linear) at layer l
W_down^l in R^{d x d_ff}  -- down projection at layer l

g_j^l(x) = SiLU(e_j^T W_gate^l x)     -- gate output for neuron j at layer l
u_j^l(x) = e_j^T W_up^l x              -- up output for neuron j at layer l
h_j^l(x) = g_j^l(x) * u_j^l(x)        -- gate product for neuron j

D = {x_1, ..., x_M}  -- calibration dataset (M hidden-state vectors)
```

---

## 2. Scoring Methods Compared

### 2.1 Activation-Only (Parent Experiment Baseline)

```
S_act(j, l) = (1/M) sum_m |h_j^l(x_m)|
```

This is the mean absolute gate product. The parent experiment showed this is
an ANTI-signal for prunability: neurons with low S_act are specialists with
high max/mean ratios. Pruning by ascending S_act is 8.9x worse than random.

### 2.2 Weight-Only

```
S_wt(j, l) = ||[W_gate^l[j,:] ; W_up^l[j,:]]||_2
           = sqrt(||W_gate^l[j,:]||_2^2 + ||W_up^l[j,:]||_2^2)
```

The L2 norm of the concatenated gate and up projection weight rows for neuron j.
This captures the "capacity" of the neuron but ignores whether it is actually used.

### 2.3 Wanda (Weight AND Activation)

```
S_wanda(j, l) = S_wt(j, l) * S_act(j, l)
              = ||[W_gate^l[j,:] ; W_up^l[j,:]]||_2 * (1/M) sum_m |h_j^l(x_m)|
```

The key idea from Sun et al. (2023): importance = weight magnitude * activation
magnitude. This should correct the specialist problem because:

- Specialist neuron: low S_act, high S_wt -> S_wanda = moderate (KEEP)
- Truly inactive neuron: low S_act, low S_wt -> S_wanda = low (PRUNE)
- Active neuron: high S_act, high S_wt -> S_wanda = high (KEEP)

### 2.4 Adaptation for Structured Pruning

Original Wanda (Sun et al., 2023) uses per-weight scoring for UNSTRUCTURED
pruning (zeroing individual weights). We adapt to STRUCTURED pruning (zeroing
entire neuron rows in gate_proj and up_proj):

- Original: score_{i,j} = |W_{i,j}| * ||X_j|| for weight at position (i,j)
- Structured: score_j = ||W_j||_2 * mean|X_j| for entire neuron j

The structured variant replaces per-weight magnitude with per-neuron weight
norm and per-input activation with mean activation across calibration data.

---

## 3. Hypothesis

### 3.1 The Specialist Correction Hypothesis

Parent experiment established:
- S_act is anti-correlated with prunability (selects specialists)
- Specialists have: low mean activation, high max activation, high weight norms

Hypothesis: multiplying by weight norm should correct this because:
```
S_wanda(specialist) = HIGH_weight * LOW_activation = MODERATE -> KEEP
S_wanda(inactive)   = LOW_weight * LOW_activation = VERY_LOW -> PRUNE
```

### 3.2 When This Fails

The correction only works if:
1. Weight norms are sufficiently variable across neurons (not approximately uniform)
2. Weight norms are negatively correlated with activation magnitude for specialists
3. The product creates a ranking that separates truly-inactive from specialist neurons

If weight norms are approximately uniform (low variance), then:
```
S_wanda(j) ≈ C * S_act(j)   where C ≈ mean(S_wt)
```
and Wanda scoring degenerates to activation-only scoring with a constant scale factor.

---

## 4. Empirical Results

### 4.1 Weight Norm Distribution

Weight norms across 116,736 neurons are approximately uniform:
```
Mean:   0.86
Std:    ~0.05 (coefficient of variation ~6%)
Range:  0.48 to 8.00
```

The coefficient of variation is only ~6%, meaning weight norms add very little
discriminative power beyond activation magnitude.

### 4.2 Spearman Rank Correlations

| Pair | rho |
|------|-----|
| activation vs wanda | 0.974 |
| weight vs wanda | 0.395 |
| activation vs weight | 0.207 |

**Critical finding**: rho(activation, wanda) = 0.974. The Wanda ranking is
97.4% correlated with the activation-only ranking. Weight norms are too uniform
to substantially re-order the scoring, confirming the degenerate case from
Section 3.2.

### 4.3 Pruning Results (18,420 neurons = 15.8%)

| Method | PPL | Delta vs Baseline | vs Random |
|--------|-----|-------------------|-----------|
| Baseline (no pruning) | 21.33 | -- | -- |
| Random (mean, 3 seeds) | 61.81 +/- 8.36 | +189.8% | 1.0x |
| Wanda (W*A) | 376.57 | +1665.6% | 6.1x WORSE |
| Activation only | 551.69 | +2486.7% | 8.9x WORSE |
| Weight only | 2179.84 | +10120.4% | 35.3x WORSE |

### 4.4 Wanda vs Random: The Ranking

Wanda scoring improvement over activation-only:
```
Activation-only: 8.9x worse than random
Wanda: 6.1x worse than random
Improvement: 8.9/6.1 = 1.46x better than activation-only
```

But still 6.1x WORSE than random. The weight norm correction provides only
a 32% reduction in ppl elevation (from 530 to 355 above baseline) -- far
short of the >2x improvement over random required by kill criterion 1.

### 4.5 Calibration Sweep

| Cal. Samples | PPL |
|-------------|-----|
| 8 | 264.22 |
| 16 | 270.53 |
| 32 | 288.69 |
| 64 | 240.40 |
| 128 | 376.57 |

Non-monotonic: more samples actually INCREASES ppl. This is because more
calibration data provides a more accurate mean activation estimate, which
for specialist neurons moves the mean LOWER (closer to true low mean),
making Wanda MORE likely to prune them. The specialist problem is
fundamental, not a calibration issue.

---

## 5. Root Cause Analysis

### 5.1 Why Wanda Fails at Structured SwiGLU Pruning

Three factors compound:

**Factor 1: Weight norm uniformity.**
In Qwen2.5-0.5B, weight norms have CV ~6%. The product S_wanda ~ S_wt * S_act
is dominated by S_act when S_wt is approximately constant. Empirically confirmed:
Spearman rho(S_act, S_wanda) = 0.974.

**Factor 2: Structured vs unstructured.**
Original Wanda was designed for UNSTRUCTURED pruning (zeroing individual weights).
In unstructured pruning, the per-weight score |W_{i,j}| varies enormously within
a neuron, allowing fine-grained selection. In structured pruning, the per-neuron
L2 norm averages out this variation, losing discriminative power.

**Factor 3: SwiGLU gate-product correlation.**
In SwiGLU, the gate product h_j = SiLU(gate_j) * up_j combines two projections.
Neurons with low MEAN gate product but high MAX are specialists by definition.
Neither weight norm nor mean activation captures the MAX/MEAN ratio, which is
the actual prunability signal. A neuron with max/mean ratio of 1965x (observed
in parent experiment) will always have low mean, regardless of weight norm.

### 5.2 Why ALL Scoring Methods Lose to Random

The fundamental problem is that ALL deterministic scoring methods (activation,
weight, Wanda) create a CORRELATED set of pruned neurons -- they systematically
target the same region of the neuron landscape. Random pruning distributes
removals uniformly across the entire network, avoiding systematic depletion
of any functional cluster.

At 15.8% pruning with structured removal, the model can tolerate distributed
neuron loss (random) but cannot tolerate targeted removal of its lowest-scoring
neurons, because those neurons disproportionately include rare-but-critical
specialists.

---

## 6. Assumptions and Their Status

| # | Assumption | Status |
|---|-----------|--------|
| 1 | Weight norms vary enough to correct activation ranking | **FALSIFIED** (CV ~6%, rho = 0.974) |
| 2 | Wanda product separates specialists from inactive neurons | **FALSIFIED** (6.1x worse than random) |
| 3 | Structured Wanda captures per-neuron importance | **FALSIFIED** (all structured methods lose to random) |
| 4 | Calibration stabilizes with more samples | **FALSIFIED** (non-monotonic, specialist problem is fundamental) |

---

## 7. Implications

### 7.1 What This Proves

1. Weight norm uniformity in Qwen2.5-0.5B makes Wanda scoring degenerate to
   activation-only scoring (97.4% correlated rankings)

2. The structured pruning adaptation of Wanda does not work for SwiGLU models
   because per-neuron weight norms lack the granularity of per-weight magnitudes

3. ALL mean-based importance scoring for structured SwiGLU pruning is dominated
   by random pruning, suggesting that MEAN statistics fundamentally cannot capture
   the specialist neuron pattern

### 7.2 What Might Work Instead

1. **MAX-based scoring**: score_j = max_over_calibration |h_j(x)| instead of mean.
   Specialists have high max, so max-based scoring would rank them correctly.

2. **Variance-based scoring**: score_j = var(h_j(x)). Specialists have high
   variance (mostly zero, occasionally large). Truly inactive neurons have low
   variance.

3. **Unstructured Wanda**: the original per-weight Wanda formulation, which has
   much higher weight-magnitude variance within neurons and has been validated
   at scale in the original paper.

4. **Post-pruning recovery**: any scoring + brief fine-tuning to redistribute
   information from pruned neurons.
