# Mathematical Foundations: Ternary Composition Scaling to N=15

## Setup

### Variables
| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model hidden dimension | 2560 (BitNet-2B-4T) |
| r | LoRA rank | 16 |
| N | Number of composed adapters | {5, 15} |
| L | Number of transformer layers | 30 |
| M | Number of LoRA modules per layer | 7 (q,k,v,o,gate,up,down) |
| W_base | Frozen ternary base weight | (d_out, d_in), values in {-1,0,1} |
| A_i | LoRA down-projection for adapter i | (d_in, r) |
| B_i | LoRA up-projection for adapter i | (r, d_out) |
| alpha | LoRA scaling factor | 20.0 |
| s_N | Per-adapter scaling | 1/N |

### Composition Formula

The output for input x under N-adapter composition with 1/N scaling:

```
y = W_base @ x + (1/N) * sum_{i=1}^{N} alpha * B_i @ A_i @ x
```

With STE ternary quantization during training, the effective A_i, B_i have
ternary structure in the forward pass but continuous gradients for backprop:

```
Q(W) = alpha * clip(round(W / alpha), -1, 1)   where alpha = mean(|W|)
```

## Scaling Analysis

### Composition Ratio

Define the composition ratio as:

```
rho(N) = PPL_composed(N) / PPL_best_individual
```

**Hypothesis**: rho scales sub-linearly with N because:

1. Each adapter contributes a signal of magnitude O(alpha/N) to each layer's output
2. The total adapter signal is sum of N terms, each scaled by 1/N
3. If adapters are orthogonal, the composed signal norm is O(alpha * sqrt(N) / N) = O(alpha / sqrt(N))
4. This DECREASES with N, so composition should improve, not degrade

**Counter-argument**: The composition ratio involves PPL, which is exp(loss).
Even if adapter signals shrink, the composed model diverges from any single
expert's optimum. The ratio rho(N) is expected to grow because:
- Each expert contributes only 1/N of its signal
- The composed model is a compromise between N different optima

**Prediction**: rho(15) / rho(5) should be between 1.0x and 2.0x if composition
is stable. Kill threshold at 2.0x.

### Orthogonality (Cosine Similarity)

For N adapters, the number of pairs is C(N,2) = N(N-1)/2.

| N | Pairs |
|---|-------|
| 5 | 10 |
| 15 | 105 |

The expected mean |cos| under random initialization in R^p (p = total adapter params):

```
E[|cos|] = sqrt(2/pi) * 1/sqrt(p)
```

For p = 21,626,880 (from multiseed validation):
```
E[|cos|_random] = sqrt(2/pi) / sqrt(21626880) = 0.000171
```

Observed at N=5: mean |cos| = 0.002 (12x above random, but well below 0.01).

**Scaling prediction**: Adding 10 more adapters from diverse domains should NOT
increase mean |cos| significantly because:
- New domains are orthogonal to existing ones (different data distributions)
- The parameter space is high-dimensional (21.6M dims)
- At d=2560, Grassmannian capacity N_max = d^2/r^2 = 25,600 >> 15

### Per-Domain Degradation (K3)

When going from N=5 to N=15 composition, each original adapter's contribution
decreases from 1/5 to 1/15 (3x reduction). The base model contribution is constant.

For domain d with adapter i:
```
PPL_d(N=5)  = f(W_base + (1/5) * alpha * sum_{j in orig5} B_j @ A_j)  evaluated on d's data
PPL_d(N=15) = f(W_base + (1/15) * alpha * sum_{j in all15} B_j @ A_j)  evaluated on d's data
```

The degradation comes from two sources:
1. **Dilution**: adapter i's contribution drops from alpha/5 to alpha/15
2. **Interference**: 10 new adapters add noise relative to domain d

If adapters are orthogonal, interference is minimal and degradation is dominated
by dilution. Expected degradation: PPL moves toward base model PPL.

**Upper bound on degradation**: In the worst case (all adapter signal lost),
PPL reverts to base. Since composed PPL at N=5 is already between individual
and base, the maximum degradation is bounded by:

```
max_degrad = (base_ppl - composed_5_ppl) / composed_5_ppl * 100%
```

For each domain from the multiseed seed 42 results:
- medical: (18.98 - 15.51) / 15.51 = +22.4% (max possible)
- code: (3.78 - 3.46) / 3.46 = +9.2%
- math: (4.54 - 4.18) / 4.18 = +8.6%
- legal: (26.93 - 24.77) / 24.77 = +8.7%
- creative: (3.51 - 3.30) / 3.30 = +6.4%

Only medical could exceed 10% even in the worst case. Code, math, legal,
creative are all well within the 10% threshold even under complete dilution.

## Worked Example

At N=15 with d=2560, r=16:
- Total adapter parameters: 30 layers * 7 modules * (d_in * 16 + 16 * d_out) per module
- Roughly 21.6M parameters per adapter
- 15 adapters: 324M parameters stored (but composed into single set at inference)
- Cosine pairs: C(15,2) = 105
- Expected mean |cos|: ~0.002 (empirical from N=5, should not increase much)
- Composition ratio: ~3.4 * k where k in [1.0, 2.0] (prediction)

## Computational Cost

- Training: 10 new adapters * 400 steps * ~0.3s/step = ~20 min
- Eval: 15 domains * 25 batches * ~0.2s/batch = ~1.2 min per condition
- Total eval: ~5 min (base + individual + N=5 + N=15)
- Cosines: 105 pairs * ~0.1s = ~10s
- **Total expected: ~90 min**

## Assumptions

1. Ternary STE training produces adapters of comparable quality across domains
2. HuggingFace datasets provide sufficient domain signal in 800 training samples
3. Validation on 25 samples of seq_len=128 gives stable PPL estimates
4. 400 training steps are sufficient for convergence
5. Single seed (42) results are representative (justified by multiseed CV=0.5%)
