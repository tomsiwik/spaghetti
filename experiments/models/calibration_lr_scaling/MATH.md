# Calibration LR Scaling with N: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Definition |
|--------|-----------|------------|
| N | scalar | Number of experts (N=2,4,8,16 tested) |
| k | scalar | Top-k selection count (k=2 fixed) |
| LR_base | scalar | Base calibration learning rate (3e-3) |
| LR_mult | scalar | LR multiplier relative to base |
| S | scalar | Number of calibration steps |
| L_joint | scalar | Joint training validation loss (target) |
| L_comp(N, LR, S) | scalar | Composition validation loss at config |
| g_N | scalar | Mean router gradient magnitude at N experts |
| cos | scalar | Mean pairwise cosine similarity between experts |

## Hypothesis

The discriminability experiment (parent) measured gradient attenuation:

```
g_N / g_2 ~ k/N
```

At N=8, k=2: g_8 ~ 0.25 * g_2 (confirmed empirically as 3.6-7.1x attenuation).

If gradient magnitude is the calibration bottleneck, Adam's effective update is:

```
delta_theta ~ LR * g / (sqrt(v) + eps)
```

where v is the second moment (accumulated gradient variance). For small v (early training), delta_theta ~ LR * g. To restore the same parameter update magnitude at N=8 as N=2:

```
LR_8 * g_8 = LR_2 * g_2
LR_8 = LR_2 * (g_2 / g_8) = LR_2 * (N/k)
```

**Predicted scaling law:**
```
LR_opt(N) = LR_base * (N/k)
Steps_opt(N) = S_base * (N/k)
```

## What Actually Happened

### Full Grid (3 seeds, mean vs-joint percentage)

| N  | LR*0.5 | LR*1.0 | LR*2.0 | LR*4.0 | LR*8.0 |
|----|--------|--------|--------|--------|--------|
| 2  | +0.78% | +0.70% | +1.02% | +5.09% | +11.97% |
| 4  | +0.28% | +0.74% | +0.51% | +1.31% | +3.72% |
| 8  | +0.62% | +0.56% | +0.99% | +0.87% | +4.78% |
| 16 | +0.76% | +0.57% | +0.73% | +3.30% | +1.72% |

### Key Observation: No Quality Gap to Close

At cos=0.0 (maximally discriminable experts, practical regime):

| N | Best quality (vs joint) | Best LR multiplier |
|---|------------------------:|--------------------:|
| 2 | +0.70% | 1.0x |
| 4 | +0.28% | 0.5x |
| 8 | +0.56% | 1.0x |
| 16 | +0.57% | 1.0x |

The quality difference across all N values is only 0.42pp (0.28% to 0.70%).
This is within noise for 3 seeds. There is NO significant quality degradation
as N increases from 2 to 16.

### Why the Hypothesis Was Wrong

The hypothesis assumed:
1. Gradient attenuation (5-7x at N=8) bottlenecks calibration quality
2. LR scaling compensates by restoring update magnitude

Both assumptions are incorrect:

**Adam's adaptive learning rate already compensates.** Adam maintains per-parameter
second moments v_t. When gradients are smaller (higher N), v_t decreases proportionally,
making the effective update LR/sqrt(v+eps) roughly invariant to gradient magnitude.
The adaptive denominator acts as an automatic gradient normalizer.

Formally:
```
Update = LR * m_t / (sqrt(v_t) + eps)
```

If g_N = alpha * g_2 (where alpha = k/N < 1):
```
m_t(N) ~ alpha * m_t(2)           [first moment scales linearly]
v_t(N) ~ alpha^2 * v_t(2)         [second moment scales quadratically]
sqrt(v_t(N)) ~ alpha * sqrt(v_t(2))
```

Therefore:
```
Update(N) = LR * alpha * m / (alpha * sqrt(v) + eps) ~ LR * m / (sqrt(v) + eps/alpha)
```

For eps << sqrt(v) (which holds after the first few steps):
```
Update(N) ~ Update(2)
```

Adam's normalization cancels the gradient attenuation. LR scaling is unnecessary.

**The quality gap is in the model, not the optimization.** At cos=0.0, all experts
are maximally discriminable. The router can easily distinguish them regardless of
gradient magnitude. The ~0.5-1% gap vs joint training comes from the composition
architecture itself (routing overhead, limited expert coverage), not from
insufficient calibration.

### Convergence Speed

All N values converge within 100 steps at default LR:

| N | Steps to within 0.5% of final | Predicted (N/k scaling) |
|---|-------------------------------:|------------------------:|
| 2 | 100 | 300 |
| 4 | 100 | 600 |
| 8 | 100 | 1200 |
| 16 | 100 | 2400 |

The predicted step scaling (N/k) is wildly wrong. 100 steps suffice for all N.

### Scaling Law Fit

Log-log regression: LR_opt = 0.76 * (N/k)^0.10

The exponent b=0.10 is effectively zero, confirming no LR scaling is needed.
r^2 = 0.067, meaning N explains only 6.7% of variance in optimal LR.

### Practical Scaling Law (Null Result)

The actual scaling law for the contribution protocol is trivially simple:

```
LR_opt(N) = LR_base       (constant, independent of N)
Steps_opt(N) = 100         (constant, independent of N)
```

No scaling needed. Adam handles it.

## Computational Cost

| Approach | Cost per step | Total budget | Quality |
|----------|--------------|-------------|---------|
| Dense backprop (killed) | O(N * d^2) | 4x baseline | +0.5pp |
| LR scaling (this, null) | O(k * d^2) | 1x baseline | no improvement |
| Default (100 steps, base LR) | O(k * d^2) | 1x baseline | same quality |

## Assumptions

1. **cos=0.0 only.** This experiment tests the practical regime where experts
   are maximally discriminable. The gradient attenuation effect may matter at
   cos>0.3, but that regime never occurs with independent training.

2. **Adam optimizer.** SGD would NOT have this property. The null result is
   specific to adaptive optimizers that normalize by gradient magnitude.

3. **Micro-scale (d=64, ~200K params).** At macro scale with d=896+, the
   absolute gradient magnitudes differ but the adaptive normalization
   argument still holds.

4. **Synthetic experts.** Generated from 2 trained LoRA adapters via projection.
   Real multi-domain experts might show slightly different dynamics.
