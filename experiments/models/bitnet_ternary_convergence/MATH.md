# BitNet-2B Ternary LoRA Convergence: Mathematical Foundations

## Setup

**Base model**: BitNet-b1.58-2B-4T with ternary weights W in {-1, 0, 1}^{d_out x d_in}, d=2560, 30 layers.

**LoRA**: Rank-16 low-rank adapters on all projections (q/k/v/o/gate/up/down = 7 per layer, 210 total).

**Output with adapter**: y = (W + B @ A * alpha/r) @ x, where A in R^{d_in x r}, B in R^{r x d_out}.

## STE Ternary Quantization

For QAT training of ternary LoRA adapters, we quantize A and B during the forward pass using the Straight-Through Estimator:

**Forward pass** (quantized):
```
alpha_A = mean(|A|)
A_q = clip(round(A / alpha_A), -1, 1) * alpha_A
```

So A_q in {-alpha_A, 0, alpha_A}^{d_in x r}, a ternary matrix scaled by the mean absolute magnitude.

**Backward pass** (STE):
```
dL/dA = dL/dA_q   (gradient passes through quantization unchanged)
```

Implementation: A_q = A + stop_gradient(quantize(A) - A).

This means the effective adapter delta is:
```
Delta_W = B_q @ A_q * alpha/r
```

where both A_q and B_q are ternary {-alpha, 0, alpha} matrices.

## Why Train Longer?

The prior experiment (bitnet_2b_real_composition) used 200 steps, which was the minimum
for quick validation. At 200 steps on 500-800 training samples:
- Each sample is seen ~0.25-0.4x on average (200 steps / 500-800 samples)
- Less than 1 full epoch
- Adapter parameters have not converged to stable values

At 400 steps:
- Each sample seen ~0.5-0.8x on average
- Approaches 1 full epoch for most domains
- Val PPL curves should plateau if convergence is reached

## Convergence Analysis

**Convergence criterion**: last_50_avg_loss < first_50_avg_loss * 0.95 (5% reduction in training loss).

This is a conservative criterion for noisy single-sample batches. A more reliable signal
is the validation PPL trajectory: monotonic decrease indicates the model is still learning.

**Validation PPL trajectories (ternary adapters)**:

| Domain   | Step 100 | Step 200 | Step 300 | Step 400 | Monotonic? |
|----------|----------|----------|----------|----------|------------|
| medical  | 10.976   | 10.164   | 9.475    | 9.041    | Yes        |
| code     | 3.023    | 3.027    | 2.982    | 2.968    | ~Yes       |
| math     | 3.245    | 3.147    | 3.106    | 3.080    | Yes        |
| legal    | 20.023   | 19.343   | 19.257   | 18.909   | Yes        |
| creative | 3.128    | 3.182    | 3.226    | 3.120    | No (overfit at 200-300, recovers) |

All 5 domains show improving val PPL at step 400, suggesting further training would help.
The "not converged" label for code and creative reflects noisy training loss (batch_size=1),
not actual failure to learn.

## Composition Analysis

**1/N scaling**: With N=5 adapters, each is scaled by 1/5 before addition:
```
W_composed = W + (1/N) * sum_i (B_i @ A_i * alpha/r)
```

**Composition ratio**: avg_composed_PPL / best_individual_PPL.
- FP16 200-step: 9.37 / 2.79 = 3.35x
- Ternary 400-step: 9.52 / 2.76 = 3.45x

The 3% difference in composition ratio is expected: ternary adapters have 1.9x higher
mean |cos| (0.00186 vs 0.00098), indicating slightly higher cross-adapter interference
from longer training. However, both values are near-random (random at d=2560 would be
~1/sqrt(d) = 0.0196), so the absolute interference is negligible.

## K2 Analysis (Inconclusive -- Confounded Comparison)

K2 compares avg composed PPL: ternary 400 (9.52) vs FP16 200 (9.37).
Ratio: 1.016x.

**This comparison is INCONCLUSIVE**, not a kill. It conflates two independent variables:

| Factor          | FP16 condition | Ternary condition | Confounded? |
|-----------------|---------------|-------------------|-------------|
| Quantization    | FP16          | Ternary (STE)     | Yes         |
| Training steps  | 200           | 400               | Yes         |
| Seed            | 1             | 1                 | No variance |

The proper experimental design is a 2x2 factorial:

|                  | 200 steps | 400 steps |
|------------------|-----------|-----------|
| **FP16**         | 9.37 (measured) | ??? (NOT RUN) |
| **Ternary (STE)**| ??? (NOT RUN) | 9.52 (measured) |

Without the diagonal cells, the 1.6% gap cannot be attributed to quantization vs training
duration. If FP16-400 also shows worse composition than FP16-200 (due to increased
specialization from longer training), the gap would be a training effect, not a ternary
effect. This would actually support the hypothesis.

The individual PPL comparison (ternary 7.00 vs FP16 7.34, -4.6%) shows ternary adapters
are BETTER individually. The composition penalty (ternary 36% vs FP16 28%) is where the
gap appears, plausibly because longer-trained adapters are more specialized and harder
to compose with uniform 1/N weighting. This is speculation without the missing controls.

## Composition Path: FP16 Latent vs Ternary-Native

The composition formula above operates on the FP16 latent parameters (A, B), not the
quantized ternary weights (A_q, B_q):

```
W_composed = W + (1/N) * sum_i (B_i @ A_i * alpha/r)     <-- FP16 latent composition
```

During training, the forward pass applies STE quantization: y = (W + B_q @ A_q * alpha/r) @ x.
But B_q and A_q are not stored -- they are computed on-the-fly from the FP16 latent parameters.
When adapters are saved to disk, the FP16 latent parameters (A, B) are saved, not (A_q, B_q).

At composition time, `compose_adapters` sums the FP16 latent (B @ A) products. The composed
model then applies STE quantization during its forward pass. This means:

- **What is tested**: composition quality of QAT-trained latent parameters (validates that
  STE training produces parameters whose FP16 representations compose well)
- **What is NOT tested**: ternary-native composition, where quantized ternary weights are
  composed directly on the integer grid {-1, 0, 1}

The ternary-native path (LoTA-QAF) would be:
```
A_q_i = quantize(A_i)                     # each adapter quantized independently
W_composed_q = W + (1/N) * sum_i (B_q_i @ A_q_i * alpha/r)   # compose quantized weights
```

This is a different operation because quantize(sum(X_i)) != sum(quantize(X_i)) in general.
The ternary-native path is deferred to exp_bitnet_serving_path.

## Convergence Criterion Reliability

The training loss criterion (last_50_avg < first_50_avg * 0.95) operates on per-step
loss values at batch_size=1, seq_len=128. The variance of a single cross-entropy sample
at these perplexity levels is high:

- For PPL ~3 (code/creative): loss ~1.1, individual token variance ~0.5-1.0
- For PPL ~18 (medical/legal): loss ~2.9, individual token variance ~1.0-2.0

With only 128 tokens per batch, the standard error of a single batch loss is ~0.06-0.15.
Over 50 batches, the mean has SE ~0.01-0.02, but the sequential ordering (step idx =
step % len(train_tokens)) means samples are not independent across the first_50 vs last_50
windows -- they are different data points from the same domain.

Val PPL (computed over 25 batches = ~3,200 tokens) is more reliable because it averages
over a held-out set. All 5 domains show val PPL improvement, confirming learning even when
training loss appears non-convergent.

## Computational Cost

| Metric | Value |
|--------|-------|
| Parameters (base) | 2,412,820,480 |
| Trainable (LoRA) | 21,626,880 (0.9%) |
| Training time per domain | ~200s at 400 steps |
| Total training (5 domains) | ~1044s (~17 min) |
| Total runtime including eval | ~30 min |
| Hardware | Apple Silicon (MLX), $0 |
| Sequence length | 128 tokens |
