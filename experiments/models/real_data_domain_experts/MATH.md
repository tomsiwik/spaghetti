# Real Data Domain Experts: Mathematical Foundations (v2, Revised)

## Architecture

**Base model**: BitNet-b1.58-2B-4T, ternary weights, d=2560, 30 layers, 7
projections per layer (q/k/v/o/gate/up/down).

**LoRA decomposition**: For each target projection W in R^{out x in}:

    y = W*x + (x @ A) @ ternary(B) * scale

where:
- A in R^{in x r}: frozen, QR-orthogonal initialized
- B in R^{r x out}: trainable, STE-ternary quantized in forward pass
- r = 16, scale = 20.0

## QR-Orthogonal A-Matrix Initialization

For N=5 experts and r=16 on d=2560: N*r = 80 << d = 2560.

**Welch bound**: mu_welch = sqrt(r * (Nr - d) / (d * (Nr - r)))

With Nr = 80, d = 2560: Nr < d, so mu_welch = 0 (perfect orthogonality
achievable). The Alternating Projection (AP) algorithm is included in the
codebase but short-circuits at the QR step (line 119: `if Nr <= d: return frames`).
**At this scale, AP never iterates -- the initialization is purely QR-based.**

AP becomes necessary when N*r > d, i.e., N > d/r = 2560/16 = 160 experts.
At that point the Welch bound becomes nonzero and AP iterates to minimize
inter-frame coherence.

**Skeleton construction**: For each of 7 projection types, generate N=5
orthonormal frames U_i in R^{d_in x r} via QR. Each frame serves as the
frozen A matrix for domain i.

**Orthogonality guarantee**: For frames U_i, U_j from QR:
    ||U_i^T U_j||_F = 0 (exact orthogonality when Nr <= d)

This means the interference bound simplifies to:
    ||delta_W_i^T delta_W_j|| = ||B_i A_i^T A_j B_j^T|| = 0

regardless of B-matrix correlation.

## STE Ternary Quantization

Forward pass quantizes B to ternary:
    alpha = mean(|B|)
    B_q = clip(round(B/alpha), -1, 1) * alpha

Backward pass uses STE:
    B_ste = B + stop_gradient(B_q - B)

This means: dL/dB is computed as if B were continuous, but the forward value
uses B_q. The ternary constraint reduces adapter storage from 4 bytes/param
(float32) to ~1.6 bits/param ({-1, 0, 1} + shared alpha).

## Trainable Parameters

Per adapter: **10,936,320 trainable params** (measured).
- 7 projections/layer x 30 layers = 210 LoRA pairs
- B matrix sizes vary by projection type (16 x out_features)

## Composition (v2, Corrected)

### Correct Multi-Expert Composition

The correct composition computes each expert's contribution through its own
A matrix and averages the LoRA outputs:

    y = W*x + (1/N) * sum_{i=1}^{N} [(x @ A_i) @ B_i] * scale

This preserves the subspace separation: each B_i's learned features are
projected through the A_i subspace they were trained with.

### Why Single-A Composition is Wrong

The v1 experiment used:

    y = W*x + (x @ A_0) @ mean(B_0, ..., B_{N-1}) * scale  [BROKEN]

Since A_i are orthogonal (A_i^T A_j = 0 for i != j), projecting x through A_0
produces a representation in A_0's column space. Applying B_i (i != 0) to this
representation is meaningless because B_i learned features in A_i's column
space, which is orthogonal to A_0's.

Formally, the single-A composed weight delta is:
    delta_W_broken = A_0 @ (1/N) sum_i B_i

The correct composed weight delta is:
    delta_W_correct = (1/N) sum_i A_i @ B_i

The difference:
    delta_W_correct - delta_W_broken
    = (1/N) sum_{i=1}^{N-1} A_i @ B_i - (1/N) A_0 @ sum_{i=1}^{N-1} B_i

The broken version discards (N-1)/N of the adapter capacity by projecting all
B matrices through the wrong subspace.

### Empirical Validation

The correct composition achieves -26.3% avg PPL improvement vs base, compared
to -8.0% for the broken version -- a 3.3x improvement. This confirms the
orthogonal subspace structure is functional, not just structural.

## Routing Head Architecture

Per-domain binary classifier:
    h = ReLU(W_1 @ z + b_1)    W_1 in R^{32 x 2560}, b_1 in R^{32}
    s = W_2 @ h + b_2           W_2 in R^{1 x 32}, b_2 in R^{1}

where z = mean_pool(hidden_states) in R^{2560}.

Parameters per head: 32*2560 + 32 + 32 + 1 = 81,985 (~82K).
Total for 5 heads: ~410K (0.004% of base model).

## Bootstrap PPL Confidence Intervals

Each domain's PPL is computed on 25 validation samples (~6400 tokens). To
quantify uncertainty, we use nonparametric bootstrap (1000 resamples):

1. For each resample, draw 25 samples with replacement from the validation set
2. Compute PPL from the resampled total_loss / total_tokens
3. Report 2.5th and 97.5th percentiles as 95% CI

This is the standard approach for ratio statistics (avg loss per token) where
per-sample variance is unknown.

## Complexity

Training cost per adapter: 200 forward+backward passes through 2.4B model.
Wall time: ~65s on M5 Pro (effective throughput ~3 iter/s).

Evaluation cost (revised): 4 model loads x 5 domains x 25 samples each = 80s.
The correct composition uses `MultiAdapterLoRALinear` which computes N separate
A_i @ B_i projections per forward pass, costing N times the single-adapter
LoRA overhead. At N=5 this is negligible vs the base model cost.

Total experiment: training (from disk) + eval (80s) + orthogonality + routing (15s)
= 95s = 1.6min (eval only; full training run was 6.8min).

Peak memory: 7.1GB eval, 17.1GB training. Well within M5 Pro 48GB.
