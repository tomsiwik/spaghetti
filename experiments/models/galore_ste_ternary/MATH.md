# GaLore + STE Integration: Mathematical Foundations

## Problem Statement

GaLore training produces weights W_galore that achieve good FP32 loss but degrade
severely (2.6x PPL ratio) when quantized post-hoc to ternary. Standard STE training
produces weights W_ste that are ternary-friendly (1.003x PPL ratio). The goal is
to combine both: GaLore's memory-efficient gradient projection with STE's ternary-aware
forward pass.

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| W | (m, n) | Full-precision latent weight matrix |
| W_q | (m, n) | Ternary-quantized weight: {-alpha, 0, +alpha} |
| alpha | scalar | Per-tensor quantization scale: mean(|W|) |
| G | (m, n) | Full gradient dL/dW |
| P | (m, r) | GaLore left projection (top-r left singular vectors of G) |
| G_proj | (r, n) | Projected gradient: P^T @ G |
| m_t, v_t | (r, n) | Adam moments in projected space |
| Delta_proj | (r, n) | Adam update in projected space |
| Delta | (m, n) | Full-rank update: P @ Delta_proj |
| r | scalar | GaLore rank (r << min(m,n)) |
| T_svd | scalar | SVD recomputation interval (steps) |

## Standard STE (Baseline)

Forward pass quantizes, backward passes through:

```
Forward:  y = x @ Q(W)^T  where Q(W) = clip(round(W/alpha), -1, 1) * alpha
Backward: dL/dW = dL/dQ(W)  (straight-through: treat Q as identity)
Update:   W <- W - lr * Adam(dL/dW)
```

Memory: Full Adam state on (m, n) matrices = 2 * m * n per weight matrix.

## GaLore (Prior Experiment, No STE)

Forward pass uses FP32 W directly (no quantization):

```
Forward:  y = x @ W^T
Backward: G = dL/dW  (full gradient, shape m x n)
Project:  G_proj = P^T @ G  (shape r x n, where r << m)
Update:   Delta_proj = Adam(G_proj, m_t, v_t)  (Adam in projected space)
Unproject: Delta = P @ Delta_proj  (shape m x n)
Apply:    W <- W + Delta
```

Memory: Adam state only on (r, n) matrices = 2 * r * n per weight (r/m reduction).

Post-hoc quantization: Q(W_final) degrades because W was never trained to be
quantization-friendly. The optimization landscape of FP32 loss != ternary loss.

## GaLore + STE (This Experiment)

Forward pass uses STE-quantized weights, backward gets STE gradients, then
GaLore projects those gradients:

```
Forward:  alpha = mean(|W|)
          W_q = clip(round(W/alpha), -1, 1) * alpha
          W_ste = W + stop_gradient(W_q - W)  [STE trick]
          y = x @ W_ste^T

Backward: G = dL/dW_ste = dL/dW  (STE passes gradient through)

Project:  G_proj = P^T @ G   (shape r x n)
          [P recomputed every T_svd steps via SVD of G]

Update:   m_t = beta1 * m_{t-1} + (1-beta1) * G_proj
          v_t = beta2 * v_{t-1} + (1-beta2) * G_proj^2
          Delta_proj = -lr * m_hat / (sqrt(v_hat) + eps)

Unproject: Delta = P @ Delta_proj  (shape m x n)
Apply:    W <- W + Delta
```

### Key Insight

The STE ensures that gradients G = dL/dW account for quantization effects
(the loss is computed on ternary-quantized outputs). GaLore then projects these
quantization-aware gradients to low-rank space for memory-efficient Adam updates.
The latent W remains FP32, allowing fine-grained gradient updates, but the
training signal comes from ternary forward passes.

### Memory Analysis

For a weight matrix W of shape (m, n), comparing optimizer state memory:

| Method | Optimizer State | Total per Weight |
|--------|----------------|------------------|
| Standard STE (Adam) | 2 * m * n (m, v) | 2mn |
| GaLore+STE | 2 * r * n (m, v in proj) + m * r (P) | r(2n + m) |
| Ratio | | r(2n+m) / (2mn) |

For d=256 (m=n=256), r=64:
- Standard: 2 * 256 * 256 = 131,072
- GaLore: 64 * (2*256 + 256) = 49,152
- Ratio: 0.375 (62.5% memory reduction in optimizer state)

For d=256, MLP 4*d (m=1024, n=256), r=64:
- Standard: 2 * 1024 * 256 = 524,288
- GaLore: 64 * (2*256 + 1024) = 98,304
- Ratio: 0.1875 (81.25% reduction)

Note: Weight memory itself is identical (both store FP32 latent W). The savings
are in optimizer state (Adam moments).

## SVD Recomputation

The projection P is recomputed every T_svd steps from the current gradient G:

```
U, S, V^T = SVD(G)
P = U[:, :r]   (top-r left singular vectors)
```

This aligns the projection with the current gradient structure. GaLore paper
recommends T_svd = 200 for good convergence.

**SVD on MLX:** Must use mx.cpu stream for SVD (GPU SVD not supported for all
shapes in MLX). This is a one-time cost every T_svd steps.

## Worked Example (d=64, r=16, vocab=27)

Architecture: 2-layer transformer, d=64, 2 heads, MLP=256
- Attention: wq, wk, wv, wo each (64, 64)
- MLP: fc1 (256, 64), fc2 (64, 256)
- Per layer: 4 * 64^2 + 2 * 64 * 256 = 49,152 params
- Total weight params: 2 * 49,152 = 98,304

Standard STE optimizer state: 2 * 98,304 = 196,608 floats
GaLore+STE optimizer state (r=16):
- Attention layers (64x64): 16 * (2*64 + 64) = 3,072 each, 4 * 3,072 = 12,288
- MLP layers (256x64 and 64x256):
  - fc1 (256,64): 16 * (2*64 + 256) = 6,144
  - fc2 (64,256): 16 * (2*256 + 64) = 9,216
- Per layer: 12,288 + 6,144 + 9,216 = 27,648
- Total: 2 * 27,648 = 55,296
- Ratio: 55,296 / 196,608 = 0.281 (71.9% reduction)

## Assumptions

1. STE gradients are well-approximated by their low-rank projection (GaLore
   paper shows this for standard gradients; ternary STE gradients may have
   different spectral properties)
2. SVD recomputation frequency T_svd=200 is adequate for STE gradients
3. The quantization scale alpha = mean(|W|) is stable under GaLore updates
   (latent W changes via low-rank delta, alpha adapts smoothly)

## Risk: STE Gradient Spectral Properties

Standard gradients for language modeling are known to be approximately low-rank
(GaLore paper, Fig 2). STE gradients pass through a non-differentiable
quantization step, which could:
- Break low-rank structure (STE adds quantization noise to gradients)
- Require higher rank r for good approximation
- Need more frequent SVD recomputation

Mitigation: We test rank r=64 (same as prior GaLore experiment) and can increase
if needed. The STE is a per-element operation that doesn't fundamentally change
the covariance structure of gradient matrices.
