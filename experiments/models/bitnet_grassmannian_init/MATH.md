# Grassmannian AP Init + Ternary QAT: Mathematical Foundations

## Setup

Let d = 64 (embedding dimension), r = 4 (LoRA rank), N = 5 (number of experts),
L = 2 (transformer layers).

### Grassmannian Packing Regime

Total subspace dimension: Nr = 5 * 4 = 20.
Ambient dimension: d = 64.

Since Nr = 20 < d = 64, we are in the **under-packed regime**. The Welch bound is:

    mu_welch = sqrt(r * (Nr - d) / (d * (Nr - r))) = 0  (when Nr <= d)

This means N = 5 perfectly orthogonal rank-4 subspaces fit inside R^64 with
zero mutual coherence. AP packing is unnecessary for geometric separation at
this N/d ratio. The experiment therefore tests a different mechanism: whether
**orthonormal initialization** (a side-effect of AP) helps ternary QAT
compared to Gaussian initialization.

### Ternary QAT with STE

During training, latent FP32 weights W are quantized to ternary in the forward pass:

    W_q = clip(round(W / alpha), -1, 1) * alpha
    alpha = mean(|W|)

The Straight-Through Estimator passes gradients through as if W_q = W:

    dL/dW_latent = dL/dW_q   (STE approximation)

After training, the final delta for expert i on weight matrix k is:

    Delta_i^k = Q(A_i^k) @ Q(B_i^k) * (alpha / r)

where Q(.) denotes ternary quantization of the trained latent weight.

### Initialization Conditions

**AP-init:** A matrices initialized as orthonormal frames from AP skeleton.
For weight matrices where d_in = d, A is the frame directly (d, r).
For d_in != d (e.g., FFN up-projection d_ff = 4d), a deterministic projection
maps the frame into the higher-dimensional space, followed by QR orthonormalization.

**Random-init:** A matrices from Gaussian N(0, 0.01^2). Not orthonormalized.

Both conditions: B matrices initialized to zero (standard LoRA).

### Why Orthonormal Init Might Help Ternary QAT

When A is orthonormal, the columns of A have unit norm and are mutually
perpendicular. Under ternary quantization:

    Q(A_ortho) preserves the approximate directional structure because
    the columns are already well-separated in direction space.

When A is Gaussian with scale 0.01, the columns have small random norms
and weaker directional separation. After ternary quantization, many entries
collapse to 0, and the surviving {-1, +1} entries depend on the magnitude
distribution relative to alpha = mean(|A|).

Orthonormal A has uniform entry magnitudes ~1/sqrt(d), so alpha ~ 1/sqrt(d)
and the quantization threshold is consistent. Gaussian A at scale 0.01 has
alpha ~ 0.008, and entries near the threshold create more quantization noise.

### Pairwise Cosine of Composed Deltas

For experts i, j with deltas Delta_i = sum_k vec(Delta_i^k), the cosine is:

    cos(i,j) = <Delta_i, Delta_j> / (||Delta_i|| * ||Delta_j||)

We measure |cos(i,j)| over all (N choose 2) = 10 pairs.

## Complexity

- AP skeleton computation: O(N^2 * r^2 * d) per iteration, 500 iterations.
  At N=5, r=4, d=64: negligible (< 1s).
- Training: O(epochs * batches * (d^2 * L)) per expert, per condition.
  30 epochs, ~6 batches, 2 layers: ~360 forward-backward passes per expert.
  10 experts total (5 AP + 5 random) * 360 = 3600 passes.
- Total: ~5 minutes on Apple Silicon CPU.

## Assumptions

1. Ternary QAT with STE converges to meaningful experts at d=64 (validated
   by bitnet_ternary_adapter_composition).
2. The AP skeleton's orthonormality is the dominant mechanism at Nr < d
   (not packing geometry, which is trivial in this regime).
3. Pairwise cosine of flattened weight deltas is a meaningful proxy for
   functional interference (consistent with prior SOLE experiments).
4. 30 epochs is sufficient for convergence (consistent with reference).

## Worked Example

At d=64, r=4, N=5:
- AP frame 0: (64, 4) orthonormal matrix, columns span a 4-dim subspace.
- AP frame 1: (64, 4) orthonormal matrix, orthogonal to frame 0.
  (Since Nr=20 < d=64, all 5 frames can be mutually orthogonal.)
- After QAT: trained B matrix B_0 (4, d_out) captures domain-specific features.
  Delta = Q(A_0) @ Q(B_0). The A_0 subspace constrains which directions the
  delta can occupy. Two experts with orthogonal A matrices produce deltas in
  orthogonal subspaces IF B matrices don't create overlap -- but ternary
  quantization of B can introduce cross-subspace leakage.
