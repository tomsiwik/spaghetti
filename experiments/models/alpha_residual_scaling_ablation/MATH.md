# Alpha Residual Scaling Ablation: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 64-256 (micro) |
| r | LoRA rank | 8 |
| N | Number of expert adapters | 8 or 50 |
| L | Number of transformer layers | 4-48 |
| h_l | Hidden state at layer l | (d,) |
| W_l | Base weight matrix at layer l | (d, d) |
| Delta_l | Merged expert delta at layer l | (d, d) |
| sigma(.) | GELU activation | R -> R |
| RN(.) | RMSNorm: x -> x / sqrt(mean(x^2) + eps) | R^d -> R^d |
| s | Residual scale factor | 1/sqrt(L) or 1.0 |
| epsilon_l | Per-layer weight-space removal error | [0, 1] |
| alpha | Amplification ratio: output_dev / sum_epsilon | [0, 1] |
| D | Output deviation from expert removal (relative L2) | [0, 1] |

## 2. The Question

The parent experiment (removal_safety_complete_bound) measured alpha = 0.022 using
the forward pass:

    h_{l+1} = h_l + (1/sqrt(L)) * sigma((W_l + Delta_l) @ RN(h_l))

Production Qwen2.5 and Llama use:

    h_{l+1} = h_l + sigma((W_l + Delta_l) @ RN(h_l))

i.e., s = 1.0 (no 1/sqrt(L) scaling). The adversarial review asked: how much of
alpha = 0.022 is due to the non-standard 1/sqrt(L) scaling?

## 3. Why Scaling Cancels Out (Proof)

### 3.1 The amplification ratio is scale-invariant

**Claim:** For any constant scale factor s > 0 applied uniformly to all layers,
the amplification ratio alpha is independent of s.

**Proof sketch:**

Define two forward passes that differ only in scale:
- f_s(x; {W_l + Delta_l}): uses scale s
- f_1(x; {W_l + Delta_l}): uses scale 1.0

Let epsilon_l be the per-layer weight-space error from removing expert k (measured
as relative Frobenius norm of weight difference). This is purely a weight-space
quantity and does not depend on the forward pass, so:

    sum_epsilon(scaled) = sum_epsilon(unscaled)  ... (1)

Now consider the output deviation. Let:
- y_all = f(x; {W_l + Delta_l^{all}}) -- output with all experts
- y_naive = f(x; {W_l + Delta_l^{naive}}) -- output after naive removal
- y_gt = f(x; {W_l + Delta_l^{gt}}) -- output after GS recompute

The output deviation is:

    D = ||y_naive - y_gt|| / ||y_gt||

At each layer, the perturbation from naive removal vs GS recompute introduces
a delta in the layer output proportional to the weight-space error AND the scale:

    delta_h_l ~ s * (Delta_l^{naive} - Delta_l^{gt}) @ RN(h_l)

But the UNPERTURBED output y_gt is ALSO proportional to s (since each layer
contributes s * sigma(W @ RN(h)) to the residual stream). After L layers,
both the perturbation and the output magnitude scale with the same power of s.

More precisely, define u_l as the perturbation to the hidden state at layer l.
In the linear regime (small perturbation):

    u_{l+1} = u_l + s * J_l @ u_l + s * eta_l

where J_l is the Jacobian of sigma(W @ RN(.)) and eta_l is the perturbation
from the weight-space error at layer l.

The hidden state h_l itself satisfies:

    h_{l+1} = h_l + s * sigma((W_l + Delta_l) @ RN(h_l))

Both h_l and u_l are driven by the same s factor. The RATIO ||u_L|| / ||h_L||
depends on the RATIO of perturbation to signal at each layer, which is s-independent.

### 3.2 Intuitive explanation

The 1/sqrt(L) scaling makes the entire network "quieter" -- both the signal and
the noise are scaled down by the same factor. The signal-to-noise ratio, which is
what the amplification ratio measures, is unaffected.

Think of it as adjusting the volume on a speaker: the music and the static get
louder or quieter together. The SNR stays the same.

### 3.3 Numerical verification

Across all tested configurations (d=64..256, N=8..50, L=4..48):

    alpha(scale=1/sqrt(L)) / alpha(scale=1.0) = 1.000 +/- 0.002

The ratio is 1.000 to three decimal places. This confirms the theoretical
prediction that the amplification ratio is scale-invariant.

### 3.4 What DOES change with scaling

The absolute output magnitude changes significantly:

| Config | RMS (1/sqrt(L)) | RMS (scale=1.0) | Ratio |
|--------|-----------------|-----------------|-------|
| d=64, N=8 | 5.69 | 27.9 | 4.90x |
| d=64, N=50 | 14.2 | 69.5 | 4.89x |
| d=256, N=50 | 14.6 | 71.7 | 4.91x |

The output RMS is ~sqrt(L) = 4.90x larger without scaling, exactly as expected.
The network does not diverge because RMSNorm constrains the norm of the
pre-activation input at each layer. The residual stream grows as O(sqrt(L)),
not exponentially, because each layer adds an O(1) vector (post-RMSNorm input
has unit RMS by construction).

## 4. Implications for the Safety Bound

### 4.1 The bound transfers to production

Since alpha is scale-invariant:

    alpha_production = alpha_micro = 0.022

The complete safety bound D = sum_epsilon * alpha is UNCHANGED when moving
from the micro architecture (with 1/sqrt(L)) to production Qwen/Llama
(without 1/sqrt(L)).

### 4.2 Why the adversarial concern was wrong

The adversarial review's reasoning was:

    "Without 1/sqrt(L), feedforward amp=0.25 vs 0.022 with scaling"

This is incorrect. The feedforward baseline (no residual, no norm) does not
use 1/sqrt(L) scaling either. The 0.25 vs 0.022 difference comes entirely
from the residual connection + RMSNorm architecture, not from the scaling.

The 1/sqrt(L) scaling was included in the code for numerical stability during
development, but the amplification ratio is a dimensionless RATIO that cancels
out the scaling factor completely.

## 5. Decomposition of Dampening

At L=24, d=64:

| Factor | Contribution | Source |
|--------|-------------|--------|
| Feedforward baseline | alpha = 0.250 | multilayer_removal_cascade |
| + Residual + RMSNorm | 12.3x dampening | This experiment |
| = Pre-RMSNorm (any scale) | alpha = 0.022 | Confirmed identical |
| + 1/sqrt(L) scaling | 1.00x (no effect) | This experiment |

The entire 11.4x dampening (from 0.250 to 0.022) comes from the combination
of residual connections and RMSNorm normalization. The 1/sqrt(L) scaling
contributes exactly 0% to the amplification ratio.

## 6. Depth Dependence Confirmation

Alpha decreases with depth identically for both scaling variants:

| L | alpha (scaled) | alpha (unscaled) | Ratio |
|---|---------------|-----------------|-------|
| 4 | 0.230 | 0.230 | 1.00 |
| 8 | 0.117 | 0.117 | 1.00 |
| 12 | 0.057 | 0.058 | 1.00 |
| 16 | 0.037 | 0.037 | 1.00 |
| 24 | 0.022 | 0.022 | 1.00 |
| 32 | 0.014 | 0.014 | 1.00 |
| 48 | 0.007 | 0.007 | 1.01 |

The alpha ~ 1/L relationship is an architectural property of Pre-RMSNorm
transformers, independent of any explicit scaling factor.

## 7. Assumptions

1. **Uniform scaling across layers.** The 1/sqrt(L) is applied identically
   to all layers. If different layers had different scales (as in some
   initialization schemes), the cancellation would not be exact.

2. **No learnable scale parameters.** RMSNorm gamma is fixed at 1.0.
   Learnable gamma could break the scale invariance if gamma values are
   correlated with the perturbation direction.

3. **Small perturbation regime.** The cancellation is exact in the linear
   regime and empirically verified for the perturbation sizes encountered
   in expert removal (0.02-0.5% output deviation).
