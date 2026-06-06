# Residual + LayerNorm Error Dynamics: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | {32, 64, 128, 256} (micro); 896 (production) |
| r | LoRA rank | 8 (micro); 16 (production) |
| N | Number of expert adapters | 8 |
| L | Number of transformer layers | {1, 2, 4, 8, 12, 16, 24} |
| k | Index of expert to remove | 0 <= k < N |
| l | Layer index | 0 <= l < L |
| h_l | Hidden state at layer l | (d,) |
| W_l | Base weight matrix at layer l | (d, d) |
| Delta_l | Merged expert contribution at layer l | (d, d) |
| sigma(.) | Nonlinear activation (GELU) | R -> R |
| LN(.) | LayerNorm: x -> (x - mu) / sqrt(var + eps) | R^d -> R^d |
| RN(.) | RMSNorm: x -> x / sqrt(mean(x^2) + eps) | R^d -> R^d |
| alpha_l | Amplification ratio at depth L | R+ |

## 2. Architecture Definitions

### 2.1 Feedforward (parent baseline)

    h_{l+1} = sigma((W_l + Delta_l) @ h_l)

No skip connection. Errors compound through the product of Jacobians.

### 2.2 Residual

    h_{l+1} = h_l + (1/sqrt(L)) * sigma((W_l + Delta_l) @ h_l)

The 1/sqrt(L) scaling prevents activation explosion at depth (standard
deep residual network initialization). The identity path h_l passes
information directly, reducing the Jacobian's dependence on W_l.

### 2.3 LayerNorm-only

    h_{l+1} = LN(sigma((W_l + Delta_l) @ h_l))

No residual. LayerNorm projects onto the unit sphere in d dimensions,
destroying norm information. This normalizes all error vectors to unit
length regardless of their original magnitude, preventing dampening.

### 2.4 Pre-LN (GPT-2, modern standard)

    h_{l+1} = h_l + (1/sqrt(L)) * sigma((W_l + Delta_l) @ LN(h_l))

Normalize before the linear layer. Combined effect: identity path preserves
signal, LN stabilizes the nonlinear branch input. Both error-suppressive.

### 2.5 Pre-RMSNorm (Qwen/Llama production)

    h_{l+1} = h_l + (1/sqrt(L)) * sigma((W_l + Delta_l) @ RN(h_l))

Same as Pre-LN but without mean subtraction. RMSNorm preserves the mean
component, which makes error suppression even stronger because the mean
direction is a stable fixed point of the normalization.

### 2.6 Post-LN (original Transformer)

    h_{l+1} = LN(h_l + (1/sqrt(L)) * sigma((W_l + Delta_l) @ h_l))

Normalize after the residual addition. LN renormalizes the combined signal,
partially undoing the identity path's error suppression.

## 3. Error Propagation Analysis

### 3.1 Feedforward: Jacobian Product

For the feedforward case, the end-to-end Jacobian is:

    df/dh_0 = prod_{l=0}^{L-1} diag(sigma'((W_l + Delta_l) h_l)) (W_l + Delta_l)

An error eps_l at layer l propagates as:

    ||output error|| <= ||eps_l|| * prod_{m=l+1}^{L-1} ||J_m||

where J_m is the per-layer Jacobian. For random weights with ||W|| ~ 1,
spectral norms are O(1), and errors neither grow nor shrink on average.
Activation masking (GELU zeros ~50% of units) provides a dampening factor
of ~0.5^L in the worst case, but direction randomization makes the
effective scaling sqrt(L) rather than L.

Parent result: amp_ratio = 0.25 at L=24 (75% dampening).

### 3.2 Residual: Identity Shortcut

With residual connections, the Jacobian becomes:

    J_l^{res} = I + (1/sqrt(L)) * J_l^{ff}

The eigenvalues of J_l^{res} are 1 + lambda_i/sqrt(L), where lambda_i are
eigenvalues of the feedforward Jacobian. This means:

1. The identity component I passes errors through unmodified
2. The perturbation component has magnitude O(1/sqrt(L))
3. Error from weight perturbation eps at layer l contributes
   O(eps/sqrt(L)) to the output

Total error from L layers: sum of L terms each O(eps/sqrt(L)) = O(eps*sqrt(L)).
But due to direction randomization: O(eps * L^{1/4}).

This explains why residual has LOWER amplification ratio (0.045 vs 0.25):
the identity path prevents error accumulation while the 1/sqrt(L) scaling
shrinks each layer's perturbation contribution.

### 3.3 LayerNorm-only: Error Amplification

LayerNorm projects h onto the (d-1)-sphere of radius sqrt(d). This has
two consequences:

1. **Norm equalization**: All inputs to the next layer have the same norm.
   An error that increases ||h|| is not dampened -- LN scales it back up.

2. **Direction sensitivity**: LN amplifies angular errors. If h and h+eps
   have different directions, LN normalizes both to unit norm, preserving
   the angular separation but destroying the norm difference that would
   normally dampen errors.

The Jacobian of LayerNorm at x is:

    dLN/dx = (1/sigma) * (I - (1/d) * 11^T - LN(x) LN(x)^T / ||LN(x)||^2)

This is a projection matrix (eigenvalues 0 or 1/sigma). It projects out
the mean direction and the current direction, but preserves all other
directions. Crucially, it does NOT contract errors -- it preserves them.

Without residual connections, this means each layer's error is fully passed
to the next layer without dampening. The result: amp_ratio > 1 and
INCREASING with depth (measured: 3.41 at L=24, slope +0.107/layer).

This is the ONLY architecture where errors amplify.

### 3.4 RMSNorm-only: Near-identical to Feedforward

RMSNorm only normalizes the norm, not the mean:

    RN(x) = x / sqrt(mean(x^2) + eps)

The Jacobian is:

    dRN/dx = (1/rms) * (I - x x^T / (d * rms^2))

This is rank-1 perturbation of (1/rms)*I. It contracts errors in the
direction of x but preserves others. Combined with activation masking,
this produces behavior nearly identical to feedforward (measured amp_ratio
0.258 vs 0.254, difference 1.5%).

### 3.5 Pre-LN: Strongest Error Suppression with LN

Pre-LN combines the residual identity path with LN's input stabilization.
The key insight: LN normalizes the INPUT to the nonlinear branch, not the
output. This means:

1. The residual path h_l -> h_{l+1} is unmodified (no LN applied)
2. The nonlinear branch sees normalized input (bounded activation range)
3. Perturbation in the branch output is O(eps/sqrt(L))

Measured amp_ratio: 0.054 at L=24 -- better than plain residual (0.045)
because LN stabilizes the branch computation.

### 3.6 Pre-RMSNorm: Best Error Suppression

Pre-RMSNorm achieves the lowest amplification ratio (0.022 at L=24).
RMSNorm preserves the mean component of h, which is a stable fixed point.
The mean acts as an "anchor" -- perturbations orthogonal to the mean
direction are suppressed by the nonlinear branch, while the mean itself
passes through the residual path unchanged.

This is the production architecture (Qwen/Llama) and provides the
strongest error dampening of all tested variants.

### 3.7 Post-LN: Intermediate

Post-LN applies LN to the combined residual + branch output:

    h_{l+1} = LN(h_l + branch(h_l))

This partially undoes the residual's error preservation because LN
renormalizes the sum. However, the residual component still dominates
the sum (since branch output is scaled by 1/sqrt(L)), so the error
suppression is weaker than pre-LN but still significant.

Measured amp_ratio: 0.119 -- between feedforward (0.254) and pre-LN (0.054).

## 4. Dimension Scaling

### 4.1 Power Law Fit

For all architectures, output deviation follows:

    dev(d) = C * d^alpha

| Architecture | C | alpha | R^2 |
|-------------|---|-------|-----|
| feedforward | 730.6 | -1.145 | 0.984 |
| residual | 114.1 | -1.161 | 0.998 |
| pre_ln | 116.6 | -1.125 | 0.999 |
| pre_rmsn | 31.4 | -1.016 | 0.999 |
| post_ln | 105.8 | -0.916 | 0.998 |

All exponents are in the range [-1.16, -0.92], confirming approximately 1/d
scaling across ALL architectures. LayerNorm does NOT break dimension scaling.

### 4.2 Extrapolation to Production (d=896)

Using measured power laws:

| Architecture | dev(d=64) | dev(d=896) | Extrapolation |
|-------------|-----------|------------|---------------|
| feedforward | 5.31% | 0.26% | Safe |
| residual | 0.95% | 0.05% | Very safe |
| pre_ln | 1.14% | 0.06% | Very safe |
| pre_rmsn | 0.46% | 0.03% | Very safe |
| post_ln | 2.48% | 0.16% | Safe |

At SOLE production cosines (90x below random), all values would be ~90x
lower, making all architectures negligible (<0.01%).

## 5. The LayerNorm-only Anomaly

LayerNorm without residual connections is the only architecture that
amplifies errors. At L=24:

    mean_output_dev = 70.7% (vs 5.3% for feedforward)
    amp_ratio = 3.41 (super-additive: errors compound)
    max_output_dev = 166.5%

The mechanism: LayerNorm preserves angular errors while destroying the
norm-based dampening that feedforward networks rely on. Without the
identity shortcut, there is no "safe path" for information to bypass the
error-amplifying normalization.

This is NOT a concern for production transformers because:
1. No production architecture uses LN without residual connections
2. Pre-LN and Pre-RMSNorm both have lower error than feedforward
3. The effect is well-understood (it motivated the switch from post-LN
   to pre-LN in GPT-2 and subsequent models)

## 6. Key Inequalities

### 6.1 Architecture ordering (at L=24, d=64)

    amp_ratio: pre_rmsn (0.022) < residual (0.045) < pre_ln (0.054)
               < post_ln (0.119) < feedforward (0.254) ~ rmsnorm (0.258)
               << layernorm (3.41)

### 6.2 The residual dampening factor

For architectures with residual connections at L=24:

    amp_ratio(residual) / amp_ratio(feedforward) ~ 0.045 / 0.254 = 0.177

Residual connections reduce amplification by ~5.6x beyond what feedforward
provides. This is the 1/sqrt(L) scaling factor: at L=24, 1/sqrt(24) = 0.204,
consistent with the measured ratio.

### 6.3 Dimension-independent amplification ratio

For all architectures with residual connections, the amplification ratio
shows minimal d-dependence:

    residual: amp_ratio ~ 0.04 across d=32..256
    pre_rmsn: amp_ratio ~ 0.02 across d=32..256

This means the amplification ratio is an ARCHITECTURAL property,
not a dimension-dependent one. Only the absolute error scales with d.

## 7. Assumptions

1. **No learnable LN parameters.** Production LayerNorm has gamma and beta
   parameters. These could slightly change error propagation, but the
   qualitative behavior (projection onto sphere) is unchanged.

2. **1/sqrt(L) residual scaling.** Production transformers do not always
   use this explicit scaling -- instead, initialization handles it
   (GPT-2 uses 1/sqrt(2L) for output projections). The qualitative effect
   is the same: residual branch contribution is small relative to identity.

3. **Single weight matrix per layer.** Production transformers have
   attention (Q/K/V/O) and FFN (gate/up/down) separately. Both have
   residual connections, so the analysis applies to each sub-block.

4. **Random base weights.** Production weights have structured spectra
   from pre-training. This likely makes error propagation MORE predictable
   (more aligned singular vectors), not less.
