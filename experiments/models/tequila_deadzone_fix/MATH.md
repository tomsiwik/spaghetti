# Tequila Minima Reactivation: Mathematical Foundations

## Problem Statement

Standard ternary quantization via STE maps each weight w_i to Q(w_i) in {-1, 0, +1}:

    Q(w_i) = clip(round(w_i / alpha), -1, 1)

where alpha = mean(|W|) is the per-tensor scaling factor.

Weights in the "deadzone" |w_i| < alpha/2 quantize to zero. During backpropagation with STE, the gradient passes through the quantization unchanged. But for weights deep in the deadzone (|w_i| << alpha/2), the gradient signal is too weak to push them past the threshold. These weights become trapped at zero: they contribute nothing to the forward pass and never escape.

**Empirical observation:** Our warm-start ternary training (d=512, 4L, 3000 steps) produces 31.3% zero weights. This represents frozen capacity.

## Tequila's Solution: Dynamic Adaptive Biases

### Step 1: Identify Dead Weights

Define the deadzone set:

    D = {i : |w_i| < Delta}

where Delta = alpha/2 (the rounding threshold). In practice, after quantization, D is simply the set of indices where Q(w_i) = 0.

### Step 2: Reactivation via Learnable Lambda

For each weight in D, instead of zeroing it out, Tequila adds a differentiable bias term. The reactivated contribution is:

    C(W) = sum_{i in D} lambda * w_i

where lambda is a learnable scalar (initialized to 1e-3), shared across the layer.

### Step 3: Modified Forward Pass

The standard ternary forward pass Y = X * Q(W) * alpha becomes:

    Y = X * Q(W) * alpha + C(W)

where C(W) is a bias vector (shape: [out_features]) computed by summing the lambda-scaled shadow weights for all dead indices along the input dimension.

More precisely, for a weight matrix W of shape [out_features, in_features]:

    C_j = lambda * sum_{i in D_j} w_{j,i}    for each output j

where D_j = {i : |w_{j,i}| < alpha_j / 2} is the deadzone set for output unit j (when alpha is per-tensor, D is the same for all j).

### Step 4: Modified Backward Pass (Mixed Gradient)

For weights in the deadzone, the gradient has two components:

    dL/dw_i = x_i * dL/dY    (STE path, often weak for dead weights)
            + lambda * dL/dY   (reactivation path, direct signal)

The lambda term provides a direct, continuous gradient that bypasses STE noise. This is the key insight: dead weights now receive a clean gradient signal proportional to lambda, regardless of their current value.

For lambda itself:

    dL/dlambda = sum_{i in D} w_i * dL/dY

### Step 5: Offline Fusion

After training, C(W) is input-independent (it only depends on the weights, not the activation). It can be precomputed and fused as a static bias:

    bias_j = lambda * sum_{i in D_j} w_{j,i}

This adds zero inference cost: just a bias vector per layer.

## Implementation for MLX

### BitLinear with Reactivation

```
class TequilaBitLinear:
    weight: [out, in]         # shadow weights (full precision)
    pre_quant_norm: RMSNorm   # Extra RMSNorm (proven essential)
    lam: scalar               # learnable reactivation parameter

    forward(x):
        x = pre_quant_norm(x)
        alpha = mean(|weight|)
        w_scaled = weight / alpha
        w_q = clip(round(w_scaled), -1, 1)   # ternary {-1, 0, +1}

        # Identify deadzone
        dead_mask = (w_q == 0)                # [out, in] boolean

        # STE for live weights
        w_ste = weight + stop_gradient(w_q * alpha - weight)

        # Standard ternary output
        Y = x @ w_ste.T

        # Reactivation bias for dead weights
        dead_weights = weight * dead_mask     # zero out live weights
        C = lam * dead_weights.sum(axis=1)    # [out_features]

        return Y + C
```

### Computational Cost

Let W be [m, n] (out x in), with fraction f of weights dead (f ~ 0.31).

**Forward pass additional cost:**
- dead_mask creation: O(mn) comparisons (fused with quantization)
- dead_weights sum: O(mn) multiply + reduce (but only f*mn are nonzero)
- bias addition: O(m) per token

Total: O(mn) additional, same order as the matmul itself.
In practice, negligible because the matmul dominates.

**Memory overhead:**
- One scalar lambda per BitLinear layer
- dead_mask is transient (computed in forward, not stored)
- C(W) is a single [out_features] vector

For our architecture (d=512, 4 layers, 6 BitLinear per layer = 24 layers):
- 24 additional scalars (lambda)
- At inference: 24 bias vectors of dim 512 or 2048 = ~50KB total

### Lambda Initialization

Per Tequila paper, lambda = 1e-3 is robust across wide range. We test:
- lambda = 0 (baseline, no reactivation)
- lambda = 1e-4
- lambda = 1e-3 (paper default)
- lambda = 1e-2

## Kill Criteria Thresholds

**K1: Zero fraction reduction**
- Baseline zero fraction: ~31.3% (from warm-start experiment)
- Target: < 20% (absolute, not relative)
- Mechanism: reactivation gradient gives dead weights a path to escape the deadzone. If lambda provides sufficient gradient signal, weights should drift past the quantization threshold during training.
- Kill condition: if even with lambda=1e-2, zero fraction stays above 20%

**K2: PPL preservation**
- Baseline PPL: ~360 (warm-start 10%), ~417 (cold-start with RMSNorm)
- Kill condition: PPL with reactivation > PPL without (on matched training)
- We compare: TequilaBitLinear vs BitLinear, both trained identically

## Worked Example (d=4, 2 outputs)

W = [[0.3, -0.1, 0.8, -0.5],
     [0.05, 0.7, -0.2, 0.4]]

alpha = mean(|W|) = mean(0.3, 0.1, 0.8, 0.5, 0.05, 0.7, 0.2, 0.4) = 0.381

Threshold = alpha/2 = 0.190

Deadzone (|w| < 0.190):
  w[0,1] = -0.1  -> dead
  w[1,0] = 0.05  -> dead

Q(W) = [[1, 0, 1, -1],
        [0, 1, -1, 1]]

lambda = 1e-3

C[0] = lambda * w[0,1] = 1e-3 * (-0.1) = -1e-4
C[1] = lambda * w[1,0] = 1e-3 * 0.05 = 5e-5

Forward: Y = X @ (Q(W) * alpha).T + C

Gradient for w[0,1] (dead):
  dL/dw[0,1] = x[1] * dL/dY[0] + lambda * dL/dY[0]
             = (x[1] + 1e-3) * dL/dY[0]

The lambda term ensures that even if x[1] is small, the dead weight still receives a gradient signal of magnitude lambda * |dL/dY|.

## Assumptions

1. **Deadzone identification is static per forward pass:** We recompute the deadzone mask every forward pass from the current shadow weights. This is correct because weights may enter or leave the deadzone as training progresses.

2. **Per-tensor alpha is sufficient:** We use alpha = mean(|W|) for the whole tensor. Per-channel alpha would change the deadzone boundaries but adds complexity.

3. **Shared lambda per layer:** Tequila uses a single lambda per layer (not per-weight). This reduces parameters and is robust per the paper's sensitivity analysis.

4. **Extra RMSNorm is orthogonal:** The pre-quantization RMSNorm (proven essential in our prior experiment) is compatible with reactivation. Both mechanisms address different failure modes: RMSNorm handles activation scale drift, reactivation handles deadzone trapping.

5. **Scale caveat:** We test at d=512 (64M params). Tequila was validated at 1B-3B. The mechanism should work at smaller scale since deadzone trapping is a local phenomenon (per-weight, not architecture-dependent), but the quantitative improvement may differ.
