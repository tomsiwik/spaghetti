# SiLU vs GELU Gamma Correction: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | {64, 128, 256, 512} (micro) |
| r | LoRA rank | 8 |
| N | Number of expert adapters | {5, 10, 25, 50} |
| L | Number of transformer layers | 24 |
| h_l | Hidden state at layer l | (d,) |
| W_l | Base weight matrix at layer l | (d, d) |
| Delta_l | Merged expert delta at layer l | (d, d) |
| gamma_l | Learnable RMSNorm scale at layer l | (d,) |
| RN(x; gamma) | gamma * x / sqrt(mean(x^2) + eps) | R^d -> R^d |
| alpha | Amplification ratio: output_dev / sum_epsilon | scalar in [0, 1] |
| D | Output deviation from expert removal | percentage |
| C(gamma, act) | Gamma correction factor for activation act | scalar >= 1.0 |

## 2. Background

The parent experiment (exp_rmsnorm_gamma_nonuniformity, PROVEN) established:
- GELU worst-case gamma correction factor: C_GELU = 1.43x (single-layer gamma=10 spike)
- Realistic gamma (log-normal sigma=0.5): C_GELU = 1.02x
- D < 0.1% at d=256, N=50 even with worst-case gamma (51x below 5% threshold)

**Gap flagged by adversarial review:** Qwen2.5/Llama use SiLU, not GELU. The 1.43x factor
needs revalidation with the production activation function.

A prior experiment (silu_gamma_correction) tested this at d={64,128,256}, N={8,50}, 3 seeds
and found SiLU correction = 1.41x (PROVEN). This experiment extends with:
- Wider d sweep: adds d=512
- Wider N sweep: {5, 10, 25, 50} instead of {8, 50}
- More seeds: 5 instead of 3
- New kill criteria aligned with the hypothesis graph

## 3. Activation Functions

### GELU (Gaussian Error Linear Unit)
$$\text{GELU}(x) = 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$$

### SiLU (Sigmoid Linear Unit, aka Swish)
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Both are smooth, non-monotone around x=0, and approach identity for large positive x.

## 4. Why the Correction Factor Depends on Activation Curvature

The amplification ratio alpha = ||u_L|| / (||h_L|| * sum_eps) is approximately
invariant to gamma because gamma is a fixed linear transformation applied to both
signal and perturbation paths through the same Jacobian. The residual correction
comes from the second-order interaction between gamma-scaled inputs and the
activation nonlinearity.

For the forward pass:
$$h_{l+1} = h_l + \text{act}((W_l + \Delta_l) \cdot \text{RN}(h_l; \gamma_l))$$

The pre-activation input z_l = (W_l + Delta_l) @ RN(h; gamma) is scaled by gamma.
Taylor-expanding the activation around z_l:
$$\text{act}(z + \delta z) \approx \text{act}(z) + \text{act}'(z)\delta z + \frac{1}{2}\text{act}''(z)(\delta z)^2$$

The second-order term act''(z) creates the correction factor. Since gamma scales z,
it changes the operating point and thus the curvature response.

### Curvature comparison

| Activation | max |act''(x)| in [-5, 5] | Location of max |
|-----------|--------------------------------|-----------------|
| GELU | 0.798 | x ~ -0.7 |
| SiLU | 0.500 | x ~ -1.3 |

SiLU has 37% lower peak curvature than GELU. Therefore:
$$C_{\text{SiLU}} \leq C_{\text{GELU}} = 1.43\text{x}$$

This is an analytical prediction confirmed in the prior experiment.

## 5. Kill Criteria

### K1: SiLU gamma correction factor differs from GELU by >1.5x

If GELU correction = 1.43x, then 1.5x would mean SiLU correction > 2.15x.
This would indicate fundamentally different nonlinear behavior requiring
separate bounds for SiLU architectures.

**Formal:** FAIL if max_d,N,gamma { C_SiLU(d, N, gamma) / C_GELU(d, N, gamma) } > 1.5

### K2: SiLU worst-case D exceeds 5% at d=256, N=50

This is the absolute safety bound. If D > 5% under any gamma profile with SiLU,
the composition is unsafe.

**Formal:** FAIL if max_gamma,seed { D_SiLU(d=256, N=50, gamma) } > 5.0%

## 6. Experimental Design

### Parameter sweep
- d in {64, 128, 256, 512}
- N in {5, 10, 25, 50}
- 5 seeds per configuration
- Gamma profiles: uniform, log-normal (sigma=0.5, 1.0), bimodal (0.2/5.0), spike (layer gamma=10)
- Both GELU and SiLU at each configuration

### Metrics per configuration
1. alpha = mean output deviation / sum per-layer weight-space error
2. C(gamma, act) = alpha(gamma) / alpha(uniform) for each activation
3. D = mean output deviation percentage
4. SiLU/GELU ratio of correction factors

### Worked example (d=64, N=8, uniform gamma)
From prior experiment:
- GELU: alpha = 0.0217, D = 0.42%
- SiLU: alpha = 0.0240, D = 0.44%
- SiLU baseline is 10% higher (more amplification), but correction ratio is similar

## 7. Assumptions

1. **Random LoRA init:** A ~ N(0, 1/d), B ~ N(0, 1/r). Production adapters are trained,
   but the cancellation argument is independent of weight structure.
2. **Pre-RMSNorm architecture:** Matches Qwen2.5/Llama layer structure.
3. **No SwiGLU gating:** Production FFN uses gated SiLU (SwiGLU). The gate adds
   another multiplicative nonlinearity. This is a known limitation.
4. **Fixed gamma:** gamma does not change between forward passes (it is a frozen
   base model parameter, not dependent on the adapter weights).
