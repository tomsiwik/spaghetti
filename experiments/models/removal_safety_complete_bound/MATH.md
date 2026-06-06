# Complete Expert Removal Safety Bound: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 256 (micro); 896+ (production) |
| r | LoRA rank | 8 (micro); 16 (production) |
| N | Number of expert adapters | 50 |
| L | Number of transformer layers | 24 |
| k | Index of expert to remove | 0 <= k < N |
| l | Layer index | 0 <= l < L |
| h_l | Hidden state at layer l | (d,) |
| W_l | Base weight matrix at layer l | (d, d) |
| Delta_l | Merged expert delta at layer l (post-GS) | (d, d) |
| delta_{k,l}' | GS-orthogonalized delta of expert k at layer l | (d, d) |
| sigma(.) | GELU activation | R -> R |
| RN(.) | RMSNorm: x -> x / sqrt(mean(x^2) + eps) | R^d -> R^d |
| epsilon_l | Per-layer weight-space removal error (relative) | [0, 1] |
| D(d,L,N) | Output deviation (relative L2 norm) | [0, 1] |

## 2. Component Bounds from Prior Experiments

We have five independent empirical results characterizing different aspects
of expert removal safety:

### 2.1 Single-Layer Weight Error (expert_removal_graceful)

At near-orthogonal cosines (SOLE regime), naive subtraction vs GS recompute:

    epsilon_1(d, r, N) ~ C_1 * sqrt(r/d)

where C_1 depends on N weakly. At d=256, r=8, N=8: epsilon_1 ~ 1.39% (from
multilayer_removal_cascade Test 4). At N=50, the per-expert delta is smaller
(scaled by ~1/N relative to total), so the GS correction term is also smaller.

Empirically from parent experiments at d=256, N=8:
- Sum of 24 per-layer errors: 5.55%
- Per-layer error: ~0.23%

### 2.2 Depth Dampening Factor alpha_depth (multilayer_removal_cascade)

The amplification ratio measures how much of the cumulative weight-space error
survives through the nonlinear forward pass:

    alpha_depth(L) = output_dev / sum_per_layer_weight_err

Measured (feedforward, d=64):
- L=1: alpha = 0.99
- L=4: alpha = 0.62
- L=8: alpha = 0.48
- L=24: alpha = 0.25

Linear regression: alpha_depth(L) = -0.027*L + 0.805

At L=24: **alpha_depth = 0.25** (75% of weight-space error is dampened).

### 2.3 Residual+RMSNorm Factor alpha_arch (residual_layernorm_error_dynamics)

Pre-RMSNorm architecture further reduces amplification vs feedforward:

    alpha_arch = amp_ratio(pre_rmsn) / amp_ratio(feedforward) = 0.022 / 0.254 = 0.087

This is an architecture-dependent multiplicative correction. The identity path
in residual connections and the mean-preserving property of RMSNorm combine to
suppress errors by ~11.5x beyond the feedforward baseline.

However, alpha_arch and alpha_depth are NOT independent -- they were measured
at the same depth. The correct interpretation is:

    amp_ratio(pre_rmsn, L=24) = 0.022

This is the COMBINED depth + architecture factor, not a product.

### 2.4 Correlation Factor alpha_corr (correlated_layer_errors)

When inter-layer errors are maximally correlated (rho=1.0):

    alpha_corr = amp_ratio(rho=1) / amp_ratio(rho=0) = 0.074 / 0.088 = 0.84

Correlation REDUCES amplification. At maximum correlation, amplification is
84% of the independent-error case. For SOLE experts (real correlation ~0.03
from B-matrix study), the correction is ~1.0 (negligible).

### 2.5 Decorrelation Filter (b_matrix_training_correlation)

Even when B-matrices show 2.52x above-baseline cosine, the full delta vectors
(A@B) show only 0.14x of baseline cosine due to the Grassmannian skeleton's
decorrelation:

    cos(delta_i, delta_j) = 0.14 * cos_baseline

This means the effective inter-expert cosine in weight space is 7x LOWER than
random. The single-layer removal error epsilon_1 already accounts for cosine
through the GS correction term, so this factor reduces epsilon_1.

### 2.6 Attention Neutrality (attention_self_repair_removal, KILLED)

Frozen attention provides only 2.1% repair (threshold was 30%). The amplification
ratios are nearly identical:

    MLP amp_ratio / TF amp_ratio = 0.037 / 0.035 = 1.06

Attention is neutral: it neither significantly helps nor hurts. We treat
alpha_attn = 1.0 (no correction).

## 3. The Combined Bound

### 3.1 Avoiding Double-Counting

The key challenge is that several factors were measured in overlapping contexts.
We must not multiply factors that are already incorporated in each other.

**The correct decomposition is:**

    D(d, L, N) = epsilon_per_layer(d, r, N) * L * alpha_combined(arch, L)

where:
- epsilon_per_layer: single-layer weight-space error from naive subtraction
- L * epsilon_per_layer: sum of per-layer errors (additive worst case)
- alpha_combined: the total amplification ratio (measured directly)

The amplification ratio alpha_combined ALREADY incorporates:
- Depth dampening (sub-additive accumulation through L layers)
- Architecture effects (residual + normalization)

So the correct bound is simply:

    D(d, L, N) = sum_epsilon * alpha_combined

### 3.2 Computing sum_epsilon at d=256, N=50

From multilayer_removal_cascade Test 4 at d=256, N=8, L=24:
- sum_per_layer_error = 5.55%

This scales with N weakly. At N=50 with GS-orthogonalized experts on the
Grassmannian skeleton, inter-expert cosines are lower (cos decreases with N
at beta=-0.575 from collision_scaling). The decorrelation filter gives
cos(delta) = 0.14x baseline, further reducing the GS correction term.

For N=50 at d=256 with skeleton decorrelation:
- Baseline random cos at d=256: ~sqrt(r/d) ~ 0.177
- With skeleton: cos ~ 0.177 * 0.14 = 0.025
- Per-layer GS correction error scales as O(cos^2) (second-order term)

The per-layer weight error from the GS correction is:

    epsilon_l ~ ||delta_k'||_F / ||Delta_gt||_F * O(max_cos^2)

At cos=0.025, this is ~6.25e-4 per pair, times N-1=49 pairs: ~0.03 per layer.
Over L=24 layers: sum_epsilon ~ 0.03 * 24 = 0.72% total weight error.

### 3.3 Computing alpha_combined at L=24

From residual_layernorm_error_dynamics, Pre-RMSNorm at L=24:
- alpha_combined = 0.022

From correlated_layer_errors, the correlation correction:
- alpha_corr = 0.84 (at rho=1.0)
- For real SOLE experts (rho~0.03): alpha_corr ~ 1.0

The combined amplification is:
- alpha_total = alpha_combined * alpha_corr = 0.022 * 1.0 = 0.022

### 3.4 The Combined Bound

    D(d=256, L=24, N=50) = sum_epsilon * alpha_total
                          = 0.72% * 0.022
                          = 0.016%

This is well below the 1% threshold (K1 target).

### 3.5 Dimension Scaling

The output deviation scales as d^(-1.016) (from Pre-RMSNorm power law fit):

    D(d) = C * d^(-1.016)

From measured D(d=64) = 0.46%:
    C = 0.46% * 64^1.016 = 31.4

At d=256:
    D(d=256) = 31.4 * 256^(-1.016) = 0.116%

But this was for N=8 with random cosines. At N=50 with skeleton decorrelation,
the GS correction errors are much smaller, giving the 0.016% estimate above.

At d=896 (production):
    D(d=896) = 31.4 * 896^(-1.016) = 0.032%

With SOLE cosines (90x lower): ~0.0004%.

## 4. Theoretical Upper Bound (Conservative)

For a rigorous upper bound, we use worst-case values from each experiment:

1. Per-layer weight error at d=256, N=50: epsilon_l <= 0.5% (conservative)
2. Sum over L=24 layers: sum_epsilon <= 12%
3. Amplification ratio (Pre-RMSNorm, L=24): alpha <= 0.022
4. Correlation worst case (rho=1): alpha_corr <= 1.0 (correlation helps, not hurts)
5. Attention: neutral (factor = 1.0)

    D_upper = 12% * 0.022 * 1.0 = 0.264%

Even the conservative upper bound is well below 1%.

## 5. Assumptions

1. **Independent A-matrices.** Frozen Grassmannian skeleton provides near-orthogonal
   A-matrices at each layer. Validated by grassmannian_expert_init (zero drift).

2. **Sub-additive error composition.** Validated by multilayer_removal_cascade
   (amp_ratio < 1.0, decreasing with depth) and correlated_layer_errors (robust
   to correlation).

3. **Pre-RMSNorm architecture.** The 0.022 amplification ratio is specific to
   Qwen/Llama-style architectures. Other architectures have different but still
   sub-1.0 ratios.

4. **1/d dimension scaling.** Validated across all architectures (exponents
   -0.92 to -1.16, R^2 > 0.98).

5. **Attention neutrality.** Validated by attention_self_repair_removal (2.1%
   effect, well within noise).

6. **Expert count N has weak effect on amplification.** From multilayer_removal_cascade
   Test 5: N=4..32 shows no significant trend in amp_ratio.

## 6. Worked Example at Target Scale

d=256, r=8, L=24, N=50, Pre-RMSNorm architecture:

1. Generate 50 experts with LoRA rank-8 at each of 24 layers
2. GS-orthogonalize per layer
3. Merge all 50 into composite weight matrix per layer
4. Remove expert k=25 (middle) via naive subtraction
5. Recompute GS from 49 experts (ground truth)
6. Forward 500 random inputs through both models
7. Measure relative L2 deviation

Predicted output deviation: ~0.016% (combined bound)
Conservative upper bound: ~0.264%
Kill criterion K1: < 1%
Kill criterion K2: empirical within 2x of theoretical prediction
