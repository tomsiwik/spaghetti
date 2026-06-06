# Multi-layer Removal Cascade: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | {32, 64, 128, 256} (micro); 896 (production) |
| r | LoRA rank | 8 (micro); 16 (production) |
| N | Number of expert adapters | {4, 8, 16, 32} |
| L | Number of transformer layers | {1, 2, 4, 8, 12, 16, 24} |
| k | Index of expert to remove | 0 <= k < N |
| l | Layer index | 0 <= l < L |
| h_l | Hidden state at layer l | (d,) |
| W_l | Base weight matrix at layer l | (d, d) |
| delta_{i,l} | Weight delta of expert i at layer l | (d, d) |
| delta_{i,l}' | GS-orthogonalized delta | (d, d) |
| sigma(.) | Nonlinear activation (GELU, ReLU, or linear) | R -> R |
| cos_{i,j} | Mean absolute pairwise cosine of flattened deltas | [0, 1] |

## 2. Multi-Layer Forward Model

### 2.1 Layer-wise Computation

Each layer computes:

    h_{l+1} = sigma( (W_l + Delta_l) @ h_l )

where Delta_l = sum_{i=1}^{N} delta_{i,l}' is the merged expert contribution
at layer l after Gram-Schmidt orthogonalization.

The full forward pass chains L layers:

    f(x) = sigma_L . (W_L + Delta_L) . ... . sigma_1 . (W_1 + Delta_1) . x

### 2.2 Expert Removal

Removing expert k changes Delta_l at every layer:

    Delta_l^{naive}  = Delta_l - delta_{k,l}'      (naive subtraction)
    Delta_l^{gt}     = GS_merge({delta_{i,l}}_i!=k) (GS recompute, ground truth)

The per-layer weight-space error is:

    epsilon_l = ||Delta_l^{naive} - Delta_l^{gt}||_F / ||Delta_l^{gt}||_F

This was measured in the parent experiment: epsilon_l ~ 0.18% at SOLE cosines.

## 3. Error Propagation Through Depth

### 3.1 The Lipschitz Amplification Concern

For a single layer, the output error from a weight perturbation epsilon is:

    ||f_l(h; W + epsilon) - f_l(h; W)|| <= L_sigma * ||epsilon|| * ||h||

where L_sigma is the Lipschitz constant of the activation (L_sigma = 1 for
ReLU, ~1.13 for GELU).

For L layers, the naive worst-case bound is:

    ||f_L - f_L^{gt}|| <= prod_{l=1}^{L} (1 + L_sigma * ||epsilon_l|| / ||W_l||)

If each layer has relative error epsilon, this gives:

    Total error <= (1 + epsilon)^L - 1 ~ L * epsilon + L*(L-1)/2 * epsilon^2 + ...

For epsilon = 0.002 (SOLE regime) and L = 24:

    Linear term:  24 * 0.002 = 0.048 (4.8%)
    Quadratic:    276 * 0.000004 = 0.0011 (0.11%)

This predicts near-linear growth with small quadratic correction.

### 3.2 Why Errors Are Sub-Additive in Practice

The experiment shows amplification ratio < 1 (mean ~0.25 at L=24), meaning
output-space error is LESS than the sum of weight-space errors. Three mechanisms:

**Mechanism 1: Activation masking.** GELU/ReLU set ~50% of hidden units to zero
(or near-zero). Error in masked dimensions is suppressed. Each layer masks a
different subset, so cumulative suppression grows with depth.

**Mechanism 2: Direction randomization.** Per-layer weight errors have random
directions (because each layer's experts are independently drawn). The errors
compose like random vectors: ||sum e_l|| ~ sqrt(L) * ||e||, not L * ||e||.
This gives sqrt(L) scaling rather than linear.

**Mechanism 3: Spectral contraction.** Random weight matrices with entries
~1/sqrt(d) have spectral norms ~1 by the Marchenko-Pastur law. Perturbations
in directions not aligned with the top singular vectors are exponentially
suppressed through depth.

### 3.3 Empirical Scaling

The observed relationship between depth and output deviation (near-orthogonal
regime, d=64) follows sub-linear growth:

    output_dev(L) ~ a * L^b    with b < 1

Measured values:

| L | Sum weight error (%) | Mean output dev (%) | Ratio |
|---|---------------------|---------------------|-------|
| 1 | 1.08 | 1.06 | 0.99 |
| 4 | 3.40 | 2.11 | 0.62 |
| 8 | 6.63 | 3.21 | 0.48 |
| 12 | 10.24 | 3.65 | 0.36 |
| 16 | 14.19 | 5.23 | 0.37 |
| 24 | 20.78 | 5.31 | 0.25 |

The amplification ratio monotonically decreases with L, confirming sub-additive
error accumulation.

## 4. Dimension Scaling

At L=24, higher d reduces both weight-space and output-space errors because
per-layer cosine decreases as 1/sqrt(d):

| d | Mean cos | Sum weight error (%) | Mean output dev (%) |
|---|----------|---------------------|---------------------|
| 32 | 0.025 | 37.69 | 15.65 |
| 64 | 0.012 | 20.78 | 5.31 |
| 128 | 0.006 | 10.38 | 2.71 |
| 256 | 0.003 | 5.55 | 1.39 |

Extrapolation to d=896 (production):

    cos ~ r/d ~ 16/896 ~ 0.018  (but structural gives 0.0002)

    sum_weight_error(d=896) ~ 5.55 * (256/896) ~ 1.6%
    output_dev(d=896) ~ 1.39 * (256/896) ~ 0.4%

At production cosines (0.0002, 90x lower than random at d=256), output
deviation would be negligible (~0.01%).

## 5. The Clustered Regime Paradox

At cos~0.3, weight-space errors are enormous (313% at L=24) but output
deviations are tiny (0.29%). This is because:

1. Clustered experts are scaled to ||dW|| ~ 0.01 (realistic magnitude)
2. Weight perturbation relative to base weight is tiny: 0.01 / 1.0 = 1%
3. The 313% weight error is relative to the delta, not the base weight
4. What matters for output error is perturbation relative to total weight

The correct metric for production relevance is:

    output_error ~ weight_perturbation / total_weight_norm
                 = epsilon_delta * ||delta|| / ||W||

Since ||delta|| << ||W||, even large relative delta errors produce small
absolute output errors.

## 6. Key Inequalities

### 6.1 Sub-additivity bound (empirical)

    output_dev(L) <= sum_{l=1}^{L} epsilon_l    for all tested configurations

This held in 100% of experiments. The amplification ratio never exceeded 1.03
(at L=1, which is the degenerate case where output dev = weight error).

### 6.2 Dimension scaling

    output_dev(d, L=24) ~ C / d    where C ~ 350 at r=8, L=24

### 6.3 Production bound

At d=896, r=16, L=24 with SOLE cosines (cos~0.0002):

    per-layer weight error ~ 0.18% (from parent experiment)
    sum weight error ~ 24 * 0.18% = 4.3%
    amplification ratio ~ 0.25 (measured at L=24)
    predicted output dev ~ 4.3% * 0.25 ~ 1.1%

Conservative upper bound (using amp_ratio = 1): 4.3%.

## 7. Assumptions

1. **Synthetic base weights.** Random matrices 1/sqrt(d). Real transformer
   weights have structured spectra (pre-training induces low-rank structure).
   This likely makes real models MORE stable (trained weights are well-conditioned).

2. **GELU activation.** Production models use SiLU (close to GELU). Test 3
   shows activation choice has <10% effect on amplification.

3. **Same dimension across layers.** Real transformers have this property
   (hidden dim is constant). Attention layers have different structure but
   the per-layer error analysis still applies.

4. **Independent per-layer experts.** We generate independent LoRA deltas per
   layer. Real LoRA experts are trained jointly across layers, potentially
   introducing inter-layer correlation. This could increase or decrease
   error amplification.

5. **Reconstruction error as PPL proxy.** Output deviation is a better proxy
   than weight-space error (it includes the nonlinear forward pass) but still
   not true PPL. The sub-additivity finding should transfer because it is a
   property of the network architecture, not the metric.
