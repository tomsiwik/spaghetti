# Channel Capacity Bound: Mathematical Foundations (REVISED)

## 1. Setup and Notation

| Symbol | Shape / Type | Definition |
|--------|-------------|------------|
| d | scalar | model embedding dimension (d=64 at micro scale) |
| N | scalar | number of composed expert domains |
| L | scalar | number of transformer layers (L=4) |
| P | scalar | per-expert signal power (normalized) |
| x_i | R^d | output of expert i's capsule pool (aggregated) |
| w_i | scalar | routing weight for expert i (from softmax router) |
| y | R^d | residual stream output after composition |
| eta | R^d | intrinsic noise (training stochasticity) |
| sigma^2 | scalar | noise variance: Var[eta_j] for each dimension j |
| rho_ij | scalar | cosine similarity between expert i and j outputs |
| SNR_0 | scalar | signal-to-noise ratio with single expert = P / sigma^2 |
| alpha | scalar | interference coupling constant = rho^2 * SNR_0 |
| C(N) | scalar | per-expert channel capacity at N composed experts |
| gap(N) | scalar | composition gap = (1 - C(N)/C(1)) * 100% |
| c_0 | scalar | calibration offset (constant, fit from data) |

## 2. Channel Model

### 2.1 The Residual Stream as a Communication Channel

The residual stream in a transformer is an additive channel. At each layer,
attention and MLP blocks ADD their outputs to the residual. When N expert
domains are composed, the MLP layer becomes:

    y = h_attn + sum_{i=1}^{N} w_i * f_i(h_attn) + eta          (1)

where h_attn is the post-attention hidden state, f_i is expert i's capsule
pool (MLP), w_i is the softmax routing weight, and eta captures training
noise and approximation error.

This is a **Gaussian Multiple-Access Channel (MAC)**: N transmitters
(experts) share a common medium (the d-dimensional residual stream).

### 2.2 Signal and Noise Powers

For a single expert i acting alone:
- Signal power: P_i = E[||f_i(x)||^2] / d
- Noise power: sigma^2 = E[||eta||^2] / d
- SNR at N=1: SNR_0 = P / sigma^2 (assuming equal power experts)

### 2.3 Inter-Expert Interference

When N experts transmit simultaneously, expert j's signal appears as noise
to expert i's signal:

    I_{i,j} = rho_{ij}^2 * P

Total interference seen by expert i:

    I_total(N) = (N-1) * rho_mean^2 * P                           (2)

where rho_mean is the average pairwise correlation.

## 3. Capacity Derivation

### 3.1 Effective SNR

    SNR_eff(N) = P / (sigma^2 + (N-1) * rho^2 * P)
               = SNR_0 / (1 + (N-1) * alpha)                     (3)

### 3.2 Per-Expert Capacity

    C(N) = (1/2) * log(1 + SNR_eff(N))                           (4)

### 3.3 Composition Gap

    gap(N) = (1 - C(N) / C(1)) * 100%                            (5)

With calibration offset:

    gap_cal(N) = gap(N) + c_0                                     (6)

### 3.4 Asymptotic Analysis

**Small N regime** (N << 1/alpha):
    gap(N) ~ (N-1) * alpha / (2 * (1 + SNR_0) * ln(1 + SNR_0)) * 100%

**Large N regime** (N >> 1/alpha):
    gap(N) -> 100% as N -> infinity

**Critical N** (where interference equals noise):
    N_critical = 1 + 1/alpha

## 4. Model Fitting

### 4.1 Free Parameters

3 free parameters: SNR_0, alpha, c_0.

Training data: N=2 (gap=-0.20%), N=5 (gap=+1.60%), N=8 (gap=+5.71%).

### 4.2 Fitted Values (Micro Scale, d=64)

| Parameter | Value | Physical Meaning |
|-----------|-------|-----------------|
| SNR_0 | 0.4277 | Low SNR at micro scale |
| rho_mean | 0.172 | Moderate correlation |
| alpha | 0.0126 | Weak coupling |
| c_0 | -1.64% | Calibration overcorrection |

### 4.3 Training Fit Quality

| Metric | Value |
|--------|-------|
| Train R^2 (N=2,5,8) | 0.944 |
| Train MSE | 0.345 |

## 5. Held-Out Validation (REVISION)

### 5.1 Validation Data Collection

Composition gaps were collected at N=3,4,6,7 using the same protocol as
N=2,5,8 (pretrain + domain fine-tune + compose + calibrate). Three seeds
(42, 123, 7) per N value.

| N | Split Method | Mean Gap | Std | Predicted Gap | Error |
|---|-------------|----------|-----|---------------|-------|
| 3 | ternary | +5.23% | 0.94% | +0.44% | -4.79% |
| 4 | quaternary | +5.35% | 1.49% | +1.44% | -3.90% |
| 6 | senary | +6.09% | 1.10% | +3.40% | -2.69% |
| 7 | septenary | +4.81% | 0.81% | +4.34% | -0.47% |

### 5.2 Validation Result

**Validation R^2 = -53.2** (catastrophic failure).
**Full-data R^2 = -0.35** (worse than mean prediction).

The model systematically underpredicts the gap at N=3,4 and the data is
non-monotonic (N=5 has lower gap than N=3,4,6,7).

## 6. Baseline Model Comparison (REVISION)

### 6.1 Models Compared

Three models fit to training data (N=2,5,8):

1. **Shannon**: gap(N) = (1 - log(1+SNR_0/(1+(N-1)*alpha))/log(1+SNR_0))*100 + c_0
   (3 parameters: SNR_0, alpha, c_0)

2. **Linear**: gap(N) = a*N + b
   (2 parameters: a, b)

3. **Power-law**: gap(N) = a*N^b + c
   (3 parameters: a, b, c)

### 6.2 Results

| Model | Train MSE | Val MSE | Val R^2 | BIC |
|-------|-----------|---------|---------|-----|
| Shannon | 0.345 | 11.42 | -53.2 | 0.10 |
| Linear | 0.296 | 11.68 | -54.4 | -1.45 |
| Power-law | 0.000 | 14.65 | -68.5 | -39.71 |

All three models fail on validation. The power-law achieves perfect training
fit (3 params / 3 points) but generalizes worst. The Shannon model is not
distinguishable from linear in predictive power.

### 6.3 Interpretation

With 3 parameters and 3 data points, ANY smooth monotonic function fits
perfectly. The training R^2 = 0.944 for Shannon was not evidence of the
channel model specifically -- it was evidence that a monotonic function can
interpolate 3 roughly monotonic points. The held-out validation confirms this.

## 7. Rate-Distortion Interpretation (REVISED -- Descriptive Only)

**CAVEAT**: The following interpretation is DESCRIPTIVE, not a proven
fundamental bound. It relabels the axes of the fitted curve but does not
prove that the bound is tight or unbeatable.

The quality-vs-N curve CAN BE INTERPRETED as a rate-distortion function:
- **Rate**: R(N) = C(N) = capacity available per expert at N
- **Distortion**: D(N) = gap(N) = quality loss

However:
1. This bound was FIT TO DATA, not derived from first principles.
2. A different composition scheme could have different alpha or functional form.
3. The claim "no composition scheme can beat it" is UNSUBSTANTIATED.
4. The model does not predict held-out data (R^2 < 0), so the rate-distortion
   interpretation is empirically unsupported.

## 8. Sensitivity Analysis (REVISION)

Perturbing each training data point by +/- 1%:
- N_max (10% gap) range: [13, 13] across all 27 perturbation combinations
- The prediction is robust to measurement noise

The instability is not in the inputs but in the model assumptions: the gap
is not a smooth monotonic function of N.

## 9. Assumptions and Their Status

| Assumption | Status After Validation |
|-----------|------------------------|
| Gaussian noise | Untested (not falsified but not confirmed) |
| Equal-power experts | Violated: different splits produce different-quality experts |
| Constant correlation (rho) | Likely violated: split quality varies with N |
| Monotonic degradation | **FALSIFIED**: gap(N=5) < gap(N=3) |
| Gap is a function of N alone | **FALSIFIED**: gap depends on split method |
| Additive interference | Correct (residual stream is additive) |

## 10. What Was Learned

1. Composition gap is NOT primarily determined by N. Domain split quality
   dominates the interference effect.

2. The channel model is a valid conceptual framework (residual stream IS
   a MAC channel) but its predictions are masked by confounding variables
   (split method, domain balance).

3. To salvage the approach: control for split quality by using a single
   8-domain partition and composing subsets, isolating the N effect.

## 11. Computational Cost

- Grid search (model fitting): < 1 second
- Data collection (N=3,4,6,7 x 3 seeds): ~4 minutes total
- Total: ~5 minutes
