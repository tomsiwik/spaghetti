# Adaptive Rank Selection: Mathematical Foundations (v2)

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model dimension | {64, 128, 256} (micro) |
| r | LoRA rank | integer in [1, d] |
| Delta | Target weight transformation | (d, d) |
| sigma_i | i-th singular value of Delta | sigma_1 >= sigma_2 >= ... >= 0 |
| U, V | Left/right singular vectors | (d, d) orthogonal |
| r_eff | Effective rank (Roy-Vetterli) | [1, d] |
| r_s | Stable rank | [1, d] |
| r_tau | Energy rank at threshold tau | integer in [1, d] |
| SNR | Signal-to-noise ratio | {5, 20, 100} |
| rho | Spearman rank correlation | [-1, 1] |
| tau | Error/energy threshold | 0.05 (error), 0.90-0.99 (energy) |

## 2. Domain Model

### 2.1 Target Transformation

A domain is modeled as a weight perturbation Delta in R^{d x d} that an expert
must learn. The "complexity" of the domain is determined by the spectral structure
of Delta.

**Exact-rank domains:** Delta = U_k S_k V_k^T + epsilon, where:
- U_k: (d, k) orthonormal, V_k: (d, k) orthonormal
- S_k = diag(sigma_1, ..., sigma_k) with sigma_i ~ Uniform(0.5, 2.0)
- epsilon: noise scaled to achieve target SNR = ||U_k S_k V_k^T||_F / ||epsilon||_F

**Spectral decay domains:** Delta = U diag(2 * gamma^i) V^T + epsilon, where:
- gamma in (0, 1) is the decay rate
- Low gamma (0.3): sharp drop, low effective complexity
- High gamma (0.98): nearly flat, high effective complexity

### 2.2 Intrinsic Dimensionality Metrics

Given the singular values sigma_1 >= ... >= sigma_d of Delta:

**Effective rank** (Roy & Vetterli, 2007):

    p_i = sigma_i^2 / sum_j sigma_j^2

    r_eff = exp(-sum_i p_i log(p_i))

This is the exponential of the Shannon entropy of the normalized squared
singular values. It equals d for a flat spectrum and 1 for a rank-1 matrix.

**Stable rank:**

    r_s = ||Delta||_F^2 / ||Delta||_2^2 = sum(sigma_i^2) / sigma_1^2

More robust to noise than r_eff; bounded by the algebraic rank.

**Energy rank at threshold tau:**

    r_tau = min{k : sum_{i=1}^k sigma_i^2 >= tau * sum_j sigma_j^2}

Tested at tau = 0.90, 0.95, 0.99. r_99 emerged as the best predictor because
it captures the long spectral tail that contributes to reconstruction error
in spectral-decay domains.

## 3. Optimal LoRA Approximation

### 3.1 Eckart-Young-Mirsky Theorem

The best rank-r approximation to Delta (in Frobenius norm) is:

    Delta_r = U_r S_r V_r^T

where U_r, S_r, V_r are the top-r components of the SVD. The reconstruction
error is:

    err(r) = ||Delta - Delta_r||_F / ||Delta||_F = sqrt(sum_{i=r+1}^d sigma_i^2) / sqrt(sum_i sigma_i^2)

### 3.2 Optimal Rank Detection

**v1 (BROKEN -- replaced):** Maximum absolute second derivative of log(err(r))
over a non-uniform rank grid. This introduced systematic bias toward high ranks
because larger grid spacing amplified curvature magnitudes. At d=64, the
detector assigned knee=48 to most domains regardless of true rank.

**v2 primary -- Kneedle algorithm** (Satopaa et al., 2011):

Given points {(r_i, err(r_i))} on a UNIFORM grid r_i = 1, 2, ..., d:

1. Normalize both axes to [0, 1]:
   x_i = (r_i - r_1) / (r_d - r_1)
   y_i = (err_i - err_d) / (err_1 - err_d)

2. Compute perpendicular distance from each point to the line connecting
   (x_1, y_1) to (x_d, y_d):

   dist_i = |dy * x_i - dx * y_i + x_d * y_1 - y_d * x_1| / sqrt(dx^2 + dy^2)

   where dx = x_d - x_1, dy = y_d - y_1.

3. The knee is at r_i* where dist_i is maximized.

This correctly identifies the transition from "steep improvement" (signal
recovery) to "flat" (noise fitting), regardless of grid uniformity.

**v2 secondary -- Threshold method:**

    r_opt = min{r : err(r) < tau}, tau = 0.05

This is simpler but conflates noise removal with signal recovery at low SNR.
At SNR=5, the threshold requires rank ~75% of d just to suppress noise below 5%,
which is not the "useful" rank for LoRA.

## 4. The Prediction Hypothesis

**Claim:** The intrinsic dimensionality of Delta (measured by r_99)
predicts the Kneedle-optimal LoRA rank.

**Formal statement:** Let r_knee(Delta) be the Kneedle-detected knee rank
and r_99(Delta) be the 99% energy rank. Then:

    Spearman_rho(r_99, r_knee) >= 0.5   ... (K1)

and the prediction r_pred = snap_to_grid(r_99(Delta)) satisfies:

    P(0.5 <= r_pred / r_knee <= 2.0) >= 0.5   ... (K2 basic)

with the strengthened requirement:

    P_pred(within 2x) - P_null(within 2x) > 0.10   ... (K2 null-beating)

where P_null is the accuracy of a constant predictor (always rank 16).

### 4.1 Why This Should Hold

For an exact-rank-k matrix with SNR >> 1:
- err(r) drops sharply at r = k, so the Kneedle knee is at r ~ k
- r_99 ~ k because 99% of energy is in the top k components
- Therefore r_99 ~ r_knee

For spectral decay sigma_i = C * gamma^i:
- The knee is gradual; Kneedle finds the elbow of the geometric curve
- r_99 is the rank capturing 99% of sum(C^2 * gamma^{2i})
  = C^2 * (1 - gamma^{2r}) / (1 - gamma^2) >= 0.99 * C^2 / (1 - gamma^2)
  => gamma^{2r} <= 0.01
  => r >= log(0.01) / (2 * log(gamma))
  => r_99 ~ -log(100) / (2 * log(gamma))
- The Kneedle knee occurs where the normalized curve departs most from
  the diagonal, which for a geometric series is proportional to 1/|log(gamma)|.

Both scale as 1/|log(gamma)|, so they are monotonically related. The
correlation should be strong (rho >> 0.5).

### 4.2 Why energy_rank_99 outperforms energy_rank_95

For spectral decay with rate gamma, the energy rank at threshold tau is:

    r_tau ~ -log(1 - tau) / (2 * |log(gamma)|)

The Kneedle knee, empirically, falls between r_95 and r_99. Using r_95
systematically underpredicts because the remaining 5% of energy in the
spectral tail is distributed across many dimensions and contributes
disproportionately to reconstruction error. r_99 captures this tail
without the need for an ad-hoc multiplier.

### 4.3 Why It Might Fail

1. **Kneedle is undefined for flat spectra.** When gamma -> 1, all singular
   values are similar, and the "knee" is arbitrary.
2. **Low-SNR noise inflation.** At SNR=5, noise adds significant energy
   in the tail, inflating r_99 beyond the useful signal rank.
3. **Real LoRA != Eckart-Young.** Training dynamics, initialization,
   and finite data mean real LoRA may not achieve the optimal rank-r fit.

## 5. Null Baseline Analysis

The null baseline predicts rank 16 for all domains. Its accuracy depends
on the distribution of Kneedle-optimal ranks:

- Exact-rank domains: knees at {2, 4, 8, 12, 16, 24, 32, 48}
  - Rank 16 is within 2x of {8, 12, 16, 24, 32} = 5 of 8 domains (62.5%)
- Spectral decay domains: knees at {4, 6, 10, 18, 25, 37, 49} (d=128, SNR=20)
  - Rank 16 is within 2x of {10, 18, 25} = 3 of 7 domains (42.9%)
- Combined: ~53% within 2x

This makes the K2 threshold of 50% nearly vacuous by itself. The
strengthened requirement (beat null by >10pp) ensures the metric provides
genuine predictive value.

## 6. Computational Complexity

| Operation | Cost |
|-----------|------|
| Generate domain | O(d^2 * k) for exact-rank, O(d^3) for spectral decay |
| SVD of Delta | O(d^3) |
| Compute metrics | O(d) (from singular values) |
| Error at all ranks (uniform grid) | O(d^2) (cumulative sum) |
| Kneedle detection | O(d) |
| Full experiment (N domains, S seeds, d dims, SNR levels) | O(N * S * d^3 * |SNR|) |

At micro scale (d=256, N=15, S=5, 3 SNR levels): total runtime < 2 seconds.

## 7. Worked Example

d = 128, true_rank = 8, SNR = 20:

1. Generate: U_(128,8) @ diag([1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.5]) @ V_(8,128)^T
2. Signal norm: sqrt(1.8^2+...+0.5^2) = sqrt(8.69) = 2.95
3. Noise norm target: 2.95/20 = 0.147
4. SVD(Delta): top-8 sigmas ~ [1.81, 1.61, 1.41, ...], tail ~ 0.01 each
5. r_eff = exp(H) where H = -sum(p_i log p_i), p ~ [0.38, 0.30, 0.23, ...] -> H ~ 1.86 -> r_eff ~ 6.4
6. r_95 = 7 (sum of top-7 squared sigmas > 0.95 * total)
7. r_99 = 8 (sum of top-8 squared sigmas > 0.99 * total)
8. Error curve: err(2) ~ 0.65, err(4) ~ 0.37, err(8) ~ 0.05, err(16) ~ 0.01
9. Kneedle on uniform grid [1..128]: normalizes, finds max distance from diagonal at r ~ 8
10. Prediction: r_pred = snap(r_99) = snap(8) = 8 -> ratio = 1.0

## 8. Assumptions

1. Target transformations are well-modeled by low-rank + noise decomposition
2. LoRA training converges to the optimal rank-r approximation (Eckart-Young)
3. The Kneedle knee of the error curve is the "right" definition of optimal rank
4. SNR in {5, 20, 100} covers the range of real fine-tuning scenarios
5. Spectral decay of real domain transformations is approximately monotone
6. Per-domain uniform rank is sufficient (vs per-layer adaptive rank a la AdaLoRA)
7. The null baseline (rank 16) is a fair comparison point for K2
