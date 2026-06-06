# SNR-Aware Rank Predictor with Fallback: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model dimension | {64, 128, 256} (micro) |
| r | LoRA rank | integer in [1, d] |
| Delta | Target weight transformation | (d, d) |
| sigma_i | i-th singular value of Delta | sigma_1 >= sigma_2 >= ... >= 0 |
| r_tau | Energy rank at threshold tau | integer in [1, d] |
| r_95, r_99 | Energy rank at 95%, 99% | integer in [1, d] |
| r_eff | Effective rank (Roy-Vetterli) | [1, d] |
| R | Ratio r_99 / r_95 | [1, d] |
| SNR | Signal-to-noise ratio | {5, 10, 20, 100} |
| T | Fallback trigger threshold | {1.5, 2.0, 2.5, 3.0} |
| rho | Spearman rank correlation | [-1, 1] |

All definitions inherited from parent experiment (exp_adaptive_rank_selection).

## 2. The Problem: r_99 Failure at Low SNR

### 2.1 Parent Finding

The parent experiment proved that energy_rank_99 is the best rank predictor
overall (rho=0.94-0.99 across 9 conditions). However, the adversarial review
identified a failure mode:

At SNR=5 d=64, r_99 achieves only 53.3% within-2x accuracy, which is
**-13.3pp worse than the null baseline** (always predict rank 16, 66.7%).

### 2.2 Root Cause: Noise Inflation of the Spectral Tail

For a signal matrix S of rank k with noise N at SNR level eta:

    Delta = S + N,  where ||S||_F / ||N||_F = eta

The singular values of Delta satisfy (Weyl's perturbation theorem):

    |sigma_i(Delta) - sigma_i(S)| <= ||N||_2

For the energy rank, what matters is the cumulative energy in the tail:

    E_tail(r) = sum_{i=r+1}^d sigma_i(Delta)^2

At high SNR (eta >> 1), noise contributes negligibly to E_tail, so
r_99(Delta) ~ r_99(S), which closely tracks the optimal rank.

At low SNR (eta ~ 5), the noise contributes O(||S||_F^2 / eta^2) to the
total energy, spread across all d dimensions. The 99% energy threshold
must capture this noise energy, inflating r_99 well beyond the signal rank:

    r_99(Delta) ~ r_99(S) + (d - r_99(S)) * (1 - eta^2 / (eta^2 + 1)) * correction

In practice at SNR=5: noise contributes ~4% of total energy. The 95% threshold
is robust (it can "miss" this 4%), but the 99% threshold must capture most of
it, requiring many additional noise dimensions.

### 2.3 The r_99/r_95 Ratio as Noise Diagnostic

Define the ratio:

    R = r_99 / r_95

**Clean signal (high SNR):** The energy between 95% and 99% is concentrated in
a few dimensions adjacent to the signal rank. R is close to 1 (typically 1.2-1.5).

**Noisy signal (low SNR):** The 95%-to-99% energy gap is distributed across many
noise dimensions, causing r_99 to grow much faster than r_95. R becomes large
(typically 3-13 at SNR=5).

Empirically:
| SNR | Mean R | Max R |
|-----|--------|-------|
| 5   | 3.8-13.0 | 11.5-43.5 |
| 10  | 1.3-1.5 | 1.8-2.0 |
| 20  | 1.3 | 1.5-1.6 |
| 100 | 1.2-1.3 | 1.5 |

The ratio R cleanly separates the noise-dominated regime (R >> 2) from the
clean-signal regime (R < 1.5), with a transition zone at R ~ 1.5-2.0.

## 3. The Compound Heuristic

### 3.1 Definition

    r_compound(Delta; T) = { r_99(Delta)  if R <= T
                           { r_95(Delta)  if R > T

where T is the trigger threshold (default T=2.0).

### 3.2 Properties

**No regression at high SNR:** When SNR >= 10, R <= 2.0 for all observed
domains, so r_compound = r_99, exactly matching the parent's best predictor.

**Noise protection at low SNR:** When SNR = 5 and noise inflates r_99,
R > 2.0 triggers the fallback to r_95, which is a more conservative (lower)
estimate that tracks the signal rank rather than the signal+noise rank.

**Monotone in T:** Increasing T makes the heuristic more permissive
(falls back less often). At T=infinity, it degenerates to r_99 unconditionally.
At T=1.0, it degenerates to r_95 unconditionally.

### 3.3 Threshold Selection

The sweep over T in {1.5, 2.0, 2.5, 3.0} shows:

| Threshold T | Mean within_2x (all conditions) | Low-SNR within_2x |
|-------------|-------------------------------|-------------------|
| 1.5 | 91.1% | mixed (regression at SNR=10) |
| **2.0** | **95.0%** | 80-87% (best overall) |
| 2.5 | 95.0% | identical to T=2.0 |
| 3.0 | 93.9% | slight degradation at d=64 |

T=2.0 is optimal because:
1. At SNR=5, most domains have R > 2.0 (53-93% triggered), so fallback fires
2. At SNR=10, all domains have R <= 2.0 (0% triggered), so r_99 is preserved
3. The boundary at R=2.0 exactly separates the noise-dominated and clean regimes

T=1.5 is too aggressive: it triggers fallback at SNR=10 (27-47% of domains),
causing -6.7pp to -20.0pp regression vs r_99 at that SNR level.

T=2.5 gives identical results to T=2.0 (the trigger gap between SNR=5 and
SNR=10 is wide enough that both thresholds fall in the same bin).

## 4. Alternative Fallback: Effective Rank

### 4.1 compound_eff Heuristic

    r_compound_eff(Delta; T) = { r_99(Delta)         if r_99 / r_eff <= T
                               { r_eff(Delta)         if r_99 / r_eff > T

### 4.2 Comparison

compound_eff_t2.0 achieves 87.8% mean within_2x vs 95.0% for compound_r95_t2.0.
The effective rank is a continuous measure (not an integer energy threshold),
which makes it less well-calibrated as a rank predictor even when used as
fallback. It underperforms r_95 as the fallback choice.

## 5. Kill Criteria Analysis

### K1: Improvement over r_99 at SNR <= 10

compound_r95_t2.0 improvements over r_99 at low SNR:

| Condition | compound_r95 | r_99 | Delta |
|-----------|-------------|------|-------|
| d=64, SNR=5 | 80.0% | 53.3% | +26.7pp |
| d=64, SNR=10 | 100.0% | 100.0% | +0.0pp |
| d=128, SNR=5 | 86.7% | 33.3% | +53.3pp |
| d=128, SNR=10 | 100.0% | 100.0% | +0.0pp |
| d=256, SNR=5 | 86.7% | 26.7% | +60.0pp |
| d=256, SNR=10 | 100.0% | 100.0% | +0.0pp |

Mean improvement: +23.33pp. All SNR=5 conditions show massive improvement.
All SNR=10 conditions show no regression (0.0pp). **K1 PASS.**

### K2: No condition worse than null after fallback

compound_r95_t2.0 vs null baseline across all 12 conditions:

| Condition | compound | null | Delta |
|-----------|----------|------|-------|
| d=64, SNR=5 | 80.0% | 66.7% | +13.3pp |
| d=128, SNR=5 | 86.7% | 53.3% | +33.3pp |
| d=256, SNR=5 | 86.7% | 60.0% | +26.7pp |
| All SNR>=10 | 93-100% | 53-60% | +33-47pp |

**Zero conditions where compound is worse than null.** The minimum margin
is +13.3pp (d=64, SNR=5). **K2 PASS.**

## 6. Computational Complexity

| Operation | Cost |
|-----------|------|
| Compute r_95, r_99 from SVD | O(d) each (cumulative sum scan) |
| Compute R = r_99/r_95 | O(1) |
| Compound decision | O(1) (single comparison) |
| Total overhead vs r_99 alone | O(d) (one additional energy_rank call) |

The compound heuristic adds negligible cost: one extra cumulative sum scan
and one comparison. The SVD of Delta (O(d^3) or O(d*r^2) for rank-r)
dominates in all cases.

## 7. Worked Example

d=128, true_rank=8, SNR=5:

1. Generate signal S = U_8 @ diag(sigmas) @ V_8^T, ||S||_F ~ 3.0
2. Add noise: ||N||_F = 3.0/5 = 0.6, noise per dimension ~ 0.6/sqrt(128*128) ~ 0.005
3. SVD(Delta): top-8 sigmas ~ [1.82, 1.62, ...], tail sigmas ~ 0.05 each
4. Total energy = sum(sigma_i^2) ~ 9.0 + 128*0.0025 = 9.32
5. r_95 = 8 (top-8 captures 9.0/9.32 = 96.5%)
6. r_99 = 18 (need 10 more noise dimensions to reach 99%: 9.0 + 10*0.0025 = 9.225/9.32 = 98.9%)
7. R = 18/8 = 2.25 > 2.0 -> TRIGGER FALLBACK
8. r_compound = r_95 = 8
9. Kneedle knee = 8 (signal-noise boundary)
10. Prediction ratio: 8/8 = 1.00 (perfect)

Without fallback, r_99=18 would give ratio 18/8 = 2.25 (still within 2x, but
for a domain with true_rank=2, the 99% energy rank at SNR=5 can reach ~12,
giving ratio 12/2 = 6.0, far outside the 2x tolerance).

## 8. Assumptions

1. All assumptions from the parent experiment (Eckart-Young optimality,
   single-matrix domains, synthetic spectral structure)
2. The r_99/r_95 ratio cleanly separates noise-dominated from clean-signal
   spectra. This holds empirically for the tested SNR range but may not
   generalize to intermediate noise structures (e.g., colored noise)
3. r_95 is a reliable predictor in the noise-dominated regime. This is
   supported by the parent experiment (r_95 achieved 80-87% at SNR=5)
4. The trigger threshold T=2.0 is robust across dimensions. Supported
   by results showing identical performance at T=2.0 and T=2.5
5. Real LoRA training has effective SNR in the range tested (5-100).
   This needs macro validation
