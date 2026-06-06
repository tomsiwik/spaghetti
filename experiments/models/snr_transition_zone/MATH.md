# SNR Transition Zone Validation: Mathematical Foundations

## 1. Setup and Notation

All notation inherited from parent experiments (exp_adaptive_rank_selection,
exp_adaptive_rank_snr_fallback). Additional definitions:

| Symbol | Definition | Range |
|--------|-----------|-------|
| SNR_tz | Transition zone SNR values | {6, 7, 8, 9} |
| SNR_anchor | Anchor SNR values from parent | {5, 10} |
| R(SNR) | Mean r_99/r_95 ratio at given SNR | [1, d] |
| R_domain(SNR) | Per-domain median r_99/r_95 ratio | [1, d] |
| f_T(SNR) | Fraction of domains with R > T | [0, 1] |

## 2. The Gap

### 2.1 Parent Findings at Boundary Points

From exp_adaptive_rank_snr_fallback:

| SNR | Mean R | % domains R > 2.0 | compound_r95_t2.0 within_2x |
|-----|--------|--------------------|-----------------------------|
| 5   | 3.8-13.0 | 53-93% | 80-87% (fallback FIRES) |
| 10  | 1.3-1.5 | 0% | 100% (fallback NEVER fires) |

The r_99/r_95 ratio jumps discontinuously between SNR=5 and SNR=10 in the
parent data because the transition zone SNR={6,7,8,9} was not sampled.

T=2.0 and T=2.5 give identical results across ALL 12 conditions because
at SNR=5 the ratio is always well above 2.5, and at SNR>=10 it is always
well below 2.0. The threshold is never actually tested in its critical range.

### 2.2 Theoretical Prediction for the Transition

For a rank-k signal in d dimensions at SNR=eta, the noise contributes:

    E_noise / E_total = 1 / (1 + eta^2)

At the energy threshold tau (0.95 or 0.99):

    r_tau ~ k + (d - k) * max(0, tau - E_signal/E_total) / (E_noise_per_dim)

where E_signal/E_total = eta^2 / (eta^2 + 1) and E_noise_per_dim = 1/((eta^2+1)*d).

For tau=0.95:
- SNR=5: E_signal/E_total = 0.962 > 0.95, so r_95 ~ k (barely needs noise dims)
- SNR=7: E_signal/E_total = 0.980 > 0.95, so r_95 ~ k
- SNR=10: E_signal/E_total = 0.990 > 0.95, so r_95 ~ k

For tau=0.99:
- SNR=5: E_signal/E_total = 0.962 < 0.99, so r_99 must capture noise dims
  Deficit: 0.99 - 0.962 = 0.028, needs ~0.028*d noise dims -> r_99 ~ k + 0.028*d
- SNR=7: E_signal/E_total = 0.980 < 0.99, deficit = 0.01, needs ~0.01*d noise dims
  -> r_99 ~ k + 0.01*d
- SNR=10: E_signal/E_total = 0.990 >= 0.99, so r_99 ~ k

### 2.3 Predicted R(SNR) Curve

The ratio R = r_99/r_95 should follow:

    R(SNR) ~ 1 + max(0, 0.99 - SNR^2/(SNR^2+1)) * d / k

This gives:
| SNR | E_signal/E_total | 0.99 deficit | R prediction (k=8, d=128) |
|-----|-----------------|--------------|---------------------------|
| 5   | 0.962 | 0.028 | 1 + 0.028*128/8 = 1.45 (underestimate, actual ~3-13) |
| 6   | 0.973 | 0.017 | ~2.3 (expected transition) |
| 7   | 0.980 | 0.010 | ~1.6 |
| 8   | 0.985 | 0.005 | ~1.3 |
| 9   | 0.988 | 0.002 | ~1.1 |
| 10  | 0.990 | 0.000 | ~1.0 |

The simple model underpredicts because it ignores the MP-law spreading of
noise singular values. But the SHAPE is correct: R(SNR) crosses the T=2.0
threshold somewhere around SNR=6-7.

## 3. Experimental Design

### 3.1 Conditions

- **Transition zone SNR**: {6, 7, 8, 9} (the gap)
- **Anchor SNR**: {5, 10} (validate consistency with parent)
- **Dimensions**: {64, 128, 256} (same as parent)
- **Total conditions**: 18 (3d x 6 SNR)

### 3.2 Key Measurements

For each condition, measure:
1. Per-domain and per-trial r_99/r_95 ratios (distribution, not just mean)
2. Fraction of domains with R > T for each T in {1.5, 2.0, 2.5, 3.0}
3. compound_r95 accuracy at each T
4. Whether T=2.0 and T=2.5 ever DISAGREE (different classification for any domain)

### 3.3 Analysis: Transition Sharpness

Fit the R(SNR) curve to determine:
- The SNR at which R crosses T=2.0 (call it SNR_cross)
- Whether the crossing is sharp (phase transition) or gradual (smooth sigmoid)
- Width of the transition zone (SNR range where 10% < f_T < 90%)

## 4. Kill Criteria

**K1: Accuracy drop in transition zone.**
The compound heuristic accuracy at SNR=7 should interpolate smoothly between
SNR=5 (80-87%) and SNR=10 (100%). If it drops >10pp below the linear
interpolation, the transition zone contains a failure mode.

Expected interpolation at SNR=7: 80% + (100%-80%) * (7-5)/(10-5) = 88%.
Kill if observed < 78%.

**K2: T=2.0 vs T=2.5 disagreement.**
At the parent's tested SNR values, T=2.0 and T=2.5 always agree. If in the
transition zone they disagree on >20% of domains, it means the threshold
choice matters and T=2.0 may not be robust.

## 5. Worked Example

d=128, true_rank=8, SNR=7:

1. Signal: S = U_8 @ diag(sigmas) @ V_8^T, ||S||_F ~ 3.0
2. Noise: ||N||_F = 3.0/7 = 0.429
3. E_signal/E_total = 9.0 / (9.0 + 0.184) = 0.980
4. r_95: 95% threshold < 98.0%, so r_95 ~ 8 (signal alone suffices)
5. r_99: 99% threshold > 98.0%, deficit = 1%, need ~1.28 noise dims
   -> r_99 ~ 9-10
6. R = 10/8 = 1.25 < 2.0 -> NO FALLBACK TRIGGER
7. compound_r95 = r_99 = 10
8. Optimal (Kneedle) ~ 8
9. Ratio: 10/8 = 1.25 (within 2x, correct)

At SNR=6:
- E_signal/E_total = 36/(36+1) = 0.973
- r_99 deficit = 1.7%, need ~2.18 noise dims -> r_99 ~ 10-11
- R = 11/8 = 1.375 -> still < 2.0, but closer to threshold
- Expect some domains (low true_rank, high noise realization) to cross T=2.0

## 6. Assumptions

1. All assumptions from parent experiments
2. The transition from noise-dominated to clean-signal is monotone in SNR
   (no reentrant behavior where R increases then decreases)
3. The theoretical R(SNR) curve applies qualitatively to both exact-rank
   and spectral-decay domain types
4. 5 seeds per domain are sufficient to estimate the per-domain median ratio
