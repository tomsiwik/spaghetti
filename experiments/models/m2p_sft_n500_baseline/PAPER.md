# PAPER: SFT n=500 Baseline Measurement — Statistical Closure for M2P-vs-SFT

**Experiment:** exp_m2p_sft_n500_baseline
**Model:** mlx-community/Qwen3-0.6B-4bit (Qwen3-0.6B, 4-bit quantized)
**Date:** 2026-04-07
**Runtime:** 891.9s (~14.9 min), n=500

---

## 1. Motivation

exp_m2p_qwen06b_gsm8k_v4 reported a quality_ratio CI_lower of 0.773, implying M2P
robustly outperforms SFT. That CI treated SFT accuracy (0.260, measured at n=200) as
a known constant. The delta method (Casella & Berger 2002, §5.5.4) shows this ignores
Var(SFT_acc) in the ratio denominator, producing an upward-biased CI_lower.

**The fix:** Remeasure SFT at n_sft=500. The delta-method Fieller CI then propagates
both Var(M2P_acc) and Var(SFT_acc), producing a calibrated, unbiased interval.

---

## 2. Prediction-vs-Measurement Table (MATH.md §D vs actual results)

| Quantity | MATH.md Prediction | Measured | Match? |
|---|---|---|---|
| SFT accuracy at n=500 | ~0.24–0.30 | **0.314** | Slightly above range |
| Wilson CI width at n=500 | ±0.039 (≈[0.22, 0.30]) | [0.275, 0.356] (width=0.081) | Wider — SFT higher than predicted |
| Two-prop z-test p-value | ~0.35 (not significant) | **0.334** | Yes — not significant |
| Corrected CI_lower (Fieller) | ~0.30 (if SFT=0.260) | **0.315** | Close; SFT=0.314 shifts it up |
| CI_lower bias (v4 – Fieller) | ~0.47 (if SFT=0.26) | **0.092** | No — bias much smaller |

**Key discrepancy:** The large predicted bias (0.471) assumed SFT=0.260 (same as n=200
point estimate). The actual SFT at n=500 came in higher at 31.4%, making the
denominator delta = 0.114 instead of 0.060. A larger denominator shrinks Term2
dramatically, so the actual bias is 0.092 rather than 0.471.

---

## 3. Statistical Results Table

### 3a. Model Accuracy Comparison

| Model | Accuracy | Correct/Total | Wilson 95% CI |
|---|---|---|---|
| Base (Qwen3-0.6B-4bit, no LoRA) | 0.200 | 40/200 (fixed from v2) | — |
| SFT (v2 adapter, n=200) | 0.260 | 52/200 (from v2) | [0.204, 0.323] |
| SFT (v2 adapter, n=500) | **0.314** | 157/500 | **[0.275, 0.356]** |
| M2P (v4, n=500) | 0.286 | 143/500 (fixed from v4) | [0.248, 0.327] |

SFT at n=500 is **above** the earlier n=200 point estimate of 26.0%. The n=200
estimate had wide uncertainty ([0.204, 0.323]); 31.4% is well within that prior CI.

### 3b. quality_ratio with Old vs New CI Method

| Method | quality_ratio | CI_lower | CI_upper | Interpretation |
|---|---|---|---|---|
| v4 method (SFT as constant, n=200 point est.) | 1.433 | **0.773** | — | "M2P robustly beats SFT" |
| v4 method (SFT as constant, actual 31.4%) | 0.754 | 0.407 | 1.102 | Gap narrowed |
| Fieller/delta (both uncertainties, n=500) | 0.754 | **0.315** | 1.194 | M2P marginally below SFT |

The quality_ratio dropped from 1.433 to 0.754 because SFT accuracy at n=500 (31.4%)
is higher than the n=200 estimate (26.0%) that was used in v4.

The Fieller CI_lower is 0.315, meaning the 95% CI for (M2P - base)/(SFT - base)
includes values from 0.315 to 1.194. Quality ratio < 1 means SFT outperforms M2P.

### 3c. Two-Proportion Z-Test (K920)

| Statistic | Value |
|---|---|
| M2P accuracy | 0.286 (n=500) |
| SFT accuracy | 0.314 (n=500) |
| Pooled proportion | 0.300 |
| z-statistic | -0.966 |
| p-value (two-tailed) | 0.334 |
| Decision | Not significant (p >= 0.05) |

The z-statistic is **negative** — SFT (31.4%) outperforms M2P (28.6%) at n=500.
The gap is 2.8pp in SFT's favor, but not statistically significant (p=0.334).

---

## 4. What Changed: Did CI_lower Drop from 0.773?

Yes, CI_lower dropped substantially from 0.773, but for a different reason than
predicted:

**Predicted mechanism:** Var(SFT_acc) term dominates → se_total >> se_v4 → big CI.
**Actual mechanism:** SFT accuracy at n=500 came in higher (31.4% vs 26.0%), making
delta = 0.114 instead of 0.060. The larger denominator shrinks both Term2 and the
quality_ratio itself.

Variance breakdown:
- Term1 (M2P noise): 0.031426
- Term2 (SFT noise): 0.018865
- Term2 as fraction of total: 37.5%

The Fieller correction still added a meaningful 37.5% more variance than the v4 method
would compute for the same SFT point estimate, but the dominant effect was SFT accuracy
being higher than assumed.

**The v4 optimism had two sources:**
1. Using n=200 SFT point estimate (26.0%) which underestimated true SFT accuracy
2. Ignoring Var(SFT_acc) in the ratio CI (Fieller correction)

---

## 5. Kill Criteria Verdicts

### K919: SFT accuracy at n=500 measured with Wilson 95% CI

**PASS.**
SFT accuracy = 0.314 (157/500), Wilson 95% CI = [0.275, 0.356].
Measurement made at n=500 with calibrated interval. K919 is unconditional.

### K920: Two-proportion z-test M2P(28.6%, n=500) vs SFT(31.4%, n=500)

**INCONCLUSIVE** (p = 0.334 >= 0.05).

z = -0.966, p = 0.334. The gap between M2P and SFT is not statistically significant.
Note: the gap is in SFT's favor (SFT higher by 2.8pp), not M2P's. The prior v4 claim
that "M2P significantly outperforms SFT" is not supported at n=500.

K920 definition: PASS if p < 0.05, INCONCLUSIVE if p >= 0.05.
Result: INCONCLUSIVE.

### K921: quality_ratio CI lower bound recalculated with Fieller/delta method

**PASS.**
Fieller CI_lower = 0.315 > 0 (K921 would only KILL if CI_lower < 0).
CI = [0.315, 1.194]. The interval includes 1.0, meaning M2P and SFT are statistically
indistinguishable. The v4-reported optimistic CI_lower=0.773 has been replaced with a
calibrated 0.315.

---

## 6. Implications

The finding from v4 (Finding #378) claimed "M2P beats SFT" with quality_ratio=1.433
and CI_lower=0.773. This experiment shows:

1. SFT at n=500 achieves 31.4%, higher than the n=200 estimate of 26.0%
2. quality_ratio is now 0.754 (SFT is actually ahead at the point estimate level)
3. Fieller CI = [0.315, 1.194] — includes 1.0, no significant difference
4. The prior CI_lower=0.773 was biased upward by ~0.46 (combining both error sources)

**Conclusion:** M2P and SFT at n=500 are statistically indistinguishable on GSM8K
with Qwen3-0.6B-4bit. Neither system significantly outperforms the other (p=0.334).
Both improve substantially over base (20.0%).

---

## 7. References

- Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *JASA* 22, 209-212.
- Newcombe, R. G. (1998). "Two-sided confidence intervals for the single proportion: comparison of seven methods." *Statistics in Medicine* 17, 857-872.
- Casella, G. & Berger, R. L. (2002). *Statistical Inference* (2nd ed.), §5.5.4 (delta method).
- Brown, L. D., Cai, T. T., & DasGupta, A. (2001). "Interval estimation for a binomial proportion." *Statistical Science* 16, 101-133.
- Cobbe et al. (arXiv:2110.14168) — GSM8K benchmark.
