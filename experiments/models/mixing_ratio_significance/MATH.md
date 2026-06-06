# Mixing Ratio Significance: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Value |
|--------|-----------|-------|
| d | Model embedding dimension | 64 |
| r | LoRA rank | 8 |
| N | Training examples per expert | 1000 |
| N_eval | Evaluation examples | 500 |
| S | Number of seeds | 20 |
| T | SGD steps | 500 |
| alpha | Synthetic mixing fraction | [0.00, 0.05, ..., 1.00] |
| W* | Ground-truth task matrix | R^{d x d}, rank r |
| A | LoRA input projection (frozen) | R^{r x d} |
| B | LoRA output projection (learned) | R^{d x r} |

All notation, data generation, and LoRA training match the parent experiment
(micro/models/synthetic_vs_real_data/MATH.md).

## 2. Statistical Testing Framework

### 2.1 Paired Design

For each seed s in {1,...,S}, we compute:

    Q_s(alpha) = quality of LoRA trained on alpha-fraction synthetic data

The paired difference for testing ratio alpha vs baseline:

    D_s(alpha) = Q_s(alpha) - Q_s(0.0)

This is a paired design because each seed shares the same W*, A initialization,
mode centers, and evaluation data. The pairing controls for initialization
variance, which is the dominant noise source.

### 2.2 Wilcoxon Signed-Rank Test

We use Wilcoxon signed-rank rather than paired t-test because:
1. n=20 is small (t-test assumes normality, which may not hold)
2. The quality metric is bounded in [0,1] (may be skewed)
3. Wilcoxon is robust to outliers

For differences D_1,...,D_S:
1. Rank |D_i| from smallest to largest
2. W+ = sum of ranks where D_i > 0
3. W- = sum of ranks where D_i < 0
4. Test statistic W = min(W+, W-)
5. p-value from the null distribution of W

H0: The distribution of D_s(alpha) is symmetric around 0.
H1 (two-sided): The distribution is not symmetric around 0.

### 2.3 Effect Size

Cohen's d for paired samples:

    d_cohen = mean(D) / std(D, ddof=1)

Interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

### 2.4 Bootstrap CI for Optimal Ratio

For the optimal ratio alpha*, we use nonparametric bootstrap:
1. Resample S seeds with replacement, B=10000 times
2. For each resample b, compute mean quality across resampled seeds at each ratio
3. Find alpha*_b = argmax_{alpha in (0,1)} mean_quality_b(alpha)
4. Report 2.5th and 97.5th percentiles of {alpha*_b} as 95% CI

This directly quantifies: "if we ran the experiment again with different seeds,
how much would the optimal ratio move?"

## 3. Power Analysis

### 3.1 Detectable Effect Size

With n=20 paired samples and alpha=0.05 (two-sided), the Wilcoxon signed-rank
test has approximately 80% power to detect effects of d_cohen >= 0.65 (medium-large).

The parent experiment reported +11.2% improvement, which corresponded to:
    mean_diff = 0.0670 - 0.0581 = 0.0089
    pooled_std ~ 0.014 (from parent's 5-seed data)
    apparent d_cohen ~ 0.0089 / 0.014 ~ 0.64

So with 20 seeds, we have approximately 80% power to detect the parent's
claimed effect if it were real. The fact that we observe d_cohen = -0.057
(wrong sign) with p=0.57 strongly suggests the parent's +11.2% was noise.

### 3.2 Why the Parent Found a Significant-Looking Effect

The parent experiment used 5 seeds with unpaired comparison. With 5 seeds:
- Standard error of mean ~ std / sqrt(5) ~ 0.014 / 2.24 ~ 0.006
- The observed difference 0.0089 is 1.4 standard errors
- This looks suggestive in a table but is far from significant
- The parent's CIs overlapped (noted by adversarial reviewer)
- 5 seeds provides approximately 35% power at d=0.64 -- coin-flip detection

## 4. Assumptions

1. **Independence across seeds.** Each seed produces an independent realization
   of W*, A, mode centers, and training data. Seeds are consecutive integers
   (42-61), which is standard for reproducibility.

2. **Exchangeability.** The Wilcoxon test assumes differences are exchangeable
   under H0. This holds because each seed creates an identically-structured
   problem with different random realizations.

3. **Same experimental setup as parent.** All hyperparameters (d=64, r=8,
   N=1000, 500 steps, lr=0.01, batch=64) match the parent exactly. The only
   change is: 20 seeds instead of 5, and 0.05-step alpha grid instead of 0.10.

## 5. Worked Example

Seed 42:
- Q(0.0) = 0.0496 (pure real)
- Q(0.2) = 0.0511 (20% synthetic)
- D = +0.0015 (tiny positive difference)

Seed 56:
- Q(0.0) = 0.0933 (pure real -- this seed got lucky)
- Q(0.2) = 0.0598 (20% synthetic)
- D = -0.0335 (large negative -- mixing HURT this seed)

Across 20 seeds: 8 positive, 12 negative differences.
Mean D = -0.00084. The effect is indistinguishable from zero.
