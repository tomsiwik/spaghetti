# Information Preservation Predictor: Mathematical Foundations

## Notation

| Symbol | Shape / Type | Description |
|--------|-------------|-------------|
| tau_k | (D,) | Flattened LoRA task vector for domain k |
| D | scalar | Total delta parameters across all layers |
| N | scalar | Number of domains |
| m | symbol | Merging method identifier |
| tau_m | (D,) | Merged delta produced by method m |
| tau_avg | (D,) | Simple average: (1/N) sum_k tau_k |
| L(m) | scalar | Validation loss of method m |
| IP(m) | scalar | Information preservation score of method m |

## Information Preservation Metrics

### Metric 1: IP vs Average

The simple average tau_avg = (1/N) sum_k tau_k is the unique lossless
linear merge -- it is the best rank-1 approximation of the mean function
across domains. Any method that produces exactly tau_avg preserves maximal
information under the linear model.

    IP_avg(m) = 1 - ||tau_m - tau_avg||_F / ||tau_avg||_F

**Range**: IP_avg = 1.0 for simple average (by construction), can be
negative if the merged delta is further from the average than the zero vector.

**Limitation**: This metric measures fidelity to the linear average, not
to the original per-domain deltas. It rewards simple averaging by definition.

### Metric 2: IP vs Originals

Measures how well the merged delta represents each original domain delta:

    IP_orig(m) = 1 - sqrt( sum_k ||tau_m - tau_k||_F^2 / sum_k ||tau_k||_F^2 )

**Range**: IP_orig < 1.0 for any single merged delta (it cannot match all
originals simultaneously unless they are identical). IP_orig = 1.0 only for
methods that keep all deltas separate (concat+calibrate).

**Decomposition**: For simple average tau_avg, we have
tau_m - tau_k = tau_avg - tau_k = -(tau_k - tau_avg), so
sum_k ||tau_avg - tau_k||^2 = sum_k ||tau_k||^2 - N ||tau_avg||^2
(by the parallel axis theorem). Thus:

    IP_orig(avg) = 1 - sqrt(1 - N ||tau_avg||^2 / sum_k ||tau_k||^2)

When deltas are orthogonal, ||tau_avg||^2 = (1/N^2) sum_k ||tau_k||^2, so:

    IP_orig(avg) = 1 - sqrt(1 - 1/N)

For N=2: IP_orig(avg) = 1 - sqrt(1/2) ~ 0.293. For N=5: IP_orig(avg) = 1 - sqrt(4/5) ~ 0.106.

**Prediction**: IP_orig of simple average should decrease with N. Confirmed
empirically: 0.296 at N=2, 0.118 at N=5.

### Metric 3: Norm Ratio

    NR(m) = ||tau_m||_F / mean_k(||tau_k||_F)

**Range**: NR > 0. NR = 1.0 means the merged delta has the same magnitude
as the average original delta. NR < 1 means signal attenuation; NR > 1 means
signal amplification (noise from rescaling).

**Why it predicts quality**: Methods that amplify (DARE at high p, DARE-TIES)
introduce noise proportional to the rescaling factor 1/(1-p). The norm ratio
directly captures this amplification.

For simple average of orthogonal deltas:
||tau_avg|| = ||(1/N) sum_k tau_k|| = (1/N) sqrt(sum_k ||tau_k||^2)
mean_k ||tau_k|| ~ sqrt(sum_k ||tau_k||^2 / N) (approximately, for equal norms)
So NR(avg) ~ 1/sqrt(N). For N=2: ~0.707. For N=5: ~0.447.

**Prediction**: Confirmed empirically: NR(avg) = 0.715 at N=2, 0.474 at N=5.

## Spearman Rank Correlation

Let r_Q(m) = quality rank of method m (1 = best = lowest val loss) and
r_IP(m) = information preservation rank (1 = best = highest IP).

Spearman's rho:

    rho = 1 - 6 sum_i d_i^2 / (n(n^2 - 1))

where d_i = r_Q(m_i) - r_IP(m_i) and n is the number of methods.

For perfect rank agreement: rho = 1.0.
Kill threshold: rho < 0.8.

## Why IP Rank Fails at Fine Resolution

The hypothesis that IP predicts quality is correct at coarse granularity
(top tier vs bottom tier) but fails at fine ranking resolution. The reason:

1. **Same-tier methods differ by < 0.5% in quality** but can have very
   different IP scores. Example: simple_avg (IP_avg = 1.0) and dare_p0.3
   (IP_avg = 0.35) differ by only 0.05% in quality.

2. **Concat+calibrate breaks the monotonicity**: it has IP = 1.0 (perfect
   preservation) but ranks 4th at N=5 due to router optimization noise.
   Information preservation is necessary but not sufficient -- the router
   introduces its own error.

3. **TIES has higher IP than dare_p0.7** (IP_avg ~ 0.03 vs -0.52) but
   worse quality (6.85% vs 2.33%). TIES preserves more of the average
   direction but introduces correlated errors through sign election,
   while DARE's uncorrelated noise averages out better.

## Worked Example (d=64, r=8, N=2)

Two domain deltas tau_A, tau_B of dimension D = 131,072 with cos(tau_A, tau_B) ~ 0.014.

Simple average: tau_avg = (tau_A + tau_B) / 2
- ||tau_avg||/mean(||tau_A||, ||tau_B||) = 1/sqrt(2) ~ 0.707  (orthogonal)
- IP_orig = 1 - sqrt(1 - 1/2) = 1 - sqrt(0.5) ~ 0.293

DARE p=0.9: tau_dare = mean of rescaled sparse copies
- Rescale factor: 10x. Random 10% survive.
- Expected ||tau_dare|| = ||tau_avg|| (unbiased), but Var is high
- Actual NR ~ 2.25 (massive noise amplification)
- IP_avg ~ -2.0 (merged is 3x further from average than zero is)

The IP metrics correctly predict that DARE p=0.9 is much worse than
simple average, and that the gap is due to noise amplification.
