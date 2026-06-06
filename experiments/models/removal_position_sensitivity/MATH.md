# Expert Removal Position Sensitivity: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 128, 256 (micro) |
| r | LoRA rank | 8 |
| N | Number of expert adapters | 50 |
| L | Number of transformer layers | 24 |
| k | Index of expert to remove (GS order) | 0 <= k < N |
| delta_i | Flattened LoRA delta of expert i (pre-GS) | (d^2,) |
| delta_i' | GS-orthogonalized delta of expert i | (d^2,) |
| Delta | Merged composite: sum of all delta_i' | (d, d) |
| Delta_{-k} | Merged composite after naive subtraction of expert k | (d, d) |
| Delta_{-k}^{GT} | Ground truth: GS recompute from N-1 experts | (d, d) |
| epsilon_k | Weight-space error: ||Delta_{-k} - Delta_{-k}^{GT}|| / ||Delta_{-k}^{GT}|| | [0, 1] |
| D_k | Output deviation from removing expert k | [0, 1] |

## 2. Gram-Schmidt Order Dependence

### 2.1 The GS Process

Given N flattened delta vectors {delta_0, ..., delta_{N-1}}, classical GS produces:

    delta_0' = delta_0                                    (unchanged)
    delta_1' = delta_1 - proj(delta_1, delta_0')
    delta_2' = delta_2 - proj(delta_2, delta_0') - proj(delta_2, delta_1')
    ...
    delta_k' = delta_k - sum_{j<k} proj(delta_k, delta_j')

where proj(v, u) = (v . u / u . u) * u.

**Key property**: delta_k' depends on ALL predecessors {delta_0', ..., delta_{k-1}'},
but NO successors. This creates an asymmetry:

- **Expert 0**: delta_0' = delta_0. No projections removed. Removing expert 0
  invalidates all corrections applied to experts 1..N-1.

- **Expert N-1**: delta_{N-1}' has been projected against all predecessors.
  But NO other expert references delta_{N-1}' in its GS computation.
  Therefore, removing expert N-1 via naive subtraction is EXACT.

### 2.2 Theorem: Last Expert Removal Is Exact

**Claim**: For the last expert in GS order, naive subtraction equals GS recompute.

**Proof**: Let S = {delta_0, ..., delta_{N-1}} and S_{-k} = S \ {delta_k}.

For k = N-1:
- Naive subtraction: Delta_{-(N-1)} = sum_{i=0}^{N-2} delta_i' = sum_{i=0}^{N-2} delta_i'
- GS recompute from S_{-(N-1)}: The GS process on {delta_0, ..., delta_{N-2}}
  produces EXACTLY the same delta_0', ..., delta_{N-2}' because none of them
  referenced delta_{N-1}' during their computation.

Therefore Delta_{-(N-1)} = Delta_{-(N-1)}^{GT} and epsilon_{N-1} = 0. QED.

This is confirmed empirically: dev(last) = 1.7e-14% (machine epsilon).

### 2.3 First Expert Error Bound

For k = 0: Removing delta_0' invalidates the projection corrections in ALL
subsequent experts. Expert j's GS correction from expert 0 was:

    correction_j = proj(delta_j, delta_0') = (delta_j . delta_0' / ||delta_0'||^2) * delta_0'

The magnitude of this correction is proportional to cos(delta_j, delta_0').
At d=256, r=8, the mean pairwise cosine is ~0.003 (from parent experiment),
so each correction is O(0.003 * ||delta_j||).

After removing expert 0 and recomputing GS, expert j no longer projects against
delta_0'. But the direction that WAS delta_0' is now partially available to all
experts, leading to slightly different orthogonalized vectors.

The total weight-space error is:

    epsilon_0 = sum_{l=1}^{L} ||naive_l - gt_l|| / ||gt_l||

where the per-layer error involves the cascaded effect of all N-1 experts
having slightly different GS solutions.

### 2.4 Monotonic Decay with Position

For expert k (0 <= k < N-1), the number of experts that depend on delta_k' is (N-1-k).
The weight-space error should scale roughly as:

    epsilon_k ~ C * (N - 1 - k) * mean_cos

because (N-1-k) experts lose their projection correction onto delta_k'.

This predicts a linearly decreasing error with position, reaching exactly zero at
k = N-1. At k = 0, the error involves all N-1 corrections.

Empirically at d=256, N=50:

| Position k | N-1-k (affected) | Predicted trend | Measured dev% |
|-----------|------------------|-----------------|---------------|
| 0 (first) | 49 | highest | 0.1640 |
| 12 (Q1) | 37 | high | 0.1299 |
| 25 (middle) | 24 | medium | 0.0983 |
| 37 (Q3) | 12 | low | 0.0758 |
| 49 (last) | 0 | zero | 0.0000 |

The linear fit dev(k) = -0.003101*k + C has R^2 = 0.946, confirming the
approximately linear relationship.

## 3. GS Retention Analysis

The GS retention ratio measures how much of the original vector survives
orthogonalization:

    rho_k = ||delta_k'|| / ||delta_k||

At d=256, all retention ratios are >0.9996, meaning GS removes <0.04% of
each vector's magnitude. This is because the pairwise cosines are very small
(~0.003), so projections are tiny.

The retention decreases monotonically with position (later experts project
against more predecessors):

| Position | Retention | Deviation |
|----------|-----------|-----------|
| first | 1.0000 | 0.1640% |
| Q1 | 0.9999 | 0.1299% |
| middle | 0.9998 | 0.0983% |
| Q3 | 0.9997 | 0.0758% |
| last | 0.9996 | 0.0000% |

The regression GS_retention vs deviation has R^2 = 0.9385, showing that
retention (equivalently, position) explains most of the variance.

## 4. Position Sensitivity Ratio

### 4.1 Excluding the Degenerate Last Position

The last expert has exactly zero error (mathematical identity). Including it
makes the max/min ratio infinite, which is technically correct but misleading.

For the MEANINGFUL positions {first, Q1, middle, Q3} at d=256:

    ratio_{first/Q3} = 0.1640 / 0.0758 = 2.16x

This marginally exceeds the 2x threshold.

### 4.2 Amplification Ratio Stability

Despite the position-dependent weight-space error, the amplification ratio
(output deviation / sum of per-layer weight errors) is remarkably stable:

| Position | Amp Ratio |
|----------|-----------|
| first | 0.0200 |
| Q1 | 0.0190 |
| middle | 0.0204 |
| Q3 | 0.0201 |

CV = 2.9%. The amplification ratio is position-independent, confirming it
is an ARCHITECTURAL constant.

The position sensitivity is entirely in the weight-space error (sum_epsilon),
not in how errors propagate through the network.

### 4.3 Position Effect Scales as N-1-k

Since deviation is proportional to the number of affected experts:

    D_k / D_0 ~ (N - 1 - k) / (N - 1)

For k=0 (first): ratio = 1.0 (maximum)
For k=N//4: ratio ~ 0.75
For k=N//2: ratio ~ 0.50
For k=3N//4: ratio ~ 0.25
For k=N-1: ratio = 0.0

This predicts max/min ratio = infinity (trivially, due to last position) and
excluding last: max/Q3 ~ 49/12 = 4.08x (theoretical) vs 2.16x (empirical).
The empirical ratio is smaller than theoretical because the linear model
overestimates; the actual error involves second-order GS corrections.

## 5. Practical Implications

### 5.1 Random Permutation Amortization

If experts are assigned GS positions uniformly at random, the expected
deviation for any removed expert is:

    E[D_k] = (1/N) * sum_{k=0}^{N-1} D_k ~ D_0 * (1/N) * sum_{k=0}^{N-1} (N-1-k)/(N-1)
           = D_0 * (1/2)

The expected deviation under random permutation is HALF the worst case.

### 5.2 All Deviations Remain Small

Even the worst case (first position, d=256) gives 0.164%, which is:
- 6x below the 1% safety threshold
- Only 1.67x the middle position used by parent experiment

At production scale (d=896), extrapolating with d^(-1.17) scaling:
- First position: ~0.038%
- Middle position: ~0.023%
- With SOLE cosines (90x lower): ~0.0004% even for worst-case position

## 6. Worked Example

d=256, r=8, L=24, N=50, Pre-RMSNorm:

1. Generate 50 experts, GS-orthogonalize per layer
2. Remove expert at position k=0 (first, worst case):
   - Per-layer weight error: sum_eps = 8.25%
   - Amplification ratio: alpha = 0.020
   - Output deviation: D = 8.25% * 0.020 = 0.164%
3. Remove expert at position k=25 (middle, parent baseline):
   - Per-layer weight error: sum_eps = 4.82%
   - Amplification ratio: alpha = 0.020
   - Output deviation: D = 4.82% * 0.020 = 0.098%
4. Remove expert at position k=49 (last):
   - Per-layer weight error: sum_eps = 0.000%
   - Output deviation: D = 0.000% (exact)

Worst-to-middle ratio: 0.164/0.098 = 1.67x
Worst-to-mean (non-last): 0.164/0.117 = 1.40x

## 7. Assumptions

1. **Classical GS (not modified GS)**. Modified GS would have different
   stability properties but the same order-dependence structure.

2. **Random LoRA initialization**. With Grassmannian skeleton (frozen A),
   the absolute cosines would be even lower, making position effects smaller.

3. **Uniform layer treatment**. Each layer uses the same GS ordering.
   In practice, different layers could use different orderings (shuffled per
   layer), which would average out position effects.

4. **Pre-RMSNorm architecture**. The amplification ratio (0.020) is specific
   to Qwen/Llama-style. The position-dependence of weight-space error is
   architecture-independent.
