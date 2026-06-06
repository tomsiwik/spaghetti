# Minimum Viable Base Dimension: Mathematical Foundations

*Revised 2026-03-16 per adversarial review.*

## Setup

We model expert composition quality as a function of base model embedding dimension d,
using all-modules LoRA (q/k/v/o/gate/up/down) at rank r=16.

### Notation

| Symbol | Definition | Shape/Type |
|--------|-----------|------------|
| d | Model embedding dimension | scalar |
| r | LoRA rank | scalar (=16) |
| d_kv | KV projection dimension = n_kv_heads * head_dim | scalar |
| d_ff | FFN intermediate dimension | scalar |
| L | Number of transformer layers | scalar |
| N | Number of experts | scalar |
| M | Number of LoRA target modules per layer (=7) | scalar |
| A_m | Frozen Stiefel frame for module m | (d_in_m, r) |
| B_m | Trained projection for module m | (r, d_out_m) |
| Delta_m | LoRA delta for module m: (1/r) A_m B_m | (d_in_m, d_out_m) |
| delta_i | Flattened concatenation of all module deltas for expert i | (D_flat,) |
| D_flat | Total flattened dimension = L * sum_m(d_in_m * d_out_m) | scalar |
| tau | Interference threshold | scalar (=0.01) |

### Module Dimensions

For Qwen2.5-family architecture with GQA:

| Module | Input dim | Output dim | Parameters per layer |
|--------|-----------|------------|---------------------|
| q_proj | d | d | d^2 |
| k_proj | d | d_kv | d * d_kv |
| v_proj | d | d_kv | d * d_kv |
| o_proj | d | d | d^2 |
| gate_proj | d | d_ff | d * d_ff |
| up_proj | d | d_ff | d * d_ff |
| down_proj | d_ff | d | d_ff * d |

Total per layer:
$$D_{layer} = 2d^2 + 2d \cdot d_{kv} + 3d \cdot d_{ff}$$

Total flattened dimension:
$$D_{flat} = L \cdot D_{layer}$$

### Concrete Values

| Model | d | d_kv | d_ff | D_layer | D_flat (L=2) |
|-------|---|------|------|---------|-------------|
| micro-64 | 64 | 32 | 256 | 61,440 | 122,880 |
| micro-256 | 256 | 64 | 1024 | 950,272 | 1,900,544 |
| Qwen2.5-0.5B | 896 | 128 | 4864 | 14,909,440 | 29,818,880 |
| Qwen2.5-7B | 3584 | 512 | 18944 | 233,046,016 | 466,092,032 |

## Theorem: Cosine Scaling with Dimensionality

### Statement

For N vectors in R^{D_flat} with independent entries, the expected pairwise cosine
scales as:

$$\mathbb{E}[|\cos(\delta_i, \delta_j)|] \approx \frac{1}{\sqrt{D_{flat}}}$$

This holds regardless of whether the vectors have LoRA structure or are purely random.

### Proof Sketch

The result follows directly from concentration of measure in high dimensions.
For two independent vectors u, v in R^D with i.i.d. entries having mean zero and
finite variance:

$$\cos(u, v) = \frac{\sum_{k=1}^{D} u_k v_k}{\|u\| \cdot \|v\|}$$

The numerator is a sum of D independent mean-zero terms, so by CLT it concentrates
with standard deviation O(1/sqrt(D)). The denominator concentrates around its
expectation. The ratio gives |cos| ~ 1/sqrt(D).

**This is the same mechanism that drives Johnson-Lindenstrauss random projections
and the "blessing of dimensionality" in high-dimensional geometry.**

For LoRA-structured deltas specifically: each flattened delta is a concatenation
of M*L sub-vectors, where each sub-vector is vec((1/r) A_m B_m). With independent
Stiefel A and random B, these sub-vectors are effectively random in their
respective subspaces. The concatenation produces a vector in R^{D_flat} that
behaves like a random vector for cosine purposes.

### Empirical Confirmation: LoRA vs Random

| d | D_flat | LoRA |cos| | Random |cos| | Ratio |
|---|--------|------------|-------------|-------|
| 64 | 122,880 | 0.002141 | 0.002257 | 0.949 |
| 128 | 491,520 | 0.001037 | 0.001121 | 0.925 |
| 256 | 1,900,544 | 0.000562 | 0.000498 | 1.128 |
| 512 | 7,602,176 | 0.000296 | 0.000298 | 0.991 |
| 896 | 29,818,880 | 0.000145 | 0.000142 | 1.023 |

**Ratios range from 0.925 to 1.128 (mean ~ 1.0).** LoRA structure does not
meaningfully alter the cosine distribution. The orthogonality guarantee is a
consequence of dimensionality alone.

This simplifies the mathematical framework: we do not need to analyze Stiefel
frames, domain-biased B matrices, or any adapter-specific structure. The only
quantity that matters is D_flat.

### Comparison with Prior Work

The structural_orthogonality_proof experiment found beta=-0.673 for cos vs d
(FFN-only adapters). This experiment finds beta=-1.049 (all-modules). The
discrepancy is explained by different D_flat scaling:

- FFN-only: D_flat = L * 3*d*d_ff ~ 12*L*d^2, so |cos| ~ 1/sqrt(12Ld^2) ~ d^{-1}
- All-modules: D_flat = L * (2d^2 + 2d*d_kv + 3d*d_ff) ~ 14*L*d^2, so |cos| ~ d^{-1}

Both should give beta ~ -1.0 for cos vs d when D_flat ~ d^2. The FFN-only beta=-0.673
likely reflects a different experimental setup (e.g., different module counts,
different d_ff/d ratios at small d, or single-layer experiments). The all-modules
beta=-1.049 is closer to the theoretical prediction.

### Observed Scaling

Empirically:
- |cos| vs d: beta = -1.049 (R^2 = 0.9947), nearly exactly 1/d
- |cos| vs D_flat: beta = -0.506 (R^2 = 0.9972), nearly exactly 1/sqrt(D_flat)

This confirms: cosine scales as 1/sqrt(D_flat), and since D_flat ~ d^2, we get
|cos| ~ d^{-1}.

## Ratio to Classical sqrt(r/d) Bound

The classical random subspace bound predicts |cos| ~ sqrt(r/d) = sqrt(16/d) for
rank-16 adapters in R^d. Our empirical ratio to this bound:

| d | Empirical |cos| | sqrt(r/d) | Ratio |
|---|-----------|----------|-------|
| 64 | 0.00214 | 0.500 | 0.004 |
| 256 | 0.00056 | 0.250 | 0.002 |
| 896 | 0.00015 | 0.134 | 0.001 |
| 3584 | 0.00003 | 0.067 | 0.001 |

The empirical cosines are 250-1000x below the sqrt(r/d) bound. **This comparison
is misleading** (as noted by the adversarial reviewer): sqrt(r/d) bounds subspaces
in R^d (single module), but our deltas live in R^{D_flat} where D_flat >> d.
The relevant bound is sqrt(r/D_flat), not sqrt(r/d). The large ratio is an artifact
of the dimensional mismatch, not evidence of beating theory.

## N_max Estimation (Qualified)

The analytical N_max estimate uses a Gaussian tail extrapolation:

$$z = \frac{\tau - \mu_{cos}}{\sigma_{cos}}, \quad N_{max} \approx \min(e^{z^2/4}, 100000)$$

where mu_cos and sigma_cos are measured from N=16 experts.

**This formula is NOT VALIDATED.** At d=256:
- Empirical binary search caps at N=128 (all pass)
- Manual test at N=2048: max|cos| = 0.004, still well below tau=0.01
- Analytical formula predicts N_max ~ 485 million (clipped to 100K)

The formula cannot be validated because the true N_max is too large to measure
empirically in tractable time. **All analytical N_max estimates (d >= 512 in the
results table) should be interpreted as "exceeds all testable values" rather
than precise numbers.**

What IS empirically verified:
- d=64: N_max = 16 (empirical, matches theory d^2/r^2 = 16)
- d=128: N_max >= 64 (empirical cap)
- d=256: N_max > 2048 (verified, theory predicts 256)
- d >= 512: N_max >> 128 (all binary search caps hit)

## Implications for Minimum Viable Base

Since |cos| < tau = 0.01 holds even at d=64 (max|cos| = 0.007), there is **no
minimum base dimension** from a composition-interference standpoint. The bottleneck
is elsewhere:

1. **Model capacity**: Smaller bases have less attention capacity and poorer
   embeddings, limiting what experts can specialize. The 0.5B base was killed
   in prior experiments not because of interference but because of insufficient
   base quality.

2. **N_max scaling**: While interference is negligible, the practical capacity
   is bounded by d^2/r^2 in the classical theory. At d=64, theory gives N_max=16
   (tight). At d=896, theory gives N_max=3136 (ample). Empirically, N_max far
   exceeds these theoretical bounds at all tested dimensions.

## Worked Example: d=896 (Qwen2.5-0.5B)

Architecture: 14 heads, 2 KV heads, head_dim=64, d_ff=4864, 24 layers.

Per-layer parameter count per LoRA module:
- q_proj: 896 * 896 = 802,816
- k_proj: 896 * 128 = 114,688
- v_proj: 896 * 128 = 114,688
- o_proj: 896 * 896 = 802,816
- gate_proj: 896 * 4864 = 4,358,144
- up_proj: 896 * 4864 = 4,358,144
- down_proj: 4864 * 896 = 4,358,144
- **Total per layer: 14,909,440**

For L=2 experiment layers: D_flat = 29,818,880

Expected |cos| ~ 1/sqrt(29,818,880) = 0.000183
Observed |cos| (LoRA) = 0.000145 (0.79x of prediction)
Observed |cos| (random baseline) = 0.000142 (0.78x of prediction)

Both LoRA and random are within 3% of each other and both below the theoretical
prediction. The small difference from 1/sqrt(D_flat) is consistent with
finite-sample variance.

At N=8 experts:
- max|cos| = 0.000434 (23x below tau=0.01)
- Signal retention = 0.9999 (essentially perfect)
- Effective rank ratio = 1.0000 (all experts fully independent)

**Conclusion**: Qwen2.5-0.5B has ample room for expert composition. The question
is not "can it support composition?" (yes, trivially, by concentration of measure)
but "does the base model quality support useful expert specialization?" (a question
that requires real training, not geometry).
