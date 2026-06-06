# Decorrelation Filter Scaling: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Embedding dimension | {64, 128, 256, 512} |
| r | LoRA rank | 8 |
| N | Number of experts | 6 |
| L | Number of layers | 2 |
| d_ff | FFN hidden dimension | {256, 512, 512, 1024} |
| A_i | Frozen A-matrix for expert i (from AP or random-orthonormal) | (d, r) |
| B_i | Trained B-matrix for expert i | (r, d_ff) |
| delta_i | Full delta vector: vec(A_i @ B_i) across all layers | (D_delta,) |
| b_i | Concatenated B-matrix vector | (D_b,) |
| D_delta | Total delta dimensionality: L * (d*d_ff + d_ff*d) | varies |
| D_b | Total B dimensionality: L * (r*d_ff + r*d) | varies |

## 2. The Decorrelation Filter

### 2.1 Definition

The decorrelation filter ratio is defined as:

    F(d) = E[|cos(delta_i^{trained}, delta_j^{trained})|] / E[|cos(delta_i^{random}, delta_j^{random})|]

where:
- delta_i^{trained} = vec(A_i @ B_i) with A_i from AP skeleton, B_i trained
- delta_i^{random} = random vectors of same dimensionality D_delta

F(d) < 1 means the trained deltas are MORE orthogonal than random (decorrelation
filter active). F(d) > 1 means trained deltas are LESS orthogonal than random.

### 2.2 Why We Expected F(d) to Decrease

The hypothesis was that F(d) decreases with d because:

1. The AP skeleton provides A_i^T A_j ~ 0 by Grassmannian packing
2. The full delta cosine decomposes as:
   cos(delta_i, delta_j) ~ sum_l [B_i^{(l)T} (A_i^{(l)T} A_j^{(l)}) B_j^{(l)}] / norms
3. As d increases, A_i^T A_j should decrease (more room for packing)
4. Therefore the B-matrix correlation should be more strongly suppressed

### 2.3 Why F(d) Actually Increases

The experimental results show F(d) ~ d^{+0.603}, meaning the filter WEAKENS with d.

The mechanism: at larger d, random vectors have stronger "natural" orthogonality
(cos ~ 1/sqrt(D_delta) ~ d^{-0.77}). This baseline shrinks rapidly. Meanwhile,
trained delta cosines barely decrease with d (d^{-0.11}). The ratio F(d) = slow/fast
increases because the denominator shrinks faster than the numerator.

### 2.4 Decomposition of Delta Cosine

The trained delta cosine has two components:

    |cos(delta_i^{trained}, delta_j^{trained})| = f(A_i^T A_j, B_i^T B_j)

The B-matrix correlation (b_corr_ratio ~ 4.4x) is roughly constant across d,
meaning B-matrices develop a FIXED amount of training-induced overlap regardless
of dimension. The A-matrix decorrelation (A_i^T A_j ~ 0) suppresses this, but
only to a fixed MULTIPLICATIVE factor, not one that improves with d.

Key insight: at d=64 with Nr=48 ~ d, the AP packing is "tight" (close to capacity).
At d=512 with Nr=48 << d, the packing is "loose" -- A_i^T A_j is already near zero
even without AP, so AP provides diminishing marginal benefit.

## 3. Scaling Laws (Empirical)

### 3.1 Power Law Fits

| Metric | Exponent (beta) | R^2 |
|--------|----------------|-----|
| Filter ratio F(d) | +0.603 | 0.874 |
| AP delta cos | -0.108 | 0.094 |
| Random delta cos | -0.770 | 0.974 |
| AP B-matrix cos | -0.267 | 0.974 |
| Random B-matrix cos | -0.259 | 0.968 |

### 3.2 The Critical Asymmetry

Random delta cos scales as d^{-0.77} (close to expected 1/sqrt(D_delta) since
D_delta ~ d^2, giving d^{-1}, but with only 2 layers the effective dimensionality
grows sub-quadratically due to the d_ff scaling choices).

Trained delta cos scales as d^{-0.11}, essentially FLAT. This means the
B-matrix training correlation creates a floor on delta cosine that does not
decrease with dimension.

### 3.3 Implication

The decorrelation filter is not dimension-dependent. It provides a fixed
suppression factor that happens to be below 1.0 at d=64 (where random baselines
are relatively high) but above 1.0 at d >= 256 (where random baselines are very
low and the trained-B correlation floor dominates).

## 4. Worked Example at d=256

- D_delta = 2 * (256*512 + 512*256) = 524,288
- Expected random |cos| ~ 1/sqrt(D_delta) = 0.00138 (measured: 0.00111)
- Trained AP delta |cos| measured: 0.00205
- Filter ratio: 0.00205 / 0.00111 = 1.86
- B-matrix |cos|: 0.031 (trained) vs 0.0076 (random) = 4.1x ratio
- The B-matrix correlation (4.1x) is NOT fully suppressed by the A-matrix
  orthogonality, because at d=256 with Nr=48, A_i^T A_j is already very small
  even for random A matrices -- AP provides little marginal benefit.

## 5. Relationship to Parent Experiments

### 5.1 b_matrix_training_correlation (d=64)

That experiment reported filter ratio 0.14x at d=64. This experiment finds 0.64x.
The discrepancy is due to:
- Different N (8 vs 6)
- Different domain structure (specific pairs vs sweep)
- Different seeds
- High variance at d=64 (our per-seed range: 0.33 to 0.88)

Both are < 1.0 at d=64, confirming the decorrelation filter IS active at small d.

### 5.2 minimum_viable_base (all d)

That experiment found LoRA/random cos ratio ~ 1.0 across all d, but with
UNTRAINED synthetic adapters. This experiment uses TRAINED adapters and finds
ratios increasing from 0.64 to 1.93 -- confirming that training-induced
B-matrix correlation is the mechanism that breaks the dimensionality-only story.

### 5.3 structural_orthogonality_characterization

That experiment found gradient cos ~ d^{-0.72}, random cos ~ d^{-0.94}.
The gradient/random ratio was 2.8x to 5.5x, INCREASING with d -- qualitatively
consistent with our finding that trained/random ratio increases with d.
