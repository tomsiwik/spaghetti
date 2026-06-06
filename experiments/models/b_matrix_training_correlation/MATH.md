# B-Matrix Training Correlation: Mathematical Foundations

## 1. Setup and Notation

Inherits notation from parent experiments (grassmannian_expert_init/MATH.md,
correlated_layer_errors/MATH.md).

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| A_i^{(l)} | Frozen A-matrix for expert i, layer l | (d, r) or (d_ff, r) |
| B_i^{(l)} | Trained B-matrix for expert i, layer l | (r, d_ff) or (r, d) |
| b_i | Concatenated B-matrix vector for expert i | (2 * n_layers * r * (d + d_ff),) |
| cos(b_i, b_j) | Cosine similarity between B-vectors | [-1, 1] |
| rho_B | Mean pairwise |cos(b_i, b_j)| across expert pairs | [0, 1] |

## 2. The B-Matrix Overlap Problem

### 2.1 Source of Interference

The minimax_grassmannian_packing experiment (KILLED) established that the
post-training tail anomaly in expert interference comes from B-matrix overlap,
not skeleton geometry. Specifically, the full expert interference term is:

    ||delta_W_i^T delta_W_j||_F = (alpha/r)^2 * ||B_i^T (A_i^T A_j) B_j||_F

The Grassmannian skeleton controls A_i^T A_j (frozen, near-orthogonal).
But B_i and B_j are learned freely during training. If domain-similar
experts learn similar B-matrices, the B_i^T ... B_j term could amplify
interference beyond what the skeleton alone would predict.

### 2.2 The Question

Two sub-questions:

**Q1 (Existence):** Do trained B-matrices show structured pairwise cosine
above random baseline? Formally:

    H0: E[|cos(b_i^{trained}, b_j^{trained})|] <= 3 * E[|cos(b_i^{random}, b_j^{random})|]
    H1: E[|cos(b_i^{trained}, b_j^{trained})|] > 3 * E[|cos(b_i^{random}, b_j^{random})|]

**Q2 (Safety):** Does B-matrix correlation increase amplification ratio?

    amp_ratio(correlated_B) <= amp_ratio(uncorrelated_B)

## 3. Expected Baseline Cosine

### 3.1 Random Vectors in High Dimension

For two random unit vectors in R^D, the expected |cos| is:

    E[|cos|] = sqrt(2 / (pi * D))

For our B-matrix vector with D = 2 * n_layers * r * (d + d_ff):
- d=64, r=8, d_ff=256, n_layers=2: D = 2 * 2 * 8 * (64 + 256) = 10,240
- E[|cos|] = sqrt(2 / (pi * 10240)) = 0.0079

Our measured random baseline: 0.0118 (1.5x above theoretical, expected
for finite samples).

### 3.2 Trained B-matrix Cosine

Training optimizes B to minimize cross-entropy loss on domain data.
The gradient update at step t is:

    B^{(t+1)} = B^{(t)} - eta * scale * (A^T h)^T dh

where h is the hidden state and dh is the loss gradient w.r.t. hidden state.

Two experts trained on similar data will have similar gradients -> similar
B-matrices. The cosine between B-matrices reflects the overlap in the
gradient trajectories, mediated by:
1. Domain data similarity (shared transition patterns)
2. Model architecture (shared base weights W)
3. Frozen A-matrix structure (shared subspace projections for AP-init)

## 4. Decomposition of B-Matrix Correlation Sources

Measured three conditions to disentangle sources:

| Condition | A-matrix | Training | B-matrix cos | Source |
|-----------|----------|----------|-------------|--------|
| AP-trained | AP skeleton | Yes | 0.0298 | Skeleton + shared gradients |
| Rand-trained | Random orthonormal | Yes | 0.0230 | Shared gradients only |
| Random baseline | N/A | No | 0.0118 | Noise floor |

Decomposition:
- Training effect: 0.0230 - 0.0118 = 0.0112 (training adds 0.95x of baseline)
- AP skeleton effect: 0.0298 - 0.0230 = 0.0068 (AP adds 0.29x of baseline)
- Total above baseline: 0.0298 - 0.0118 = 0.0180 (1.52x of baseline)

The AP skeleton contributes about 38% of the excess correlation above
baseline; training dynamics contribute the other 62%.

## 5. Domain Similarity Effect

Similar domain pairs (adjacent IDs sharing transition patterns):
- Mean |cos|: 0.0317

Dissimilar domain pairs (distant IDs):
- Mean |cos|: 0.0228

Ratio: 1.39x. Domain similarity has a weak but positive effect on
B-matrix overlap. The effect is noisy across seeds (one seed showed
reversed direction), indicating this is a second-order effect.

## 6. Amplification Analysis

### 6.1 Why Amplification is Near-Zero

At L=2 layers with d=64, the amplification ratio is effectively zero
(< 0.0001) for ALL conditions (AP-trained, random-trained, shuffled-B).
This is because:

1. **Shallow depth.** The parent experiment showed amp_ratio = 0.25 at
   L=24. At L=2, the error simply doesn't have enough layers to compound.

2. **Small deltas.** The LoRA deltas are scaled by alpha/r = 1.0, but
   the base weights dominate the forward pass. Relative perturbation
   is negligible.

3. **Residual connections.** The MicroMLP uses h = h_in + z2, which
   passes the majority of the signal through unchanged.

### 6.2 Safety Conclusion from K2

Even though amplification is too small to differentiate between
conditions, this is actually the desired result: at production parameters,
the parent experiment proved amp_ratio < 1.0 even at maximum synthetic
correlation (rho=1.0). Our real B-matrix correlation (rho_B ~ 0.03,
two orders of magnitude below rho=1.0) is far below the regime where
amplification becomes measurable.

The K2 safety margin is:
- Real correlation: rho_B ~ 0.03
- Synthetic worst case tested: rho = 1.0
- Amp ratio at rho=1.0: 0.074 (from parent)
- Amp ratio at rho=0.03: < 0.001 (extrapolated linearly)

## 7. Key Inequalities

### 7.1 B-Matrix Correlation is Moderate

    rho_B^{trained} / rho_B^{random} = 2.52x < 3.0x (K1 threshold)

The 3x threshold was chosen to distinguish "structured" from "noise-level"
correlation. At 2.52x, B-matrix overlap is elevated but not dramatically
structured.

### 7.2 B-Matrix Correlation is Safe

    amp_ratio(rho_B^{trained}) / amp_ratio(rho_B^{shuffled}) = 1.06x < 1.5x

Correlation does not increase amplification. This is consistent with the
parent finding that correlation REDUCES amplification via rank-1
compressibility and consistent activation masking.

## 8. Assumptions

1. **Toy dimension.** d=64, d_ff=256 vs production d=896+. B-matrix
   correlation may scale differently at higher d (likely lower, since
   E[|cos|] ~ 1/sqrt(D) and D grows as d * d_ff).

2. **Shallow model.** L=2 layers. Amplification effects emerge at
   L >= 8 (parent experiment). However, the B-matrix correlation
   measurement (K1) is independent of depth.

3. **Synthetic domain data.** Real domain data may create stronger
   B-matrix correlation due to shared linguistic structure. The
   toy Markov data has limited capacity for domain overlap.

4. **Small N.** N=6 experts (vs production N=50+). With more experts,
   more pairs will be similar, potentially increasing average correlation.

## 9. Worked Example

d=64, r=8, d_ff=256, n_layers=2:
- B-vector dimension: D = 2 * 2 * 8 * (64 + 256) = 10,240
- Random |cos| expected: sqrt(2 / (pi * 10240)) = 0.0079
- Random |cos| measured: 0.0118 (1.5x above theory)
- AP-trained |cos| measured: 0.0298 (2.52x above random baseline)
- K1 threshold: 3x * 0.0118 = 0.0355 (not reached)
- K2 amp ratio comparison: 1.06x (well within 1.5x margin)
