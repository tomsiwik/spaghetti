# Composition Weight Sensitivity: Mathematical Foundations

## 1. Setup

### 1.1 Notation

| Symbol | Definition | Shape |
|--------|-----------|-------|
| W_s | Frozen skeleton (base) weights | R^{d_out x d_in} |
| dW_i | Expert i delta (= (alpha/r) * B_i @ A_i) | R^{d_out x d_in} |
| N | Number of experts in the composed model | scalar |
| r | LoRA rank | scalar |
| d | Model embedding dimension | scalar |
| L | Number of layers | scalar |
| L(W, x) | Loss of model W on data x | scalar |

### 1.2 Two Pre-Merge Strategies

**Sum composition** (SOLE default):

    W_composed = W_s + sum_{i=1}^{N} dW_i

Each expert contributes its full delta. At large N, the total perturbation
grows as O(N * ||dW||).

**Averaged composition** (dilution-aware):

    W_composed = W_s + (1/N) * sum_{i=1}^{N} dW_i

Each expert contributes (1/N) of its delta. Total perturbation stays bounded.

## 2. Dilution Analysis

### 2.1 Per-Expert Effective Weight

Under averaged composition, expert i's effective contribution to the composed
weight is:

    dW_i^{eff} = (1/N) * dW_i

The Frobenius norm of this effective contribution is:

    ||dW_i^{eff}||_F = (1/N) * ||dW_i||_F

At N=500 with a typical ||dW_i||_F, each expert contributes 0.2% of its
original signal. The question is whether this 0.2% is detectable.

### 2.2 Marginal Contribution

The marginal contribution of expert i is defined as:

    MC_i = L(W_s + compose(all except i)) - L(W_s + compose(all))

For averaged composition:

    MC_i^{avg} = L(W_s + (1/(N-1)) * sum_{j!=i} dW_j) - L(W_s + (1/N) * sum_j dW_j)

Note: removing expert i changes both the numerator (missing dW_i) AND the
denominator (1/(N-1) vs 1/N). This means the remaining N-1 experts each
contribute SLIGHTLY MORE weight after removal: (1/(N-1)) vs (1/N).

The change in effective weight per remaining expert is:

    delta_w = 1/(N-1) - 1/N = 1/(N(N-1))

At N=100, this is 1/9900 ~ 0.01% change per expert. So the marginal
contribution is dominated by the absent expert's signal, not the reweighting.

For sum composition:

    MC_i^{sum} = L(W_s + sum_{j!=i} dW_j) - L(W_s + sum_j dW_j)

This cleanly measures expert i's contribution without reweighting effects.

### 2.3 Signal vs Noise

The noise floor is established by replacing one expert's delta with a random
perturbation of matched Frobenius norm:

    dW_noise ~ N(0, sigma^2 I), scaled so ||dW_noise||_F = ||dW_i^{eff}||_F

The signal-to-noise ratio (SNR) for expert i is:

    SNR_i = |MC_i| / sigma(noise replacements)

An expert's contribution is "detectable" when SNR > 1.

### 2.4 Expected Scaling of Marginal Contribution

Under orthogonality (cos(dW_i, dW_j) ~ 0), each expert's contribution
to the loss is approximately independent. First-order Taylor expansion:

    L(W_s + sum_j dW_j) ~ L(W_s) + sum_j <grad L, dW_j> + O(||dW||^2)

The marginal contribution of expert i under sum composition:

    MC_i^{sum} ~ -<grad L|_{W_composed}, dW_i>

Under averaged composition, the effective gradient changes because the
composed model is different, but to first order:

    MC_i^{avg} ~ -(1/N) * <grad L|_{W_composed}, dW_i>

This predicts MC_i^{avg} ~ O(1/N), i.e., power law exponent -1.

### 2.5 Noise Floor Scaling

The noise delta has norm matched to (1/N) * ||dW_i||, so:

    noise_contribution ~ O(1/N) * random_direction

The variance of the noise contribution to the loss is:

    Var(noise) ~ (1/N^2) * ||dW_i||^2 * avg(grad_magnitude^2) / D

where D is the parameter count. The noise floor standard deviation scales as:

    sigma_noise ~ O(1/N) * ||dW_i||_F / sqrt(D)

Since both signal and noise scale as O(1/N), the SNR should be approximately
N-independent (constant) under orthogonality, governed by:

    SNR ~ sqrt(D) * |<grad L, dW_i / ||dW_i||>|

This is the directional alignment between the expert delta and the loss
gradient, amplified by the square root of the dimension count.

**Key prediction:** Under structural orthogonality, SNR should NOT degrade
with N. Dilution affects both signal and noise equally.

**STATUS OF THIS PREDICTION:** The micro-scale experiment cannot validate
this prediction. The measured power law exponent for marginal contribution
is +0.57 (increasing with N), which directly contradicts the predicted
-1.0 (decreasing as 1/N). At expert improvement of 0.00%, all
measurements are dominated by floating-point noise, making scaling
analysis meaningless. The prediction remains theoretically motivated but
empirically untested.

### 2.6 Signal/Noise Symmetry Requirement

The signal (marginal contribution) and noise floor must use the same
experimental operation for the SNR to be meaningful.

**Signal operation (leave-one-out):**

    MC_i = L(W_s + (1/(N-1)) * sum_{j!=i} dW_j) - L(W_s + (1/N) * sum_j dW_j)

This removes expert i and reweights the remaining N-1 experts from 1/N
to 1/(N-1). The loss change reflects both (a) the missing expert signal
and (b) the reweighting of remaining experts by delta_w = 1/(N(N-1)).

**Noise operation (symmetric LOO):**

    noise_j = L(W_s + (1/(N-1)) * sum_{j!=last} dW_j) - L(W_s + (1/N) * (sum_{j!=last} dW_j + dW_noise))

This replaces one expert with noise, then measures the LOO removal of
the noise expert. The "without" condition is compose_avg(N-1 real),
identical to the signal's "without" condition when removing the last
expert. The "with" condition is compose_avg(N-1 real + 1 noise).

Both operations see the same reweighting (1/N to 1/(N-1)) when one
slot is removed. The noise floor then measures the variance of LOO
marginal contributions when the removed expert is random noise rather
than a trained expert. This ensures SNR compares like with like.

**Why the original asymmetry was problematic:** The v1 noise experiment
replaced one expert with noise but kept all N experts at 1/N weight.
This meant signal saw the 1/(N(N-1)) reweighting effect but noise did
not. At the micro-scale magnitudes (1e-9 to 1e-11), this systematic
difference could dominate the actual expert-removal signal, making SNR
reflect reweighting artifacts rather than expert detectability.

**Impact on results:** After the symmetry fix, the qualitative pattern
is unchanged (SNR increases with N), and the quantitative values are
similar (v1: 2.7/16.5/19.9 at N=20/50/100; v2: 2.2/13.6/16.0). This
suggests the reweighting asymmetry was not the dominant effect, but the
fix makes the measurement formally correct.

## 3. Kill Criteria Formalization

### K1: Per-expert signal below noise at N < 50

    KILL if exists N < 50 such that mean(SNR_i, i in sampled experts) < 1.0

This would mean individual experts are undetectable in the composed model
at moderate scale, indicating pre-merge is fundamentally limited.

### K2: Catastrophic dilution at N = 100 vs N = 10

    KILL if |avg_gap(N=100) - avg_gap(N=10)| > 20%

where avg_gap(N) = (L(W_composed_avg) - L(W_base)) / L(W_base) * 100%.

## 4. Worked Example (d=64, r=8, N=4)

Parameters per expert delta: 2 * L * (d*r + r*d_ff) = 2 * 4 * (64*8 + 8*256)
= 2 * 4 * (512 + 2048) = 20,480

Total parameter space: sum of all weight matrix params
= L * (d*d_ff + d_ff*d) + d*V + d*V = 4*(64*256 + 256*64) + 64*32 + 64*32
= 4*32768 + 4096 = 135,168

Ratio: 20,480 / 135,168 = 15.2% of parameter space per expert

Random baseline cos ~ sqrt(r/d) = sqrt(8/64) = 0.354
Measured cos ~ 0.002 (167x below bound, per lora_flow_comparison)

At N=4 with averaging, each expert contributes 25% of its delta.
Expected marginal contribution: measurable (large fraction of total effect).

At N=100 with averaging, each expert contributes 1% of its delta.
Expected marginal contribution: smaller but still detectable if SNR is
N-independent (as predicted by Section 2.5).

## 5. Assumptions

1. **Structural orthogonality holds** at d=64: cos ~ 0.002 between
   independently trained LoRA experts.
2. **Expert deltas have similar magnitude**: ||dW_i||_F ~ ||dW_j||_F
   for all i, j (ensured by identical training procedure).
3. **Loss landscape is locally smooth**: first-order Taylor approximation
   is valid for the marginal contribution analysis.
4. **Domain-specific experts provide meaningful signal**: at micro scale,
   expert specialization may be weak (known limitation from
   lora_flow_comparison).
5. **Noise floor is established by norm-matched random perturbation**:
   this is a conservative noise model (structured noise could be higher).

## 6. Complexity

| Operation | Cost |
|-----------|------|
| Training N experts | O(N * EXPERT_STEPS * B * d^2) |
| Pre-merge (sum or avg) | O(N * L * d^2) |
| Leave-one-out eval (k samples) | O(k * eval_cost) |
| Full N sweep | O(sum(N_i) * expert_train_cost + sum(N_i * k_i) * eval_cost) |

At micro scale (d=64, N_max=100): ~3-5 minutes total.
