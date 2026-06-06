# Synthetic vs Real Data Quality: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Value/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro) |
| r | LoRA rank | 8 |
| N | Training examples per expert | 1000 |
| N_eval | Evaluation examples | 500 |
| K | Number of experts per regime | 4 |
| S | Number of seeds | 5 |
| T | SGD steps | 500 |
| W* | Ground-truth task matrix | R^{d x d}, rank r |
| A | LoRA input projection (frozen) | R^{r x d} |
| B | LoRA output projection (learned) | R^{d x r} |
| sigma_s | Synthetic label noise std | 0.05 |
| sigma_r | Real label noise std | 0.30 |

## 2. Data Source Models

### 2.1 Ground Truth Task

W* is a rank-r matrix in R^{d x d}, constructed as:

    W* = U @ V * 0.1

where U in R^{d x r}, V in R^{r x d} with entries ~ N(0, 1).

This ensures the task is exactly representable by a rank-r LoRA (B @ A)
when A spans the correct subspace. With random A, the achievable quality
is bounded by the alignment between A's row space and V.

### 2.2 Synthetic Data Distribution

Synthetic data models LLM-generated "textbook quality" examples (Phi-1 style):

    x ~ MixtureOfGaussians(K_s modes, alpha_s = 0.5, sigma_input = 0.3)
    y = x @ W* + epsilon,  epsilon ~ N(0, sigma_s^2 I)

Properties:
- K_s = 5 modes: Few distinct patterns (LLM generates from learned manifold)
- alpha_s = 0.5 (Dirichlet): Concentrated weights -> mode collapse
- sigma_input = 0.3: Tight clusters around mode centers
- sigma_s = 0.05: Very clean labels (LLM outputs are coherent)
- Systematic bias: x += 0.25 * bias_direction (the "LLM accent")

Calibration:
- Phi-1 (Gunasekar et al. 2023): synthetic-only achieves 50.6% on HumanEval
- Orca (Mukherjee et al. 2023): 88% agreement with teacher on diverse tasks
- Shumailov et al. 2024: recursive generation causes diversity loss

### 2.3 Real Data Distribution

Real data models naturally-occurring code (codeparrot-clean style):

    x ~ MixtureOfGaussians(K_r modes, alpha_r = 2.0, sigma_input = 0.8)
    y = x @ W* + epsilon,  epsilon ~ N(0, sigma_r^2 I)

Properties:
- K_r = 20 modes: Many distinct coding styles and paradigms
- alpha_r = 2.0 (Dirichlet): Near-uniform weights -> broad coverage
- sigma_input = 0.8: Wide spread -> diverse input space coverage
- sigma_r = 0.30: Substantial label noise (real code contains bugs)
- No systematic bias

### 2.4 Coverage-Quality Tradeoff

The fundamental tension: for a rank-r LoRA to generalize, it needs training
inputs that cover enough of R^d to constrain the B matrix. Formally:

    Quality(B @ A, X_eval) depends on:
    1. rank(X_train^T @ X_train)  -- coverage of input space
    2. sigma_noise                  -- label quality
    3. alignment(rowspace(A), rowspace(V))  -- A quality

Synthetic data optimizes (2) at the expense of (1).
Real data optimizes (1) at the expense of (2).

### 2.5 Effective Rank as Coverage Metric

Effective rank (Roy & Vetterli 2007):

    r_eff(X) = exp(H(p))

where p_i = sigma_i / sum(sigma_j) are the normalized singular values of X
and H(p) = -sum(p_i * log(p_i)) is the Shannon entropy.

At d=64 with N=1000 samples, both sources achieve r_eff ~ 63
(near-maximal), because even 5 modes with N=1000 samples fill the
64-dimensional space. The effective rank metric is insensitive to
MODE STRUCTURE -- it only measures dimensionality coverage.

The actual diversity difference is in the CONDITIONAL distribution:
synthetic concentrates probability mass on 5 modes with Dirichlet
alpha=0.5, while real distributes across 20 modes with alpha=2.0.
This means synthetic training data has fewer distinct gradient
directions, leading to poorer exploration of the loss landscape.

## 3. LoRA Training Dynamics

### 3.1 Frozen-A SGD

With A fixed, the LoRA update B @ A approximates W* by solving:

    min_B  E_{(x,y)} ||x @ A^T @ B^T - y||^2

This is a linear regression in the projected space. The gradient:

    grad_B = (1/|batch|) * error^T @ x_batch @ A^T

where error = y_pred - y_true.

With cosine-decayed learning rate:

    lr_t = lr_base * (0.1 + 0.9 * 0.5 * (1 + cos(pi * t / T)))

### 3.2 Convergence under Different Data Types

For synthetic data (low noise, concentrated inputs):
- Clean gradients: grad_B has low variance -> fast convergence
- But: gradient directions are concentrated in the span of 5 mode centers
- The learned B @ A approximates W* well in the MODE SUBSPACE
  but poorly in the complementary space

For real data (high noise, diverse inputs):
- Noisy gradients: grad_B has high variance -> slower convergence
- But: gradient directions span nearly all of R^d
- The learned B @ A is a noisy-but-unbiased approximation of W*

### 3.3 Quality Metric

    Quality(A, B) = 1 - ||X_eval @ (B @ A)^T - X_eval @ W*||_F / ||X_eval @ W*||_F

This is 1 minus the relative Frobenius error. Quality in [0, 1] where
1 = perfect reconstruction.

Evaluation on UNIFORM X_eval (standard normal) tests generalization.
Evaluation on SOURCE-MATCHED X_eval tests in-distribution performance.

## 4. Mixing Analysis

### 4.1 Mixed Data

For mixing ratio alpha in [0, 1] (fraction synthetic):

    X_mix = [X_synth[1:alpha*N]; X_real[1:(1-alpha)*N]]

The effective label noise is:

    sigma_eff^2 = alpha * sigma_s^2 + (1-alpha) * sigma_r^2

At alpha=0.2 (optimal): sigma_eff^2 = 0.2*0.0025 + 0.8*0.09 = 0.0725

The coverage is dominated by the real data fraction, while the
synthetic fraction provides clean gradient signal in high-density regions.

### 4.2 Optimal Mixing Ratio

Empirically, the optimal mixing ratio is alpha* ~ 0.2 (20% synthetic).
This aligns with the "accumulate, not replace" finding from the
literature: adding synthetic alongside real preserves diversity while
adding clean signal.

The mixing benefit is bounded by the coverage-noise tradeoff:
- Too much synthetic (alpha > 0.5): coverage loss dominates
- Too little synthetic (alpha < 0.1): noise is not reduced enough
- Sweet spot (alpha ~ 0.1-0.3): best of both worlds

## 5. Orthogonality Analysis

### 5.1 Within-Regime Orthogonality

Mean |cos| between expert delta_W matrices:
- Within synthetic: |cos| ~ 0.106
- Within real: |cos| ~ 0.110
- Within mixed: |cos| ~ 0.112

All values are comparable, suggesting that data source does NOT
significantly affect expert orthogonality at d=64. This is consistent
with the structural orthogonality result (cos ~ 1/sqrt(d) ~ 0.125
is the random baseline at d=64).

### 5.2 Cross-Regime Orthogonality

Cross synthetic-real: |cos| ~ 0.095

Slightly LOWER than within-regime, suggesting that different data
sources actually INCREASE inter-expert orthogonality. This is because
synthetic and real experts learn different aspects of W* (mode subspace
vs full space), which are naturally somewhat orthogonal.

### 5.3 Subspace Angles

Minimum principal angles between B subspaces:
- Synthetic: 0.75 deg (VERY similar subspaces)
- Real: 2.22 deg (more diverse subspaces)
- Mixed: 2.73 deg (most diverse)

Synthetic experts converge to similar B subspaces because they all
train on data from the same 5 modes. Real experts, training on diverse
data, explore different parts of the optimization landscape.

## 6. Contamination Risk Model

### 6.1 Benchmark Overlap

For HumanEval (164 problems):

    Synthetic: P(overlap) = 0.10 per example
    Expected overlap = 1000 * 0.10 = 100 examples
    P(at least one benchmark contaminated) = 1 - (1-0.10)^1000 ~ 1.0

    Real: P(overlap) = 0.02 per example
    Expected overlap = 1000 * 0.02 = 20 examples
    P(at least one benchmark contaminated) = 1 - (1-0.02)^1000 ~ 1.0

### 6.2 Expected Accuracy Boost

    boost = (expected_overlap / benchmark_size) * memorization_factor

With memorization_factor = 0.30 (30% boost on exact matches):
- Synthetic: (100/164) * 0.30 * 100 = 18.3% expected boost
- Real: (20/164) * 0.30 * 100 = 3.7% expected boost

The 5x difference means synthetic-trained experts may show
artificially inflated benchmark scores that do not reflect genuine
capability improvement.

## 7. Assumptions and Limitations

1. **Linear task**: W* is a linear map. Real tasks are nonlinear. The
   coverage-quality tradeoff may behave differently with nonlinear tasks,
   though the direction should hold (coverage still matters for generalization).

2. **Gaussian mixture model**: Real data distributions are more complex
   than Gaussians. But the MODE CONCENTRATION property (synthetic has fewer,
   more concentrated modes) is well-calibrated from Phi-1/Orca literature.

3. **Label noise model**: We use additive Gaussian noise. Real noise
   patterns are more structured (systematic errors, not random).
   Structured noise could be worse or better than Gaussian depending on alignment.

4. **Contamination model**: The 10% overlap rate for synthetic data is
   calibrated from Xu et al. 2024 but is domain-dependent. Code benchmarks
   (HumanEval) are more contaminated than reasoning benchmarks (MATH).

5. **Scale**: d=64 is much smaller than production (d=4096). At larger d,
   the coverage advantage of real data may diminish because even
   concentrated modes in high dimensions provide broader coverage.
   The orthogonality finding (no effect of data source) is expected
   to STRENGTHEN at larger d per the structural orthogonality theorem.

## 8. Worked Example at Micro Scale

d=64, r=8, N=1000, 5 modes (synthetic), 20 modes (real):

1. W* = U(64x8) @ V(8x64) * 0.1 -- rank-8 target
2. Synthetic: 5 unit-norm mode centers, Dirichlet(0.5) weights
   -> ~200 examples per mode (concentrated)
   -> sigma_s = 0.05 noise
   -> 200 gradient steps with clean signal in 5 directions

3. Real: 20 unit-norm mode centers, Dirichlet(2.0) weights
   -> ~50 examples per mode (spread)
   -> sigma_r = 0.30 noise
   -> 200 gradient steps with noisy signal in 20 directions

4. Quality on uniform eval:
   - Synthetic: 0.025 (learns 5 directions well, misses the rest)
   - Real: 0.060 (learns all 64 directions noisily)
   - Mixed(0.2): 0.067 (clean signal on 5 + noisy on 20 -> best of both)

5. Orthogonality: |cos| ~ 0.10 for all regimes (d=64 dominates data source)
