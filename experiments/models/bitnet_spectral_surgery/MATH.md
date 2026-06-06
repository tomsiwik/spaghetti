# Spectral Surgery: Mathematical Foundations

## Setup

**Base model**: BitNet-b1.58-2B-4T (d=2560, 30 layers, ternary weights)
**LoRA**: rank r=16, scale alpha=20.0, applied to 7 modules per layer (q/k/v/o/gate/up/down)
**Total LoRA layers**: L = 210

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| A_l | (d_in, r) | LoRA A matrix for layer l (MLX convention: x @ A @ B) |
| B_l | (r, d_out) | LoRA B matrix for layer l |
| alpha | scalar | LoRA scale factor (20.0) |
| Delta_W_l | (d_in, d_out) | Effective low-rank update = A_l @ B_l * alpha |
| U_l | (d_in, r) | Left singular vectors of Delta_W_l |
| sigma_l | (r,) | Singular values of Delta_W_l |
| V_l^T | (r, d_out) | Right singular vectors of Delta_W_l |
| G_l | (d_in, d_out) | Gradient dL/d(Delta_W_l) on calibration set |
| g_k | scalar | Sensitivity of k-th singular component |
| N_cal | scalar | Number of calibration examples (128) |

## Algorithm: Spectral Surgery (arXiv 2603.03995)

### Step 1: SVD Decomposition

For each LoRA layer l:

    Delta_W_l = A_l @ B_l * alpha = U_l * diag(sigma_l) * V_l^T

Since rank(Delta_W_l) <= r = 16, this gives exactly r non-zero singular values.

### Step 2: Gradient Sensitivity

Compute the loss gradient w.r.t. the update matrix, aggregated over N_cal calibration examples:

    G_l = (1/N_cal) * sum_{i=1}^{N_cal} dL(x_i)/d(Delta_W_l)

The per-singular-value sensitivity is the directional derivative along u_k * v_k^T:

    g_k = u_k^T @ G_l @ v_k    for k = 1, ..., r

**Efficient computation** (avoiding materializing the full G matrix):

Since the autograd gives us dL/dA and dL/dB, and:
- dL/dA = G_l @ (B * alpha)^T = G_l @ B^T * alpha
- dL/dB = A^T @ G_l * alpha

We reconstruct g_k via:

    G_l = (1/alpha) * dL/dA @ pinv(B^T)

where pinv(B^T) = (B @ B^T)^{-1} @ B for full-rank B (r x d_out, r << d_out).

Then g_k = diag(U_r^T @ G_l @ V_r)_k, computed as:

    g_k = (1/alpha) * diag(U_r^T @ dL/dA @ (B @ B^T)^{-1} @ B @ V_r)_k

This requires only (r x r) matrix operations after the initial (d_in x r) @ (r x r) product.

### Step 3: Signed Update

Normalize sensitivities:

    g_tilde_k = g_k / ||g||_2

Split into positive (detrimental) and negative (beneficial):

    g_k^+ = max(g_tilde_k, 0)
    g_k^- = max(-g_tilde_k, 0)

Compute effective gradient with asymmetric scaling:

    g_k^eff = eta_sup * g_k^+ + eta_amp * g_k^-

where eta_sup = 1.0 (suppress detrimental), eta_amp = 0.5 (amplify beneficial).

Reweight:

    sigma'_k = sigma_k * exp(-g_k^eff)

### Step 4: Nuclear-Norm Renormalization

Preserve the L1 mass of the singular value spectrum:

    sigma'_final = sigma' * (||sigma||_1 / ||sigma'||_1)

This prevents over-amplification while maintaining total adapter magnitude.

### Step 5: Reconstruct LoRA Factors

    A_new = U_r * diag(sqrt(sigma'_final / alpha))    (d_in, r)
    B_new = diag(sqrt(sigma'_final / alpha)) * V_r^T   (r, d_out)

Such that A_new @ B_new * alpha = U_r * diag(sigma'_final) * V_r^T = Delta_W'.

## Computational Complexity

Per LoRA layer:
- SVD of (d_in x d_out): O(d_in * d_out * r) (thin SVD, since rank <= r)
- Pseudo-inverse: O(r^3) for (B @ B^T)^{-1}
- Sensitivity computation: O(d_in * r^2 + r^3)
- Reweighting: O(r)
- Reconstruction: O(d_in * r + r * d_out)

Per adapter (gradient aggregation dominates):
- N_cal forward+backward passes through full model: O(N_cal * model_FLOPs)
- SVD + reweighting for L layers: O(L * d_in * d_out * r)

**Critical insight**: The gradient computation (128 backprop passes through 2.4B model)
dominates wall-clock time. The SVD/reweighting is negligible (<5% of total time).

## Numerical Example (micro scale)

For q_proj in layer 0:
- d_in = 2560, d_out = 2560, r = 16, alpha = 20.0
- Delta_W = A @ B * 20.0  (2560 x 2560, rank 16)
- SVD yields sigma = [s_1, ..., s_16]
- After gradient estimation on 128 examples:
  - Some g_k > 0 (detrimental directions): sigma_k suppressed by exp(-eta_sup * g_k^+)
  - Some g_k < 0 (beneficial directions): sigma_k amplified by exp(-eta_amp * g_k^-)
  - Nuclear-norm renormalization rescales to preserve total mass

## Why It Failed on BitNet-2B LoRA

1. **Already efficient spectrum**: Our rank-16 adapters trained for only 200 iterations have a
   spectrum that is already concentrated. The "inefficient spectrum" problem from the paper
   (where beneficial effects concentrate in a subset of singular directions) assumes
   longer training where noise accumulates in unused directions.

2. **Nuclear-norm preservation prevents improvement**: The L1 constraint means surgery is
   zero-sum: suppressing one direction requires amplifying another. With an already-efficient
   spectrum, this redistributes energy without net benefit.

3. **Gradient estimation noise at micro scale**: With 128 calibration examples and a 2.4B
   parameter model, the gradient estimates are noisy. The sensitivity scores g_k may not
   reliably distinguish detrimental from beneficial directions.

## Assumptions

1. LoRA update matrix has an "inefficient spectrum" -- beneficial effects concentrate in
   a subset of singular directions (VIOLATED: our short-trained adapters are already efficient)
2. Gradient on calibration set reliably estimates per-direction sensitivity (PARTIALLY MET:
   128 examples may be insufficient for noisy BitNet gradients)
3. Nuclear-norm preservation is the right constraint (QUESTIONABLE: preserves total magnitude
   but prevents net improvement when spectrum is already well-distributed)
