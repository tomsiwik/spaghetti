# SOTA Ternary Training Techniques: Mathematical Foundations

## 0. Failure Mode & Impossibility Structure

### The Failure Mode: Deadzone Trapping in STE

Standard ternary quantization maps weights w to Q(w) in {-1, 0, +1} via:

    Q(w) = clip(round(w / alpha), -1, 1),  alpha = mean(|W|)

Weights with |w| < alpha/2 quantize to zero (the "deadzone"). During backpropagation, STE passes gradients through unchanged:

    dL/dw_dead = dL/dY * x_i      (no alpha scaling, no directional signal)
    dL/dw_live = dL/dY * x_i * alpha   (scaled by alpha)

Dead weights receive the raw upstream gradient noise, but since Q(w) = 0, the weight's contribution to Y is exactly zero. The gradient tells the weight how to change to reduce loss **if it were contributing to the output**, but it is not. This creates two trapping mechanisms:

**Mechanism 1: Alpha-coupling equilibrium.** alpha = mean(|W|) is a function of the weight distribution itself. For weights ~ N(0, sigma), the fraction quantized to zero is:

    P(|w| < alpha/2) = erf(alpha / (2*sqrt(2)*sigma))

Since alpha = sqrt(2/pi) * sigma for half-normal mean, we get:

    P(|w| < alpha/2) = erf(1/(2*sqrt(2))) = 0.3108...

This 31.1% is a **statistical fixed point**. If some dead weights escape (reducing the zero fraction), mean(|W|) increases, raising the threshold and trapping new weights. The equilibrium is self-reinforcing.

**Mechanism 2: Gradient noise dominance.** For a weight deep in the deadzone (|w| << alpha/2), the STE gradient is purely from input activations. Without the alpha scaling that active weights receive, the signal-to-noise ratio is:

    SNR_dead = E[dL/dY * x_i] / Var[dL/dY * x_i]^{1/2}

For typical training with batch noise, this SNR is too low to consistently push the weight past the threshold alpha/2 within a reasonable number of steps.

### What Mathematical Structure Makes Trapping Impossible?

Three classes of solutions exist, each attacking a different trapping mechanism:

**Class A: Smooth the quantization function (Hestia).** Replace Q(w) with a differentiable surrogate:

    pi_tau(q|w) = exp(-(w/gamma - q)^2 / tau) / Z(w, tau)
    w_eff = gamma * sum_q q * pi_tau(q|w)

**Impossibility guarantee:** Lemma 4.2 of Hestia proves that as tau -> 0+, the surrogate pi_tau(q|w) converges pointwise to the hard quantizer Q(w) almost everywhere. At finite tau > 0, the gradient d(w_eff)/dw is nonzero everywhere (including the deadzone), so no weight can be trapped. The Hessian-trace guided annealing ensures that tau decreases slowly for high-curvature layers, preventing premature trapping.

**Class B: Decouple alpha from weight distribution (TernaryLM).** Replace alpha = mean(|W|) with:

    tau = 0.5 * std(W)

This breaks the alpha-coupling because std(W) responds differently to weight migration than mean(|W|). When weights escape the deadzone, std(W) can decrease (tighter distribution), lowering the threshold and making escape easier -- the opposite of alpha-coupling's self-reinforcing trap.

**Class C: Fix the gradient, not the quantizer (FOGZO).** Replace the biased STE gradient with a debiased estimate:

    v_i = sqrt(beta) * s_i * g_hat + sqrt(1-beta) * u_i
    G = (1/n) * sum_i [(L(theta + eps*v_i) - L(theta - eps*v_i)) / (2*eps)] * v_i

**Impossibility guarantee:** If the STE gradient is orthogonal to the true gradient (which happens at quantization boundaries), the STE component's projection onto the finite-difference evaluates to zero, and only the unbiased ZO component survives. Dead weights receive a gradient correlated with the true loss landscape, not STE noise.

## 1. Mechanism Definitions

### 1.1 Standard BitNet b1.58 (Baseline)

**Forward pass:**
```
x_norm = RMSNorm(x)                           # [batch, seq, d]
alpha = mean(|W|)                              # scalar, per-tensor
W_scaled = W / alpha                           # [d_out, d_in]
W_q = clip(round(W_scaled), -1, 1)            # ternary {-1, 0, +1}
W_ste = W + stop_gradient(W_q * alpha - W)     # STE trick
Y = x_norm @ W_ste.T                          # [batch, seq, d_out]
```

**Complexity:** O(d_out * d_in) for quantization, O(batch * seq * d_out * d_in) for matmul.

### 1.2 Tequila Minima Reactivation (arXiv:2509.23800, Tencent/AngelSlim)

**Key modification:** Dead weights contribute as input-independent bias.

```
D = {(j,i) : |W_{j,i}| < alpha/2}             # deadzone set
C_j = lambda * sum_{i in D_j} W_{j,i}         # [d_out] bias vector
Y = x_norm @ W_ste.T + C                      # add reactivation bias
```

**Gradient for dead weight (j,i):**
    dL/dW_{j,i} = x_i * dL/dY_j + lambda * dL/dY_j = (x_i + lambda) * dL/dY_j

**Lambda gradient:**
    dL/dlambda = sum_{(j,i) in D} W_{j,i} * dL/dY_j

**Inference:** C is precomputed and fused as a static bias. Zero additional cost.

**Our prior result (exp_tequila_deadzone_fix):**
- K1 FAIL: Zero fraction unchanged at 32% (threshold 20%)
- K2 PASS: PPL improved -6.7% (463 -> 432)
- Conclusion: Bias compensation works; actual weight reactivation requires more training budget (paper shows improvement at 1B-3B scale with 10B tokens, we tested 64M params with 2M tokens)

### 1.3 Hestia (arXiv:2601.20745)

**Key idea:** Replace hard quantizer with temperature-controlled Softmax surrogate, guided by per-tensor Hessian-trace sensitivity.

**Surrogate quantizer:**
```
pi_tau(q|w) = exp(-(w/gamma - q)^2 / tau) / Z(w, tau)    # q in {-1, 0, 1}
w_eff = gamma * sum_q q * pi_tau(q|w)                      # differentiable
```

**Temperature annealing:**
```
s_i = HutchPP_trace(H_i) / sum_j HutchPP_trace(H_j)      # normalized sensitivity
tau_i(t) = tau_bar(t) * exp(alpha_anneal * s_i)            # per-tensor
tau_bar(t) = cosine_decay(tau_max, tau_min, t)             # global schedule
```

High-curvature tensors get higher temperature longer (slower discretization).

**Gradient (always nonzero):**
```
dw_eff/dw = gamma * sum_q q * d(pi_tau(q|w))/dw
d(pi_tau(q|w))/dw = pi_tau(q|w) * [-(2(w/gamma - q))/(gamma*tau) + 2*E_pi[(w/gamma-q')/(gamma*tau)]]
```

**Convergence:** As tau -> 0+, pi_tau(q|w) -> delta(q - Q(w)) a.e., and the derivative approaches sum of Dirac deltas at boundaries {-0.5*gamma, +0.5*gamma}.

**Complexity overhead:** Hutch++ trace estimation requires O(m) matrix-vector products per tensor (m = rank parameter, typically 3-5). This is the main training cost.

### 1.4 FOGZO (arXiv:2510.23926)

**Key idea:** Use zeroth-order gradient estimation guided by the (biased) STE gradient as prior.

**Algorithm (per step):**
```
1. Compute STE gradient g via standard backprop
2. Normalize: g_hat = g / ||g||
3. For i = 1..n:
   s_i ~ Uniform({-1, +1})          # random sign
   u_i ~ N(0, I) / ||N(0, I)||      # random unit vector
   v_i = sqrt(beta)*s_i*g_hat + sqrt(1-beta)*u_i
   f_plus = L(theta + eps*v_i)       # forward pass
   f_minus = L(theta - eps*v_i)      # forward pass
4. G = (1/n) * sum_i [(f_plus - f_minus)/(2*eps)] * v_i
5. Update: theta <- theta - lr * G
```

**Overhead:** n=1 means 2 extra forward passes per step (3x training time). n=0 degenerates to pure STE.

**Key property:** When STE gradient g is orthogonal to true gradient g*, the projection g_hat^T g* = 0, so the biased component vanishes and only the unbiased ZO component contributes.

### 1.5 TernaryLM (arXiv:2602.07374)

**Key modification:** Dynamic layer-wise threshold based on std, not mean.

```
tau_l = 0.5 * std(W_l)                        # per-layer threshold
Q(w) = sign(w) * I(|w| >= tau_l)              # quantize
alpha_l = mean(|W_l[|W_l| >= tau_l]|)         # scale from active weights only
Y = alpha_l * Q(W_l) @ x                      # forward
```

**Why this helps:** For W ~ N(0, sigma):
- Old threshold: alpha/2 = mean(|W|)/2 = sigma*sqrt(2/pi)/2 = 0.399*sigma
- New threshold: 0.5*std(W) = 0.5*sigma

The new threshold is higher (0.5*sigma vs 0.399*sigma), which means MORE weights are zeroed initially, but the coupling dynamics differ: as training progresses and the distribution shifts, std(W) responds proportionally rather than creating a self-reinforcing trap.

**Training recipe:** AdamW (beta1=0.9, beta2=0.95, wd=1e-5), cosine LR schedule peaking at 1e-3, 1000-step warmup. RMSNorm (not LayerNorm). Mixed activations: SiLU for attention, GELU for MLP.

### 1.6 PT2-LLM (arXiv:2510.03267, post-training only)

**Key idea:** Asymmetric ternary quantization for pre-trained models without retraining.

```
mu_j = mean(W[j, :])                          # row-wise shift
W_centered = W[j, :] - mu_j                   # center
T_j = ternary_round(W_centered / alpha_j)     # quantize centered
W_hat[j, :] = alpha_j * T_j + mu_j            # dequantize: asymmetric
```

**Refinement via Iterative Ternary Fitting (ITF):**
```
repeat:
    T* = argmin_T ||W - alpha*T - mu||^2       # optimal rounding given alpha, mu
    alpha* = (W - mu)^T T* / (T*^T T*)         # optimal scale given T
```

**Relevance:** Not for our training-from-scratch case, but the asymmetric insight (weights may have non-zero mean per row) could inform our BitLinear init.

### 1.7 MatMul-Free LM (arXiv:2406.02528)

**Key idea:** Replace all matmul operations with ternary weight + addition/subtraction.

Since W in {-1, 0, +1}, Y = W @ x becomes:
```
Y_j = sum_{i: W_{j,i}=+1} x_i - sum_{i: W_{j,i}=-1} x_i
```

No multiplication needed. Pure addition and conditional sign-flip.

**Architecture:** Uses MLGRU (MatMul-free Linear Gated Recurrent Unit) instead of attention for sequence mixing. Replaces softmax attention entirely.

**Relevance:** Shows that ternary weights are sufficient for language modeling up to 2.7B params. The matmul elimination is complementary to our approach (we use standard attention with ternary linear layers).

### 1.8 1-Bit Wonder (arXiv:2602.15563)

**Key idea:** K-means quantization achieves 1.25 bits/weight average.

```
1-bit weight base + 16-bit scale per block of 64 weights
Average: 1 + 16/64 = 1.25 bits/weight
```

Scaled to 31B params fitting in 7.7 GB. Outperforms 12B at 4.25 bits.

**Relevance:** Shows extreme quantization can scale. Could inform our ternary adapter compression.

## 2. Why Each Mechanism Works

| Method | Core Mechanism | Why It Works |
|--------|---------------|-------------|
| Tequila | Bias from dead weights | Gives dead weights a forward-pass contribution, providing clean gradient signal via lambda term |
| Hestia | Smooth surrogate | Eliminates flat regions in loss landscape; every weight gets nonzero gradient at finite temperature |
| FOGZO | ZO gradient correction | Debiases STE gradient at quantization boundaries where STE is maximally wrong |
| TernaryLM | std-based threshold | Breaks alpha-coupling equilibrium by decoupling threshold from mean(|W|) |
| PT2-LLM | Asymmetric shift | Handles non-zero-mean weight distributions that symmetric quantizers mishandle |

## 3. What Breaks Each Mechanism

| Method | Failure Condition |
|--------|-------------------|
| Tequila | Does not reduce zero fraction at micro scale (need 10B+ tokens). Bias helps PPL but capacity remains frozen. Alpha-coupling persists. |
| Hestia | Hutch++ overhead: O(m) extra mat-vecs per tensor per step. If m is too small, sensitivity estimates are noisy, causing premature or delayed annealing. If tau schedule is wrong, weights may not converge to discrete values. |
| FOGZO | 3x training time (n=1). At n=0, degenerates to STE. The ZO estimate has variance O(d/eps^2), so in very high dimensions, may need many perturbations. |
| TernaryLM | Higher initial threshold (0.5*std > 0.399*std) means more zeros initially. The std-coupling may still create an equilibrium, just at a different zero fraction. No proof that it converges to fewer zeros than mean-based. |
| PT2-LLM | Post-training only. Cannot be used for training-from-scratch. ITF may not converge for all weight distributions. |

## 4. Assumptions

1. **Gaussian weight distribution:** The 31.1% zero fraction analysis assumes W ~ N(0, sigma). If the distribution is heavy-tailed or bimodal, the equilibrium point shifts. (Justified: empirically observed 31.3% matches Gaussian prediction.)

2. **alpha = mean(|W|) is per-tensor:** Some implementations use per-channel alpha, which changes the deadzone geometry. Our BitLinear uses per-tensor. (Justified: matches BitNet b1.58 paper.)

3. **STE passes gradients unchanged in the deadzone:** Some STE variants clip gradients for |w| > some threshold (clipped STE). We use identity STE. (Justified: standard in BitNet.)

4. **Training budget matters:** Tequila's weight reactivation effect requires ~10B tokens at 1B+ params. At micro scale (2M tokens, 64M params), only the bias effect is observable. (Justified: Tequila paper Figure 8.)

## 5. Complexity Analysis

| Method | Training Overhead | Memory Overhead | Inference Overhead |
|--------|-------------------|-----------------|-------------------|
| Tequila | ~O(mn) per layer (mask + sum) | 1 scalar lambda per layer | Zero (bias fused) |
| Hestia | O(m * d_out * d_in) for Hutch++ | Temperature state per tensor | Zero (converges to hard Q) |
| FOGZO | 3x forward passes (n=1) | 2 perturbation tensors | Zero (training only) |
| TernaryLM | O(d_in) per layer (compute std) | None | Zero (same Q function) |

For our architecture (d=2560, 32 layers, 6 BitLinear per layer = 192 layers):
- Tequila: 192 extra scalars + 192 bias vectors (negligible)
- Hestia: 192 temperature states + Hutch++ per step (significant)
- FOGZO: 2 extra full-model forward passes per step (significant)
- TernaryLM: 192 std computations per step (negligible)

## 6. Worked Example (d=4, 2 outputs)

**Standard BitNet:**
```
W = [[0.30, -0.10, 0.80, -0.50],
     [0.05,  0.70, -0.20,  0.40]]

alpha = mean(|W|) = 0.381
threshold = alpha/2 = 0.190

Dead: W[0,1]=-0.10, W[1,0]=0.05   (2/8 = 25% zeros)
Q(W) = [[1, 0, 1, -1],
        [0, 1, -1, 1]]
```

**With TernaryLM threshold:**
```
std(W) = std([0.30, 0.10, 0.80, 0.50, 0.05, 0.70, 0.20, 0.40]) = 0.252
threshold = 0.5 * 0.252 = 0.126

Dead: W[0,1]=-0.10, W[1,0]=0.05   (same 2 weights, higher threshold)
Actually: |0.10| < 0.126 -> dead, |0.05| < 0.126 -> dead  (same result here)
```

**With Hestia (tau=1.0, gamma=0.381):**
```
For W[0,1] = -0.10:
  pi(q=-1) = exp(-(-0.10/0.381 - (-1))^2/1.0) = exp(-0.498) = 0.608
  pi(q=0)  = exp(-(-0.10/0.381 - 0)^2/1.0)    = exp(-0.069) = 0.933
  pi(q=+1) = exp(-(-0.10/0.381 - 1)^2/1.0)     = exp(-1.414) = 0.243
  Z = 1.784
  w_eff = 0.381 * [(-1)*0.341 + 0*0.523 + 1*0.136] = 0.381 * (-0.205) = -0.078

Crucially: w_eff != 0, so gradient flows through this weight.
dw_eff/dw is nonzero -> weight can update toward optimal value.
```

**With Tequila (lambda=1e-3):**
```
C[0] = 1e-3 * (-0.10) = -1e-4
C[1] = 1e-3 * (0.05) = 5e-5
Y = x @ (Q(W)*alpha).T + C    (bias added)
```

## 7. Connection to Our Architecture

### Impact on Composable Ternary Experts

Our architecture is: ternary base + Grassmannian LoRA adapters (rank-16, d=2560).

1. **Dead weights in base affect adapter capacity.** The 31.3% dead weights represent frozen capacity in the base model. Adapters must compensate for this missing capacity via their rank-16 perturbations. Reducing dead weights expands the base's representational space, potentially allowing adapters to specialize more narrowly.

2. **Tequila bias is immediately deployable.** The -6.7% PPL improvement from bias fusion costs zero inference overhead and is compatible with our existing BitLinear + runtime LoRA serving. The bias vector is per-output-dimension and adds to the base model output before adapter composition.

3. **Hestia is training-only.** At inference, Hestia converges to the same hard quantizer as BitNet. No serving changes needed. But the training pipeline must be modified to track temperatures and compute Hutch++ traces.

4. **TernaryLM threshold change is trivial to implement.** Replace `alpha = mean(|W|)` with `tau = 0.5 * std(W)` in our BitLinear forward pass. One line change. But the effect on composition quality is unknown -- different zero patterns may affect adapter interference differently.

5. **Adapter training (not just base).** All these techniques apply to adapter training too. Our ternary B-matrices (15.8x compression) have their own deadzone patterns. Improving ternary training for adapters could yield compounding benefits.

### Production Comparison

| System | Training Method | Dead Weight Handling |
|--------|----------------|---------------------|
| BitNet b1.58 2B4T | STE + absmean | None (31% zeros accepted) |
| Falcon-Edge | STE + absmean + Triton kernels | None (standard STE) |
| MatMul-Free LM | STE + ternary weights | None (addition-only inference) |
| Tequila | STE + bias reactivation | Bias compensation (-6.7% PPL) |
| Hestia | Smooth surrogate + Hessian | Eliminated during training |
| TernaryLM | STE + std-based threshold | Partially addressed via threshold |
| **Ours (proposed)** | **TernaryLM threshold + Tequila bias** | **Both: fewer zeros + bias compensation** |
