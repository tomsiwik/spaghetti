# Attention Self-Repair After Expert Removal: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | {32, 64, 128} (micro); 896 (production) |
| r | LoRA rank | 8 |
| N | Number of expert adapters | {4, 8, 16} |
| L | Number of transformer layers | {1, 2, 4, 8, 12, 16} |
| k | Index of expert to remove | 0 <= k < N |
| l | Layer index | 0 <= l < L |
| h_l | Hidden state at layer l | (batch, d) |
| W_l | Base MLP weight matrix at layer l | (d, d) |
| Delta_l | Merged expert contribution at layer l | (d, d) |
| W_q^l, W_k^l, W_v^l, W_o^l | Attention projection matrices | (d, d) |
| n_h | Number of attention heads | d / d_head |
| d_head | Head dimension | 16 (fixed) |
| sigma(.) | GELU activation | R -> R |
| RN(.) | RMSNorm: x -> x / sqrt(mean(x^2) + eps) | R^d -> R^d |
| rho_repair | Self-repair ratio: 1 - dev_tf / dev_mlp | R, >0 means repair |

## 2. Architecture Definitions

### 2.1 MLP-Only Model (Baseline)

The parent-proven Pre-RMSNorm residual architecture:

    h_{l+1} = h_l + (1/sqrt(L)) * sigma((W_l + Delta_l) @ RN(h_l))

Total sub-layers: L. Scale factor: 1/sqrt(L).

### 2.2 Transformer Model (With Attention)

Each layer has two sub-layers:

    Attention:  h' = h_l + (1/sqrt(2L)) * Attn(RN(h_l))
    MLP:        h_{l+1} = h' + (1/sqrt(2L)) * sigma((W_l + Delta_l) @ RN(h'))

Total sub-layers: 2L. Scale factor: 1/sqrt(2L).

Multi-head attention:

    Q = RN(h) @ W_q^T,  K = RN(h) @ W_k^T,  V = RN(h) @ W_v^T
    Attn_h(Q, K, V) = softmax(Q_h K_h^T / sqrt(d_head)) V_h
    Attn(h) = Concat(Attn_1, ..., Attn_{n_h}) @ W_o^T

Expert deltas apply ONLY to MLP sub-layers (attention weights are frozen base).

## 3. Self-Repair Hypothesis

### 3.1 Mechanism

When expert k is removed, the weight perturbation at each layer is:

    epsilon_l = Delta_l^{with k} - Delta_l^{without k}

In MLP-only model, this perturbation propagates through residual connections:

    h_L^{perturbed} - h_L^{clean} = sum_l (1/sqrt(L)) * J_l * epsilon_l

where J_l is the effective Jacobian from layer l to output.

In transformer model, attention between MLP sub-layers could:
- **Redistribute**: softmax creates data-dependent routing. If epsilon_l
  pushes activations into a subspace that attention de-weights, the
  perturbation is suppressed.
- **Compensate**: other attention heads may increase their contribution
  to restore the pre-perturbation representation (McGill et al. 2024).

### 3.2 Why Self-Repair Might Fail

The key insight from the experiment: attention operates on the hidden state
BETWEEN the perturbation source (MLP) and the next layer. But:

1. **Attention is linear in V**: Attn(h) = softmax(QK^T/sqrt(d)) V.
   The perturbation epsilon propagates through V linearly. Softmax only
   affects the weighting, not the value space.

2. **Random frozen attention**: without training to compensate, frozen
   attention weights have no mechanism to selectively suppress perturbations.
   Self-repair in McGill et al. requires TRAINED heads that have learned
   redundant representations.

3. **Scale factor difference**: transformer uses 1/sqrt(2L) vs MLP's
   1/sqrt(L). This means each MLP sub-layer contributes sqrt(2) less
   in the transformer, but there are also attention sub-layers contributing
   noise. Net effect is roughly neutral.

### 3.3 Predicted Self-Repair Ratio

For random (untrained) attention weights:

    rho_repair = 1 - dev_tf / dev_mlp
              ~ 1 - sqrt(2L) / (sqrt(L) * sqrt(2))  (scale factor ratio)
              ~ 0  (no self-repair)

The prediction is that random attention provides negligible self-repair
because it lacks the trained redundancy that enables compensation.

## 4. Amplification Analysis

### 4.1 Amplification Ratio

Both models show sub-additive error (amp_ratio << 1):

    amp_ratio = mean_output_deviation / sum_per_layer_error

At L=16: MLP amp_ratio = 0.037, Transformer amp_ratio = 0.035.
Difference: ~4%, far below 30% threshold.

### 4.2 Depth Scaling

Both architectures show identical depth scaling of amplification ratio:

    amp_ratio ~ C * L^{-alpha}

The attention sub-layers do not change the fundamental scaling law.

## 5. Worked Example

d=64, r=8, N=8, L=12, n_heads=4 (head_dim=16):

- MLP-only output deviation: 0.57%
- Transformer output deviation: 0.54%
- Self-repair ratio: 4.5% (far below 30% threshold)
- MLP amplification ratio: 0.057
- Transformer amplification ratio: 0.054

The 4.5% difference is within noise (std across seeds: 6.2%).
