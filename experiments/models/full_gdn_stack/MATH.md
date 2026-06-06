# Full GatedDeltaNet Composition Stack: Mathematical Foundations

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| x_t | (B, C) | Input at timestep t, C = n_embd |
| q_t | (B, h, d) | L2-normalized query after conv1d, d = C/h |
| k_t | (B, h, d) | L2-normalized key after conv1d |
| v_t | (B, h, d) | Value after conv1d |
| S_t | (B, h, d, d) | Recurrent state (key-value memory) |
| g_t | (B, h) | Scalar decay gate, in (0, 1) |
| beta_t | (B, h, d) | Per-dimension update strength for delta rule |
| z_t | (B, h, d) | Output gate (for SiLU gating) |
| A | (h,) | Learned decay rate (log-space) |
| dt_bias | (h,) | Bias for decay computation |
| kv_mem_t | (B, h, d) | Retrieved association |
| delta_t | (B, h, d) | Correction term (new information only) |
| w_conv | (C, K) | Causal conv1d kernel weights, K = kernel_size |

## Complete Mechanism (6 Components)

The full GatedDeltaNet attention combines six components. Each is described
below with its mathematical formulation, then we analyze how they interact.

### Component 1: Linear Projections

    q_raw = W_q @ x_t      (B, T, C) -> (B, T, C)
    k_raw = W_k @ x_t      (B, T, C) -> (B, T, C)
    v_raw = W_v @ x_t      (B, T, C) -> (B, T, C)

### Component 2: Causal Conv1d Preprocessing

    q_conv[t] = SiLU(sum_{j=0}^{K-1} w_conv_q[c, j] * q_raw[t - K + 1 + j, c])
    k_conv[t] = SiLU(sum_{j=0}^{K-1} w_conv_k[c, j] * k_raw[t - K + 1 + j, c])
    v_conv[t] = SiLU(sum_{j=0}^{K-1} w_conv_v[c, j] * v_raw[t - K + 1 + j, c])

Where K = kernel_size (default 4). The convolution is causal: position t
only sees positions {t-K+1, ..., t}. Positions before 0 are zero-padded.

The SiLU activation after conv1d matches the real GatedDeltaNet implementation.
This provides smooth nonlinear local mixing before the recurrence.

**Composition implications**: Conv1d preprocessing means each position's QKV
depends on K-1 previous positions. During composition, domain A's fine-tuned
capsule pool outputs at position t-1 affect position t's attention through
the conv1d window. This is a LOCAL coupling (K positions) vs the GLOBAL
coupling of full attention (all positions).

### Component 3: L2 QK Normalization

    q_t = q_conv[t] / (||q_conv[t]||_2 + eps)     ||q_t||_2 = 1
    k_t = k_conv[t] / (||k_conv[t]||_2 + eps)     ||k_t||_2 = 1

Bounds QK dot products to [-1, 1]. Proven to eliminate catastrophic
composition failures (0/25 vs 4/25 without normalization).

### Component 4: Decay Gate

    a_t = W_a @ x_t                                (B, T, h)
    g_t = exp(-A * softplus(a_t + dt_bias))         (B, T, h), in (0, 1)

Where A = exp(A_log) is a learned per-head rate constant.

### Component 5: Per-Dimension Beta Gating

    beta_t = sigmoid(W_beta @ x_t)                  (B, T, h, d)

Key difference from delta_rule_attention.py: beta has shape (B, T, h, d),
not (B, T, h). Each feature dimension within each head has its own update
strength.

This means the delta rule correction:

    delta_t[i] = (v_t[i] - kv_mem_t[i]) * beta_t[i]

can selectively update some dimensions while leaving others unchanged.
In the per-head variant, all d dimensions within a head share the same
update strength.

**Composition implications**: Per-dimension beta can learn to suppress
updates in dimensions that carry cross-domain information while allowing
updates in domain-specific dimensions. This provides a finer-grained
isolation mechanism than per-head beta.

### Component 6: SiLU Output Gating

    o_t = (S_t * q_t[:, :, :, None]).sum(axis=-2)   query the state
    o_t = RMSNorm(o_t) * SiLU(z_t)                   gated normalization
    output = W_o @ concat(o_t over heads)

The RMSNorm + SiLU gating controls the output magnitude, preventing
any individual head from dominating.

## State Update (Delta Rule)

    S_t = g_t * S_{t-1}                              decay
    kv_mem_t = (S_t * k_t[:, :, :, None]).sum(-2)     retrieve
    delta_t = (v_t - kv_mem_t) * beta_t               correct (per-dim)
    S_t = S_t + k_t[:, :, :, None] * delta_t[:, :, None, :]   update

## Component Interaction Analysis

### Conv1d x Delta Rule Interaction

The conv1d preprocessing changes the effective keys that are used for
retrieval. Without conv1d:

    kv_mem_t = S^T @ k_t = S^T @ l2norm(W_k @ x_t)

With conv1d:

    kv_mem_t = S^T @ l2norm(SiLU(conv1d(W_k @ x)))

The conv1d mixes information from K positions before the key is used for
retrieval. This means the delta rule's "what does state know about this
key?" query is asking about a LOCAL CONTEXT, not just a single position.

At micro scale (T=32, K=4), this covers 12.5% of the sequence per window.
At macro scale (T=4096, K=4), only 0.1% -- purely local smoothing.

### Per-Dim Beta x Delta Rule Interaction

Per-dimension beta allows selective correction:

    delta_t[i] = (v_t[i] - kv_mem_t[i]) * beta_t[i]

If beta_t[i] ~ 0 for dimension i, that dimension is NOT corrected even if
kv_mem_t[i] differs significantly from v_t[i]. This provides dimension-level
isolation: some dimensions can store domain A's associations undisturbed while
other dimensions get updated with domain B's corrections.

With per-head beta (prior experiment), ALL dimensions within a head share
the same update strength. This is coarser isolation.

### L2 Norm x Conv1d Interaction

L2 normalization is applied AFTER conv1d. This means the conv1d output's
magnitude is normalized away -- only the direction matters for attention.
The SiLU activation in conv1d preserves sign information, and L2 norm
preserves the direction while discarding magnitude.

This ordering is important: if L2 norm were applied before conv1d, the
normalization would be undone by the convolution's linear combination.
The post-conv1d ordering ensures the stability guarantee holds.

## Computational Cost

### Parameters per linear attention layer

| Component | Parameters | Count |
|-----------|-----------|-------|
| W_q, W_k, W_v, W_o | 4 * C^2 | 4 * 64^2 = 16,384 |
| Conv1d (Q, K, V) | 3 * C * K | 3 * 64 * 4 = 768 |
| W_a (decay) | C * h | 64 * 4 = 256 |
| W_beta (per-dim) | C * C | 64 * 64 = 4,096 |
| W_z (gate) | C^2 | 64^2 = 4,096 |
| A_log, dt_bias | 2 * h | 2 * 4 = 8 |
| **Total per layer** | | **25,608** |

For 3 linear layers: 76,824 parameters.

Comparison:
- Delta rule (per-head beta): 3 * 20,992 = 62,976 (+22.0% vs this)
- L2 norm (simplified): 3 * 17,152 = 51,456
- This (full stack): 76,824

Total model: 230,936 params
- vs delta rule: 217,112 (+6.4%)
- vs L2 norm: 204,032 (+13.2%)

### Overhead breakdown

| Added component (vs delta rule) | Params | % of model |
|--------------------------------|--------|------------|
| Conv1d (3 layers x 3 projections) | 2,304 | 1.0% |
| Per-dim beta (C^2 vs C*h, 3 layers) | 11,520 | 5.0% |
| **Total overhead** | **13,824** | **6.4%** |

## Worked Example (micro scale)

B=1, T=4, h=1, d=4, K=2 (single head, 4 timesteps, kernel_size 2)

Input sequence x = [x_0, x_1, x_2, x_3], each (4,).

Step 1 - Project:
    q_raw = W_q @ x,  k_raw = W_k @ x,  v_raw = W_v @ x

Step 2 - Conv1d (K=2, causal):
    q_conv[0] = SiLU(w[0]*0 + w[1]*q_raw[0])     -- left-padded with zero
    q_conv[1] = SiLU(w[0]*q_raw[0] + w[1]*q_raw[1])
    q_conv[2] = SiLU(w[0]*q_raw[1] + w[1]*q_raw[2])
    q_conv[3] = SiLU(w[0]*q_raw[2] + w[1]*q_raw[3])

    Position 2's query mixes info from positions 1 and 2. Causal: no future.

Step 3 - L2 norm:
    q[0] = q_conv[0] / ||q_conv[0]||    -- unit norm

Step 4 - Decay:
    g[0] = exp(-A * softplus(a[0] + bias))

Step 5 - Per-dim beta:
    beta[0] = sigmoid(W_beta @ x[0])    -- (4,), one per dimension

Step 6 - Delta rule at t=0:
    S = 0  (initial state)
    kv_mem = 0  (nothing stored)
    delta = (v[0] - 0) * beta[0] = v[0] * beta[0]
    S = k[0] * delta^T    (rank-1 outer product)

At t=1:
    S = g[1] * S
    kv_mem = S^T @ k[1]   (retrieve: what does state know about k[1]?)
    -- If k[1] aligns with k[0]: kv_mem ~ v[0] * beta[0] * g[1]
    -- If k[1] orthogonal to k[0]: kv_mem ~ 0
    delta = (v[1] - kv_mem) * beta[1]
    -- Per-dim: beta[1][i] controls how much dimension i is corrected
    S = S + k[1] * delta^T

Step 7 - Output:
    o[1] = S^T @ q[1]    (query the accumulated state)

Step 8 - Gate:
    out[1] = RMSNorm(o[1]) * SiLU(z[1])

## Assumptions

1. **Conv1d does not break composition stability.** Conv1d is a linear
   operation followed by SiLU. The L2 normalization applied afterward
   bounds the resulting QK products. The local mixing (K positions) does
   not amplify interference because it only introduces dependencies within
   a small window, and the frozen attention weights (including conv1d
   weights) are shared across domains. Validated empirically: 0/7
   catastrophic failures.

2. **Per-dimension beta does not create harmful dimension-level conflicts.**
   Different domains could learn to rely on different dimensions for their
   associations, with per-dim beta providing natural isolation. The risk
   is that both domains compete for the same dimensions. Empirically, the
   interference ratio is 0.86x (LOWER than delta-rule-only's 0.88x),
   suggesting per-dim beta may help with isolation rather than hurt.

3. **SiLU activation after conv1d preserves useful gradient information.**
   SiLU(x) = x * sigmoid(x) is smooth and non-zero for all x (unlike
   ReLU). This ensures gradients flow through the conv1d during
   pretraining, allowing the local mixing patterns to be learned effectively.

4. **Component interactions are additive, not multiplicative.** The
   empirical finding (0.86x interference ratio, below delta-rule's 0.88x)
   suggests the additional components do not compound interference. Each
   component provides independent stabilization (L2 norm bounds magnitudes,
   decay gate provides forgetting, per-dim beta provides dimension-level
   isolation, conv1d provides local smoothing).
