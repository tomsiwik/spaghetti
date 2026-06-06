# Value Norm Dynamics Under L2-Normalized QK Composition: Mathematical Foundations

## Notation

| Symbol | Shape | Definition |
|--------|-------|------------|
| x_t | (d,) | Hidden state at position t |
| d | scalar | Model dimension (d=64 at micro) |
| h | scalar | Number of attention heads (h=4) |
| d_h | scalar | Head dimension (d_h = d/h = 16) |
| T | scalar | Sequence length (T=32) |
| q_t, k_t, v_t | (d_h,) | Query, key, value at position t per head |
| q_hat, k_hat | (d_h,) | L2-normalized query, key (unit norm) |
| g_t | scalar | Forget gate at position t per head, g_t in (0, 1) |
| S_t | (d_h, d_h) | Recurrent state matrix at position t per head |
| W_v | (d, d) | Value projection weight matrix |

## Background: The State Boundedness Argument

The parent experiment (exp_l2_norm_composition_stability) established that
L2 normalization of Q and K bounds QK products to [-1, 1], preventing
catastrophic state accumulation. The recurrent state update is:

    S_t = g_t * S_{t-1} + k_hat_t @ v_t^T

The Frobenius norm of the state satisfies:

    ||S_t||_F <= g_t * ||S_{t-1}||_F + ||k_hat_t||_2 * ||v_t||_2

Since ||k_hat_t||_2 = 1 by L2 normalization:

    ||S_t||_F <= g_t * ||S_{t-1}||_F + ||v_t||_2         ... (1)

This bound is correct but depends on ||v_t||_2 being bounded. The output
computation is:

    o_t = q_hat_t @ S_t

With ||q_hat_t||_2 = 1:

    ||o_t||_2 <= ||S_t||_F                                 ... (2)

Combining (1) over T steps with constant gate g:

    ||S_T||_F <= sum_{t=1}^{T} g^{T-t} * ||v_t||_2

If all ||v_t||_2 <= V_max (a constant), then:

    ||S_T||_F <= V_max * (1 - g^T) / (1 - g)              ... (3)

For g < 1, this is bounded by V_max / (1 - g). At g = 0.9, T = 32:

    ||S_T||_F <= V_max * 10 * (1 - 0.9^32) = V_max * 9.64

The state is bounded iff V_max is bounded.

## The Question: Is V_max Bounded During Composition?

Value vectors are computed as v_t = W_v @ RMSNorm(x_t), where x_t is the
hidden state. The norm of v_t depends on:

1. **The norm of the normalized input** RMSNorm(x_t). By construction of
   RMSNorm (dividing by RMS), the output has roughly unit variance per
   dimension, so ||RMSNorm(x_t)||_2 is approximately sqrt(d_h) = 4.

2. **The spectral norm of W_v**. This is a training dynamics property.
   Under Adam optimization, weights are implicitly regularized by the
   learning rate and beta parameters, but there is no hard bound.

3. **The composition effect**. During composition, hidden states x_t
   receive contributions from independently-trained capsule pools.
   These pools were trained on different data and may produce hidden
   states with different magnitude distributions than what the attention
   layers saw during training.

The concern (from adversarial review): even with bounded QK products,
if ||v_t||_2 grows without bound during composition training (calibration),
the state S_t can still diverge through the value contribution term
in equation (1).

## Theoretical Bound (Why Growth Should Be Limited)

During router calibration, only the router weights are trainable. The
attention weights (W_q, W_k, W_v, W_g, W_o) are frozen. Therefore:

    v_t = W_v @ RMSNorm(x_t)

where W_v is fixed. The only thing that changes is x_t, through:

1. The capsule pool routing changes (router weights update)
2. This changes the MLP contribution to the residual stream
3. This changes the input to subsequent layers

Since the router only selects which capsule groups contribute (via
softmax weights), and all capsule group outputs are bounded by their
fixed weights, the change in x_t is bounded by the difference between
the worst and best capsule group contributions.

Let C_max = max over all groups of ||group_i(x)||_2 for any input x.
The routing change can shift x by at most:

    ||delta_x|| <= 2 * k * C_max    (switching k groups completely)

For d=64 with typical trained capsule groups, C_max ~ O(d) = O(64).
The total hidden state magnitude is O(d) plus the routing perturbation,
which is also O(d). So the value norm ||v_t||_2 = ||W_v|| * ||RMSNorm(x_t)||_2
is bounded by ||W_v|| * sqrt(d_h), and ||W_v|| is fixed during calibration.

This predicts that value norm growth during calibration should be
O(1) -- essentially no growth, because the value projection weights
are frozen and the RMSNorm limits the input magnitude.

## Growth Ratio Definition

We define the growth ratio as:

    R_max = max_{t in composition} ||v_t||_max / ||v_t||_baseline

where ||v_t||_max is the maximum per-head mean value norm observed at
any point during the composition pipeline, and ||v_t||_baseline is the
value norm after pretraining (before any domain-specific fine-tuning).

Similarly for the mean growth ratio:

    R_mean = max_{t in composition} mean(||v_t||) / mean(||v_t||)_baseline

## Kill Criteria (Formal)

**Kill criterion 1**: R_max > 10 for any seed. This would indicate that
value norms grow an order of magnitude during composition, breaking the
state boundedness argument in equation (3).

**Kill criterion 2**: |Pearson(R, gap)| > 0.5 where R is the growth
ratio vector and gap is the composition gap vector across seeds. This
would indicate that value norm growth is a mechanism for quality
degradation, not just incidental.

## Worked Example (d=64, h=4, d_h=16)

**Post-pretraining baseline**:
- W_v has spectral norm ~ 2.5 (typical for Adam-trained d=64 networks)
- RMSNorm output has ||.||_2 ~ sqrt(16) = 4.0
- Expected ||v_t||_2 ~ 2.5 * 4.0 = 10.0 per head

**During composition**:
- Router switches from (group_0, group_1) to (group_2, group_3)
- Capsule output changes by delta ~ 2.0 (typical)
- RMSNorm absorbs magnitude: ||RMSNorm(x + delta)||_2 still ~ 4.0
- ||v_t||_2 ~ 2.5 * 4.0 = 10.0 (unchanged)
- Growth ratio R ~ 1.0

**State boundedness check**:
- At R = 1.0, g = 0.9: ||S_T||_F <= 10.0 * 9.64 = 96.4
- Output: ||o_t||_2 <= 96.4 (bounded, within normal range)

## Assumptions

1. **Router calibration does not update attention weights.** The
   experiment protocol freezes everything except router weights during
   calibration. If attention weights were also updated, value norms
   could change through W_v updates, not just input changes.

2. **RMSNorm prevents magnitude explosion in the residual stream.**
   RMSNorm divides by the RMS of the input, normalizing the scale.
   Without normalization, residual stream magnitudes could grow with
   depth, causing value norms to increase in later layers.

3. **Capsule group outputs are bounded.** Each capsule group is a
   two-layer ReLU MLP (A -> relu -> B). With fixed weights, the output
   is bounded for bounded inputs. The RMSNorm on input to the capsule
   pool ensures inputs are bounded.

4. **The growth pattern at micro scale (d=64, T=32) transfers to macro
   scale (d=896, T=4096).** The theoretical argument (frozen W_v +
   RMSNorm + bounded inputs = bounded values) does not depend on
   scale. However, longer sequences accumulate more state, so even
   small per-step value norm increases compound over more steps. At
   T=4096, the factor (1 - g^T)/(1 - g) is closer to 1/(1-g), so
   the state bound tightens to V_max/(1-g).
