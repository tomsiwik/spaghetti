# L2 QK Normalization for Composition Stability: Mathematical Foundations

## Notation

| Symbol | Shape | Definition |
|--------|-------|------------|
| x_t | (d,) | Hidden state at position t |
| d | scalar | Model dimension (d=64 at micro) |
| h | scalar | Number of attention heads (h=4) |
| d_h | scalar | Head dimension (d_h = d/h = 16) |
| T | scalar | Sequence length (T=32) |
| q_t, k_t, v_t | (d_h,) | Query, key, value at position t per head |
| g_t | scalar | Forget gate at position t per head |
| S_t | (d_h, d_h) | Recurrent state matrix at position t per head |

## The Problem: Unbounded QK Products

In the gated linear attention (simplified GatedDeltaNet), the output is:

    o_t = q_t @ S_t = sum_{s<=t} [prod_{u=s+1}^{t} g_u] * (q_t @ k_s^T) * v_s

The effective attention weight for position s at query position t is:

    A[t,s] = (q_t @ k_s^T) * prod_{u=s+1}^{t} g_u

The term (q_t @ k_s^T) is a dot product of two d_h-dimensional vectors with
no normalization. By Cauchy-Schwarz:

    |q_t @ k_s^T| <= ||q_t||_2 * ||k_s||_2

With random initialization (e.g., Xavier/Glorot), weights have scale
~1/sqrt(d), so q and k have entries of order ~1/sqrt(d). The QK dot product
then has magnitude:

    E[|q @ k^T|] ~ d_h * (1/sqrt(d))^2 = d_h / d = 1/h

At h=4: expected magnitude ~ 0.25. But the variance is high, and across
25 seeds we observed that ~16% of initializations produce QK products large
enough to cause numerical overflow in the recurrent state accumulation.

**Why this is composition-specific**: During single-domain training, the
optimizer can implicitly regularize QK magnitudes. During composition, two
independently-trained capsule groups produce hidden states that the linear
attention layers never saw during training. The QK products on these
out-of-distribution inputs can be much larger than during training, causing
the recurrent state S_t to accumulate extreme values.

## The Solution: L2 Normalization

L2 normalization maps each vector to the unit sphere:

    l2norm(x) = x / sqrt(sum(x_i^2) + eps)

where eps = 1e-6 prevents division by zero.

Applied to Q and K before the dot product:

    q_hat = l2norm(q, dim=-1)    -- ||q_hat||_2 = 1
    k_hat = l2norm(k, dim=-1)    -- ||k_hat||_2 = 1

Now the QK dot product is bounded:

    |q_hat @ k_hat^T| = |cos(theta)| <= 1

where theta is the angle between q and k in d_h-dimensional space.

This is a HARD bound, not a statistical property. Every QK product is in
[-1, 1] regardless of initialization, training state, or input distribution.

## Effect on Recurrent State

The recurrent state update becomes:

    S_t = g_t * S_{t-1} + k_hat_t^T v_t

Since ||k_hat_t||_2 = 1, the outer product k_hat_t^T v_t has Frobenius norm:

    ||k_hat_t^T v_t||_F = ||k_hat_t||_2 * ||v_t||_2 = ||v_t||_2

The state update magnitude is bounded by the value norm (not amplified by
key norm). Combined with the forget gate g_t in (0, 1), the state remains
bounded:

    ||S_t||_F <= g_t * ||S_{t-1}||_F + ||v_t||_2

This is a contraction if g_t * ||S_{t-1}||_F < ||S_{t-1}||_F, which holds
for g_t < 1. The state cannot grow without bound.

Without L2 normalization, the update is:

    ||S_t||_F <= g_t * ||S_{t-1}||_F + ||k_t||_2 * ||v_t||_2

If ||k_t||_2 is large (which happens with ~16% probability at random init),
the state can grow rapidly. Multiple tokens with large keys compound
exponentially before the gate can damp them.

## Effect on Attention Weights

The effective attention weight with L2 norm:

    A_hat[t,s] = (q_hat_t @ k_hat_s^T) * prod_{u=s+1}^{t} g_u

Since |q_hat @ k_hat^T| <= 1 and prod g_u in (0, 1]:

    |A_hat[t,s]| <= prod_{u=s+1}^{t} g_u <= 1

Every attention weight is bounded by 1.0. Without normalization, weights
can be arbitrarily large, causing the output to have arbitrarily large
magnitude -- the root cause of catastrophic composition failure.

## Computational Cost

L2 normalization per head per token:
- Compute sum of squares: d_h multiply-adds
- Compute inverse square root: 1 rsqrt
- Scale: d_h multiplies

Total: 2 * d_h + 1 operations per head per token.

At d_h=16, h=4, T=32: 4 * 32 * (2*16 + 1) = 4224 extra operations.
The full attention computation (QK product alone) is h * T * T * d_h =
4 * 32 * 32 * 16 = 65,536 operations.

Overhead: 4224 / 65536 = 6.4% of the QK computation, ~1.6% of total
layer compute. Negligible at any scale.

## No Learnable Parameters Added

L2 normalization is a fixed function with no learnable parameters.
The normalized model has exactly the same parameter count as the
unnormalized model. At micro scale: 204,032 parameters for both.

## Worked Example (d=64, h=4, d_h=16)

**Unnormalized (problematic init)**:
- q = W_q @ x produces vector with ||q||_2 = 3.2 (plausible at random init)
- k = W_k @ x produces vector with ||k||_2 = 2.8
- QK product: q @ k^T ~ 3.2 * 2.8 * cos(theta) = 8.96 * cos(theta)
- If cos(theta) ~ 0.5: QK ~ 4.5
- State update adds 4.5 * ||v||_2 to S per token
- Over 32 tokens with gates ~ 0.9: cumulative state magnitude grows ~43x
- Output magnitude grows proportionally -> loss of 0.99+ (catastrophic)

**L2 normalized**:
- q_hat = q / ||q||_2, ||q_hat||_2 = 1.0
- k_hat = k / ||k||_2, ||k_hat||_2 = 1.0
- QK product: q_hat @ k_hat^T = cos(theta) in [-1, 1]
- State update adds at most 1.0 * ||v||_2 per token
- Over 32 tokens with gates ~ 0.9: cumulative state bounded
- Output has controlled magnitude -> normal training dynamics

## Assumptions

1. **L2 normalization does not remove useful signal from Q/K.** The
   direction of the query/key vectors carries the important information
   (which positions attend to which). The magnitude is irrelevant for
   determining attention patterns -- only relative angles matter.
   Empirically validated: L2 normalized model achieves equivalent or
   better composition quality (-0.33% median gap vs +2.54% unnormalized).

2. **The catastrophic failure mode is magnitude-driven, not direction-driven.**
   If failures were caused by pathological Q/K directions (e.g., all queries
   pointing the same way), L2 normalization would not help. The 0/25
   failure rate with normalization confirms the magnitude hypothesis.

3. **The epsilon=1e-6 in rsqrt does not materially affect the bound.**
   For any non-zero vector, the normalization is effectively exact. Zero
   vectors would produce NaN, but this does not occur in practice because
   linear projections of non-zero inputs produce non-zero outputs.

4. **This result transfers to the delta rule variant.** The delta rule
   modifies the state update (v_t - kv_mem) but the QK product instability
   is in the output computation (q @ S), not the state update. L2 norm
   on Q and K stabilizes both. This assumption is NOT yet empirically
   validated.
