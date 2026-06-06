# Delta Rule Gated Linear Attention: Mathematical Foundations

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| x_t | (B, C) | Input at timestep t, C = n_embd |
| q_t | (B, h, d) | L2-normalized query, d = C/h |
| k_t | (B, h, d) | L2-normalized key |
| v_t | (B, h, d) | Value |
| S_t | (B, h, d, d) | Recurrent state (key-value memory) |
| g_t | (B, h) | Scalar decay gate, in (0, 1) |
| beta_t | (B, h) | Update strength for delta rule |
| z_t | (B, h, d) | Output gate (for SiLU gating) |
| A | (h,) | Learned decay rate (log-space) |
| dt_bias | (h,) | Bias for decay computation |
| kv_mem_t | (B, h, d) | Retrieved association (what state knows about k_t) |
| delta_t | (B, h, d) | Correction term (new information only) |

## Delta Rule State Update

### Simplified variant (prior experiments)

The simplified gated linear recurrence accumulates information naively:

    S_t = g_t * S_{t-1} + k_t * v_t^T

This adds the full outer product k_t v_t^T at each step. When multiple
domains share state, their associations simply pile up.

### Delta rule (this experiment)

The delta rule retrieves what the state already knows before storing:

    kv_mem_t = (S_{t-1} * k_t[:, :, None]).sum(axis=-2)     -- retrieve
    delta_t = (v_t - kv_mem_t) * beta_t[:, :, None]          -- correct
    S_t = g_t[:, :, None, None] * S_{t-1} + k_t[:, :, :, None] * delta_t[:, :, None, :]

Step by step:

1. **Retrieve**: kv_mem_t = S_{t-1}^T @ k_t computes what the current state
   would output for key k_t. Shape: (B, h, d).

2. **Correct**: delta_t = (v_t - kv_mem_t) * beta_t. If the state already
   stores the correct association for k_t, delta_t is near zero (no update).
   If the state stores a different association, delta_t captures the correction.
   beta_t scales the update strength.

3. **Update**: S_t = g_t * S_{t-1} + k_t * delta_t^T. The state decays by
   g_t and accumulates only the correction, not the full value.

### Decay gate

The decay gate uses the same parameterization as real GatedDeltaNet:

    a_t = W_a @ x_t                        -- input-dependent component
    g_t = exp(-A * softplus(a_t + dt_bias)) -- in (0, 1)

where A = exp(A_log) is a learned per-head rate constant.

### Output with SiLU gating

    o_t = (S_t * q_t[:, :, :, None]).sum(axis=-2)   -- query the state
    o_t = RMSNorm(o_t) * SiLU(z_t)                   -- gated normalization
    output = W_o @ concat(o_t over heads)

The RMSNorm + SiLU gating matches real GatedDeltaNet's Qwen3_5RMSNormGated.

## Interference Analysis

### Why the delta rule creates cross-domain interference

Consider two domains A and B composed into a single model. During training:

- Domain A fine-tunes capsule groups while attention (including state S) is shared
- Domain B fine-tunes different capsule groups on the same shared attention

At composition time, the state S processes inputs from both domains. When
domain B presents key k_B:

    kv_mem = S^T @ k_B

If S contains domain A's associations (learned during A's portion of the
data), kv_mem retrieves A's stored values. The correction becomes:

    delta = (v_B - kv_mem_A) * beta

This correction explicitly depends on domain A's stored content. In the
simplified variant, there is no retrieval -- the update is just k_B * v_B^T
regardless of what S contains. The delta rule creates an active coupling
between domains through the shared state memory.

### Hypothesis: this could reverse interference ordering

In the simplified variant, linear attention showed 0.59x the interference of
full attention (linear < full). The reasoning was that the gated recurrence
provides natural isolation through the decay gate.

The delta rule could reverse this because:
1. The retrieval step actively queries cross-domain content
2. Corrections are computed relative to existing state (not absolute)
3. The decay gate helps less when associations are actively being corrected

### Empirical finding: the hypothesis is WRONG

The delta rule ratio is 0.74x (linear/full), maintaining the ordering.
The simplified variant measured 0.86x in this experiment (different from the
0.59x in the original 5-seed experiment, likely due to different seed set).
In both cases, linear < full.

Possible explanations:
1. L2 normalization bounds the magnitude of kv_mem retrievals, limiting
   cross-domain interference even when the delta rule retrieves across domains
2. The decay gate g_t still provides isolation -- old associations decay
   before being retrieved by new domain keys
3. beta_t learns to modulate update strength, dampening corrections when
   the retrieval is unreliable

## Computational Cost

### Parameters per linear attention layer

| Component | Parameters | Count |
|-----------|-----------|-------|
| W_q, W_k, W_v, W_o | 4 * C^2 | 4 * 64^2 = 16,384 |
| W_a (decay) | C * h | 64 * 4 = 256 |
| W_beta (update) | C * h | 64 * 4 = 256 |
| W_z (gate) | C^2 | 64^2 = 4,096 |
| A_log, dt_bias | 2 * h | 2 * 4 = 8 |
| **Total per layer** | | **20,992** |

For 3 linear layers: 62,976 parameters.
L2 norm variant (no delta rule): 3 * 17,152 = 51,456 parameters.
Overhead: +22.4% per linear attention layer.

Total model: 217,112 vs 204,032 (L2 norm) = +6.4% overhead.

### FLOPs per linear attention layer

The sequential recurrence adds O(T * d^2 * h) FLOPs for the delta rule
retrieval and correction. At T=32, d=16, h=4, this is 32,768 FLOPs per layer
-- small compared to the QKV projections (3 * 64^2 = 12,288 each).

The materialized attention approach used by the simplified variant is
O(T^2 * d * h) = 32^2 * 16 * 4 = 65,536 FLOPs. The delta rule recurrence
is competitive at micro scale and strictly better at longer sequences.

## Worked Example (micro scale)

B=1, T=4, h=1, d=4 (single head, 4 timesteps, 4-dim)

State evolution for one head processing a sequence:

    t=0: S_0 = g_0 * 0 + k_0 * ((v_0 - 0) * beta_0)^T
         = k_0 * (v_0 * beta_0)^T        (first step: no prior state)

    t=1: kv_mem_1 = S_0^T @ k_1           (retrieve: what does state know about k_1?)
         delta_1 = (v_1 - kv_mem_1) * beta_1
         S_1 = g_1 * S_0 + k_1 * delta_1^T

    t=2: kv_mem_2 = S_1^T @ k_2
         delta_2 = (v_2 - kv_mem_2) * beta_2
         S_2 = g_2 * S_1 + k_2 * delta_2^T

If k_1 is similar to k_0, then kv_mem_1 ~ v_0 * beta_0 (retrieves the first
association). delta_1 = (v_1 - v_0*beta_0) * beta_1, which is large if
v_1 != v_0 (the state needs to correct itself).

If k_2 is orthogonal to both k_0 and k_1, then kv_mem_2 ~ 0 and
delta_2 ~ v_2 * beta_2 (no prior information to correct against).

## Assumptions

1. **L2 normalization bounds interference**: Because ||q||=||k||=1,
   the retrieved kv_mem has bounded magnitude, preventing runaway corrections.
   Validated empirically (0/7 catastrophic failures).

2. **Micro-scale recurrence is equivalent to chunked**: At T=32, the explicit
   sequential recurrence produces identical results to the chunk-based
   implementation used at macro scale. Both are exact; the difference is
   computational efficiency at longer sequences.

3. **Capsule group fine-tuning does not modify attention weights**: The
   composition protocol freezes attention during domain-specific fine-tuning.
   The delta rule's state dynamics during composition reflect the shared
   base model's attention, not domain-specialized attention.

4. **Interference metric (cosine distance) captures meaningful signal**:
   The per-layer cosine distance between domain-specialized capsule outputs
   measures how differently the two domains behave through each layer.
   This is a proxy for interference, not a direct measurement of the
   delta rule's retrieval-correction dynamics.
