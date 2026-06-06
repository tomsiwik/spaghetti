# Hybrid Attention Composition: Mathematical Foundations (Revised)

## Notation

| Symbol | Shape | Definition |
|--------|-------|------------|
| x_t | (d,) | Hidden state at position t |
| d | scalar | Model dimension (d=64 at micro) |
| h | scalar | Number of attention heads (h=4) |
| d_h | scalar | Head dimension (d_h = d/h = 16) |
| T | scalar | Sequence length (T=32) |
| G | scalar | Number of capsule groups per layer |
| P | scalar | Capsules per group |
| k | scalar | Top-k group routing |
| L | scalar | Number of transformer layers (L=4) |
| N | scalar | Number of composed domain pools |

## Full Attention (Baseline)

Standard causal self-attention at layer l for head j:

    Q_j = x W_q^j,  K_j = x W_k^j,  V_j = x W_v^j
    A_j[t,s] = softmax_s(Q_j[t] @ K_j[s]^T / sqrt(d_h))  for s <= t
    o_j[t] = sum_{s<=t} A_j[t,s] * V_j[s]

Properties relevant to composition:
- A[t,s] depends on ALL tokens s <= t through the softmax denominator
- Changing V at any position s <= t changes the output at position t
- This creates dense dependency: composition changes propagate through all attended positions
- The 1/sqrt(d_h) scaling prevents softmax saturation -- this is specific to softmax attention

## Gated Linear Attention (Simplified Gated Linear Recurrence)

**Important**: This is a simplified variant that omits the delta rule,
per-dimension beta gating, SiLU output gating, L2 normalization, and conv1d
preprocessing from the full GatedDeltaNet. See Assumptions section.

Linear attention with a learnable forget gate at layer l for head j:

    Q_j = x W_q^j,  K_j = x W_k^j,  V_j = x W_v^j
    g_j[t] = sigmoid(x_t @ w_g^j)           -- scalar gate per head
    S_j[t] = g_j[t] * S_j[t-1] + K_j[t]^T V_j[t]    -- (d_h, d_h) state matrix
    o_j[t] = Q_j[t] @ S_j[t]

**No 1/sqrt(d_h) scaling is applied to QK products.** The 1/sqrt(d_h) scaling
is a softmax-attention convention: it prevents the softmax from entering its
saturation region where gradients vanish. Without softmax, there is no such
saturation concern. The reference GatedDeltaNet uses L2 normalization of Q and
K instead, which bounds the QK product magnitude differently.

Note: Removing the scaling means QK product magnitudes scale with d_h. At
d_h=16 (micro) this is manageable but can cause numerical instability in
some random initializations (empirically observed at ~20% failure rate).
L2 normalization would address this.

Unrolling the recurrence:

    S_j[t] = sum_{s=0}^{t} [prod_{u=s+1}^{t} g_j[u]] * K_j[s]^T V_j[s]

So the effective attention weights are:

    A_j^{lin}[t,s] = Q_j[t] @ K_j[s]^T * prod_{u=s+1}^{t} g_j[u]    for s <= t

Key differences from full attention:
1. No softmax normalization -- weights are NOT a probability distribution
2. Exponential decay via gate products: older tokens are geometrically downweighted
3. No denominator coupling: changing one V[s] affects output at t independently of other positions
4. No QK scaling -- magnitudes are unbounded (unlike L2-normalized real GatedDeltaNet)

## The Delta Rule (What This Model Omits)

Full GatedDeltaNet uses a retrieval-and-correction state update:

    S = alpha_t * S                                    -- decay
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)       -- retrieve from memory
    delta = (v_t - kv_mem) * beta_t                    -- correction
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)    -- update

The critical difference: `v_t - kv_mem` computes the difference between the
new value and what memory already predicts for that key. This means:
- The state stores CORRECTIONS, not raw associations
- Redundant information is not stored twice
- When composed domains share keys, the delta rule causes cross-domain
  retrieval and correction -- domain B's values get "corrected" against
  domain A's stored associations
- The beta gate (per-dimension, not per-head) modulates update strength
  independently for each feature dimension

This retrieval-and-correction mechanism could cause MORE or LESS interference
during composition than the naive additive model tested here. It is the
single most important omission and the primary reason results may not
transfer to real GatedDeltaNet.

## Composition Interference Analysis

When capsule groups from domain A and domain B are composed, the hidden state
at layer l becomes:

    h_l = attn_l(norm(h_{l-1})) + capsule_pool_composed(norm(h_{l-1} + attn_out))

The interference is the difference between the composed hidden state and
what each domain's specialist would produce in isolation.

### Full Attention Interference

For full attention, changing the MLP outputs at layer l-1 (from composition)
changes ALL subsequent attention patterns because:

    A[t,s] = softmax(... + noise_from_composition ...)

The softmax denominator couples all positions: noise at any position s
redistributes probability mass across ALL attended positions. This is
DENSE interference.

### Linear Attention Interference

For gated linear attention, the state update is:

    S[t] = g[t] * S[t-1] + k[t]^T v[t]

Composition-induced changes in k[t] or v[t] only ADD to the state at position t.
They do not redistribute existing state. The gate g[t] provides natural
forgetting, so old interference is geometrically damped.

The interference at position t from a composition change at position s < t is:

    delta_o[t] = Q[t] @ [prod_{u=s+1}^{t} g[u]] * delta(K[s]^T V[s])

This decays EXPONENTIALLY with distance (t-s), governed by gate values.
With typical gate values g ~ 0.5-0.9, interference from 10 positions ago
is attenuated by factor (0.7)^10 ~ 0.03.

### Predicted vs Observed Interference Ordering

Prediction: Linear attention layers should show LESS composition interference.

Observation (5 seeds, excluding Layer 0):
- Mean linear (Layers 1-2): 0.1694
- Mean full (Layer 3): 0.2858
- Ratio: 0.59x -- prediction confirmed

Depth confound check (full_attn model, all layers full attention):
- Layer 1 mean: 0.5864 (HIGHEST)
- Layer 2 mean: 0.4215
- Layer 3 mean: 0.4373
- Layer 3 - Layer 2 gap: +0.016 (negligible)
- No monotonic depth trend, so the lower interference in hybrid linear layers
  is not attributable to depth position alone.

### Layer 0 Zero Interference (Not Attention-Type Dependent)

Layer 0 shows exactly zero interference in BOTH full_attn and hybrid models.
This occurs because the base model's first-layer attention weights are shared
identically (same pretrained base, only capsule groups differ). The input to
Layer 0's attention is identical in base and composed models (same embeddings,
same norm). Therefore Layer 0 should be EXCLUDED from interference ratio
computations.

## Capsule Pool (Unchanged)

The capsule pool is identical in both attention types:

    MLP(x) = sum_{i in top_k} w_i * B_i @ ReLU(A_i @ x)

where w_i are softmax routing weights. Capsule composition concatenates
groups from different domains:

    Pool_composed = [Pool_A; Pool_B]

The router re-normalizes over the combined 2G groups.

## Computational Cost

Per layer, per token:

| Component | Full Attention | Linear Attention | Difference |
|-----------|---------------|-----------------|------------|
| Q/K/V proj | 3 * d^2 | 3 * d^2 | same |
| Gate proj | 0 | d * h | +d*h |
| Attention | 2 * T * d | 2 * T * d | same at micro |
| Output proj | d^2 | d^2 | same |
| Capsule pool | same | same | same |
| **Total** | 4d^2 + 2Td | 4d^2 + 2Td + dh | ~0.4% more |

At d=64, h=4: the gate adds 256 parameters per layer, negligible.

## Worked Example (d=64, h=4, T=32, G=4, P=64, L=4)

Full attention model:
- Per layer: 4 * 64^2 = 16,384 (attention) + 2 * 64 * 256 = 32,768 (capsule) = ~49K
- Total (4 layers): ~196K + embeddings (~3.5K) = ~200K params

Hybrid model (3 linear + 1 full):
- Linear layers add 3 * (64 * 4) = 768 params for gates
- Total: ~200K + 768 = ~201K params
- Param increase: <0.4%

Composition (N=2 domains, 2G groups composed):
- Capsule params double: 4 * 2 * 2 * 64 * 64 = 131,072 (capsule A+B matrices)
- Router params increase: 4 * 64 * 8 = 2,048 (8 groups instead of 4)
- Attention/embedding params shared (not doubled)

## Assumptions

1. **Simplified gated linear recurrence captures the essential gating
   mechanism but NOT the full GatedDeltaNet.** We omit: (a) the delta rule
   (retrieval-and-correction), which fundamentally changes state accumulation;
   (b) per-dimension beta gating; (c) SiLU output gating; (d) L2 key/value
   normalization (which prevents the numerical instability observed here);
   (e) conv1d preprocessing. Results are valid for the simplified variant
   only.

2. At T=32, the O(n^2) vs O(n) complexity difference is irrelevant --
   what matters is the information flow pattern.

3. Composition interference is dominated by attention-layer effects,
   not capsule-pool effects (validated by previous finding: "shared
   attention is the composition bottleneck").

4. 3:1 linear:full ratio at L=4 (3 linear + 1 full) is a valid micro
   approximation of Qwen3.5's 18:6 ratio at L=24.

5. **No QK scaling is applied in linear attention.** The 1/sqrt(d_h) scaling
   is a softmax-specific convention. Without it, QK product magnitudes are
   unbounded, which causes numerical instability in ~20% of random
   initializations at micro scale. Real GatedDeltaNet addresses this with
   L2 normalization of Q and K.
