# Pure-Linear Composition Control: Mathematical Foundations

## Notation

| Symbol | Shape | Definition |
|--------|-------|------------|
| x_t | (d,) | Hidden state at position t |
| d | scalar | Model dimension (d=64 at micro) |
| h | scalar | Number of attention heads (h=4) |
| d_h | scalar | Head dimension (d_h = d/h = 16) |
| T | scalar | Sequence length (T=32) |
| G | scalar | Number of capsule groups per layer |
| P | scalar | Capsules per group (P=64) |
| k | scalar | Top-k group routing (k=2) |
| L | scalar | Number of transformer layers (L=4) |
| N | scalar | Number of composed domain pools (N=2) |
| S_t^j | (d_h, d_h) | Recurrent state for head j at position t |

## The Question

The hybrid attention experiment validated that a 3:1 linear:full attention
ratio (3 GatedDeltaNet layers + 1 full attention layer) is composition-
compatible. But this leaves an ambiguity:

1. **Linear-compatible hypothesis**: Linear attention is inherently compatible
   with composition, and the full attention layer is unnecessary scaffolding.

2. **Scaffolding hypothesis**: Linear attention needs at least one full
   attention layer to provide global context integration that enables
   composition. Without it, the composed domains cannot coordinate.

This experiment tests hypothesis 1 by removing all full attention layers
(4:0 pure-linear configuration).

## Why Scaffolding Might Be Needed

Full attention computes:

    A[t,s] = softmax_s(q_t @ k_s^T / sqrt(d_h))    for s <= t
    o_t = sum_{s<=t} A[t,s] * v_s

This creates dense coupling: the softmax denominator makes every token's
attention weight depend on ALL other tokens. This provides a global
"consensus" mechanism that could help domain-specific capsule outputs
integrate.

In contrast, GatedDeltaNet's linear attention computes:

    S_t = g_t * S_{t-1} + k_t^T @ delta_t
    o_t = S_t^T @ q_t

The recurrent state S_t integrates information incrementally. Each position
only sees past positions through the state, with exponential decay via g_t.
There is no global denominator coupling.

## Why Scaffolding Might NOT Be Needed

The composition protocol freezes attention weights and only fine-tunes
capsule groups. During composition, attention weights are shared (identical
across domains). The composition interference comes entirely from the
capsule pool MLP outputs, not from attention mechanisms.

Formally, the layer computation is:

    h_l = x + attn(norm1(x)) + capsule_pool(norm2(x + attn(norm1(x))))

Since attn() uses shared (frozen) weights, the only domain-specific
component is capsule_pool(). The attention mechanism processes the SAME
hidden states regardless of which domain's capsules are active.

This suggests the attention type (linear vs full) should have minimal
effect on composition quality, because attention is not the source of
domain-specific computation.

## Interference Analysis for Pure-Linear

In the pure-linear 4:0 configuration, all layers use GatedDeltaNet.
The state update at each layer l for head j is:

    S_l^j[t] = g_l^j[t] * S_l^j[t-1] + k_l^j[t]^T @ delta_l^j[t]

where delta is the retrieval-corrected value:

    kv_mem = S^T @ k_t            (retrieve from memory)
    delta = (v_t - kv_mem) * beta  (correction)

The key prediction: removing the full attention layer at position 3
eliminates the one source of global (non-decaying) information integration.
If composition requires this global view, pure-linear should degrade.

If composition does NOT require it (because the composition signal is
entirely in the capsule pool, not the attention), pure-linear should
perform comparably.

## Computational Cost

Per layer, per token:

| Component | Full Attention | GatedDeltaNet | Difference |
|-----------|---------------|---------------|------------|
| Q/K/V proj | 3 * d^2 | 3 * d^2 | same |
| Conv1d | 0 | 3 * d * k_conv | +3dk |
| Decay/Beta/Gate | 0 | 3 * d * h + d^2 + d^2 | +3dh+2d^2 |
| Attention/Recurrence | 2 * T * d | d * T (sequential) | GDN is O(T) vs O(T^2) |
| Output proj | d^2 | d^2 | same |

At d=64, h=4, T=32, k_conv=4:
- Full attention: 4 * 64^2 + 2 * 32 * 64 = 16,384 + 4,096 = 20,480 MADs
- GatedDeltaNet: 4 * 64^2 + 3*64*4 + 3*64*4 + 2*64^2 + 64*32 = 16,384 + 768 + 768 + 8,192 + 2,048 = 28,160 MADs

At micro scale, GatedDeltaNet is actually more expensive per layer due to
the extra projections. At macro scale (T >> d), the O(T) vs O(T^2) wins.

## Param Count Comparison

| Config | Params |
|--------|--------|
| Hybrid 3:1 (3 GDN + 1 full) | 230,936 |
| Pure linear 4:0 (4 GDN) | 240,160 |
| Full attention 0:4 (4 full) | 202,688 |

Pure-linear has 4.0% more params than hybrid because the last layer's
GatedDeltaNet has more projections (beta, z, conv1d, decay) than causal
self-attention. This is a controlled confound: the extra capacity could
slightly favor pure-linear.

## Worked Example

d=64, h=4, d_h=16, T=32, G=4, P=64, L=4, N=2

Composition protocol:
1. Pretrain: 300 steps on all data, all params trainable
2. Domain A: 300 steps, only capsule groups trainable (attention frozen)
3. Domain B: 300 steps, only capsule groups trainable (attention frozen)
4. Compose: concatenate capsule groups (4+4=8 groups), double top-k (2+2=4)
5. Calibrate: 100 steps, only router trainable

Kill criterion: composed loss of pure-linear > composed loss of hybrid * 1.05

With observed values:
- Hybrid composed mean: 0.5059
- Pure composed mean: 0.5110
- Degradation: (0.5110 - 0.5059) / 0.5059 = +1.02%
- Threshold: 5%
- Result: PASS (1.02% < 5%)

## Assumptions

1. The full GatedDeltaNet stack (L2 norm + delta rule + conv1d + per-dim
   beta + SiLU gate + decay) is used in all linear layers. This is the
   same configuration validated in the full_gdn_stack experiment.

2. The composition protocol (freeze attention, train capsules, compose,
   calibrate) isolates the effect of attention type on composition.

3. 7 seeds provide sufficient statistical power to detect a 5% effect
   with the observed variance (~1% std).

4. The param count difference (+4% for pure-linear) is a minor confound
   that, if anything, favors pure-linear.
