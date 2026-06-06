# Parallel Block Capsule Composition: Mathematical Foundations

## Notation

| Symbol | Shape | Definition |
|--------|-------|------------|
| x_l | (B, T, d) | Hidden state at layer l |
| d | scalar | Model dimension (d=64 at micro) |
| h | scalar | Number of attention heads (h=4) |
| d_h | scalar | Head dimension (d_h = d/h = 16) |
| T | scalar | Sequence length (T=32) |
| G | scalar | Number of capsule groups per layer |
| P | scalar | Capsules per group |
| k | scalar | Top-k group routing |
| L | scalar | Number of transformer layers (L=4) |
| N | scalar | Number of composed domain pools |

## Sequential Block (Standard Pre-Norm)

The standard transformer block uses two normalizations and sequential computation:

    h_l = x_l + Attn(Norm1(x_l))
    x_{l+1} = h_l + CapsulePool(Norm2(h_l))

Expanding Norm2's input:

    Norm2(h_l) = Norm2(x_l + Attn(Norm1(x_l)))

The capsule pool sees a feature vector that INCLUDES attention output.
When attention weights are shared across composed domains, this means
the capsule adapter operates on attention-processed (and potentially
interfered) features.

## Parallel Block (Tiny Aya Style)

The parallel block uses a single normalization with parallel branches:

    n_l = Norm(x_l)
    x_{l+1} = x_l + Attn(n_l) + CapsulePool(n_l)

Both attention and capsule pool receive the SAME normalized input.
The capsule pool's input does NOT depend on attention output.

## Composition Interference Analysis

### Sequential Block Interference Pathway

When domains A and B are composed via capsule group concatenation:

1. Attention is shared (same weights for both domains)
2. At layer l, attention produces output based on all prior-layer
   activations, which may already carry composition interference
3. Capsule pool at layer l receives:

       input_capsule = Norm2(x_l + Attn(Norm1(x_l)))

4. Any interference in Attn output propagates into the capsule input
5. This creates a SERIAL interference chain:

       interference_attn_l -> capsule_input_l -> capsule_output_l -> interference_attn_{l+1}

### Parallel Block Interference Pathway

With parallel blocks:

1. Attention is still shared (same composition bottleneck exists)
2. At layer l, capsule pool receives:

       input_capsule = Norm(x_l)

3. This is the SAME input that attention receives -- no sequential dependency
4. Interference pathway is PARALLEL, not serial:

       x_l -> Norm(x_l) -> [Attn: interference_attn_l]
                         -> [CapsulePool: adapter_output_l]
       x_{l+1} = x_l + interference_attn_l + adapter_output_l

5. The adapter output at layer l is independent of attention interference
   at layer l (though it still inherits interference from layers < l)

### Theoretical Prediction

The sequential block creates a depth-2 interference chain per layer:
   interference propagates Attn -> CapsulePool within each layer.

The parallel block reduces this to depth-1:
   interference from Attn and CapsulePool are additive at the residual,
   not cascaded.

Over L layers, the total interference chain depth is:
- Sequential: 2L (each layer contributes 2 steps)
- Parallel: L (each layer contributes 1 step, with parallel branches)

**Prediction**: Parallel blocks should show EQUAL or LOWER composition gap
compared to sequential blocks, because the interference pathway is shorter.

The effect may be small because:
1. The dominant interference source is shared attention weights (Exp 4),
   which is present in both architectures
2. Capsule-pool interference is typically small relative to attention
3. At micro scale (L=4), the chain depth difference (8 vs 4) is modest

## Capsule Pool (Unchanged)

The capsule pool is identical in both architectures:

    MLP(x) = sum_{i in top_k} w_i * B_i @ ReLU(A_i @ x)

Capsule composition concatenates groups from different domains:

    Pool_composed = [Pool_A; Pool_B]

The router re-normalizes over the combined 2G groups.

## Computational Cost

Per layer, per token:

| Component | Sequential | Parallel | Notes |
|-----------|-----------|----------|-------|
| Norm layers | 2 (Norm1, Norm2) | 1 (Norm) | RMSNorm has 0 learnable params |
| Attention | d^2 * 4 | d^2 * 4 | same Q/K/V/O projections |
| Capsule pool | same | same | same A/B/router |
| Norm compute | 2 * O(d) | 1 * O(d) | Parallel saves one norm pass |
| **Total params** | **identical** | **identical** | RMSNorm has no weight param |

The parallel block saves one normalization COMPUTATION per layer (not params,
since our RMSNorm has no learnable parameters). At micro scale this is
negligible (~1% of layer FLOPs). The main benefit is the shorter
interference chain, not compute savings.

In practice, the parallel block enables ~30-40% higher fine-tuning throughput
because the capsule pool and attention can be computed concurrently on hardware
that supports parallel execution. At micro scale with MLX, the effect comes from
a simpler computation graph (one norm call instead of two sequential ones).

## Worked Example (d=64, h=4, T=32, G=4, P=64, L=4)

Sequential model:
- Per layer: 4 * 64^2 (attention) + 2 * 64 * 256 (capsule A+B) + 64 * 4 (router) = ~49K params
- Total (4 layers): ~196K + embeddings (~3.5K) = ~200K params
- Norm layers: 8 (2 per layer) but 0 learnable params each

Parallel model:
- Per layer: identical param count (4 * 64^2 + 2 * 64 * 256 + 64 * 4 = ~49K)
- Total: ~200K params (same as sequential)
- Norm layers: 4 (1 per layer) but 0 learnable params each

Composition (N=2 domains, 2G groups composed):
- Capsule params double in both architectures
- Router params increase: 4 * 64 * 8 = 2,048
- Attention/embedding params shared (same in both)

## Assumptions

1. **Parallel blocks change only the information flow pattern, not capacity.**
   Both architectures have identical parameter counts. The only difference
   is whether the capsule pool sees pre-attention or post-attention features.

2. **RMSNorm has no learnable parameters in our implementation.** Unlike
   Tiny Aya's CohereLayerNorm (which has a scale parameter per dimension),
   our RMSNorm is a pure function. This means the parameter count difference
   between 1-norm and 2-norm blocks is exactly zero.

3. **The composition protocol is unchanged.** Both architectures use the
   same freeze-attention/train-capsules/compose/calibrate pipeline.

4. **Shared attention remains the composition bottleneck in both architectures.**
   Parallel blocks do not eliminate the shared attention bottleneck; they
   reduce its secondary effect (interference propagation into capsule input).
