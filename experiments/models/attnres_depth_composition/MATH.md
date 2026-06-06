# AttnRes Depth-Wise Attention for LoRA Composition: Mathematical Foundation

## 1. Mechanism Definition

### Standard Residual Connection

In a transformer with L layers, the hidden state evolves as:

```
h_0 = Embed(x)                    ∈ R^{n×d}
h_l = h_{l-1} + f_l(h_{l-1})      for l = 1, ..., L
```

where f_l is the l-th transformer block (attention + FFN with pre-norm).

**Hybrid residual design:** In our implementation, AttnRes replaces only the
inter-block residual connection. The intra-block attention sublayer still uses
a standard additive residual: h = x + Attn(RMSNorm(x)), and v_l is the FFN
output of this sublayer (not the full block output including the internal
residual). This means the PreNorm dilution argument applies to the inter-block
accumulation (which IS replaced by depth attention), but the attention sublayer
within each block still has the standard dilution pattern. At L=4 this is a
minor distinction; at deeper scales, it means approximately half the residual
connections remain additive.

Each layer's contribution is v_l = f_l(h_{l-1}). Unrolling:

```
h_L = h_0 + Σ_{l=1}^{L} v_l
```

All layers contribute with unit weight. No mechanism exists to amplify or
attenuate any layer's contribution post-hoc.

### PreNorm Dilution Problem

With PreNorm (RMSNorm before each sublayer), the effective contribution of
layer l to the final representation shrinks as the residual stream grows.
After L layers:

```
||h_L|| ~ O(√L) · ||v_l||
```

because L independent contributions accumulate in the residual stream.
The RELATIVE contribution of any single layer is ~1/L. For L=32 (typical),
each layer contributes ~3% to the final hidden state norm.

Consequence for LoRA: An adapter ΔW_l = B_l A_l at layer l produces a
perturbation δ_l = ΔW_l · h_{l-1}. Under standard residuals, this perturbation
is added with unit weight regardless of its importance. In deep layers, h_{l-1}
is large (accumulated residuals), so ||δ_l|| may be large in absolute terms
but the PreNorm at subsequent layers effectively renormalizes, making the
relative contribution of δ_l shrink as 1/(L-l).

### AttnRes: Depth-Wise Softmax Attention

AttnRes (Kimi, arXiv 2603.15031) replaces the uniform accumulation with
learned depth-wise attention:

```
h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
```

where v_i are the layer outputs (including v_0 = h_0 from embedding), and
the attention weights are:

```
α_{i→l} = softmax_i(w_l^T · RMSNorm(v_i))
```

Key components:
- w_l ∈ R^d: learned pseudo-query for layer l (NOT input-dependent)
- RMSNorm(v_i): normalizes each layer's output before attention scoring
- softmax over depth dimension i ∈ {0, ..., l-1}

**Shapes:**
- v_i ∈ R^{n×d} (sequence length × model dim) per layer output
- w_l ∈ R^d (one pseudo-query per layer, shared across all positions)
- Score s_{i→l} = w_l^T · RMSNorm(v_i) ∈ R^n (one score per position, per depth)
- α_{i→l} ∈ R^n (attention weight per position per depth), softmax over depth dim

**Parameter overhead per layer:** d parameters for w_l
**Total overhead:** L × d parameters
**At d=128, L=4:** 512 parameters (negligible vs model params)

### Zero Initialization

```
w_l = 0 ∈ R^d initially
```

When w_l = 0: s_{i→l} = 0^T · RMSNorm(v_i) = 0 for all i.
Therefore α_{i→l} = softmax(0, 0, ..., 0) = 1/l (uniform over l preceding layers).

This means AttnRes STARTS as uniform averaging (similar to standard residual
but normalized to sum=1 instead of sum=l), then learns to specialize during
training. This is the key property ensuring stable training from initialization.

### Block AttnRes Variant

For L layers divided into N blocks of size B = L/N:

```
Block b contains layers {(b-1)·B+1, ..., b·B}
```

Within each block: standard residual connections (additive).
Between blocks: AttnRes attention over block outputs.

For our micro model with L=4 layers and N=1 block: this degenerates to full
AttnRes (every layer is in one block, attention over all layer outputs). We
use N=2 blocks of 2 layers each to test the block mechanism.

## 2. Why It Works (for Composition)

### Standard Residuals and Adapter Dilution

Under standard residuals with N composed adapters at 1/N scaling:

```
h_l^composed = h_{l-1}^composed + f_l(h_{l-1}^composed) + (1/N) Σ_{j=1}^{N} δ_l^j
```

where δ_l^j = ΔW_l^j · h_{l-1} is adapter j's contribution at layer l.

The adapter signal at layer l is diluted by:
1. 1/N scaling (N adapters share the residual stream)
2. PreNorm renormalization at subsequent layers
3. No mechanism to amplify important adapter contributions

### AttnRes Advantage for Composition

Under AttnRes, the model can learn depth-attention weights that:

1. **Amplify adapter-heavy layers:** If adapters contribute most at layers
   l=2,3, the pseudo-query can learn to assign higher α for those layers
2. **Attenuate base-dominated layers:** Layers where adapters have minimal
   effect can be downweighted, reducing dilution
3. **Input-adaptive (per-position):** Because w_l^T · RMSNorm(v_i) depends
   on v_i which includes adapter contributions, the attention is implicitly
   input-dependent through the layer outputs

The key mathematical property: softmax normalization bounds the hidden state
growth. Instead of ||h_L|| ~ O(L·||v||), we get ||h_L|| ≤ max_i ||v_i||
(bounded by the largest individual contribution, not their sum).

## 3. What Breaks It

### Insufficient Depth

At L=4 layers, the PreNorm dilution effect is weak (~25% per layer vs ~3%
at L=32). The benefit of selective depth attention may be negligible at small
depth. This is a KNOWN limitation of the micro experiment.

If K2 fails, this is the most likely explanation: the mechanism is real but
only manifests at L >> 4.

### Softmax Temperature

With L=4, the softmax has only 4 entries. Even learned pseudo-queries can
only produce attention distributions between uniform (1/4 each) and one-hot.
The dynamic range may be insufficient for fine-grained routing.

### Training Instability from Softmax Normalization

Standard residuals preserve the gradient flow: ∂h_L/∂h_l = I + ∂f_l/∂h_l.
AttnRes introduces softmax, which can create vanishing gradients for layers
with low attention weight. If α_{i→l} → 0, layer i receives no gradient
signal through the residual path.

Kill criterion K1 catches this: if AttnRes training is unstable, base
quality will degrade >10%.

### Uniform Attention = No Benefit

If the learned pseudo-queries remain near zero (uniform attention), AttnRes
degenerates to average-pooling over depth, which is WORSE than standard
residuals (sum vs average). K3 directly tests this.

## 4. Assumptions

1. **PreNorm dilution is a real problem at L=4:** May be too small to matter.
   Justified by: the experiment measures the effect, not assumes it.

2. **LoRA adapters create non-uniform per-layer contributions:** Supported by
   prior finding that adapter ΔW norms vary across layers (micro/models/
   bitnet_2b_real_composition). If all layers contribute equally, depth
   attention has nothing to specialize on.

3. **d parameters per layer is sufficient for depth routing:** The pseudo-query
   only needs to distinguish L=4 layer outputs. At d=128, this is massive
   overparameterization for a 4-way classification.

4. **Training signal propagates through softmax attention:** Standard in
   sequence-level attention (works in all transformers). Depth-level is
   analogous but with L tokens instead of n.

## 5. Complexity Analysis

| Component | Standard | AttnRes |
|-----------|----------|---------|
| Parameters | 0 (no residual params) | L × d = 4 × 128 = 512 |
| Forward pass per layer | O(1) addition | O(l × d) for attention scores |
| Total forward | O(L × d_model) | O(L^2 × d) for depth attention |
| Memory | O(d) per layer output | O(L × d) to store all v_i |

At L=4, d=128: overhead is trivial. At L=32, d=4096: L^2 × d = 32K scores
per position, still negligible vs O(n^2 × d) sequence attention.

## 6. Worked Example (d=4, L=3)

Suppose layer outputs after training:
```
v_0 = [1, 0, 0, 0]  (embedding)
v_1 = [0, 1, 0, 0]  (layer 1: base)
v_2 = [0, 0, 1, 1]  (layer 2: adapter-heavy, larger norm)
```

Pseudo-query for layer 3 (learned to attend to adapter layer):
```
w_3 = [0, 0, 0.5, 0.5]  (aligns with v_2's direction)
```

After RMSNorm (unit norm):
```
RMSNorm(v_0) = [1, 0, 0, 0]
RMSNorm(v_1) = [0, 1, 0, 0]
RMSNorm(v_2) = [0, 0, 1/√2, 1/√2]
```

Scores: w_3^T · RMSNorm(v_i):
```
s_0 = 0·1 + 0·0 + 0.5·0 + 0.5·0 = 0
s_1 = 0·0 + 0·1 + 0.5·0 + 0.5·0 = 0
s_2 = 0·0 + 0·0 + 0.5/√2 + 0.5/√2 = 1/√2 ≈ 0.707
```

Softmax: α = softmax([0, 0, 0.707])
  e^0 = 1.0, e^0.707 = 2.028, sum = 1 + 1 + 2.028 = 4.028
  α = [1/4.028, 1/4.028, 2.028/4.028] = [0.248, 0.248, 0.503]

Layer 3 input:
```
h_3 = 0.248·v_0 + 0.248·v_1 + 0.503·v_2
    = [0.248, 0.248, 0.503, 0.503]
```

vs standard residual: h_3 = v_0 + v_1 + v_2 = [1, 1, 1, 1] (uniform weight).

The adapter-heavy layer (v_2) gets ~2x more weight under AttnRes.

## 7. Connection to Architecture

In our SOLE architecture, adapters are composed per-layer with 1/N scaling:
```
h_l = h_{l-1} + f_l(h_{l-1}) + (1/N) Σ_j δ_l^j
```

AttnRes would replace the residual accumulation mechanism, affecting BOTH
the base model f_l contributions AND the adapter δ_l^j contributions. This
is a more fundamental change than routing (which selects WHICH adapters to
include) — it changes HOW all contributions are accumulated across depth.

Production context: Kimi's K2 (32B) uses AttnRes with 8 blocks at 48 layers.
MoDA (2603.15619) unifies sequence and depth attention, achieving +2.11% at
1.5B. Neither paper tests the interaction with LoRA composition, which is
the novel contribution of this experiment.

If AttnRes improves composition, it could be combined with our existing
Grassmannian skeleton + routing heads + 1/N scaling for a compounding benefit.
