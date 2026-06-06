# XSA for Adapter Composition: Mathematical Foundations

## 1. Background: Self-Value Bias in Attention

Standard self-attention computes output for position i:

$$y_i = \sum_{j=1}^{T} a_{i,j} v_j$$

where a_{i,j} = softmax(q_i^T k_j / sqrt(d_h))_j.

Decompose into self-component and context:

$$y_i = a_{i,i} v_i + c_i, \quad c_i = \sum_{j \neq i} a_{i,j} v_j$$

**Problem:** Because softmax outputs are always positive, a_{i,i} > 0 always.
Combined with positive correlation among value vectors within a sequence,
this creates "attention similarity bias": cos(y_i, v_i) >> 0. The attention
output wastes capacity redundantly encoding self-information that already
flows through the residual connection.

## 2. XSA: Exclusive Self-Attention

XSA removes the self-value component via orthogonal projection:

$$z_i = y_i - \frac{y_i^T v_i}{\|v_i\|^2} v_i$$

Properties (by construction):
- cos(z_i, v_i) = 0 (zero self-similarity)
- Context preserved: component of c_i orthogonal to v_i passes through unchanged
- Zero new parameters
- Compute cost: O(T * d_h) per head (dot product + projection per position)

### Numerical Example (d_h = 32, T = 16)

Per-head compute for XSA:
- dot(y_i, v_i): d_h multiplies = 32 FLOPs per position
- ||v_i||^2: d_h multiplies = 32 FLOPs per position
- scale + subtract: 2 * d_h = 64 FLOPs per position
- Total: 128 * T = 2048 FLOPs per head

Standard attention cost: O(T^2 * d_h) = 16^2 * 32 = 8192 FLOPs per head
XSA overhead: 2048/8192 = 25% extra FLOPs (but mostly element-wise, fast on GPU)

## 3. Why XSA Should Help Adapter Composition

### The Interference Mechanism

When k adapters are pre-merged:
$$W_{merged} = W_{base} + \frac{1}{k} \sum_{i=1}^{k} B_i A_i^T$$

The attention output becomes:
$$y_i^{merged} = \text{Attn}(x W_Q^{merged}, x W_K^{merged}, x W_V^{merged})$$

Each adapter perturbs Q, K, and V projections. Cross-terms arise because:
1. Query perturbation delta_Q interacts with all key perturbations delta_K_j
2. Value perturbation delta_V changes what information is gathered
3. LayerNorm + softmax nonlinearities amplify these cross-terms

### XSA's Compositional Benefit

**Hypothesis:** The self-value bias amplifies adapter interference because
it couples each position's output to its own (perturbed) value vector.

With k merged adapters, the value at position i is:
$$v_i^{merged} = x_i (W_V + \frac{1}{k} \sum B_j A_j^T)$$

The self-value component a_{i,i} * v_i^{merged} contains ALL adapter
perturbations scaled by the self-attention weight. This is a direct
interference channel — each adapter's V perturbation at position i
contributes to the output regardless of whether that adapter is relevant.

XSA removes this channel. After projection:
$$z_i = y_i - \text{proj}_{v_i^{merged}}(y_i)$$

The remaining output z_i contains only the cross-position context, which
is more robustly averaged across multiple positions and thus less sensitive
to single-adapter perturbations.

### Expected Effect Size

At micro scale (d=128, H=4, L=4, r=8), the effect may be small because:
1. d_h = 32 limits expressiveness per head
2. Short sequences (T ~ 24) mean few positions to gather context from
3. Toy domains have limited vocabulary overlap

We use 3-seed validation to distinguish signal from noise.

## 4. Experimental Design

### Architecture
- Vocabulary: ~40 chars (toy character-level)
- d_model = 128, n_heads = 4, d_head = 32
- n_layers = 4
- FFN hidden = 512 (4x expansion, SwiGLU)
- Max sequence length = 32
- Ternary base (post-quantization from FP16)

### Adapter Setup
- LoRA rank r = 8 (A frozen Grassmannian, B trained with STE for ternary)
- 5 domains: arithmetic, reverse, repeat, sort, parity
- Train each adapter independently on its domain
- Two conditions: standard attention vs XSA in last 2 layers

### XSA Application
- Apply XSA to last 2 layers only (layers 2 and 3)
- Per parameter-golf: full-depth hurts, last 3-4 layers captures most benefit
- At L=4, last 2 layers = 50% of network depth (proportional to 3-4/11 = 27-36%)

### Metrics
1. **Single-adapter PPL** per domain (K1: XSA vs standard, < 3% degradation)
2. **Composition ratio** = PPL(composed) / PPL(single-best) (K2: XSA < standard)
3. **Per-domain composed PPL** (K3: XSA wins on >= 3/5 domains)
4. **Adapter cosine similarity** |cos(delta_W_i, delta_W_j)| (diagnostic)
5. **Attention self-similarity** cos(y_i, v_i) before/after XSA (diagnostic)

### Kill Criteria
- K1: XSA degrades single-adapter quality > 3% PPL on any domain -> KILL
- K2: XSA composition ratio >= no-XSA ratio (3-seed mean) -> KILL
- K3: XSA+composition worse than no-XSA on >= 3/5 domains -> KILL

## 5. Dimensions and Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| x | (B, T, d) | Input embeddings, d=128 |
| W_QKV | (d, 3d) | Fused QKV projection |
| q, k, v | (B, H, T, d_h) | Per-head Q/K/V, H=4, d_h=32 |
| attn_weights | (B, H, T, T) | Attention scores after softmax |
| y | (B, H, T, d_h) | Attention output pre-XSA |
| z | (B, H, T, d_h) | Attention output post-XSA |
| A_i | (d_in, r) | LoRA A matrix (frozen), r=8 |
| B_i | (r, d_out) | LoRA B matrix (trained, ternary) |
| delta_W_i | (d_in, d_out) | Effective adapter: B_i @ A_i^T |

## 6. Assumptions

1. Self-value bias is present at micro scale (d_h=32) — validated by parameter-golf
2. Residual connection adequately carries self-information even without self-value in attention
3. Ternary quantization does not eliminate the self-value bias (quantization is post-softmax)
4. XSA in last 2/4 layers is proportionally similar to 3-4/11 in parameter-golf
5. Cross-position context (c_i) is more robust to adapter perturbations than self-value (a_{i,i} * v_i)
