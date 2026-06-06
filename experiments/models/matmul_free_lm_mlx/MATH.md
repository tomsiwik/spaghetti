# MatMul-Free LM on MLX: Mathematical Foundations

## Overview

The MatMul-free LM (Zhu et al., 2024, arxiv 2406.02528) eliminates all dense
matrix multiplications from a language model by combining:
1. **Ternary BitLinear layers** — weights in {-1, 0, +1}, so `x @ W^T` becomes
   additions/subtractions only (no floating-point multiply)
2. **HGRN2 Token Mixer** — replaces self-attention with a gated linear recurrence
   (no QK^T matmul)
3. **GLU Channel Mixer** — replaces MLP with gated linear unit using element-wise
   products (Hadamard, not matmul)

## 1. BitLinear (Ternary Dense Layer)

### Forward pass

Given input `x in R^{B x T x d}` and latent weights `W in R^{d_out x d_in}`:

```
alpha = mean(|W|)                           # per-tensor scale
W_scaled = W / (alpha + eps)                # normalize
W_q = clip(round(W_scaled), -1, 1) * alpha  # quantize to {-a, 0, +a}
W_ste = W + sg(W_q - W)                     # STE: forward=W_q, backward=W
y = RMSNorm(x) @ W_ste^T                   # the "matmul" is ternary accumulation
```

At inference (post-quantization), `y` requires only additions and subtractions
since `W_q[i,j] in {-alpha, 0, +alpha}`. For each output:
```
y[..., i] = alpha * (sum of x[...,j] where W_q[i,j]=+1)
           - alpha * (sum of x[...,j] where W_q[i,j]=-1)
```

FLOPs: O(d_in * d_out) additions vs O(d_in * d_out) multiply-adds for dense.

### Extra RMSNorm (arxiv 2505.08823)

From warmstart experiment learnings: placing an RMSNorm before the quantized
matmul is the single most impactful modification for ternary training
(76% gap reduction). Every BitLinear includes this pre-quantization norm.

## 2. HGRN2 Token Mixer (Gated Linear Recurrence)

### Motivation

Standard self-attention computes `Q @ K^T` — an O(T^2 d) matmul. The
HGRN2 token mixer replaces this with a linear recurrence that is O(T d^2/H)
where H is the number of heads, and uses no matmul.

### Architecture

For input `x in R^{B x T x d}`:

```
# Project to gates and values (using BitLinear, no matmul)
q = BitLinear_q(x)    # R^{B x T x d}
k = BitLinear_k(x)    # R^{B x T x d}
v = BitLinear_v(x)    # R^{B x T x d}

# Reshape to heads: each head has dim d_h = d/H
q = reshape(q, [B, T, H, d_h])
k = reshape(k, [B, T, H, d_h])
v = reshape(v, [B, T, H, d_h])

# Forget gate (sigmoid) — controls information retention
f = sigmoid(BitLinear_f(x))   # R^{B x T x d}, reshaped to [B, T, H, d_h]

# Lower-rank key for state expansion: k -> (alpha, beta) of rank r
# alpha in R^{B x T x H x r}, beta in R^{B x T x H x r}
# where r << d_h is the expansion rank
# Outer product: kv = alpha^T @ beta gives R^{r x r} per head per time
# But we use a simpler formulation for micro scale:

# Recurrent state update (the core no-matmul operation):
# S_t = f_t * S_{t-1} + k_t^T * v_t   (outer product, element-wise gate)
# o_t = q_t * S_t                       (element-wise, then sum over key dim)

# For efficiency at micro scale, we use the "simplified HGRN" form:
# g_t = sigmoid(W_g @ x_t)              # forget gate
# i_t = W_i @ x_t                       # input (value)
# h_t = g_t * h_{t-1} + (1 - g_t) * i_t # gated recurrence

# Output projection
o = BitLinear_o(h)   # R^{B x T x d}
```

### Simplified Form (our implementation)

We implement a simplified HGRN that captures the core mechanism without
the state expansion. For each head:

```
g_t = sigmoid(lower_bound + W_g @ x_t)     # forget gate, lower_bound=log(0.9) ensures retention
i_t = silu(W_i @ x_t) * W_v @ x_t          # gated input (GLU-style)
h_t = g_t * h_{t-1} + (1 - g_t) * i_t      # linear recurrence
```

This is O(T * d) per step — linear in sequence length, no matmul for attention.

### Complexity comparison

| Operation | Self-Attention | HGRN Token Mixer |
|-----------|---------------|------------------|
| QK^T      | O(T^2 d)     | None             |
| Recurrence| None          | O(T d)           |
| Total     | O(T^2 d + Td) | O(Td)           |

## 3. GLU Channel Mixer

Replaces the standard 2-layer MLP `x -> W1(ReLU(W2(x)))` with:

```
gate = BitLinear_gate(x)        # R^{B x T x d_ff}
value = BitLinear_up(x)         # R^{B x T x d_ff}
y = silu(gate) * value          # element-wise Hadamard product
y = BitLinear_down(y)           # R^{B x T x d}
```

The SiLU(gate) * value is a Hadamard product (O(d_ff) element-wise multiply),
NOT a matrix multiplication. Combined with ternary BitLinear projections,
the entire channel mixer uses zero dense matmuls.

## 4. Full Model Architecture

```
x = Embedding(tokens) + PositionEmbedding(positions)
for each layer:
    x = x + TokenMixer(RMSNorm(x))   # HGRN recurrence
    x = x + ChannelMixer(RMSNorm(x)) # GLU with Hadamard
x = RMSNorm(x)
logits = LMHead(x)                    # BitLinear projection
```

### Parameter count at micro scale

| Component | Shape | Params | Count |
|-----------|-------|--------|-------|
| Embedding | V x d = 28 x 256 | 7,168 | 1 |
| Pos Embed | T x d = 32 x 256 | 8,192 | 1 |
| BitLinear (gate) | d x d = 256 x 256 | 65,536 | 6 |
| BitLinear (input) | d x d = 256 x 256 | 65,536 | 6 |
| BitLinear (value) | d x d = 256 x 256 | 65,536 | 6 |
| BitLinear (output) | d x d = 256 x 256 | 65,536 | 6 |
| GLU gate | d x 4d = 256 x 1024 | 262,144 | 6 |
| GLU up | d x 4d = 256 x 1024 | 262,144 | 6 |
| GLU down | 4d x d = 1024 x 256 | 262,144 | 6 |
| RMSNorm | d = 256 | 256 | ~18 |
| LM Head | d x V = 256 x 28 | 7,168 | 1 |
| **Total** | | | **~6.3M** |

With RMSNorm inside BitLinear (extra norm): add ~30 more RMSNorms = ~7,680.

## 5. LoRA on HGRNBit

LoRA can be applied to any linear projection in HGRNBit. The candidate layers:

- Token Mixer: W_gate, W_input, W_value, W_output (4 per layer)
- Channel Mixer: W_gate_up, W_value_up, W_down (3 per layer)

For rank-8 LoRA on all 7 layers per block, 6 blocks:
- LoRA params per layer: (d * r + r * d_out) = 256*8 + 8*256 = 4,096 per projection
- Total LoRA: 7 projections * 6 layers * 4,096 = 172,032 params
- LoRA/Total ratio: 172K / 6.3M = 2.7%

### Composition

LoRA composition works identically to Transformer case:
```
W_composed = W_base + sum_i (B_i @ A_i) / N
```

The key question is whether the HGRN recurrence creates implicit coupling
between adapters that breaks the compositional guarantee. The Grassmannian
skeleton should still work because:
1. Each BitLinear is still a linear projection (LoRA applies as normal)
2. The recurrence is element-wise gating (no cross-adapter interaction)
3. Hadamard products in GLU are element-wise (no cross-adapter interaction)

## 6. Kill Criteria Thresholds

- K1: val_loss > 2.0 within 2000 steps. The FP32 Transformer baseline achieves
  ~1.5 val loss on the names dataset at d=256. A matmul-free model should reach
  comparable range. Loss > 2.0 indicates fundamental convergence failure.

- K2: LoRA adapters incompatible. This would mean either (a) BitLinear doesn't
  support LoRA wrapping, or (b) the recurrence structure prevents gradient flow
  through LoRA params. Both are testable.

- K3: composition_ratio > 1.5. Our Transformer baseline achieves ~1.02-1.1.
  The HGRN recurrence could amplify composition interference through the
  sequential gating. Ratio > 1.5 means the architecture is fundamentally
  hostile to composition.
