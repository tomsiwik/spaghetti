# Mathematical Foundations: Dense GPT

This document derives the mathematics for each component of the dense GPT model
defined in `gpt.py`.

---

## 1. Embeddings and Positional Encoding

Tokens are embedded via two learned lookup tables that are summed element-wise:

$$
\mathbf{x}_t = \mathbf{W}_{te}[\text{token}_t] + \mathbf{W}_{pe}[t], \quad t \in \{0, \ldots, T-1\}
$$

where

- $\mathbf{W}_{te} \in \mathbb{R}^{V \times d}$ — token embedding matrix (vocabulary size $V$, model dimension $d$)
- $\mathbf{W}_{pe} \in \mathbb{R}^{T_{\max} \times d}$ — position embedding matrix (block size $T_{\max}$)

Both are **learned absolute embeddings**. No sinusoidal or rotary encoding is used. The
position index is simply `mx.arange(T)` passed to the position embedding table.

After summation, the sequence is passed through a pre-block `RMSNorm` (`norm0`) before
entering the layer stack.

---

## 2. RMSNorm

Standard LayerNorm computes mean and variance, then re-scales with learnable gain and
bias. RMSNorm (Zhang & Sennrich, 2019) drops the mean-centering and the learnable
parameters entirely, keeping only the RMS re-scaling:

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon}}
$$

In code:

```python
ms = mx.mean(x * x, axis=-1, keepdims=True)   # mean square
return x * mx.rsqrt(ms + self.eps)             # x / sqrt(ms + eps)
```

**Why no learnable gain/bias?** This is a deliberate simplification. The gain $\gamma$
and bias $\beta$ present in LayerNorm add $2d$ parameters per norm layer and require
additional Adam state. At micro scale the extra expressiveness is unnecessary and the
parameters add noise to comparisons across model variants. The model still learns
scale-appropriate representations because all downstream weights (projection matrices,
output head) can compensate.

---

## 3. Causal Self-Attention

### 3.1 Projections

For each position, the $d$-dimensional residual stream is linearly projected into queries,
keys, and values. With $h$ heads and head dimension $d_k = d / h$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\qquad W_Q, W_K, W_V \in \mathbb{R}^{d \times d}
$$

Each result is reshaped to $(B, h, T, d_k)$ so heads operate in parallel.

### 3.2 Scaled Dot-Product Attention

For a single head:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

where $M \in \mathbb{R}^{T \times T}$ is the **causal mask**:

$$
M_{ij} =
\begin{cases}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}
$$

The mask is constructed via `mx.triu(..., k=1)` filled with $-\infty$, added to the raw
attention logits before softmax. After softmax, the $-\infty$ positions become 0,
preventing any position from attending to future tokens.

The scale factor $1/\sqrt{d_k}$ prevents dot products from growing large in magnitude
(which would push softmax into near-zero gradient regions) as $d_k$ increases.

### 3.3 Output Projection

After multi-head attention, the $h$ head outputs are concatenated (via reshape to
$(B, T, d)$) and projected:

$$
\text{out} = \text{concat}(\text{head}_1, \ldots, \text{head}_h)\, W_O, \quad W_O \in \mathbb{R}^{d \times d}
$$

All four projection matrices ($W_Q, W_K, W_V, W_O$) have `bias=False`.

---

## 4. MLP (Feed-Forward Network)

Each block contains a two-layer MLP with a 4x hidden expansion:

$$
\text{MLP}(\mathbf{x}) = \mathbf{x} W_2 \cdot \text{ReLU}(\mathbf{x} W_1)
$$

More precisely:

$$
\mathbf{h} = \text{ReLU}(\mathbf{x} W_1), \quad W_1 \in \mathbb{R}^{d \times 4d}
$$
$$
\text{MLP}(\mathbf{x}) = \mathbf{h} W_2, \quad W_2 \in \mathbb{R}^{4d \times d}
$$

The expansion ratio 4 is inherited from the original Transformer (Vaswani et al., 2017)
and GPT-2. ReLU is used in place of GELU (see PAPER.md for rationale). Both weight
matrices have `bias=False`.

---

## 5. Transformer Block

Each of the $L$ blocks applies attention and MLP as residual branches, with pre-norm:

$$
\mathbf{x} \leftarrow \mathbf{x} + \text{Attn}(\text{RMSNorm}(\mathbf{x}))
$$
$$
\mathbf{x} \leftarrow \mathbf{x} + \text{MLP}(\text{RMSNorm}(\mathbf{x}))
$$

Pre-norm (normalize before the sublayer) is generally more stable at small scale than
post-norm, and is the standard for modern transformer variants.

---

## 6. Language Model Head

The final hidden state for each position is projected to vocabulary logits:

$$
\ell_t = \mathbf{x}_t W_{\text{head}}, \quad W_{\text{head}} \in \mathbb{R}^{d \times V}
$$

The model outputs raw logits; cross-entropy loss is computed externally by the training
loop. `bias=False` on `lm_head`.

---

## 7. Parameter Count

Let:
- $V$ = vocab size
- $T$ = block size (max sequence length)
- $d$ = `n_embd`
- $h$ = `n_head`
- $L$ = `n_layer`
- $d_k = d / h$ (head dimension, used for clarity but cancels out)

### Per-component count

| Component | Parameters |
|---|---|
| Token embedding $W_{te}$ | $V \cdot d$ |
| Position embedding $W_{pe}$ | $T \cdot d$ |
| Pre-block `norm0` (no params) | $0$ |
| Attention per block: $W_Q, W_K, W_V, W_O$ | $4 \cdot d^2$ |
| Attention norm (no params) | $0$ |
| MLP per block: $W_1, W_2$ | $d \cdot 4d + 4d \cdot d = 8d^2$ |
| MLP norm (no params) | $0$ |
| LM head $W_{\text{head}}$ | $d \cdot V$ |

### Total

$$
N_{\text{params}} = 2Vd + Td + L \cdot 12d^2
$$

Breaking down the per-layer $12d^2$: four attention matrices ($4d^2$) plus two MLP
matrices ($8d^2$).

### Example (default config)

Default: $V=28$, $T=32$, $d=64$, $h=4$, $L=4$.

$$
N = 2 \cdot 28 \cdot 64 + 32 \cdot 64 + 4 \cdot 12 \cdot 64^2
= 3{,}584 + 2{,}048 + 196{,}608 = 202{,}240
$$

At this scale the layer stack dominates; embeddings and head contribute less than 3% of
total parameters.
