# Ternary Base From Scratch: Mathematical Foundations

## Notation

| Symbol | Definition | Shape |
|--------|-----------|-------|
| $d$ | model dimension | scalar (256) |
| $L$ | number of layers | scalar (6) |
| $H$ | number of heads | scalar (4) |
| $h = d/H$ | head dimension | scalar (64) |
| $V$ | vocabulary size | scalar (28, a-z + BOS + pad) |
| $T$ | sequence length (block size) | scalar (32) |
| $r$ | LoRA rank | scalar (8) |
| $N$ | number of domain adapters | scalar (5) |
| $\alpha$ | absmean scale factor | scalar per weight matrix |
| $W$ | latent FP32 weight | varies per layer |
| $\hat{W}$ | ternary weight $\in \{-\alpha, 0, +\alpha\}$ | same as $W$ |

## 1. BitLinear: Ternary Weights with STE

### Forward Pass (Quantized)

For each weight matrix $W \in \mathbb{R}^{m \times n}$:

$$\alpha = \text{mean}(|W|)$$

$$\hat{W} = \alpha \cdot \text{clip}(\text{round}(W / \alpha), -1, 1)$$

The forward pass computes $y = \hat{W} x$ using the quantized weights.

### Backward Pass (Straight-Through Estimator)

The STE passes gradients through the non-differentiable quantization:

$$\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial \hat{W}}$$

This is exact when $|W_{ij}/\alpha| < 1.5$ (the round function is locally identity).
The clip operation kills gradients for weights already saturated at $\pm 1$.

### Implementation

In MLX, we maintain latent FP32 weights $W$ and compute $\hat{W}$ in the forward pass.
The `stop_gradient` trick implements STE:

```python
w_q = quantize(w)              # {-alpha, 0, alpha}
w_ste = w + stop_gradient(w_q - w)  # forward: w_q, backward: dL/dw
```

This works because:
- Forward: `w_ste = w + (w_q - w) = w_q` (quantized)
- Backward: $\nabla_w [w + \text{sg}(w_q - w)] = I$ (straight-through)

## 2. Architecture

Standard transformer with BitLinear replacing all nn.Linear layers except embeddings:

$$x_0 = E_{\text{tok}}[\text{tokens}] + E_{\text{pos}}[\text{positions}]$$

For each layer $\ell = 1, \ldots, L$:

$$h_\ell = \text{RMSNorm}(x_{\ell-1})$$
$$x_\ell = x_{\ell-1} + \text{Attn}_\ell(h_\ell) + \text{MLP}_\ell(\text{RMSNorm}(x_{\ell-1} + \text{Attn}_\ell(h_\ell)))$$

Where attention uses quantized Q, K, V, O projections and MLP uses quantized fc1, fc2.

### Parameter Count

Embeddings (FP32, not quantized):
- Token embedding: $V \times d = 28 \times 256 = 7{,}168$
- Position embedding: $T \times d = 32 \times 256 = 8{,}192$

Per layer (BitLinear):
- Q, K, V, O projections: $4 \times d^2 = 4 \times 65{,}536 = 262{,}144$
- MLP (fc1 + fc2): $d \times 4d + 4d \times d = 2 \times 262{,}144 = 524{,}288$
- Layer norms (FP32): $2 \times d = 512$

Total per layer: $786{,}944$
Total 6 layers: $4{,}721{,}664$
LM head (BitLinear): $d \times V = 7{,}168$
Final norm: $256$

**Total: ~4.74M parameters** (latent FP32 for training, ternary at inference)

### Effective Ternary Storage

At inference, ternary weights need only 2 bits per parameter:
- Ternary weight matrices: $4{,}721{,}664 + 7{,}168 = 4{,}728{,}832$ params at 2 bits = 1.13 MB
- Plus per-matrix scale factors $\alpha$: negligible
- Plus FP32 embeddings and norms: ~60 KB
- **Total inference: ~1.2 MB** vs ~18 MB for FP32

## 3. Expected PPL Gap

From BitNet b1.58 (Ma et al., 2024), the ternary-from-scratch gap is:
- ~0% at 3.9B parameters
- ~2-5% at 700M parameters
- At 4.7M parameters (our scale), gap is unknown but expected to be larger

The kill criterion K2 allows 3x the FP32 baseline PPL. For names dataset at d=256:
- FP32 baseline: ~16 PPL (from prior experiments)
- Random baseline: $V = 28 \Rightarrow \text{PPL} = 28$ (uniform distribution)
- Kill threshold: $16 \times 3 = 48$ PPL

## 4. Adapter Composition on Ternary Base

### Ternary LoRA with Grassmannian Init

Each adapter $i$ is a rank-$r$ LoRA: $\Delta W_i = B_i A_i$

- $A_i \in \mathbb{R}^{r \times d}$: frozen, pre-computed on Grassmannian $\text{Gr}(r, d)$
- $B_i \in \mathbb{R}^{d \times r}$: trained, then quantized to ternary

Grassmannian guarantee: $A_i^T A_j \approx 0$ for $i \neq j$, giving:

$$\|\Delta W_i^T \Delta W_j\| \leq \frac{\alpha_i \alpha_j}{r^2} \|B_i\| \|A_i^T A_j\| \|B_j\| \approx 0$$

### Composition

$N$-adapter composition with equal weights:

$$W_{\text{composed}} = \hat{W}_{\text{base}} + \frac{1}{N} \sum_{i=1}^{N} \Delta W_i$$

The composition ratio measures degradation:

$$\gamma = \frac{\text{PPL}_{\text{composed}}}{\text{mean}(\text{PPL}_{\text{single}_i})}$$

Kill criterion K3: $\gamma < 2.0$ (prior work on BitNet achieves $\gamma \approx 1.1$).

### Orthogonality Metric

For adapter pairs $(i, j)$, we measure cosine similarity of the flattened deltas:

$$\cos(\Delta W_i, \Delta W_j) = \frac{\text{vec}(\Delta W_i) \cdot \text{vec}(\Delta W_j)}{\|\text{vec}(\Delta W_i)\| \|\text{vec}(\Delta W_j)\|}$$

Success criterion S3: mean $|\cos| < 0.05$.

## 5. Worked Example (d=64, r=4, L=2)

Tiny sanity check:
- Latent weight $W \in \mathbb{R}^{64 \times 64}$, $\alpha = 0.02$ (init scale)
- $\hat{W}$: most entries round to 0 at init (since $|W_{ij}|/\alpha \approx 1$ for Gaussian init)
- After training: weights cluster at $\{-\alpha, 0, +\alpha\}$, sparsity ~40-50%
- Adapter: $A \in \mathbb{R}^{4 \times 64}$ (frozen), $B \in \mathbb{R}^{64 \times 4}$ (ternary trained)
- $\Delta W = BA \in \mathbb{R}^{64 \times 64}$, effective rank 4
- Storage: $64 \times 4 = 256$ ternary params = 64 bytes
