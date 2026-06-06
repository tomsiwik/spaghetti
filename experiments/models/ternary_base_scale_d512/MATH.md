# Ternary Base Scale d=512: Mathematical Foundations

## Notation

| Symbol | Definition | Shape |
|--------|-----------|-------|
| $d$ | model dimension | scalar (512) |
| $L$ | number of layers | scalar (8) |
| $H$ | number of heads | scalar (8) |
| $h = d/H$ | head dimension | scalar (64) |
| $V$ | vocabulary size (GPT-2 BPE) | scalar (50,257) |
| $T$ | sequence length (block size) | scalar (128) |
| $\alpha$ | absmean scale factor | scalar per weight matrix |
| $W$ | latent FP32 weight | varies per layer |
| $\hat{W}$ | ternary weight $\in \{-\alpha, 0, +\alpha\}$ | same as $W$ |
| $B$ | batch size | scalar (32) |

## 1. BitLinear: Ternary Weights with STE

Identical mechanism to d=256 experiment. For each weight matrix $W \in \mathbb{R}^{m \times n}$:

$$\alpha = \text{mean}(|W|)$$
$$\hat{W} = \alpha \cdot \text{clip}(\text{round}(W / \alpha), -1, 1)$$

Forward: $y = \hat{W}x$ (ternary). Backward: $\nabla_W \approx \nabla_{\hat{W}}$ (STE).

Implementation: $w_{\text{ste}} = w + \text{stop\_gradient}(\hat{w} - w)$.

## 2. Architecture

### Differences from d=256 Experiment

| Parameter | d=256 (prior) | d=512 (this) |
|-----------|---------------|--------------|
| d_model | 256 | 512 |
| n_layers | 6 | 8 |
| n_heads | 4 | 8 |
| head_dim | 64 | 64 |
| MLP dim | 1024 | 2048 |
| vocab_size | 27 (char) | 50,257 (BPE) |
| block_size | 32 | 128 |
| Total params | 4.7M | 76.7M |

### Parameter Count

Embeddings (FP32, not quantized):
- Token embedding: $V \times d = 50{,}257 \times 512 = 25{,}731{,}584$
- Position embedding: $T \times d = 128 \times 512 = 65{,}536$

Per layer (BitLinear):
- Q, K, V, O projections: $4 \times d^2 = 4 \times 262{,}144 = 1{,}048{,}576$
- MLP (fc1 + fc2): $d \times 4d + 4d \times d = 2 \times 1{,}048{,}576 = 2{,}097{,}152$
- Layer norms (FP32): $2 \times d = 1{,}024$

Total per layer: $3{,}146{,}752$
Total 8 layers: $25{,}174{,}016$
LM head (BitLinear): $d \times V = 25{,}731{,}584$
Final norm: $512$

**Total: ~76.7M parameters**

### Memory Budget

Training (FP32 latent + Adam):
- Model weights: 76.7M * 4B = 307 MB
- Adam state (m, v): 76.7M * 4B * 2 = 614 MB
- Gradients: 307 MB
- **Total parameter memory: ~1.2 GB**

Activations (batch_size=32, seq_len=128):
- Per-layer activations: $B \times T \times d \times 4$ = 8.4 MB
- 8 layers with attention intermediates: ~200 MB
- **Total activation memory: ~200 MB**

**Total: ~1.5 GB** (well within 40 GB usable on M5 Pro 48GB)

## 3. Expected PPL Gap (Key Prediction)

From LEARNINGS.md of the d=256 experiment:

> "Expect 1.5-2.5x PPL ratio on real tasks at sub-3B scale"

This prediction is grounded in:
- **pQuant**: 1-bit QAT scales sublinearly -- gap widens before narrowing at 3B+
- **BitNet b1.58**: ~0% gap at 3.9B, ~2-5% at 700M, unknown at 77M
- **Spectra**: 3.9B ternary matches FP16 on reasoning, slight lag on web corpora

At 77M params with real English text (50K vocab BPE), we predict:
- FP32 baseline PPL: ~40-80 (small model on real text is high)
- Ternary PPL: 1.5-2.5x FP32 = 60-200
- Kill threshold (K2): 2x FP32 PPL

The task is now genuinely hard (50K vocab, real English), so the quantization constraint will be binding -- unlike the overcapacity regime of the d=256/vocab-27 experiment.

## 4. Deadzone Analysis

### Definition
A weight is "deadzoned" if its latent FP32 value is close enough to zero that STE cannot push it across a ternary boundary within remaining training. The ternary boundaries are at $\pm \alpha/2$ where $\alpha = \text{mean}(|W|)$.

### Expected Behavior
- BitNet b1.58 reports 30-40% zeros as healthy learned sparsity
- Sparse-BitNet (2603.05168) reports natural 42% sparsity
- K3 threshold: 40% zeros (above which we consider it pathological)

### What We Track
- Zero-weight fraction per 1K steps (evolution over training)
- Per-layer zero fraction (to detect if deep layers trap worse, per STE blind spot hypothesis)
- Overall deadzone at convergence

## 5. STE Depth Concern

From LEARNINGS:
> "STE 'blind spot' grows with depth -- 6-layer was fine, 8-layer may show issues"

The STE gradient approximation error accumulates through layers:
$$\epsilon_\ell = \prod_{i=1}^{\ell} (I + \delta_i)$$

where $\delta_i$ is the per-layer quantization-gradient mismatch. At 8 layers, this product may become significant. We monitor convergence speed and final loss to detect this.

## 6. Training Schedule

### FP32 Baseline
- Steps: 5,000
- LR: 3e-4 with cosine warmup (500 steps warmup)
- Batch: 32 sequences of 128 tokens = 4,096 tokens/step
- Total tokens: 5K * 4096 = 20.5M tokens

### Ternary STE
- Steps: 10,000 (K1 kill if no convergence)
- LR: 1e-3 (3.3x larger than FP32, per LEARNINGS recommendation for STE)
- Same batch config: 4,096 tokens/step
- Total tokens: 10K * 4096 = 41M tokens

### Why Higher LR for Ternary
From LEARNINGS:
> "Large learning rates (3-10x default) may be needed for STE to push latent weights across ternary boundaries"

The STE gradient is an approximation. Larger LR compensates for the fact that the true gradient direction is partially masked by the quantization step-function.
