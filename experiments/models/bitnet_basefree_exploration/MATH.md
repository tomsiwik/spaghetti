# Base-Free Composition: Mathematical Foundations

## Setup

Let $\mathcal{M}$ be a transformer with $L = 30$ layers. Each layer $\ell$ has
$M = 7$ linear projections (q, k, v, o, gate, up, down). The base weight matrix
for projection $m$ in layer $\ell$ is $W_{\ell,m} \in \{-1, 0, 1\}^{d_{out} \times d_{in}}$
(ternary, BitNet-2B-4T architecture, $d = 2560$).

A LoRA adapter $i$ adds rank-$r$ perturbation:
$$\Delta W_{\ell,m}^{(i)} = B_{\ell,m}^{(i)} A_{\ell,m}^{(i)} \in \mathbb{R}^{d_{out} \times d_{in}}$$
where $A \in \mathbb{R}^{d_{in} \times r}$, $B \in \mathbb{R}^{r \times d_{out}}$, $r = 16$.

Composed model with $N = 5$ adapters under $1/N$ scaling:
$$\hat{W}_{\ell,m} = W_{\ell,m} + \frac{1}{N} \sum_{i=1}^{N} \alpha \cdot B_{\ell,m}^{(i)} A_{\ell,m}^{(i)}$$
where $\alpha = 20.0$ is the LoRA scale factor.

## Layer Criticality Model

Define the **criticality** of layer $\ell$ as the relative PPL increase when
that layer's base weights are zeroed:

$$c_\ell = \frac{\text{PPL}(\mathcal{M}_{W_\ell = 0}) - \text{PPL}(\mathcal{M})}{\text{PPL}(\mathcal{M})}$$

**Empirical observation**: Criticality follows a U-shaped profile across depth:

| Layer range | Mean $c_\ell$ | Interpretation |
|-------------|---------------|----------------|
| 0-4 (input) | 58,269% | Embedding interface |
| 5-9 (early) | 7.8% | Feature extraction |
| 10-16 (middle) | 3.5% | Abstract representation |
| 17-23 (late) | 7.8% | Feature synthesis |
| 24-29 (output) | 17.8% | Logit projection |

The U-shape is consistent with residual stream theory: the first and last
layers perform critical format transformations (input embedding -> internal
representation -> output logits), while middle layers perform incremental
refinements that partially overlap.

## Progressive Ablation Model

Let $\sigma$ be the ordering of layers by ascending criticality. Define
$K$-layer ablation as zeroing the $K$ least critical layers:

$$\mathcal{M}_K = \mathcal{M} \text{ with } W_{\sigma(1)}, \ldots, W_{\sigma(K)} = 0$$

**Empirical PPL ratios** (relative to composed baseline):

| $K$ | PPL ratio | Effective layers remaining |
|-----|-----------|--------------------------|
| 0 | 1.00x | 30 |
| 1 | 1.03x | 29 |
| 3 | 1.13x | 27 |
| 5 | 1.40x | 25 |
| 10 | 3.29x | 20 |
| 15 | 28.1x | 15 |
| 20 | 324,918x | 10 |

The relationship is approximately:
$$\text{PPL ratio}(K) \sim \exp\left(\beta \cdot K^\gamma\right)$$

Fitting to the data gives $\gamma \approx 2.1$, indicating **super-exponential
degradation** — each additional zeroed layer is worse than the last, because
the critical layers are loaded toward the ends.

## Scaffold Replacement Analysis

Replacing ALL base weights with random ternary $\tilde{W}_{\ell,m} \sim \text{Uniform}\{-1, 0, 1\}$
scaled to match the Frobenius norm of the original:

$$\tilde{W}_{\ell,m} = T_{\ell,m} \cdot \frac{\|W_{\ell,m}\|_F}{\|T_{\ell,m}\|_F}$$
where $T_{\ell,m} \sim \text{Uniform}\{-1, 0, 1\}^{d_{out} \times d_{in}}$.

**Result**: PPL = $3.19 \times 10^8$ vs base PPL = 11.55.

This is a ratio of $2.76 \times 10^7$, far exceeding the 5x kill threshold.
The random ternary scaffold has no meaningful computation — it is equivalent
to a completely untrained model.

### Why Norm-Matching Is Insufficient

Norm-matching preserves $\|W\|_F$ but not:
1. **Spectral structure**: The pretrained weights have learned singular value
   distributions that encode meaningful transformations
2. **Cross-layer coherence**: Layer $\ell$ expects specific activation
   distributions from layer $\ell - 1$
3. **Attention patterns**: Q/K/V matrices must produce coherent attention;
   random Q/K produce uniform attention

### Why Adapters Cannot Compensate

The adapter contribution per layer is:
$$\|\Delta W\|_F = \alpha \cdot \|BA\|_F \approx \alpha \cdot \sigma_{\max}(B) \cdot \sigma_{\max}(A) \cdot \sqrt{r}$$

With $\alpha = 20$, $r = 16$, and typical $\|A\| \sim 0.02$, $\|B\| \sim 0.01$:
$$\|\Delta W\|_F \approx 20 \cdot 0.01 \cdot 0.02 \cdot 4 = 0.016$$

But $\|W\|_F \sim \sqrt{d_{in} \cdot d_{out} / 3} \approx 50$ for ternary
weights with $\sim$2/3 nonzero entries.

The adapter perturbation is $\sim 0.03\%$ of the base weight norm — it is a
fine-tuning signal on top of massive base computation, not a replacement for it.

## Comparison with Toy-Scale Prior Art

At $d = 64$ (micro/models/base_free_composition/), skeleton-only gave:
- Base loss ratio: 6.94x
- Expert loss ratio: 1.27x

At $d = 2560$ (this experiment), skeleton-only gives:
- PPL ratio: $\sim 10^7$x

The degradation is catastrophically worse at scale. This is because:
1. At $d = 64$, the model is tiny (4 layers) and adapters are relatively
   large compared to base weights
2. At $d = 2560$, the model is deep (30 layers) and cascading errors
   through 30 layers of random computation destroy all signal
3. The ratio of adapter rank to model dimension is $r/d = 16/2560 = 0.006$
   (vs $8/64 = 0.125$ at toy scale) — adapters are proportionally 20x smaller

## Implications

1. **Base-free composition is NOT viable** at $d = 2560$ by scaffold replacement.
   The pretrained base is not a "scaffold" — it IS the model.

2. **Middle layers ARE partially replaceable**: 9/30 layers can be individually
   zeroed with <5% PPL impact. But this is a single-layer effect; it does not
   compose (zeroing 5 least-critical simultaneously gives 40% impact).

3. **True base-free requires training-from-scratch**: The SVD decomposition
   approach (base_free_composition at toy scale) or ReLoRA (train base
   incrementally) are the only viable paths. Adapter-only composition on a
   random scaffold is fundamentally impossible.

## Worked Example

For layer 14 (least critical, $c_{14} = 2.57\%$):

- Composed PPL with pretrained layer 14: 10.24
- Composed PPL with zeroed layer 14: 10.51
- PPL increase: $(10.51 - 10.24) / 10.24 = 2.6\%$
- The LoRA adapters and residual stream carry enough signal through this
  layer that the base weights add only marginal value

For layer 0 (most critical, $c_0 = 290,373\%$):
- Composed PPL with pretrained layer 0: 10.24
- Composed PPL with zeroed layer 0: 29,758
- PPL increase: $(29758 - 10.24) / 10.24 = 290,373\%$
- This layer converts token embeddings into the internal representation
  format expected by all subsequent layers. Without it, the entire
  computation is incoherent.
