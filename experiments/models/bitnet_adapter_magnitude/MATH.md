# BitNet Adapter Magnitude Analysis: Mathematical Framework

## Setup

Consider a frozen base model with weight matrices $W \in \mathbb{R}^{d_{in} \times d_{out}}$
and N LoRA adapters producing deltas $\Delta_i = B_i A_i$ where $A_i \in \mathbb{R}^{d_{in} \times r}$,
$B_i \in \mathbb{R}^{r \times d_{out}}$.

Two base types:
- **FP16 base**: $W \in \mathbb{R}^{d_{in} \times d_{out}}$, continuous weights
- **Ternary base**: $\tilde{W} = \alpha \cdot Q(W)$ where $Q(W) \in \{-1, 0, 1\}^{d_{in} \times d_{out}}$
  and $\alpha = \text{mean}(|W|)$ (BitNet absmean quantization)

## Key Quantities

### Delta Norm
$$\|\Delta_i\|_F = \|B_i A_i\|_F$$

### Norm Variance (Kill Criterion K1)
$$\text{Var}(\|\Delta\|) = \frac{1}{N} \sum_{i=1}^{N} (\|\Delta_i\|_F - \bar{\|\Delta\|})^2$$

where $\bar{\|\Delta\|} = \frac{1}{N}\sum_i \|\Delta_i\|_F$.

### Max/Min Norm Ratio (Kill Criterion K2)
$$\rho = \frac{\max_i \|\Delta_i\|_F}{\min_i \|\Delta_i\|_F}$$

### Composition Delta Norm
Under equal-weight 1/N composition:
$$\left\|\frac{1}{N}\sum_{i=1}^{N} \Delta_i\right\|_F$$

Under perfect orthogonality ($\Delta_i^\top \Delta_j = 0$ for $i \neq j$):
$$\left\|\frac{1}{N}\sum_i \Delta_i\right\|_F = \frac{1}{N}\sqrt{\sum_i \|\Delta_i\|_F^2}$$

**Composition efficiency** $\eta$: ratio of actual to expected (orthogonal) composition norm.
$\eta > 1$ indicates constructive interference (adapters partially aligned).

### Signal Strength
$$s_i = \frac{\|\Delta_i\|_F}{\|W\|_F}$$

How large the adapter perturbation is relative to the base weight.

### Activation Dynamic Range
For hidden state $h$ at layer $l$ with effective weight $W + \Delta$:
$$h^{(l+1)} = \sigma(h^{(l)} (W + \Delta))$$

The post-FFN activation norm $\|h^{(l)}\|$ characterizes the dynamic range.

## Worked Example (d=64, r=4, N=5)

From seed=42:

| Metric | FP16 | Ternary |
|--------|------|---------|
| Mean $\|\Delta\|_F$ | 10.54 | 9.67 |
| Std $\|\Delta\|_F$ | 1.23 | 1.77 |
| Var $\|\Delta\|_F$ | 1.52 | 3.13 |
| CV | 0.117 | 0.183 |
| Max/Min ratio | 1.42 | 1.73 |
| Composition norm (1/N) | 6.14 | 6.08 |
| Expected (ortho) | 4.75 | 4.40 |
| Efficiency $\eta$ | 1.29 | 1.38 |
| Post-FFN activation (L1) | 2.23 | 1.14 |
| Signal strength | 0.49 | 0.63 |

Key observations:
1. Ternary norm variance is 2.1x HIGHER than FP16 -- K1 fails
2. Ternary activations are 1.96x SMALLER -- compressed dynamic range
3. Ternary signal strength is 1.31x HIGHER -- adapters are proportionally larger

## Why Magnitude Bounding Fails

The hypothesis assumed ternary base constrains adapter gradient landscape to produce
more uniform delta norms. This is incorrect because:

1. LoRA parameters (A, B) are always FP16 regardless of base type
2. The gradient $\nabla_A L = W^\top \nabla_h B^\top$ depends on base weight
   structure, not magnitude uniformity
3. Ternary weights create sparser gradient signals (many zero entries), which
   actually increases variance in how different domains use the weight space

## What Actually Differs: Activation Compression

The ternary base consistently produces 2x smaller activations:
- Layer 0 post-FFN: 0.47 vs 0.85 (1.8x)
- Layer 1 post-FFN: 1.13 vs 2.23 (2.0x)

This means the logit scale is smaller and more uniform, which explains why
1/N composition is more stable: the absolute perturbation from each adapter
contributes less to the logit-space distribution even if the relative
perturbation (signal strength) is larger.

## Complexity

All measurements are O(N * P) where P is total parameters. No training required
beyond what the composition stability experiment already does. Runtime: ~400s
for 3 seeds on Apple Silicon CPU.
