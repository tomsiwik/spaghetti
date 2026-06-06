# BitNet Composition Stability: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro), 2048-6144 (BitNet production) |
| d_ff | MLP intermediate dimension | 4d |
| L | Number of transformer layers | 2 (micro) |
| r | LoRA rank | 4 (micro), 16 (production) |
| N | Number of composed adapters | 5 |
| W | Base weight matrix | (d_in, d_out), values in R (FP16) or {-1, 0, 1} (ternary) |
| alpha | Absmean scale factor | R+, alpha = mean(|W|) |
| W_t | Ternary weight | {-1, 0, 1}^{d_in x d_out} |
| Delta_i | LoRA adapter delta | B_i @ A_i, rank-r |
| A_i | LoRA down-projection | (d_in, r) |
| B_i | LoRA up-projection | (r, d_out) |

## 2. BitNet Absmean Quantization

### 2.1 Quantization Rule

Given a trained FP16 weight matrix W in R^{d_in x d_out}:

    alpha = mean(|W|)
    W_t = RoundClip(W / alpha, -1, 1)

where RoundClip(x, a, b) = clip(round(x), a, b).

The effective ternary weight is:

    W_eff = alpha * W_t

This is a lossy operation. The quantization error is:

    E = W - W_eff = W - alpha * RoundClip(W/alpha, -1, 1)

### 2.2 Sparsity

Empirically, ternary quantization produces ~31% zero weights (micro scale).
This matches the BitNet paper's observation: for normally-distributed weights,
P(|w/alpha| < 0.5) ~ 31% when alpha = mean(|W|) for Gaussian weights.

## 3. LoRA Composition on Ternary vs FP16 Base

### 3.1 Effective Output

For a single layer with N composed LoRA adapters at equal weight:

**FP16:**

    y = (W + (1/N) * sum_i Delta_i) @ x

**Ternary:**

    y = (alpha * W_t + (1/N) * sum_i Delta_i) @ x

### 3.2 Signal Decomposition

The output decomposes into base signal and adapter signal:

    y = W_base @ x + (1/N) * sum_i (Delta_i @ x)
      = y_base + y_adapter

**Key insight:** The base-to-adapter signal ratio determines composition stability.

For FP16: ||W|| ~ O(d * sigma_W) where sigma_W has continuous variance
For ternary: ||alpha * W_t|| ~ alpha * sqrt(d_in * d_out * (1 - s)) where s is sparsity

The ternary base signal is MORE PREDICTABLE because:
1. Weight magnitudes are exactly {-alpha, 0, alpha} (zero variance within active weights)
2. The only variation is the sparsity pattern (which is discrete)

### 3.3 Composition Interference Analysis

When composing N adapters, the interference term is:

    I = (1/N) * sum_{i != j} (Delta_i @ x)^T (Delta_j @ x)

This interference depends on the correlation between adapter activations, NOT
directly on the base weights. However, the base weights affect interference
through the gradient landscape during training:

**FP16 base:** Continuous weight space allows adapters to learn magnitude-dependent
features. Different adapters may develop different magnitude scales for their deltas,
leading to one adapter dominating the composition.

**Ternary base:** The discrete {-1, 0, 1} weight space constrains the feature space.
All adapters operate on the same discrete routing mask, so their learned deltas
tend to have more uniform magnitude.

### 3.4 Composition Ratio Analysis

Define the composition ratio R = PPL_composed / PPL_base.

- R < 1: composition improves over base (adapters recover quality)
- R = 1: composition is neutral
- R > 1: composition degrades (interference)
- R > 100: catastrophic failure (kill criterion K1)

**Why ternary gets R < 1:** The ternary base has higher base PPL (quantization loss).
The LoRA adapters partially recover this loss. After composition, the composed
model is better than the raw ternary base because the adapter signal compensates
for quantization error.

**Why FP16 gets R ~ 1:** The FP16 base is already well-trained. Adapters provide
domain specialization but equal-weight averaging dilutes each to 1/N of their
effect. The diluted adapters approximately cancel out (R ~ 1).

## 4. Delta Norm Analysis

### 4.1 Adapter Magnitude Bounding

**Hypothesis:** Ternary base produces adapters with more uniform magnitudes.

Measured:
- FP16 delta norm CV (coefficient of variation): 0.107 +/- 0.030
- Ternary delta norm CV: 0.199 +/- 0.013

**Result:** The hypothesis is REJECTED at micro scale. Ternary adapters have
HIGHER norm variance, not lower. The composition stability comes from a
different mechanism than magnitude bounding.

### 4.2 Cross-Adapter Cosine Similarity

- FP16 mean |cos|: 0.260 +/- 0.016
- Ternary mean |cos|: 0.275 +/- 0.028

The cosine similarities are comparable, ruling out orthogonality improvement
as the explanation for ternary's composition advantage.

## 5. The True Mechanism: Quantization Recovery

The composition stability improvement is NOT from:
- Magnitude bounding (CV is higher for ternary)
- Improved orthogonality (cosine similarities are comparable)

It IS from:
- **Quantization recovery:** The ternary base has significant quantization loss
  (PPL 6-9 vs FP16 PPL 2-6). Each LoRA adapter recovers part of this loss.
  Equal-weight composition of N recovery adapters retains partial recovery,
  yielding R < 1.
- **Lower baseline:** The denominator (ternary base PPL) is higher, making
  the ratio smaller even if the composed PPL is similar in absolute terms.

## 6. Implications for Production SOLE

### 6.1 Composition is Stable on Ternary Base

At N=5, equal-weight composition on a ternary base produces R = 0.63 (3 seeds).
This is well below the K1 threshold of 100x. In absolute terms, the composed
model is BETTER than the ternary base alone.

### 6.2 But the Mechanism is Not What We Expected

The stability comes from the adapter's ability to recover quantization loss,
not from bounded weight magnitudes or reduced interference. This means:

- At production scale (BitNet-2B or 30B), the effect depends on the quantization
  gap between the ternary base and FP16 equivalent.
- If the ternary base is trained from scratch (BitNet recipe), it has no
  quantization loss -- the "recovery" mechanism disappears.
- The critical question is whether natively-trained BitNet bases also show
  stable composition, for a different reason (the routing mask hypothesis).

### 6.3 FP16 Composition at Micro Scale

FP16 composition ratio R = 1.01 (3 seeds) shows composition is approximately
neutral at micro scale. This contrasts with the macro result (PPL in trillions),
suggesting the catastrophe emerges at larger scale / more complex data.

## 7. Complexity

| Operation | Cost |
|-----------|------|
| Ternary quantization | O(d_in * d_out) per layer |
| LoRA forward | O(d_in * r + r * d_out) per layer |
| Composition (N adapters) | O(N * d_in * r * d_out) (pre-merge, one-time) |
| Inference (post-merge) | O(d_in * d_out) per layer (same as base) |

## 8. Assumptions and Limitations

1. **Post-training quantization, not native BitNet:** Real BitNet is trained
   from scratch with ternary constraints. Our experiment quantizes a trained FP16
   model. Native BitNet may behave differently.

2. **Micro scale (d=64, r=4):** At production scale (d=2048+, r=16), the
   dynamics may differ. The FP16 composition catastrophe seen at macro was not
   reproduced at micro.

3. **Toy data:** Character-level tasks, not natural language. The composition
   dynamics on real language with complex semantics may differ.

4. **Equal-weight only:** PPL-probe or other weighted composition may change
   the relative advantage of ternary vs FP16 bases.
