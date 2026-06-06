# BitNet-2B-4T Real Composition: Mathematical Foundations

## Notation

| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| W_tern | Ternary base weight matrix | (d_out, d_in), values in {-1, 0, 1} |
| s | Weight scale (per-layer scalar) | scalar (bfloat16) |
| A_i | LoRA down-projection for expert i | (d_in, r) |
| B_i | LoRA up-projection for expert i | (r, d_out) |
| alpha | LoRA scale factor | scalar (=20.0) |
| N | Number of composed experts | scalar (=5) |
| r | LoRA rank | scalar (=16) |
| d | Hidden dimension | scalar (=2560) |

## Base Model Forward Pass

BitNet-b1.58 uses ternary weights with a scale factor:

$$y = s \cdot W_{tern} \cdot x$$

where W_tern in {-1, 0, 1}^{d_out x d_in} and s is a per-layer bfloat16 scalar
learned during pre-training.

The key property: the base weight matrix has NO magnitude variance between
elements. Every non-zero weight contributes equally (up to sign). This creates
a fundamentally different composition landscape than FP16 models.

## LoRA on Ternary Weights

For a single expert i:

$$y_i = s \cdot W_{tern} \cdot x + \alpha \cdot B_i \cdot A_i \cdot x$$

The adapter contribution is FP16 (continuous), added to the scaled ternary output.
The LoRA matrices are trained with the ternary base frozen.

## Composition via Naive Addition (1/N scaling)

For N experts composed with equal weights:

$$y_{composed} = s \cdot W_{tern} \cdot x + \frac{\alpha}{N} \sum_{i=1}^{N} B_i \cdot A_i \cdot x$$

This is equivalent to:

$$y_{composed} = s \cdot W_{tern} \cdot x + \alpha \cdot \bar{B} \cdot \bar{A} \cdot x$$

where the merged adapter is:

$$\bar{\Delta} = \frac{1}{N} \sum_{i=1}^{N} B_i \cdot A_i$$

## Orthogonality Analysis

The pairwise cosine similarity between flattened adapter vectors:

$$\cos(\theta_{ij}) = \frac{\text{vec}(\Delta_i)^T \text{vec}(\Delta_j)}{||\text{vec}(\Delta_i)|| \cdot ||\text{vec}(\Delta_j)||}$$

For adapters on a d=2560 model at rank-16, the theoretical random baseline
(Grassmannian concentration) is:

$$E[|\cos|] \approx \sqrt{\frac{2}{\pi \cdot d_{eff}}}$$

where d_eff is the effective dimensionality of the adapter parameter space.
With 210 LoRA layers, each contributing r*(d_in + d_out) parameters:
d_eff = 21,626,880. Expected |cos| ~ 0.0005.

**Observed**: mean |cos| = 0.0010, consistent with near-random directions.

## Composition Ratio

The composition ratio measures quality degradation:

$$R = \frac{\text{PPL}_{composed}}{\text{PPL}_{best\_individual}}$$

**Observed**: R = 3.59 at N=5. This is within the 10x kill threshold.

Note: for 1/N scaling, each expert contributes only 1/5 of its full signal.
The "dilution penalty" is expected and is the cost of equal-weight composition.

## Unit-Weight vs 1/N Scaling

Surprisingly, unit-weight composition (no 1/N) produced BETTER PPL (7.90 vs 7.96)
than 1/N scaling. This is unusual and suggests that at d=2560 with rank-16 adapters,
the adapters are sufficiently orthogonal that full-strength composition does not
cause interference.

## Computational Cost

| Operation | FLOPs per token |
|-----------|----------------|
| Base forward (ternary matmul) | 2 * d^2 * L * 7 = 2.75 GFLOP |
| LoRA forward (per expert) | 2 * d * r * L * 7 = 17.2 MFLOP |
| Pre-merge composition | One-time: N * 2 * r * d * L * 7 = 86.0 MFLOP |

Pre-merge composition amortizes to zero at inference time. The base model's
ternary weights mean the base forward pass is integer arithmetic (addition only),
with the LoRA delta being the only FP16 computation.

## Memory Analysis

| Component | Size |
|-----------|------|
| Packed ternary base | 490 MB (2 bits/weight) |
| Unpacked bfloat16 base (training only) | 3.9 GB |
| Single LoRA adapter (rank-16) | 41.4 MB |
| 5 adapters | 207 MB |
| Merged adapter | 41.4 MB |

At inference with packed ternary base + merged adapter: ~531 MB total.

## Assumptions

1. LoRA adapters are additive and commutative (order-independent composition)
2. The ternary base provides a sufficient representation space for 5 domains
3. 200 training steps is sufficient for domain adaptation (NOT full convergence)
4. Eval on 25 validation samples provides a reliable PPL estimate
5. The MLX bfloat16 unpacking preserves the ternary base quality exactly (verified: max diff = 0.0)
