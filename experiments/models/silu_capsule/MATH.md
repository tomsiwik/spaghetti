# SiLU vs ReLU Capsule Activation — Mathematical Analysis

## Setup

Two-layer MLP capsule pool with activation function σ:

```
y = B · σ(A · x)
```

where A ∈ ℝ^{P×d}, B ∈ ℝ^{d×P}, x ∈ ℝ^d.

## Activation Functions

**ReLU**: σ(z) = max(0, z)
- Hard sparsity: exactly 50% zeros for symmetric input
- Dead neurons: if a_i^T x < 0 for all x in the data, neuron i is permanently dead
- Gradient: ∂σ/∂z = 1 if z > 0, else 0 (no gradient through dead neurons)

**SiLU (Swish)**: σ(z) = z · sigmoid(z) = z / (1 + e^{-z})
- Smooth: σ'(z) = σ(z) + sigmoid(z)(1 - σ(z))
- Min value: σ(-1.278) ≈ -0.278 (not zero)
- Never exactly zero except at z = 0
- Gradient never exactly zero: neurons can always recover

## Composition Identity

The composition protocol concatenates A matrices vertically and B matrices horizontally:

```
A_composed = [A_1; A_2]     (2P × d)
B_composed = [B_1 | B_2]    (d × 2P)
```

**Claim**: For ANY activation σ with σ(0) = 0 (not needed) and ANY B:

```
B_composed · σ(A_composed · x) = B_1 · σ(A_1 · x) + B_2 · σ(A_2 · x)
```

**Proof**: This follows directly from the block structure:
```
σ(A_composed · x) = [σ(A_1 · x); σ(A_2 · x)]
B_composed · [h_1; h_2] = B_1 · h_1 + B_2 · h_2
```

This holds for ANY activation function, not just ReLU. The identity depends on the LINEAR structure of B, not on properties of σ.

**Zero-init identity**: With B_new = 0:
```
y = 0 · σ(A · x) = 0
```
This holds regardless of σ. Adding a new zero-initialized pool doesn't change output.

## Sparsity Analysis

### ReLU Sparsity
For input z ~ N(0, σ²): P(ReLU(z) = 0) = 0.5 (exact).

Dead capsule = capsule where a_i^T x < 0 for ALL training examples. At d=64, this probability depends on the angular distribution of data.

### SiLU Effective Sparsity
SiLU never produces exact zero, so we define:
- **Effective sparsity**: fraction of activations with |σ(z)| < ε
- **Near-dead capsule**: capsule where E[|σ(a_i^T x)|] < ε

For z ~ N(0, σ²), effective sparsity at ε = 0.01:
```
P(|SiLU(z)| < 0.01) ≈ P(|z| < 0.01) + P(z ≈ -1.278)
```
This is much lower than ReLU's 50% — SiLU activations are denser.

## Gradient Flow Comparison

**ReLU**: ∂L/∂a_i = 0 when a_i^T x < 0 (dead zone — no learning possible)

**SiLU**: ∂L/∂a_i always nonzero (smooth gradient everywhere)
```
∂SiLU/∂z = SiLU(z)/z + sigmoid(z)(1 - SiLU(z)/z)
         = sigmoid(z) + z · sigmoid(z)(1 - sigmoid(z))
```

Key implication: SiLU capsules can always recover from near-dead states. ReLU capsules cannot recover once truly dead without explicit intervention.

## Hypothesis

At macro scale (d=896), the input distribution from Qwen's SwiGLU-trained hidden states may be better matched to SiLU than ReLU. The 0% dead capsules observed could be:
1. **Scale effect**: higher d → lower death probability (holds for both activations)
2. **Distribution match**: SiLU matches the pretrained distribution, reducing the probability of neurons landing in low-activation regions

This experiment isolates (2) by testing at micro scale (d=64) where death is observable.
