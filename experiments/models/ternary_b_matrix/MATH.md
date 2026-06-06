# Ternary B-Matrix: Mathematical Foundations

## Setup

Standard LoRA decomposes a weight update as:

    DeltaW = B @ A,    B in R^{out x r}, A in R^{r x in}

In our architecture, A is frozen on the Grassmannian Gr(r, d) -- orthogonal across
adapters by construction. Currently B is FP32 (trainable). This experiment makes B
ternary: B_q in {-alpha, 0, +alpha}^{out x r}.

## Ternary Quantization of B

### Absmean Quantization (BitNet b1.58 style)

For a weight matrix W:

    alpha = mean(|W|)
    W_scaled = W / alpha
    W_q = clip(round(W_scaled), -1, 1)

Forward pass uses W_q * alpha. Backward pass uses Straight-Through Estimator (STE):

    grad(W) = grad(W_q * alpha)   (gradient passes through quantization)

### STE Training (QAT)

During training, B_latent is maintained in FP32. Each forward pass:

    alpha_B = mean(|B_latent|)
    B_q = clip(round(B_latent / alpha_B), -1, 1) * alpha_B
    B_ste = B_latent + stop_gradient(B_q - B_latent)

This gives B_q in forward, but gradients flow to B_latent.

### Post-Training Quantization (PTQ)

Train B in FP32 normally. After training, quantize:

    alpha_B = mean(|B_trained|)
    B_ptq = clip(round(B_trained / alpha_B), -1, 1)

Store only B_ptq (int2 per element) + alpha_B (one float per adapter per layer).

## Composition with Ternary B

### Single adapter forward pass

    y = base(x) + x @ A_i^T @ B_i^T

With ternary B_i = alpha_i * T_i where T_i in {-1,0,+1}:

    y = base(x) + alpha_i * (x @ A_i^T @ T_i^T)

### N-adapter composed forward pass

    y = base(x) + (1/N) * sum_i [alpha_i * (x @ A_i^T @ T_i^T)]

### Composition via weight merging

For weight merging (apply delta to base):

    W_composed = W_base + (1/N) * sum_i [B_i @ A_i]

With ternary B_i:

    W_composed = W_base + (1/N) * sum_i [alpha_i * T_i @ A_i]

Since A_i are orthogonal across adapters (Grassmannian):
- A_i^T A_j approx 0 for i != j
- Cross-adapter interference: ||DeltaW_i^T DeltaW_j|| <= alpha_i * alpha_j * ||T_i||_F * ||A_i^T A_j||_F * ||T_j||_F
- With orthogonal A: interference -> 0 regardless of B correlation

Key insight: **ternary B does not change the orthogonality guarantee from Grassmannian A.**
The interference bound depends on A_i^T A_j, not on the precision of B.

## Storage Analysis

Per-adapter B-matrix storage:

| Representation | Bits per element | Total for rank=8, d=128, 4 layers x 7 matrices |
|---------------|-----------------|------------------------------------------------|
| FP32 B        | 32              | 128 * 8 * 28 = 28672 params * 4B = 112 KB      |
| Ternary B     | 2 (log2(3))     | 28672 * 0.25B = 7 KB + 28 scale floats         |

Compression ratio: ~16x per adapter.

## Composition Cost

| Method | FP32 B | Ternary B |
|--------|--------|-----------|
| Runtime LoRA | N matmuls (x @ A^T @ B^T) | N matmuls but B is sparse/ternary |
| Weight merge | N matmuls (B @ A) + N additions | Same, but B @ A = alpha * T @ A (ternary matmul) |

With fully ternary adapters, the merge B @ A becomes: for each output row, sum/subtract
selected A rows according to T's {-1,0,+1} pattern. This is pure addition, no multiply.

## Expected Quality Impact

Prior evidence:
- Ternary BASE weights (STE): PPL ratio 1.003x vs FP32 (essentially no loss)
- Ternary A-matrix adapters (QAT): composition ratio 1.022x
- B-matrix typically has higher variance than A (it's the learned part)
- Quantizing B to ternary is MORE aggressive since B carries the task-specific signal

Hypothesis: ratio < 1.5 is achievable because:
1. B-matrices at rank 8 have only 8 values per output dimension -- limited capacity to lose
2. STE has proven effective on this architecture at both base and adapter level
3. The Grassmannian orthogonality guarantee is independent of B precision

## Worked Example (d=8, r=2, 2 adapters)

A_1 = [[1, 0, 0, 0, 0, 0, 0, 0],    (spans dims 0-1)
        [0, 1, 0, 0, 0, 0, 0, 0]]

A_2 = [[0, 0, 1, 0, 0, 0, 0, 0],    (spans dims 2-3)
        [0, 0, 0, 1, 0, 0, 0, 0]]

FP32 B_1 = [[0.3, -0.7],     B_1_ternary = alpha_1 * [[1, -1],
             [0.1, 0.5]]                               [0, 1]]
             alpha_1 = mean(|B_1|) = 0.4

DeltaW_1 = B_1 @ A_1 = alpha_1 * T_1 @ A_1
         = 0.4 * [[1, -1, 0, ..., 0],
                   [0,  1, 0, ..., 0]]

DeltaW_2 = B_2 @ A_2 -> occupies dims 2-3 only

Cross-interference: DeltaW_1^T @ DeltaW_2 = 0 (orthogonal A subspaces).

Quantization error: ||B_1 - B_1_q||_F / ||B_1||_F
  = ||[[0.3-0.4, -0.7+0.4], [0.1-0, 0.5-0.4]]||_F / ||B_1||_F
  = ||[[-0.1, -0.3], [0.1, 0.1]]||_F / ||B_1||_F
  = sqrt(0.01+0.09+0.01+0.01) / sqrt(0.09+0.49+0.01+0.25)
  = 0.346 / 0.917 = 0.377

This 37.7% relative error per element is typical for aggressive quantization, but
the *composition* error is filtered by the Grassmannian: each adapter's quantization
error stays in its own subspace.

## Assumptions

1. The Grassmannian orthogonality guarantee on A is sufficient to keep composition
   quality high even when B precision is reduced to ternary
2. STE training at rank 8 provides enough gradient signal to learn good ternary B
3. Per-channel scale (alpha per adapter per layer) is sufficient; we do not need
   per-row or per-column scales
4. The character-level names task is representative enough to test the mechanism
