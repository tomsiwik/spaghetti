# LoRA Rank Sensitivity for Composition: Mathematical Foundations

## 1. Setup

We have a pretrained base model with weights W_base. For each domain k in
{1, ..., N}, we fine-tune LoRA adapters of rank r on the MLP layers while
freezing all base weights.

Each LoRA adapter at rank r consists of:
- A_k in R^{d_in x r} (down-projection)
- B_k in R^{r x d_out} (up-projection)

The delta is:
  dW_k = (alpha / r) * A_k @ B_k  in R^{d_in x d_out}

The effective weight for domain k:
  W_k = W_base + dW_k

## 2. Rank and Information Capacity

The rank r bounds the information content of the adapter. The delta dW_k
lives in a subspace of dimension at most r. The total number of trainable
parameters per domain scales linearly with r:

  P(r) = n_layer * (d * r + r * 4d + 4d * r + r * d) = n_layer * 10 * d * r

For our micro setup (d=64, n_layer=4):
  P(r) = 4 * 10 * 64 * r = 2560 * r

| r  | P(r)    | P(r)/P_base |
|----|---------|-------------|
| 2  | 5,120   | 2.5%        |
| 4  | 10,240  | 5.0%        |
| 8  | 20,480  | 10.1%       |
| 16 | 40,960  | 20.2%       |
| 32 | 81,920  | 40.3%       |
| 64 | 163,840 | 80.7%       |

where P_base ~ 203K is the base model parameter count.

## 3. Rate-Distortion Framing

In information theory, rate-distortion theory characterizes the minimum
number of bits R(D) needed to represent a source with distortion <= D.

By analogy, LoRA rank r is the "rate" (representation capacity), and
composition quality gap (vs joint training) is the "distortion":

  D(r) = L_composed(r) / L_joint - 1

The rate-distortion prediction: there exists a critical rank r* such that:
- For r < r*, D(r) degrades significantly (insufficient capacity)
- For r >= r*, D(r) plateaus (additional capacity is wasted)

## 4. Effective Rank

The nominal rank r is an upper bound on the delta's dimensionality.
The effective rank measures the actual dimensionality used.

Given delta dW with SVD decomposition dW = U S V^T where S = diag(s_1, ..., s_r):

Normalize: p_i = s_i / sum_j s_j

Shannon entropy: H = -sum_i p_i * log(p_i)

Effective rank: r_eff = exp(H)

Properties:
- r_eff = 1 when all information is in one singular direction (rank-1)
- r_eff = r when all singular values are equal (uniform)
- 1 <= r_eff <= r always

## 5. Orthogonality vs Rank

The cosine similarity between two domain deltas at rank r:

  cos(dW_A, dW_B) = <vec(dW_A), vec(dW_B)> / (||dW_A|| * ||dW_B||)

where vec() flattens the matrix.

Hypothesis: Higher rank provides more dimensions for the deltas to be
orthogonal in. In R^d, two random vectors have expected cosine ~ 0 with
variance ~ 1/d. The delta lives in a subspace of dimension r * (d_in + d_out),
so higher r means more dimensions, potentially lower cosine.

However, the optimization dynamics may counteract this: higher-rank adapters
may converge to the same low-dimensional manifold as lower-rank ones (if the
task has inherent low dimensionality).

## 6. Shared Fraction vs Rank

The shared/unique decomposition (from exp_lora_procrustes_linear):

  dW_shared = (1/N) * sum_k dW_k
  dW_unique_k = dW_k - dW_shared

Shared fraction:
  f_shared = ||dW_shared||_total / (||dW_shared||_total + ||dW_unique||_total)

At N=2 with near-orthogonal deltas (cos ~ 0):
  ||dW_shared||^2 = (||dW_A||^2 + ||dW_B||^2 + 2<dW_A, dW_B>) / 4
  ||dW_unique||^2 ~ (||dW_A||^2 + ||dW_B||^2 - 2<dW_A, dW_B>) / 4

When cos ~ 0: f_shared ~ 0.5 regardless of rank. The shared fraction is
determined by the angle between deltas, not their rank.

## 7. Dead Neuron Rate

Dead neurons are ReLU units in the MLP that output 0 for all inputs in a
profiling dataset. The dead neuron rate is:

  D = |{i : max_x relu(fc1(x))_i = 0}| / (n_layer * 4d)

The LoRA delta modifies the pre-ReLU activations:
  pre_relu_k(x) = (W_fc1 + dW_fc1_k) @ x

Larger rank means larger potential change to pre-ReLU values, which could
either revive or kill neurons. The net effect is not predictable a priori.

## 8. Worked Example (d=64, r=8)

Per the existing MATH.md in lora_procrustes:
- fc1: A in R^{64x8}, B in R^{8x256}
- fc2: A in R^{256x8}, B in R^{8x64}
- Params per domain per layer: 5,120
- Total per domain: 20,480

Full delta: dW_fc1 in R^{64x256} = 16,384 elements, but rank-8.
Only 8 directions in 16,384-dimensional space are used.

## 9. Assumptions

1. LoRA deltas are small relative to base weights.
2. Domains are trained from the same frozen base.
3. The task has some inherent dimensionality d_task that bounds
   the useful effective rank regardless of nominal rank r.
4. 300 training steps is sufficient for the LoRA to converge to
   its rank-constrained optimum.

## 10. Falsification Criteria

| Criterion | Threshold | Kill if |
|-----------|-----------|---------|
| Quality range across ranks (TA) | >1pp | <1pp |
| Orthogonality-rank correlation | r^2 >= 0.2 | r^2 < 0.2 |

## 11. Computational Cost

Per rank per seed: ~1.5 minutes
  - Joint baseline: 600 steps (~10s)
  - Base pretraining: 300 steps (~5s)
  - LoRA fine-tuning: 2 x 300 steps (~10s)
  - Router calibration: 100 steps (~5s)
  - Dead neuron profiling: 20 batches (~2s)
  - SVD for effective rank: negligible

Total: 6 ranks x 3 seeds = 18 runs ~ 27 minutes.
