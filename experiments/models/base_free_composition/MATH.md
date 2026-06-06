# Base-Free Composition: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro) |
| d_ff | FFN intermediate dimension | 4d = 256 |
| r | LoRA rank (expert) | 8 |
| L | Number of transformer layers | 4 |
| N | Number of domain experts | 4 |
| k | SVD truncation rank for base delta | {4, 8, 16, 32, full} |
| alpha | LoRA scaling factor | 1.0 |
| W_p | Pretrained weight matrix | R^{d_out x d_in} |
| W_s | Skeleton (random init) weight matrix | R^{d_out x d_in} |
| Delta | Base delta: W_p - W_s | R^{d_out x d_in} |

## 2. Delta Decomposition

### 2.1 The Base-as-Adapter Thesis

Any pretrained weight matrix can be decomposed relative to its random
initialization:

    W_pretrained = W_skeleton + Delta_base

where Delta_base = W_pretrained - W_skeleton captures ALL knowledge
acquired during pretraining. This is exact (no approximation).

The key insight: if Delta_base can be approximated at low rank, the
"pretrained base" is expressible as an adapter -- making the entire
model (base + experts) composable.

### 2.2 SVD Approximation

For a weight matrix Delta in R^{m x n}, the rank-k SVD approximation is:

    Delta_k = U_k @ diag(sigma_1, ..., sigma_k) @ V_k^T

where U_k in R^{m x k}, V_k in R^{k x n} are the top-k left/right
singular vectors.

By the Eckart-Young theorem, this is the best rank-k approximation in
Frobenius norm:

    ||Delta - Delta_k||_F = sqrt(sum_{i=k+1}^{min(m,n)} sigma_i^2)

### 2.3 Reconstruction Error

The relative reconstruction error is:

    epsilon(k) = ||Delta - Delta_k||_F / ||Delta||_F

Experimental results (3-seed average):

| Rank k | epsilon(k) | Description |
|--------|-----------|-------------|
| full | 0.000 | Exact reconstruction |
| 32 | 0.203 | Half of d=64, retains ~80% of delta energy |
| 16 | 0.419 | Quarter of d, retains ~58% |
| 8 | 0.614 | Same as LoRA expert rank |
| 4 | 0.754 | Aggressive truncation |

### 2.4 Storage Cost of Base Adapter

At rank k, the base adapter for one weight matrix W in R^{m x n}
requires:

    Storage(k) = k * (m + n) + k  (U, V, and singular values)

For our micro model (d=64, d_ff=256):

| Weight | Full rank params | Rank-32 | Rank-16 | Rank-8 |
|--------|-----------------|---------|---------|--------|
| fc1 (64x256) | 16,384 | 10,272 | 5,136 | 2,568 |
| fc2 (256x64) | 16,384 | 10,272 | 5,136 | 2,568 |
| wq/wk/wv/wo (64x64) | 4,096 | 3,104 | 1,552 | 776 |

Total base adapter at rank-16: ~42K params (vs ~50K full model).
At rank-8: ~21K params (~42% of full model).

## 3. Composition on Delta-Reconstructed Base

### 3.1 The Composition Stack

The full model under base-free composition is:

    W_composed = W_skeleton + Delta_base_k + sum_{i=1}^{N} dW_i

where dW_i = (alpha/r) * B_i @ A_i are LoRA expert deltas.

Equivalently:

    W_composed = W_pretrained_approx + sum_{i=1}^{N} dW_i

where W_pretrained_approx = W_skeleton + Delta_base_k.

### 3.2 Error Propagation

The approximation error in the base propagates to expert quality:

    L(W_composed) = L(W_exact_composed) + O(epsilon(k))

where epsilon(k) is the relative reconstruction error.

Experimental verification (3-seed average):

| Condition | Base Loss Ratio | Expert Loss Ratio | mean|cos| |
|-----------|----------------|-------------------|-----------|
| pretrained | 1.000 | 1.000 | 0.068 |
| delta_full | 1.000 | 1.000 | 0.068 |
| delta_r32 | 1.001 | 1.001 | 0.068 |
| delta_r16 | 1.019 | 1.014 | 0.083 |
| delta_r8 | 1.100 | 1.050 | 0.155 |
| delta_r4 | 1.229 | 1.095 | 0.220 |
| skeleton | 6.936 | 1.272 | 0.424 |

Key observation: expert loss ratio degrades MORE SLOWLY than base
loss ratio. At rank-8 with 10% base quality loss, expert loss is
only 5% worse. The LoRA experts partially compensate for base
approximation error.

### 3.3 Orthogonality Degradation

As the base quality degrades, expert deltas become less orthogonal:

    mean|cos|(k) ~ f(epsilon(k))

The relationship is approximately linear in our data:

| epsilon(k) | mean|cos| | cos/cos_pretrained |
|-----------|-----------|-------------------|
| 0.000 | 0.068 | 1.00x |
| 0.203 | 0.068 | 1.01x |
| 0.419 | 0.083 | 1.22x |
| 0.614 | 0.155 | 2.30x |
| 0.754 | 0.220 | 3.24x |
| 1.000 | 0.424 | 6.27x |

At macro scale (d=896), the absolute cosine values are ~100x smaller,
so even a 3x degradation at rank-8 would give mean|cos| ~ 0.0006 --
well within the safe composition zone.

## 4. Effective Rank Analysis

### 4.1 Pretrained vs Delta Effective Rank

The effective rank of the delta (r_eff(Delta)) is generally LOWER
than the effective rank of the pretrained weights:

| Matrix | r_eff(pretrained) | r_eff(delta) |
|--------|------------------|-------------|
| Mean (3 seeds) | 48.9 | 39.6 |

This is expected: the delta captures the LEARNED information, which
is more structured (lower rank) than the full weights which include
the random initialization component.

### 4.2 Implications for Base Adapter Rank

The delta's effective rank of ~40 at d=64 suggests that:

    r_needed / d ~ 40/64 ~ 0.625

At macro scale (d=3584 for Qwen2.5-7B):

    r_needed ~ 0.625 * 3584 ~ 2,240

This is high, but the effective rank at macro scale may be proportionally
lower due to the higher redundancy in larger models. BitDelta's finding
that fine-tuning deltas are 1-bit compressible suggests the base delta
may also be highly compressible at scale.

## 5. Cost Analysis

### 5.1 Decomposition Cost

SVD decomposition of all weight matrices: ~0.001 seconds (negligible).
This is a one-time offline operation, not per-inference.

### 5.2 Storage Cost Comparison

At macro scale (Qwen2.5-7B, d=3584, d_ff=18944):

| Format | Params per layer | Total (32 layers) | Size |
|--------|-----------------|-------------------|------|
| Full model | 179.7M | 5.75B | 11.5 GB |
| Base adapter (rank-512) | 18.0M | 576M | 1.15 GB |
| Base adapter (rank-256) | 9.0M | 288M | 576 MB |
| LoRA expert (rank-16) | 1.21M | 38.6M | 77 MB |

A rank-256 base adapter is ~5% of the full model size, comparable
to ~7 LoRA experts in storage. The entire model becomes:

    skeleton (free, random) + base_adapter (576 MB) + N experts (77 MB each)

## 6. Assumptions and Limitations

1. **Micro scale (d=64)**: The delta effective rank ratio (40/64 = 0.63)
   may not hold at macro scale. Larger models may be more compressible.

2. **Eckart-Young optimality**: SVD gives the optimal rank-k
   approximation in Frobenius norm, but not necessarily the best
   approximation for downstream task performance.

3. **Static decomposition**: The delta is computed AFTER pretraining.
   This is a decomposition strategy, not a training strategy. Training
   directly in low-rank (ReLoRA) may yield different results.

4. **Layer-wise independence**: We apply SVD independently to each
   weight matrix. Cross-layer structure is not exploited.

5. **Non-trivial skeleton**: The skeleton is the random initialization,
   not a zero matrix. The quality of the skeleton affects the delta's
   structure. A better skeleton (e.g., a smaller pretrained model)
   would yield a more compressible delta.

## 7. Worked Example (d=64, r_expert=8, k_base=16)

Setup:
- fc1 weight: W_p in R^{256 x 64}, W_s in R^{256 x 64}
- Delta = W_p - W_s in R^{256 x 64}
- SVD: Delta = U @ diag(S) @ V^T, with 64 singular values

Truncation at k=16:
- Keep top 16 singular values: Delta_16 = U[:,:16] @ diag(S[:16]) @ V^T[:16,:]
- Reconstruction error: epsilon(16) = 0.419 (42% of delta energy lost)
- Base loss ratio: 1.019 (1.9% quality loss)

Expert training on W_s + Delta_16:
- Fresh LoRA: A in R^{64 x 8}, B in R^{8 x 256}
- After 300 steps: expert loss ratio = 1.014 (1.4% quality loss)
- cos(expert_1, expert_2) ~ 0.083 (vs 0.068 on full pretrained)
- cos degradation: 1.22x

Base adapter storage for this layer: 16 * (256 + 64) + 16 = 5,136 params
Expert LoRA storage: 8 * (64 + 256) = 2,560 params
Ratio: base adapter is 2x a single expert at rank-16.
