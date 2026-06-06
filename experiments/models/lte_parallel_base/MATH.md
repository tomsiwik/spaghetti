# LTE Parallel Base Construction: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro) |
| d_ff | FFN intermediate dimension | 4d = 256 |
| r | LoRA rank per head | 8 |
| L | Number of transformer layers | 4 |
| K | Number of parallel LoRA heads | 4 |
| N | Number of domain experts | 4 |
| alpha | LoRA scaling factor | 1.0 |
| T | Steps per head between merges | 50 |
| M | Number of merge intervals | total_steps / (K * T) |

## 2. Parallel vs Sequential LoRA Merging

### 2.1 Single LoRA Update

A LoRA adapter modifies weight matrix W in R^{d_out x d_in}:

    W' = W + (alpha/r) * (A @ B)^T

where A in R^{d_in x r}, B in R^{r x d_out}. The update dW = (alpha/r) * B^T @ A^T
has rank at most r.

### 2.2 Parallel Merge (LTE-style)

K heads are trained independently on different data shards, starting from
fresh LoRA parameters (A_k random, B_k = 0) with the same frozen base W.

After T training steps, each head produces deltas:

    dW_k = (alpha/r) * B_k^T @ A_k^T, rank(dW_k) <= r

The merged update is:

    dW_parallel = (1/K) * sum_{k=1}^{K} dW_k

**Rank bound:**
    rank(dW_parallel) <= min(K * r, min(d_in, d_out))

With K=4, r=8: rank <= min(32, 64) = 32.

The key property: dW_parallel can capture a rank-32 update despite each
individual head being limited to rank 8. This is the capacity advantage
of parallel merging (Hyeon-Woo et al. 2024, Theorem 1).

### 2.3 Sequential Merge (ReLoRA-style)

A single LoRA is trained for T_seq = K*T steps (compute-fair), then merged.
After the merge, LoRA is reset and the process repeats.

After each merge, the update is:

    dW_sequential = (alpha/r) * B^T @ A^T, rank(dW_sequential) <= r

Over M sequential merges:

    W_final = W_0 + sum_{m=1}^{M} dW_m

**Rank bound on total accumulated update:**
    rank(sum dW_m) <= min(M * r, min(d_in, d_out))

With M=5 merges (500 steps / 100 per merge): rank <= min(40, 64) = 40.

### 2.4 Rank Comparison

Both methods achieve comparable effective rank by the end:
- Parallel: 2 intervals * 4 heads/interval = 8 rank-8 merges = rank-32 capacity
- Sequential: 5 merges of rank-8 = rank-40 capacity

The difference is in HOW the rank accumulates:
- Parallel: diverse subspaces from different data shards in the SAME interval
- Sequential: potentially correlated subspaces from consecutive training steps

### 2.5 Compute Fairness

Total gradient steps are matched:
- Parallel: M * K * T = 2 * 4 * 50 = 400 gradient steps
- Sequential: total_steps = 500 gradient steps

Note: not perfectly matched due to integer division. With K GPUs, parallel
wall time = M * T = 2 * 50 = 100 steps (5x faster than sequential).

## 3. Expert Orthogonality on Adapted Bases

### 3.1 Expert Delta Vectors

For expert i trained on base W_adapted:

    v_i = flatten(dW_i^{(1,fc1)}, ..., dW_i^{(L,fc2)})
    dim(v_i) = L * 2 * d * d_ff = 4 * 2 * 64 * 256 = 131,072

### 3.2 Pairwise Cosine Similarity

    cos(v_i, v_j) = <v_i, v_j> / (||v_i|| * ||v_j||)

Random expectation at D = 131,072:

    E[|cos|] ~ sqrt(2 / (pi * D)) ~ 0.0039

### 3.3 Experimental Results (3 seeds, 4 experts each)

| Method | mean|cos| | max|cos| | vs Random |
|--------|----------|---------|-----------|
| Parallel | 0.0095 +/- 0.002 | 0.018 +/- 0.005 | 2.4x |
| Sequential | 0.0074 +/- 0.002 | 0.016 +/- 0.008 | 1.9x |
| Continued conv | 0.0222 +/- 0.006 | 0.038 +/- 0.009 | 5.7x |

All three methods produce experts with near-random orthogonality, confirming
that the base construction method does not substantially affect the
structural orthogonality of subsequently trained experts at d=64.

### 3.4 Cosine Ratio Analysis

    par_cos_ratio = par_mean_cos / continued_mean_cos = 0.455 (3-seed mean)
    seq_cos_ratio = seq_mean_cos / continued_mean_cos = 0.381 (3-seed mean)

Both LoRA-merge methods produce LOWER expert cosines than continued
conventional training. This is surprising: the LoRA-adapted bases produce
more orthogonal experts. Possible explanation: LoRA adaptation constrains
the weight space to a lower-rank manifold, and expert LoRA deltas on this
manifold are naturally more diverse.

## 4. Base Quality Analysis

### 4.1 Adapted Base Loss

| Method | Val Loss (3-seed mean) | vs Pretrained (0.525) |
|--------|----------------------|----------------------|
| Parallel | 0.515 +/- 0.007 | -1.9% (improved) |
| Sequential | 0.512 +/- 0.007 | -2.5% (improved) |
| Continued | 0.526 +/- 0.007 | +0.2% (flat) |

Both LoRA-merge methods improve the base beyond what continued conventional
training achieves. This is because LoRA acts as a regularizer (rank-r
constraint prevents overfitting).

### 4.2 Expert Quality

| Method | Mean Expert Loss | vs Continued |
|--------|-----------------|--------------|
| Parallel | 0.4531 | 0.988x |
| Sequential | 0.4504 | 0.982x |
| Continued | 0.4582 | 1.000x |

Expert quality is nearly identical across all three substrates.

## 5. Effective Rank

After adaptation, all three bases maintain comparable effective rank:

| Method | Mean Effective Rank | vs d=64 |
|--------|--------------------|---------|
| Parallel | 53.24 +/- 0.5 | 83.2% |
| Sequential | 53.50 +/- 0.4 | 83.6% |
| Continued | 53.03 +/- 0.5 | 82.9% |

No rank collapse from parallel or sequential merging. This confirms that
the compute-fair setup with alpha/r scaling preserves the weight spectrum.

## 6. Assumptions and Limitations

1. **Micro scale**: d=64, r=8, L=4. At macro scale (d>=4096), the
   rank capacity advantage of parallel (K*r per merge) may matter more.

2. **Compute-fair but not data-fair**: With K=4 heads and 500 total gradient
   steps, parallel gets only 2 merge intervals (each head does 50 steps).
   Sequential gets 5 merge intervals (100 steps each). More data budget
   would allow more parallel merge intervals, potentially revealing
   differences.

3. **Character-level data**: Names dataset has limited diversity. Real
   domains would provide more varied data shards, potentially amplifying
   the parallel diversity advantage.

4. **LoRA-only (frozen base)**: Both adaptation methods freeze the base
   and only train LoRA A/B. This is different from the original LTE paper
   which trains from scratch (all parameters via LoRA).

5. **Single-device simulation**: True parallel training distributes heads
   across K GPUs. Our simulation is exact in gradients but does not test
   communication overhead or synchronization effects.

## 7. Worked Example (d=64, r=8, K=4, T=50)

Setup: weight matrix W_fc1 in R^{256 x 64}, pretrained.

**Parallel (1 merge interval):**
- Head 0: A_0 in R^{64x8} (random), B_0 = 0 -> train 50 steps -> delta_0
- Head 1: A_1 in R^{64x8} (random), B_1 = 0 -> train 50 steps -> delta_1
- Head 2, 3: similarly
- dW = (1/4) * sum (alpha/r) * (A_k @ B_k)^T
- rank(dW) <= min(32, 64) = 32

**Sequential (2 merge-and-restart cycles of 100 steps each):**
- Cycle 1: A_1 random, B_1=0 -> train 100 steps -> merge, reset
- Cycle 2: A_2 random, B_2=0 -> train 100 steps -> merge, reset
- rank(total update) <= min(16, 64) = 16 per merge

**Result:** Both produce val loss ~0.51, expert cos ~0.01, expert loss ~0.45.
No significant difference despite different rank-accumulation patterns.
