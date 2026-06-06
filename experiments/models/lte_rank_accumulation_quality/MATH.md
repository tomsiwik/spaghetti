# LTE Rank Accumulation Quality at d=256: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 256 (this experiment), 64 (parent) |
| d_ff | FFN intermediate dimension | 4d = 1024 |
| r | LoRA rank per head | 8 |
| L | Number of transformer layers | 4 |
| K | Number of parallel LoRA heads | 4 |
| N | Number of domain experts | 4 |
| alpha | LoRA scaling factor | 1.0 |
| T | Steps per head between merges | 25 |
| M_par | Parallel merge intervals | total_steps / (K * T) = 4 |
| M_seq | Sequential merge intervals | total_steps / merge_every_seq = 4 |

## 2. The Rank Capacity Hypothesis

### 2.1 Core Claim

At d=64, r/d = 12.5%. A single rank-8 LoRA update covers a large fraction of
the weight space, so K parallel rank-8 updates (effective rank 32) vs one
rank-8 update (effective rank 8) cannot show meaningful difference -- the
space is too small for the 4x capacity ratio to matter.

At d=256, r/d = 3.1%. Each rank-8 update covers only 3.1% of the space.
K=4 parallel updates cover 12.5% per interval vs 3.1% for sequential --
a ratio that should produce measurable quality differences if the LTE
rank accumulation advantage is real.

### 2.2 Per-Interval Rank Capacity

**Parallel (1 merge interval):**

    dW_parallel = (1/K) * sum_{k=1}^{K} (alpha/r) * B_k^T @ A_k^T

    rank(dW_parallel) <= min(K * r, d) = min(32, 256) = 32

Coverage: 32/256 = 12.5% of dimension space.

**Sequential (1 merge interval):**

    dW_sequential = (alpha/r) * B^T @ A^T

    rank(dW_sequential) <= r = 8

Coverage: 8/256 = 3.1% of dimension space.

**Capacity ratio:** 32/8 = 4x.

### 2.3 Total Rank Accumulation Over Full Adaptation

With 400 total gradient steps:

**Parallel:** 4 intervals, each with effective rank 32.
Total rank capacity = min(4 * 32, 256) = min(128, 256) = 128.

**Sequential:** 4 merges (at steps 100, 200, 300, 400).
Each merge contributes rank 8.
Total rank capacity = min(4 * 8, 256) = min(32, 256) = 32.

**Accumulated rank ratio:** 128/32 = 4x.

At d=64 (parent experiment):
- Parallel: 2 intervals * 32 = 64 capacity, but min(64, 64) = 64 (saturated!)
- Sequential: 5 merges * 8 = 40 capacity, min(40, 64) = 40

At d=64, parallel is already SATURATED (rank capacity = d). This is why
no advantage was observed. At d=256, neither method saturates, so the
true 4x capacity ratio is testable.

## 3. Experimental Results

### 3.1 Base Quality After Adaptation (3 seeds)

| Method | Val Loss (mean +/- range) | vs Pretrained | vs Continued |
|--------|--------------------------|---------------|--------------|
| Parallel (K=4) | 0.658 [0.648, 0.667] | -12.0% | **0.670x** |
| Sequential | 0.640 [0.633, 0.654] | -14.4% | **0.653x** |
| Continued conv | 1.035 [0.761, 1.318] | +38.5% | 1.000x |

Both LoRA methods vastly outperform continued conventional at d=256. This
gap is much larger than at d=64 (-2% vs +0.2%). Sequential is slightly
better than parallel on base loss (2.3% gap).

### 3.2 Expert Orthogonality (3 seeds, 4 experts, 6 pairs/seed)

| Method | mean|cos| | Range | vs Continued |
|--------|----------|-------|--------------|
| Parallel | 0.036 | [0.024, 0.042] | 0.439x |
| Sequential | 0.022 | [0.013, 0.044] | 0.385x |
| Continued | 0.097 | [0.038, 0.152] | 1.000x |

**CRITICAL: The par_vs_seq_cos ratio is HIGH VARIANCE.**

- Seed 42: par/seq = 0.71 (parallel better)
- Seed 123: par/seq = 3.17 (sequential much better)
- Seed 7: par/seq = 2.30 (sequential better)

Mean ratio: 2.06, CI [0.71, 3.17]. Includes 1.0 but skewed toward
sequential being better at orthogonality.

This contrasts with d=64 where par/seq was 1.46, CI [0.80, 2.20].

### 3.3 Expert Quality (3 seeds, 4 experts)

| Method | Mean Expert Loss | vs Continued |
|--------|-----------------|--------------|
| Parallel | 0.5776 | 0.913x |
| Sequential | 0.5724 | 0.906x |
| Continued | 0.6530 | 1.000x |

Expert quality is close between parallel and sequential (0.7% gap).
Both are much better than continued conventional (9% gap at d=256 vs
1.5% at d=64).

### 3.4 Effective Rank

| Method | Mean Effective Rank | vs d |
|--------|--------------------|----|
| Parallel | 157.79 | 61.6% |
| Sequential | 157.96 | 61.7% |
| Continued | 152.44 | 59.5% |

All methods maintain high effective rank. The LoRA-adapted bases have
slightly higher effective rank than continued conventional, consistent
with the d=64 finding.

## 4. Scaling Analysis: d=64 vs d=256

### 4.1 Head-to-Head Comparison

| Metric (par/seq) | d=64 | d=256 | Change |
|-----------------|------|-------|--------|
| Base loss ratio | 1.007 | 1.023 | +0.016 |
| Expert cos ratio | 1.46 | 2.06 | +0.60 |
| Expert loss ratio | 1.006 | 1.007 | +0.001 |

**Base loss:** The gap widened slightly (from 0.7% to 2.3%). Parallel
is consistently slightly worse than sequential on base loss.

**Expert cosines:** The ratio increased (from 1.46 to 2.06), meaning
parallel produces LESS orthogonal experts at d=256 than at d=64 relative
to sequential. This is the opposite of the hypothesized direction.

**Expert loss:** No meaningful change (<0.1%).

### 4.2 The Anti-Pattern: Why Parallel May Hurt Orthogonality

The rank capacity advantage of parallel is a theoretical maximum.
In practice, K heads trained on different shards of the SAME distribution
(character-level names) may explore OVERLAPPING subspaces. When their
deltas are averaged, the high-rank property of the sum reflects redundancy,
not diversity.

Furthermore, the averaging step (1/K * sum) DILUTES each head's contribution.
At d=256, each head's rank-8 update covers 3.1% of space. After averaging
4 such updates, the effective magnitude per subspace direction is 1/4 of
what a single sequential update achieves. Sequential gets the full undiluted
gradient signal for 100 steps, potentially reaching deeper into relevant
subspaces.

## 5. Assumptions and Limitations

1. **Character-level data lacks shard diversity.** The quintary split
   (a-e, f-j, ...) provides minimal distribution shift between shards.
   Real domain-diverse data (code vs medical vs legal) would give parallel
   heads more distinct subspaces to explore.

2. **d=256 is still mid-scale.** The LTE paper targets d=4096+. Effects
   may differ at true macro scale.

3. **Equal merge intervals.** At d=256, both parallel and sequential
   get 4 merge intervals. The parent experiment had unequal intervals
   (2 parallel, 5 sequential) due to integer division at d=64.

4. **Reset-after-merge only.** The no-reset variant (LTE's claimed
   advantage) was not tested due to the known instability at micro scale.

5. **Fixed K=4.** Larger K (8, 16) might show different scaling behavior.

## 6. Worked Example (d=256, r=8, K=4, T=25)

Weight matrix W_fc1 in R^{1024 x 256}, pretrained.

**Parallel (1 interval of 4 heads, 25 steps each):**
- Head k: A_k in R^{256x8}, B_k in R^{8x1024}, trained 25 steps
- dW_k = (1/8) * B_k^T @ A_k^T, rank(dW_k) <= 8
- dW_avg = (1/4) * sum dW_k
- rank(dW_avg) <= min(32, 256) = 32
- Each direction's magnitude: ~1/4 of individual head's update

**Sequential (1 cycle of 100 steps):**
- A in R^{256x8}, B in R^{8x1024}, trained 100 steps (4x longer)
- dW = (1/8) * B^T @ A^T, rank(dW) <= 8
- Full gradient signal for 100 steps in 8 directions

**Key difference:** Sequential concentrates 100 steps of gradient into
8 directions. Parallel distributes 25 steps across 32 potential directions
(4 heads x 8 rank), but each direction gets 1/4 the gradient magnitude
and only 1/4 the training steps.

The theoretical rank advantage (32 vs 8) is real but offset by:
(a) 1/K magnitude dilution from averaging
(b) 1/K fewer training steps per head
(c) Potential subspace overlap when data shards are similar
