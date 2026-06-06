# ReLoRA Merge Cycle Scaling: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro) |
| r | LoRA rank | 8 |
| alpha | LoRA scaling factor | 1.0 (scaling = alpha/r = 0.125) |
| K | Number of ReLoRA merge cycles | {5, 25, 50, 100, 200} |
| T | Total pretraining steps (fixed budget) | 2000 |
| T_c | Steps per merge cycle = T/K | {400, 80, 40, 20, 10} |
| N | Number of domain experts | 4 |
| L | Number of transformer layers | 4 |
| M | Number of LoRA modules per layer | 2 (fc1, fc2, FFN-only) |

## 2. The Merge Cycle Mechanism

### 2.1 Single Merge Cycle

At cycle k (k = 1, ..., K):

1. Train LoRA for T_c steps: learn delta_k = (alpha/r) * B_k @ A_k
2. Merge into base: W_base <- W_base + delta_k
3. Reset A_k to random, B_k to zero
4. Reset optimizer state (momentum, variance)

After K cycles, the accumulated base perturbation is:

    W_final = W_0 + sum_{k=1}^{K} delta_k

### 2.2 Accumulated Perturbation Properties

Each delta_k is rank-r. The accumulated sum has rank at most K*r:

    rank(W_final - W_0) <= K * r

For our sweep:
- K=5, r=8: effective rank <= 40 (out of d=64)
- K=25, r=8: effective rank <= 200 (>> d=64, so full-rank)
- K=50+: always full-rank at d=64

At production scale (d=3584, K=200, r=16): rank <= 3200, still sub-full-rank.

### 2.3 Potential Degradation Mechanisms

**Hypothesis (weight bias):** Each optimizer reset at cycle boundaries introduces
a transient period where gradients are computed with reset momentum (from zero).
This could create systematic bias if the reset gradient direction differs from
the steady-state direction. After K resets:

    bias_K ~ K * epsilon_reset

where epsilon_reset is the per-reset bias magnitude.

**Hypothesis (rank saturation):** At K*r >= d, the accumulated perturbation
spans the full space. Expert LoRA deltas (rank-r) then operate in a space
that has been uniformly perturbed, which should NOT increase pairwise cosine
(perturbation is isotropic).

**Hypothesis (training quality):** More cycles with shorter T_c means less
learning per cycle. Adam momentum resets waste the first ~10 steps of each
cycle rebuilding momentum estimates. The fraction of "wasted" steps:

    waste_fraction ~ 10 / T_c = 10K / T

At K=200, T=2000: waste_fraction = 100% (every step is "wasted" warmup).
At K=5, T=2000: waste_fraction = 2.5%.

This predicts base quality (not composition) degrades with K.

## 3. Kill Criteria Derivation

### 3.1 K1: cos_ratio > 5x at K=200

From the original micro experiment (K=5, d=64): cos_ratio = 1.77x (CI [0.77, 2.64]).
From the macro experiment (K=3, d=3584): cos_ratio = 0.882x.

If merge-induced bias is the dominant mechanism, cos_ratio should scale with K.
A 5x threshold at K=200 allows for ~2.5x growth factor (200/5 = 40x more cycles,
but 5x/1.77x = 2.8x growth). This is conservative: it assumes sub-linear scaling.

### 3.2 K2: loss_ratio > 1.50 at K=200

The original micro result showed loss_ratio = 1.052 at K=5.

If loss degrades linearly with K due to training inefficiency:

    loss_ratio(K) ~ 1 + beta * K

At K=5: beta = (1.052 - 1) / 5 = 0.0104
Predicted at K=200: loss_ratio = 1 + 0.0104 * 200 = 3.08

But this linear model is overly pessimistic. A sqrt or log model:

    loss_ratio(K) ~ 1 + alpha * K^gamma

The 1.50 threshold allows substantial degradation while still being useful
for production (50% quality loss is significant but not catastrophic).

## 4. Experimental Design

### 4.1 Fixed Budget Design

Fix T=2000 total steps for all K. Vary K in {5, 25, 50, 100, 200}.
This means steps_per_cycle T_c = T/K varies from 400 to 10.

**Rationale:** This design isolates the effect of merge frequency from total
training budget. At K=200 with T_c=10, each cycle barely learns anything
before resetting -- this stress-tests the merge mechanism itself.

**Alternative (fixed T_c, variable T):** Would keep per-cycle learning constant
but conflate K effects with total training budget effects.

### 4.2 Composition Measurement

For each K:
1. Train ReLoRA base (T steps, K merge cycles)
2. Train conventional base (T steps, no merges) -- SAME budget, SAME seed
3. Train N=4 domain LoRA experts on each base (300 steps each)
4. Measure pairwise cosine similarity of expert deltas
5. Compute cos_ratio = mean|cos|_relora / mean|cos|_conv
6. Compute loss_ratio = mean_loss_relora / mean_loss_conv

### 4.3 Seeds

Two seeds (42, 7) per K value. This gives 10 data points total.
The cosine metric is known to be high-variance across seeds (original
experiment had CI width of ~2x), so we prioritize covering K-space
over per-K replication.

## 5. Worked Example

### Setup
- K=100, T=2000, T_c=20, d=64, r=8

### Per-cycle dynamics
- 20 steps per cycle with Adam optimizer
- First ~5 steps rebuild momentum from zero
- Last 15 steps contribute meaningful gradient descent
- Effective learning: 15/20 = 75% efficiency

### Accumulated perturbation
- 100 cycles of rank-8 LoRA
- Theoretical max rank of perturbation: min(800, 64) = 64 (full rank)
- Each cycle adds ~O(lr * T_c * gradient_norm) perturbation magnitude

### Expected cos_ratio
- If merge-induced bias scales as sqrt(K): sqrt(100/5) * 1.77 = 7.9x
- If bias scales as log(K): log(100)/log(5) * 1.77 = 5.1x
- If bias is constant (no K-dependence): ~1.77x

### Observed (from experiment)
- cos_ratio at K=100: 3.25 (between constant and log scaling)
- But high variance: seed 42 gave 4.26x, seed 7 gave 2.24x

## 6. Assumptions

1. **Fixed budget is the right control.** Fixing total steps rather than
   per-cycle steps isolates merge frequency effects. Production ReLoRA
   operates with fixed total budget and chooses K.

2. **Micro-scale cosine trends transfer to macro.** At d=64, the parameter
   space is very small (expert deltas ~8K elements). Random cosine baseline
   E[|cos|] ~ 0.013. At d=3584, E[|cos|] ~ 1e-5. Any systematic bias that
   matters at d=3584 should also be detectable at d=64.

3. **Two seeds provide directional signal.** The cosine metric has high
   intrinsic variance. Two seeds cannot give tight CIs, but they reveal
   whether the trend is monotonic (loss_ratio) or noisy (cos_ratio).

4. **Adam optimizer resets are the dominant K-dependent mechanism.**
   The merge operation itself (W += delta) is exact. Only the optimizer
   state reset introduces K-dependent effects.
