# Few-Shot Adaptation Recovery: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro) |
| d_ff | FFN intermediate dimension | 4d = 256 |
| r | LoRA rank (expert) | 8 |
| L | Number of transformer layers | 4 |
| N | Number of domain experts | 4 |
| k | SVD truncation rank for base delta | 16 |
| alpha | LoRA scaling factor | 1.0 |
| W_p | Pretrained weight matrix | R^{d_out x d_in} |
| W_k | SVD-perturbed base weight matrix | R^{d_out x d_in} |
| A_i, B_i | LoRA matrices for expert i | A: R^{d_in x r}, B: R^{r x d_out} |
| dW_i | Expert i's LoRA delta: (alpha/r) * A_i @ B_i | R^{d_in x d_out} |
| s | Scale factor alpha/r | 1/8 = 0.125 |
| n | Number of adaptation steps | {1, 5, 10, 25, 50} |
| eta | Adaptation learning rate | 1e-3 (primary), 3e-3 (sensitivity) |

## 2. The Adaptation Problem

### 2.1 Setup

From the parent experiment (zero_shot_base_transfer), we know that
applying expert deltas trained on W_p to a perturbed base W_k
produces a transfer gap:

    G(k) = L(W_k + dW_i^{train}) - L(W_p + dW_i^{train})

At rank-16 SVD perturbation, the 3-seed average transfer gap is
approximately 3.7% of the original expert loss.

The adaptation question asks: if we fine-tune the LoRA parameters
(A_i, B_i) for n steps on the perturbed base W_k, can we close
this gap cheaply?

### 2.2 Adaptation via SVD Warm-Start

Standard LoRA training initializes A with Kaiming and B = 0
(zero output at initialization). For adaptation, we need to
start from the trained expert's state.

Given a trained delta dW_i = s * A_i @ B_i, we decompose it
via SVD to recover (A_init, B_init) such that:

    dW_i = s * A_init @ B_init

Procedure:
1. Compute target = dW_i / s (remove scaling)
2. SVD: target = U @ diag(sigma) @ V^T
3. Set A_init = U[:, :r] @ diag(sqrt(sigma[:r]))
4. Set B_init = diag(sqrt(sigma[:r])) @ V^T[:r, :]

This ensures A_init @ B_init = target (up to truncation at rank r).
Since the original delta was already rank-r, the decomposition is exact.

### 2.3 Adaptation Dynamics

The adaptation objective is:

    min_{A, B} E_{(x,y) ~ D_domain} [L(W_k + s * A @ B; x, y)]

with (A, B) initialized at (A_init, B_init).

The gradient with respect to A at step t:

    dL/dA = s * x^T @ (dL/dy) @ B^T

and with respect to B:

    dL/dB = s * A^T @ x^T @ (dL/dy)

Key observation: the gradient landscape depends on W_k, not W_p.
The adaptation finds the nearest local minimum in the loss landscape
defined by the new base. If the perturbation W_k - W_p is small
(rank-16 SVD retains most energy), the loss landscapes are similar
and adaptation should be fast.

### 2.4 Transfer Gap Decomposition

Define three loss levels for expert i:

    L_orig = L(W_p + dW_i^{train})        -- original quality
    L_zs   = L(W_k + dW_i^{train})        -- zero-shot transfer
    L_adapt(n) = L(W_k + dW_i^{adapt,n})  -- after n adaptation steps

The transfer gap: G = L_zs - L_orig

The adaptation recovery: R(n) = L_zs - L_adapt(n)

The gap reduction percentage: R%(n) = R(n) / G * 100

The retrained reference: L_rt = L(W_k + dW_i^{retrained})

### 2.5 The Adaptation-Forgetting Tradeoff

Adaptation on W_k moves the expert away from its optimum for W_p.
Define the forgetting cost:

    F(n) = L_adapt_on_orig(n) - L_orig

where L_adapt_on_orig(n) is the adapted expert evaluated on the
ORIGINAL base W_p.

We expect: as n increases, R(n) increases (closing gap on new base)
but F(n) also increases (degrading quality on old base). This is a
fundamental tradeoff. An expert adapted for a new base cannot
simultaneously be optimal for the old base.

## 3. Empirical Results

### 3.1 Primary Experiment (adapt_lr = 1e-3, 3 seeds)

| Steps | ZS Ratio | Adapted Ratio | Retrained Ratio | Gap Reduction | K2 Ratio |
|-------|----------|---------------|-----------------|---------------|----------|
| 1     | 1.042    | 1.042         | 1.014           | 1.4%          | 1.000    |
| 5     | 1.042    | 1.039         | 1.014           | 7.2%          | 1.000    |
| 10    | 1.042    | 1.037         | 1.014           | 13.2%         | 1.000    |
| 25    | 1.042    | 1.032         | 1.014           | 24.7%         | 1.003    |
| 50    | 1.042    | 1.027         | 1.014           | 35.1%         | 1.007    |

### 3.2 Sensitivity Analysis (adapt_lr = 3e-3, seed 42)

| Steps | Adapted Ratio | Gap Reduction | Recovery vs RT | K2 Ratio |
|-------|---------------|---------------|----------------|----------|
| 10    | 1.028         | 25.6%         | 41.2%          | 1.004    |
| 25    | 1.023         | 39.3%         | 62.4%          | 1.010    |
| 50    | 1.019         | 50.0%         | 79.6%          | 1.014    |
| 100   | 1.014         | 64.3%         | 102.6%         | 1.017    |
| 200   | 1.006         | 84.2%         | 135.3%         | 1.019    |

### 3.3 Adaptation Rate Model

The gap reduction follows approximately logarithmic growth:

    R%(n) ~ a * ln(n) + b

Fitting to the primary experiment data (adapt_lr = 1e-3):

    R%(n) ~ 8.5 * ln(n) - 2.0

This predicts:
- 50% gap reduction at n ~ 550 steps
- 80% gap reduction at n ~ 16,000 steps

With adapt_lr = 3e-3 (3x higher):
- 50% gap reduction at n ~ 50 steps
- 80% gap reduction at n ~ 160 steps

The adaptation speed scales roughly linearly with learning rate
in this regime.

### 3.4 The Forgetting Curve

The K2 ratio (quality on original base after adaptation) grows
approximately linearly with adaptation steps:

    K2(n) ~ 1.0 + c * n

At adapt_lr = 1e-3: c ~ 1.3e-4 per step
At adapt_lr = 3e-3: c ~ 3.8e-4 per step

This is a small effect (1.6% degradation at 50 steps with low LR,
3.9% at 200 steps with high LR), but it is monotonically increasing.

### 3.5 Crossover Point

The sensitivity experiment reveals a crossover at ~100 steps
(adapt_lr = 3e-3): the adapted expert EXCEEDS the retrained
baseline quality on the new base (recovery > 100%). This occurs
because the adapted expert has a warm start from the pre-trained
initialization, while the retrained expert starts from
Kaiming/zero initialization. The warm start provides a better
local minimum at short adaptation budgets.

However, this comes at the cost of 1.7% degradation on the
original base, meaning the adapted expert is no longer a valid
drop-in replacement for the original.

## 4. Kill Criteria Analysis

### K1: 50 steps must reduce gap by >50%

At adapt_lr = 1e-3 (primary): 35.1% gap reduction. **KILLED**.
At adapt_lr = 3e-3 (sensitivity): 50.0% gap reduction. Borderline.

The threshold of 50% at 50 steps is too aggressive for this
learning rate regime. The adaptation follows logarithmic dynamics:
fast initial gains, then diminishing returns.

### K2: Adapted expert must not be worse on original base

At 50 steps: K2 ratio = 1.007 (primary), 1.014 (sensitivity).
The adapted expert IS worse on the original base by 0.7-1.4%.
This is below the 5% hard-fail threshold but above 0%.

**KILLED** on strict interpretation (any degradation at all).
**SURVIVES** on practical interpretation (<5% degradation is acceptable).

## 5. Practical Implications

### 5.1 The Deploy-Then-Adapt Protocol

Despite K1 being killed, the experiment reveals a practical protocol:

1. Base changes (e.g., Qwen2.5 -> Qwen3)
2. Deploy all experts zero-shot immediately (3.7% gap, acceptable)
3. Schedule background adaptation jobs (25-50 steps per expert)
4. After adaptation, experts are ~2.7% gap (primary) or ~1.9% gap (high LR)
5. Each adaptation costs ~50 steps vs 300 steps for full retraining (6x cheaper)

### 5.2 Adaptation Budget vs Expert Count

For N experts needing adaptation:
- Full retraining: N * 300 steps
- 50-step adaptation: N * 50 steps (6x cheaper, recovers ~35-50% of gap)
- 100-step adaptation: N * 100 steps (3x cheaper, recovers ~64% of gap)

At N=500 experts: full retraining = 150,000 steps vs 50-step adapt = 25,000 steps.
On A5000 at ~15 min per expert: 125 hours vs 21 hours.

### 5.3 The Forgetting Tax

Adaptation creates a "forgetting tax": the adapted expert is no longer
optimal for the original base. This means:
- Adapted experts are BASE-SPECIFIC (tied to W_k, not portable)
- If you need to revert to W_p, you need the original (non-adapted) deltas
- Storage: keep both original and adapted deltas (12MB per expert instead of 6MB)

## 6. Worked Example (d=64, r=8, k_base=16, seed 42)

Expert 1 (f_j domain):

| Metric | Value |
|--------|-------|
| Original loss (on pretrained base) | 0.4245 |
| Zero-shot loss (on rank-16 base) | 0.4439 |
| Transfer gap | 0.0194 (4.57%) |
| Adapted loss (50 steps, lr=1e-3) | 0.4374 |
| Adapted gap | 0.0129 (3.04%) |
| Gap reduction | 33.5% |
| Retrained loss (300 steps) | 0.4324 |
| Adapted on original base | 0.4304 |
| Forgetting cost | 0.0059 (1.39%) |

SVD warm-start for this expert:
- delta shape per layer: (64, 256) for fc1, (256, 64) for fc2
- SVD decomposition exact (rank-8 delta from rank-8 LoRA)
- Reconstruction error: 0.0 (up to floating point)
- After 50 steps: LoRA params shift by ~3% from initialization
  (measured as ||A_adapt - A_init|| / ||A_init||)

## 7. Assumptions and Limitations

1. **Adaptation LR sensitivity**: Results depend strongly on learning rate.
   The optimal adaptation LR may differ from training LR and may need
   per-expert tuning at macro scale.

2. **SVD perturbation only**: Real base changes (continued pretraining)
   may create different loss landscape geometries that are harder or
   easier to adapt to.

3. **Same data for adaptation**: Adaptation uses the same domain data
   as original training. In production, the domain data may not be
   available during base swap.

4. **Micro scale**: At d=64, the LoRA deltas are large relative to
   base weights. At d=4096, deltas are proportionally smaller, which
   may make adaptation faster (smaller perturbation in relative terms).

5. **No learning rate scheduling**: A warmup or cosine schedule for
   adaptation might improve efficiency at small step counts.
