# Amplification Factor Scaling: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Values |
|--------|-----------|--------|
| d | Model embedding dimension | {64, 128, 256} |
| d_ff | FFN intermediate dimension | 4d |
| r | LoRA rank (expert) | 8 (fixed) |
| L | Number of transformer layers | 4 |
| N | Number of domain experts | 4 |
| k | SVD truncation rank for base delta | d/2, d/4, d/8, d/16 |
| W_p | Pretrained weight matrix | R^{d_out x d_in} |
| W_s | Skeleton (random init) weight | R^{d_out x d_in} |
| Delta | Base delta: W_p - W_s | R^{d_out x d_in} |
| dW_i | Expert i's LoRA delta | R^{d_out x d_in} |
| c | Amplification factor | scalar |

## 2. The Amplification Factor

### 2.1 Definition

From the parent experiment (zero_shot_base_transfer), we observed that
zero-shot experts amplify base error. The amplification factor c quantifies
this:

    L_expert_zs(k) / L_expert_pretrained = 1 + c * (L_base(k) / L_base_pretrained - 1)

Rearranging:

    c = (L_expert_zs(k) / L_expert_pretrained - 1) / (L_base(k) / L_base_pretrained - 1)

where:
- L_expert_zs(k): expert loss on SVD rank-k base (zero-shot, no retraining)
- L_expert_pretrained: expert loss on original pretrained base
- L_base(k): base model loss on SVD rank-k reconstruction
- L_base_pretrained: original pretrained base loss

When c > 1: experts amplify base error (worse than base degradation alone)
When c = 1: experts track base error exactly (neutral)
When c < 1: experts are robust to base perturbation (partial compensation)

### 2.2 Why c Might Depend on d

The LoRA delta has shape dW_i in R^{d_out x d_in}. For MLP layers:
- fc1: dW in R^{4d x d}, so ||dW|| / ||W|| ~ r/d (relative magnitude)
- fc2: dW in R^{d x 4d}, so ||dW|| / ||W|| ~ r/d

At larger d with fixed r:
- The LoRA delta is a SMALLER perturbation relative to base weights
- The SVD base approximation error is spread across MORE dimensions
- Each expert delta interacts with MORE base weight dimensions

Hypothesis: since ||dW|| / ||W|| ~ r/d, the amplification c should
decrease roughly as O(r/d) or O(1/sqrt(d)), making zero-shot transfer
SAFER at larger model dimensions.

### 2.3 First-Order Perturbation Analysis

Consider the loss function L(W + dW) where W is the base weight and
dW is the expert delta. Taylor-expanding around W:

    L(W + dW) ~ L(W) + grad_L(W)^T dW + (1/2) dW^T H dW

When we perturb the base to W' = W + epsilon (SVD error):

    L(W' + dW) ~ L(W') + grad_L(W')^T dW + (1/2) dW^T H' dW

The amplification comes from the change in gradient landscape:

    grad_L(W') != grad_L(W)

The difference depends on:
    ||grad_L(W') - grad_L(W)|| ~ ||H|| * ||epsilon||

Since the expert delta dW was optimized for grad_L(W), not grad_L(W'),
the misalignment between the expert's correction and the actual gradient
causes amplification.

The key scaling insight: ||epsilon|| / ||W|| stays roughly constant
for a given SVD rank ratio (k/d), but ||dW|| / ||W|| ~ r/d decreases.
So the RELATIVE impact of the gradient misalignment on the expert
output scales as:

    c ~ (||dW|| * ||H|| * ||epsilon||) / (||epsilon|| * ||L||)
      ~ ||dW|| / ||W|| * condition_number
      ~ r/d * kappa

If the condition number kappa doesn't grow faster than d/r, then
c stays bounded or decreases with d.

## 3. Experimental Design

### 3.1 Matched Rank Ratios

To make fair comparisons across dimensions, we use SVD ranks as
fixed fractions of d:

| d | SVD Ranks (k) | Rank Ratios (k/d) |
|---|---------------|-------------------|
| 64 | 32, 16, 8, 4 | 0.50, 0.25, 0.125, 0.0625 |
| 128 | 64, 32, 16, 8 | 0.50, 0.25, 0.125, 0.0625 |
| 256 | 128, 64, 32, 16 | 0.50, 0.25, 0.125, 0.0625 |

This ensures each dimension sees the same "relative base quality"
at each rank ratio, isolating the effect of d on amplification.

### 3.2 Training Scaling

To ensure comparable convergence across dimensions:
- d=64: 1000 pretrain / 300 expert steps, lr=3e-3
- d=128: 1500 pretrain / 400 expert steps, lr=2e-3
- d=256: 2000 pretrain / 500 expert steps, lr=1e-3

Larger models get more training steps (more parameters) and lower
learning rates (better convergence at larger scale).

### 3.3 Head Dimension

We keep head_dim = d/n_head = 16 constant, scaling n_head with d:
- d=64: n_head=4
- d=128: n_head=8
- d=256: n_head=16

This mirrors production model scaling (head_dim typically 64-128,
with more heads at larger d).

## 4. Power Law Fit

We fit c(d) = a * d^alpha via log-log linear regression:

    log(c) = log(a) + alpha * log(d)

Possible outcomes:
- alpha < 0: c decreases with d (SURVIVES, good for SOLE scaling)
- alpha = 0: c constant with d (SURVIVES, amplification bounded)
- 0 < alpha < 1: c grows sub-linearly (SURVIVES, but concerning)
- alpha >= 1: c grows linearly or faster (KILLED, K1)

NOTE: The parent MATH.md reported c ~ 0.06-0.10, but that used a
different definition: expert/base ratio ~ 1 + c * epsilon, where
epsilon is the SVD reconstruction error (a different quantity than
base_loss_ratio - 1). Our definition here (c = expert_excess / base_excess)
produces c ~ 1.4-2.0, consistent with the parent data when the
denominators are reconciled.

## 5. Empirical Results

### 5.1 Measurements at Matched Rank Ratio d/16

| d | SVD Rank | Base Ratio (3-seed mean) | Expert Ratio | c (mean +/- std) |
|---|---------|-------------------------|-------------|------------------|
| 64 | 4 | 1.229 | 1.321 | 1.42 +/- 0.18 |
| 128 | 8 | 1.161 | 1.250 | 1.58 +/- 0.19 |
| 256 | 16 | 1.072 | 1.135 | 1.90 +/- 0.18 |

Power law fit: c(d) = 0.343 * d^0.210
R-squared: 0.975
p-value: 0.10

### 5.2 The Perturbation-Magnitude Confound

At fixed d, c increases as perturbation decreases. This is a measurement
artifact from dividing by small numbers. Example at d=64:

| SVD Rank | base_excess | expert_excess | c |
|---------|------------|--------------|---|
| 4 | 0.229 | 0.321 | 1.42 |
| 8 | 0.100 | 0.167 | 1.71 |
| 16 | 0.019 | 0.042 | 2.30 |
| 32 | 0.001 | 0.002 | 1.66 |

At small perturbation, second-order (Hessian) terms dominate,
inflating c beyond its linear-regime value. At large perturbation
(base_excess > 0.05), c stabilizes around 1.4-1.9.

### 5.3 The Critical Observation: Base Perturbation Decreases with d

At matched rank ratio d/4, base_loss_ratio across dimensions:
- d=64: 1.019 (1.9% base degradation)
- d=128: 1.003 (0.3% base degradation)
- d=256: 0.999 (no measurable degradation)

SVD approximation quality improves dramatically with d. This means
even with constant c, the ABSOLUTE expert quality loss decreases:

    expert_loss_ratio = 1 + c * (base_loss_ratio - 1)

At d=256 with rank-64 (k/d=0.25): expert_ratio = 1 + 1.9 * 0 ~ 1.00

### 5.4 Net Effect on Zero-Shot Transfer Quality

Expert quality loss at matched rank ratio d/4:
- d=64: expert_ratio = 1.042 (4.2% loss)
- d=128: expert_ratio = 1.012 (1.2% loss)
- d=256: expert_ratio = 1.004 (0.4% loss)

Zero-shot transfer gets dramatically BETTER at scale despite c
growing mildly, because base approximation improves even faster.

## 6. Assumptions and Limitations

1. **Fixed LoRA rank**: r=8 across all dimensions. In production,
   rank might scale with d (e.g., r=16 at d=4096). The r/d ratio
   would remain similar, so the scaling law should generalize.

2. **Same data and task**: Using the same character-level names dataset
   at all dimensions. Larger models may be over-parameterized for this
   toy task, which could affect convergence differently.

3. **SVD perturbation only**: Real base model changes involve continued
   pretraining, not SVD truncation. The perturbation structure differs.

4. **3 data points for power law**: d={64, 128, 256} gives only 3
   points for the log-log fit. R-squared may be artificially high.
   Directional trend is more important than exact exponent.

5. **Convergence matching**: Different training budgets at each d are
   heuristic. Under-training at larger d would inflate c artificially.
