# Concat+Calibrate N=5 Calibration Budget: Mathematical Analysis

## Setup

The concat+calibrate method composes N LoRA expert deltas via a learned router.

**Model dimensions:**
- d = 64 (embedding dimension)
- N = 5 (number of domain experts)
- L = 4 (transformer layers)
- r = 8 (LoRA rank)
- top_k = 2 (experts selected per token)

**Router architecture (per layer):**
- Router: R_l in R^{d x N}, bias-free linear
- Input: h in R^{B x T x d} (hidden states after layer norm)
- Scores: s = h @ R_l^T, s in R^{B x T x N}
- Probs: p = softmax(s), top-k masking, renormalization
- Output: y = sum_{e in top_k} p_e * MLP_e(h)

**Router parameters:**
- Per layer: d * N = 64 * 5 = 320 parameters
- Total: L * d * N = 4 * 320 = 1,280 router parameters
- These are the ONLY trainable parameters during calibration

## Calibration Optimization

**Objective:** minimize NTP loss over mixed-domain training data, updating only
router weights R_l for each layer l.

**Optimization:**
- Adam optimizer, lr = 3e-3
- Round-robin over N=5 domain training sets
- Each step: sample batch (B=32 sequences, T=32 tokens) from one domain

**Effective training signal per budget:**

| Budget | Total tokens | Tokens per domain | Router updates | Steps per domain |
|--------|-------------|-------------------|----------------|------------------|
| 100 | 102,400 | ~20,480 | 1,280 updates | ~20 |
| 200 | 204,800 | ~40,960 | 1,280 updates | ~40 |
| 300 | 307,200 | ~61,440 | 1,280 updates | ~60 |
| 500 | 512,000 | ~102,400 | 1,280 updates | ~100 |

At N=5, each domain gets only steps/5 gradient updates for routing.
At 100 steps, that is 20 steps per domain -- potentially insufficient for
the router to learn meaningful routing preferences among 5 experts.

## Expected Router Convergence

For N=5 with top_k=2, the router at each layer must learn to partition the
softmax probability mass across 5 experts. With 1,280 parameters total:

- Parameter-to-data ratio at 100 steps: 102,400 / 1,280 = 80x
- Parameter-to-data ratio at 500 steps: 512,000 / 1,280 = 400x

Both ratios are high (well above overfitting risk). The router has FAR more
data than parameters at every budget level. This suggests the failure is NOT
statistical underfitting of the router.

## Overfitting Analysis

The 500-step result being WORSE than 300 steps contradicts the underfitting
hypothesis. Possible explanations:

1. **Router overfitting to training data**: unlikely given 1,280 params and
   512K tokens (400x ratio).

2. **Optimization instability**: the loss landscape for routing N=5 experts
   with cross-entropy is non-convex. Longer training may push the router
   into a suboptimal routing pattern that happens to fit training batches
   but generalizes poorly.

3. **Domain round-robin creates optimization conflict**: at step s, the
   router is optimized for domain s%5. This could create oscillatory
   behavior in routing weights -- the router "forgets" domain A's routing
   while optimizing for domain B, especially with 5 domains competing.

4. **Fundamental capacity limitation**: with rank-8 LoRA, each expert has
   limited capacity (2 * L * d * r = 2 * 4 * 64 * 8 = 4,096 parameters
   per expert). The experts themselves may not be differentiated enough
   for routing to add value vs simple averaging.

## Comparison to N=2

At N=2 with 100 steps:
- Each domain gets 50 routing steps
- Router has 2 * 64 = 128 params per layer = 512 total
- top_k=2 means ALL experts are always selected (routing degenerates to
  learned weighted average)

The N=2 success may be because top_k=2 of 2 is simply learned averaging,
while top_k=2 of 5 requires genuine expert selection.

## Numerical Results

| Budget | Mean Val Loss | Std | vs Joint | vs Simple Avg |
|--------|-------------|-----|----------|---------------|
| 100 | 0.5215 | 0.0054 | +4.06% | +1.18% |
| 200 | 0.5220 | 0.0043 | +4.16% | +1.28% |
| 300 | 0.5158 | 0.0037 | +2.94% | +0.08% |
| 500 | 0.5226 | 0.0082 | +4.30% | +1.41% |

Baselines (N=5):
- Joint training: 0.5011 (std 0.0054)
- Simple average: 0.5154 (std 0.0068)
