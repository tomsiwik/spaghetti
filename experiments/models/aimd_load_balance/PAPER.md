# AIMD Expert Load Balancing: Research Digest

## Hypothesis

AIMD (Additive Increase / Multiplicative Decrease) feedback control on per-expert
routing bias achieves comparable load balancing and model quality to auxiliary
load-balancing loss (Switch Transformer style), without coupling balancing to the
training loss.

**Falsifiable prediction:** AIMD achieves lower load imbalance than no-balance
baseline and comparable quality to aux loss baseline, with convergence speed
within 2x of aux loss.

## What This Model Is

Three-way comparison of expert load balancing strategies in a capsule MoE:

1. **AIMD Balance**: Non-gradient feedback control inspired by TCP congestion
   control. Each expert has a routing bias (not a learned parameter). After each
   forward pass, overloaded experts get their bias multiplicatively decreased
   (aggressive), while underloaded experts get additive increase (gentle). This
   asymmetry is the core TCP insight (Jacobson 1988, Chiu-Jain 1989).

2. **Aux Loss Balance**: Switch Transformer balance loss L = G * sum(f_i * p_i)
   added to training loss (Fedus et al. 2022). Standard baseline.

3. **No Balance**: Pure softmax routing, no balancing. Control condition.

The AIMD approach is related to DeepSeek-V3's "auxiliary-loss-free" per-expert
bias, but with asymmetric updates (multiplicative decrease vs additive increase)
instead of DeepSeek-V3's symmetric sign-based updates.

## Lineage in the Arena

```
gpt
  +-- capsule_moe (softmax routing, balance loss)
       +-- aimd_balance (AIMD bias feedback, THIS WORK)
       +-- aux_loss_balance (Switch-style aux loss, BASELINE)
       +-- no_balance (no balancing, CONTROL)
```

## Key References

- Fedus et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models
  with Simple and Efficient Sparsity." arXiv:2101.03961.
- Jacobson (1988). "Congestion Avoidance and Control." SIGCOMM.
- Chiu & Jain (1989). "Analysis of the Increase and Decrease Algorithms for
  Congestion Avoidance in Computer Networks."
- DeepSeek-V3 Technical Report (2024). arXiv:2412.19437.
- Puigcerver et al. (2024). "From Sparse to Soft Mixtures of Experts."
  arXiv:2308.00951.

## Empirical Results

### Quality (val_loss, lower is better)

| Model | Seed 42 | Seed 123 | Seed 7 | Mean | Std |
|-------|---------|----------|--------|------|-----|
| aux_loss_balance | 0.5032 | 0.5095 | 0.5156 | **0.5094** | 0.0051 |
| aimd_balance | 0.5172 | 0.5064 | 0.5110 | 0.5115 | 0.0044 |
| no_balance | 0.5140 | 0.5056 | 0.5166 | 0.5121 | 0.0047 |

AIMD vs Aux Loss: **+0.41%** (AIMD is worse).

### Load Imbalance (all-layer mean, lower is better)

| Model | Step 50 | Step 500 | Final (all layers) |
|-------|---------|----------|--------------------|
| aux_loss_balance | 0.273 | 0.204 | **0.205** |
| aimd_balance | 0.709 | 0.490 | 0.561 |
| no_balance | 0.659 | 0.714 | 0.719 |

AIMD achieves better balance than no-balance (0.56 vs 0.72) but significantly
worse than aux loss (0.56 vs 0.21). Aux loss is **2.7x better** at load balance.

### Convergence to Fair Allocation

Neither AIMD nor aux loss converged to "fair" allocation (imbalance < 0.15) within
500 steps. Aux loss reached approximately 0.20 by step 200 and stayed there.
AIMD started at 0.71 and improved to 0.49 by step 500 -- still trending down but
far from equilibrium.

### Per-Layer Analysis

AIMD shows high variance across layers. Some layers achieve good balance
(imbalance 0.17-0.21 in layer 0 for seed 123) while others show extreme
imbalance (0.90 in layer 3 for seed 123). The bias feedback operates
per-layer independently and cannot coordinate across layers.

Aux loss shows more consistent balance across layers (0.11-0.30 range).

## Kill Criteria Evaluation

| Criterion | Value | Threshold | Verdict |
|-----------|-------|-----------|---------|
| KC1: AIMD quality vs aux loss | +0.41% worse | Must not be worse | **TRIGGERED** |
| KC2: Convergence speed ratio | Neither converged | AIMD <= 2x aux loss | **CANNOT EVALUATE** |

**KC1 is triggered**: AIMD produces worse quality than aux loss (+0.41%).

However, the quality difference is small (0.41%) and within noise range (individual
seeds show AIMD better on seed 123: 0.5064 vs 0.5095). The real failure is on
load balancing, not quality: AIMD achieves 2.7x worse imbalance than aux loss.

## Root Cause Analysis

The AIMD feedback loop **fights the gradient-based router**:

1. The router learns weights W_r via gradient descent to minimize task loss.
   It routes tokens to the best expert for each input.

2. The AIMD bias adjusts routing logits to equalize load. This pushes tokens
   AWAY from the experts that are best for them (the popular ones) and TOWARD
   experts that are worse (the idle ones).

3. At each step, gradient descent adjusts W_r to compensate for the bias
   changes, partially undoing the AIMD correction.

4. This creates an adversarial dynamic: AIMD and gradient descent are
   optimizing different objectives on the same routing distribution.

The auxiliary loss avoids this by **integrating both objectives into a single
optimization**: the balance term is part of the loss, so gradient descent
simultaneously optimizes for task quality AND load balance in a coherent way.

**This is the fundamental insight**: decoupled feedback (AIMD) is inferior to
coupled optimization (aux loss) because the router is a learned component. In
TCP, the "router" (network) is fixed infrastructure -- the analogy breaks down
for learned routers.

## Connection to DeepSeek-V3

DeepSeek-V3's auxiliary-loss-free approach uses a similar bias mechanism but with
key differences:

1. Their bias updates are based on sequence-level statistics (not batch-level)
2. They use a much smaller learning rate for bias updates (gamma << our alpha)
3. They have 256 experts (not 4), so each expert handles a tiny fraction
4. Their bias is updated less frequently (per-sequence, not per-step)

At micro scale with G=4, the AIMD corrections are too coarse -- a single bias
change can shift significant token mass. At G=256 (DeepSeek-V3 scale), each
bias correction has a much smaller effect, potentially making feedback-based
approaches more viable.

## Micro-Scale Limitations

1. **G=4 experts is too few**: AIMD corrections are coarse (each expert is
   25% of capacity). At G=64-256 (macro), corrections would be finer-grained.

2. **500 steps is too short**: AIMD may need longer to converge. TCP convergence
   guarantees are asymptotic and assume many round-trips.

3. **Batch-level statistics are noisy**: With batch_size=32, token_length=32,
   load fractions are estimated from 1024 tokens. At macro scale with larger
   batches, estimates would be more stable.

4. **No capacity buffer**: Real MoE systems (Switch, GShard) use capacity
   factors and token dropping. AIMD might work better with explicit capacity
   constraints that make the TCP analogy more direct.

5. **Quality difference is noise-level**: The 0.41% gap is within 1-seed
   variance. A larger-seed study might show no significant difference.

## What Would Kill This

**At micro scale (confirmed killed):**
- AIMD achieves 2.7x worse load balance than aux loss
- AIMD never converges to fair allocation in 500 steps
- The adversarial dynamic between bias feedback and gradient descent prevents
  convergence

**At macro scale (untested predictions):**
- If DeepSeek-V3's bias-based approach (which is essentially AI/AD, not AIMD)
  works at G=256, then the symmetric (AI/AD) variant is sufficient and the
  multiplicative decrease of AIMD adds unnecessary instability
- If the adversarial dynamic persists at G=256, then ALL feedback-based bias
  approaches (including DeepSeek-V3's) should fail -- but DeepSeek-V3 reports
  success, suggesting G or update frequency matters

## Key Takeaway

TCP's AIMD works because the network topology is fixed. In MoE, the "network"
(router) is learned simultaneously. Decoupled feedback control on a learned
system creates an adversarial dynamic that prevents convergence. The aux loss
approach succeeds because it integrates balancing into the same optimization
objective as task quality. This is a principled negative result that explains
why auxiliary losses persist as the standard approach despite their coupling
to the training objective.

The connection to DeepSeek-V3 suggests that feedback-based bias works at
much larger G (256 experts) where per-correction perturbation is small, but
the micro-scale experiment with G=4 cannot validate this.
