# Combined Dead Capsule + Gate-Product Pruning for SwiGLU: Research Digest

## Hypothesis

Combining dead capsule pruning (activation frequency = 0) with gate-product
magnitude pruning (mean |gate*up| < tau) will yield >5pp higher pruning rates
than either method alone, because the two criteria identify complementary
parameter sets.

## What This Model Is

A SwiGLU transformer (matching Qwen3.5 MLP architecture) with post-training
pruning applied using the UNION of two criteria:
1. **Dead capsules**: gate product is zero for all calibration inputs
2. **Gate-product-low**: mean gate product magnitude below threshold tau

The experiment profiles both criteria on the same trained models and measures
(a) set overlap between dead and gate-prunable capsules, and (b) whether
combining both criteria exceeds either alone.

## Lineage in the Arena

```
gpt
 +-- silu_capsule
      +-- swiglu_gate_pruning       (parent: gate-product pruning)
           +-- swiglu_combined_dead_capsule  (this experiment)

(Also informed by:)
gpt
 +-- relu_router
      +-- dead_capsule_pruning      (parent: dead capsule pruning for ReLU)
```

## Key References

- Dead capsule pruning (Exp 9): tau=0 exact, 57% dead in ReLU composed models
- SwiGLU gate pruning (Exp 16): 66.5% prunable at tau=0.05 with aux sparsity loss
- SiLU pruning (Exp 15): 0% prunable, floor ~0.046 -- KILLED
- Pruning controls (Exp 10): dead capsule pruning is general ReLU, not composition-specific

## Empirical Results

### Setup
- Architecture: SwiGLU GPT, d=64, n_head=4, n_layer=4, P=128 capsules/layer
- Training: 300 steps, batch_size=32, lr=3e-3, single domain (a-m names)
- Aux losses: l1_target_sparsity=0.50, l1_coeff=0.01, balance_coeff=0.01
- Profiling: 20 batches on joint validation set
- 3 seeds: 42, 123, 7

### Dead Capsule Rates (tau=0, exact)

| Layer | Seed 42 | Seed 123 | Seed 7 | Mean |
|-------|---------|----------|--------|------|
| 0     | 0.0%    | 0.0%     | 0.0%   | 0.0% |
| 1     | 0.0%    | 0.0%     | 0.0%   | 0.0% |
| 2     | 0.0%    | 0.0%     | 0.0%   | 0.0% |
| 3     | 0.0%    | 0.0%     | 0.0%   | 0.0% |

**Zero dead capsules across all seeds, all layers.** Minimum fire frequency
across all 512 capsules and 3 seeds: 0.998 (>99.8% of positions fire).

### Gate-Product Pruning (3-seed mean, dead_tau=0)

| gate_tau | Dead% | Gate% | Combined% | Quality delta | Combined advantage |
|----------|-------|-------|-----------|---------------|--------------------|
| 0.005    | 0.0%  | 0.0%  | 0.0%      | +0.00%        | +0.0pp             |
| 0.010    | 0.0%  | 0.0%  | 0.0%      | +0.00%        | +0.0pp             |
| 0.020    | 0.0%  | 1.7%  | 1.7%      | +0.00%        | +0.0pp             |
| 0.050    | 0.0%  | 57.0% | 57.0%     | +0.79%        | +0.0pp             |
| 0.100    | 0.0%  | 99.1% | 99.1%     | +17.71%       | +0.0pp             |

Combined advantage is **exactly 0.0pp** at every threshold because the dead
capsule set is empty.

### Gate Product Distribution

Minimum gate product mean_abs across all capsules:
- Seed 42: 0.012 (layer 1)
- Seed 123: 0.033 (layer 2)
- Seed 7: 0.023 (layer 2)

This is 2-7x above the dead_epsilon threshold (1e-6), confirming that SiLU's
nonzero floor propagates through the gate product.

## Kill Criteria Evaluation

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| KC1: Combined > best single by >5pp | 5pp | 0.0pp advantage | **KILL** |
| KC2: Quality degrades >3% | 3% | +0.79% at best combo | PASS |

**KILLED on KC1.** Combined pruning provides exactly zero advantage over
gate-product pruning alone because the dead capsule set is empty.

## Root Cause Analysis

The hypothesis assumes dead capsules exist in SwiGLU models. They do not.

**Why ReLU has dead capsules but SwiGLU does not:**

| Property | ReLU | SiLU/SwiGLU |
|----------|------|-------------|
| Activation at z=0 | Exactly 0 | ~0 but positive |
| Activation for z<0 | Exactly 0 | Negative (small) |
| Floor | 0 (hard) | ~4.5e-5 (soft) |
| Dead capsules at micro | 57% | 0% |
| Pruning mechanism | Frequency-based (fires/doesn't) | Magnitude-based (small/large) |

SiLU(z) = z * sigmoid(z) is strictly positive for z > 0 and strictly negative
for z < 0, with |SiLU(z)| > 0 for all finite z. The gate product inherits this
property: it can be very small but never exactly zero.

**This is not a bug -- it is a fundamental architectural difference.** Dead
capsule pruning is a ReLU-specific technique. For SwiGLU, magnitude-based
pruning (gate-product threshold) is the correct and sufficient approach.

## What Was Learned

1. **Dead capsule pruning does not apply to SwiGLU architectures.** The concept
   of "dead" neurons (never fire) is specific to ReLU-family activations with
   hard zeros. SiLU/SwiGLU neurons always fire, with varying magnitude.

2. **The two pruning criteria are not complementary -- one is vacuous.** Rather
   than identifying overlapping vs complementary sets, we found that the dead
   capsule set is always empty for SwiGLU models.

3. **Gate-product magnitude pruning IS the right criterion for SwiGLU.** The
   parent experiment's finding (57-80% prunable at tau=0.05 with <1.3% quality
   loss) stands as the best available pruning approach for SwiGLU.

4. **Pruning taxonomy depends on activation function.** The correct pruning
   criterion is:
   - ReLU: frequency-based (dead capsule pruning, exact at tau=0)
   - SiLU/SwiGLU: magnitude-based (gate-product pruning, approximate)
   - These are not combinable because only one applies per architecture.

## Micro-Scale Limitations

- Tested only at d=64, P=128. At larger scale, numerical precision could
  potentially create near-zero gate products, but the SiLU floor is
  architecture-level, not scale-dependent.
- Only tested single-domain training. Composition (concatenating capsule pools
  from different domains) was not tested, but the zero-dead finding is about
  the activation function, not the composition mechanism.
- The parent experiment used aux sparsity loss, which encourages small gate
  products. Without aux loss, the gate product distribution shifts higher
  (as shown in threshold sensitivity experiments).

## What Would Kill This

This experiment is already killed on KC1. The finding (SwiGLU has no dead
capsules) is robust and architecture-level:

- **At micro scale:** Would need SiLU to produce exact zeros, which is
  mathematically impossible for finite inputs.
- **At macro scale:** The SiLU floor is even more robust at higher precision
  and larger dimensions. Confirmed by macro experiments showing 0% dead
  capsules at d=896.
