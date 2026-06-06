# Training-Time Composition Compatibility: Research Digest

## Hypothesis

An auxiliary loss during domain-specific fine-tuning reduces the composition gap
(independently-composed vs jointly-trained) by at least 50% relative to the
no-auxiliary baseline.

## Verdict: KILL

The auxiliary losses do NOT reduce the zero-shot composition gap. In fact,
all auxiliary conditions produce WORSE zero-shot composition than the no-aux
baseline. However, a secondary finding is significant: the combined auxiliary
loss improves WEIGHT-AVERAGING composition to match (or slightly beat) joint
training.

## What This Model Is

TrainingCompatGPT extends ReLURouterGPT with two auxiliary losses applied
during domain-specific fine-tuning:

1. **Weight orthogonality loss** (L_ortho): Penalizes capsule weight deltas
   that align with base model weight directions. Inspired by InfLoRA's
   orthogonality constraints for continual learning.

2. **Output-norm matching loss** (L_norm): Penalizes deviation of capsule pool
   output norms from base model output norms. Addresses magnitude mismatch
   between independently-trained domain pools.

The base is snapshotted after pretraining. During fine-tuning, these losses
push domain-specific deltas to be orthogonal to the base AND to each other
(indirectly), while maintaining similar output magnitudes.

## Lineage in the Arena

```
gpt (dense baseline)
 |-- capsule_moe (routed capsule groups)
      |-- relu_router (self-routing ReLU MLP)
           |-- training_compat (+ ortho + norm aux losses) <-- THIS
```

## Key References

- InfLoRA (2024): Orthogonality constraints on LoRA for continual learning
- MoRAM (2025): Self-routing LoRA adapters as associative memory
- MoE-Adapters4CL (2024): MoE adapters on frozen base for continual learning

## Empirical Results

### Zero-Shot Composition (concatenation, no calibration)

| Condition | Avg Loss | Gap vs Joint | Gap Reduction |
|-----------|----------|-------------|---------------|
| Joint     | 0.5287   | baseline    | --            |
| no_aux    | 0.5626   | +6.4%       | baseline      |
| ortho_only| 0.5652   | +6.9%       | -7.8% (WORSE) |
| norm_only | 0.5744   | +8.6%       | -34.9% (WORSE)|
| combined  | 0.5655   | +6.9%       | -8.5% (WORSE) |

**All auxiliary conditions worsen zero-shot composition.** The kill criterion
requires >50% reduction; the best condition (none) achieves 0% reduction.

### Weight-Averaging Composition (no calibration needed)

| Condition | Avg Loss | Gap vs Joint |
|-----------|----------|-------------|
| Joint     | 0.5287   | baseline    |
| no_aux    | 0.5354   | +1.3%       |
| ortho_only| 0.5312   | +0.5%       |
| norm_only | 0.5310   | +0.4%       |
| combined  | 0.5278   | **-0.2%**   |

**The combined condition achieves weight-averaged composition BETTER than joint
training.** This is a significant secondary finding: the auxiliary losses make
weight averaging more effective even though they make concatenation worse.

### Diagnostic Measurements (3-seed means)

**Weight delta orthogonality** (cosine similarity between domain deltas):
| Condition | Mean cos | Max cos |
|-----------|---------|---------|
| no_aux    | 0.170   | 0.205   |
| ortho_only| 0.109   | 0.167   |
| norm_only | 0.223   | 0.345   |
| combined  | 0.132   | 0.272   |

The ortho loss DOES reduce inter-domain cosine similarity (0.170 -> 0.109),
but this does not translate to better concatenation composition. Norm loss
INCREASES cosine similarity (0.170 -> 0.223), likely because constraining
output norms forces deltas into more similar directions.

**Cross-domain output norm variance** (lower = more matched):
| Condition | Norm variance |
|-----------|---------------|
| no_aux    | 794.53        |
| ortho_only| 11.05         |
| norm_only | 0.056         |
| combined  | 0.050         |

The norm loss dramatically reduces output magnitude variance (794 -> 0.05).
This explains why weight averaging improves: when norms are matched, averaging
produces a balanced blend. But for concatenation, matched norms mean the sum
is 2x the expected magnitude, which the network hasn't been trained to handle.

## Why Concatenation Gets Worse

The key insight from this experiment: **auxiliary losses that improve weight
averaging HURT concatenation, and vice versa.**

For weight averaging: success requires similar magnitudes and directions.
The auxiliary losses push toward this.

For concatenation: success requires that pool_1(x) + pool_2(x) approximates
pool_joint(x). But the auxiliary losses constrain the pools in ways that
make their SUM different from what joint training would produce:

1. Ortho loss pushes deltas away from base subspace, changing the function
   computed by each pool in ways that may increase the summed error
2. Norm loss equalizes magnitudes, making the sum 2x louder than expected
3. The function-space gap is NOT about weight-space properties (orthogonality,
   norms) but about the nonlinear interaction f(Ax) where f=ReLU

This confirms FINDINGS.md: "The gap is directional (function-space), not
scalar (loudness)."

## Micro-Scale Limitations

- Only N=2 domains tested (binary split a-m vs n-z)
- d=64, P=128 -- orthogonal complement dimension may be too small for
  the ortho loss to work as intended
- Single coefficient values (0.1) tested -- did not sweep hyperparameters
- Only ReLU activation tested (SiLU may behave differently)
- 200 fine-tuning steps -- short training may not show full effect

## What Would Kill This

Already killed for zero-shot concatenation composition. The kill criterion
is unambiguous: no auxiliary condition achieved >0% gap reduction.

For the weight-averaging finding to be considered robust:
- Need to verify at N=5 domains (not just N=2)
- Need to verify the -0.2% advantage holds with more seeds (could be noise)
- Need to test at macro scale where domains are more distinct

## Positive Findings Despite Kill

1. **Weight averaging + combined aux loss = joint quality.** The combined
   condition achieves -0.2% vs joint, meaning zero-calibration composition
   that matches joint training. Previous best was +1.3% (no_aux weight avg).

2. **Ortho loss does reduce delta cosine similarity.** From 0.170 to 0.109
   mean cosine. The mechanism works at the weight level; it just doesn't
   translate to function-space improvement for concatenation.

3. **Norm loss dramatically equalizes output magnitudes.** Variance drops
   from 794 to 0.05. This is a practical tool for making weight averaging
   more reliable.

4. **The composability-specialization tradeoff is real.** Auxiliary losses
   constrain the solution space, which hurts per-domain fine-tuning loss
   (visible in higher training losses for aux conditions). The pools
   specialize less but compose better via averaging.

## Implications for the Research Program

The function-space gap cannot be closed by training-time weight-space
regularization alone. The nonlinear composition problem is fundamental:

```
ReLU(A_1 @ x) + ReLU(A_2 @ x) != ReLU((A_1 + A_2) @ x)
```

No amount of orthogonality or norm matching changes this inequality.
Weight averaging sidesteps it by staying in the same parameter space
(averaging A_1 and A_2 rather than summing their outputs).

This suggests the remaining research directions for closing the gap are:
1. Better weight-space merging (TIES, DARE, task arithmetic with learned scales)
2. Calibration-based approaches (the existing 100-step router calibration)
3. Accepting the gap and optimizing the composition protocol instead
