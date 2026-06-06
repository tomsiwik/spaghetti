# Combined Parallel Block + Pure-Linear Attention Composition: Research Digest

## Hypothesis

Combined parallel blocks (Tiny Aya style) and pure-linear attention (all
GatedDeltaNet) will NOT degrade capsule composition quality by more than 5%
compared to the validated sequential + hybrid 3:1 baseline.

## Verdict: PASS (degradation +1.48%, threshold 5%)

The combined architecture composes within 1.48% of the sequential+hybrid
baseline. Zero catastrophic failures across 20 runs (5 seeds x 4 conditions).
The two modifications compose well together with approximately additive
effects (interaction term +0.31%).

## What This Model Is

This experiment combines two individually-proven architectural simplifications
and tests whether they interact destructively when applied simultaneously:

1. **Parallel blocks** (proven in exp_parallel_block_capsules): Single
   normalization per layer, attention and capsule pool branches from same
   normalized input. Eliminates the sequential norm1->attn->norm2->mlp chain.

2. **Pure-linear attention** (proven in exp_pure_linear_composition): All
   layers use full GatedDeltaNet linear attention. No full causal self-attention
   layer anywhere.

The combined architecture is maximally simplified:
```
x_{l+1} = x_l + GatedDeltaNet(Norm(x_l)) + CapsulePool(Norm(x_l))
```

This is the simplest composition-safe transformer block: one norm, two
parallel branches, one residual update. Every layer is identical (all linear
attention, no special "anchor" layer).

## Lineage in the Arena

```
gpt (dense baseline)
 |-- capsule_moe (routed capsule groups)
      |-- parallel_capsule_moe (parallel blocks, PROVEN)
      |-- full_gdn_stack_capsule_moe (full GatedDeltaNet, 3:1)
           |-- pure_linear_composition (4:0 all-linear, PROVEN)
           |-- THIS: parallel + pure-linear combined
```

## Key References

- **Parallel blocks experiment** (micro/models/parallel_block_capsules/):
  -0.39pp composition gap vs sequential, 30% faster fine-tuning. 3 seeds.
- **Pure-linear composition** (micro/models/pure_linear_composition/):
  +1.02% degradation vs hybrid 3:1, zero catastrophic failures. 7 seeds.
- **Full GatedDeltaNet stack** (micro/models/full_gdn_stack/): All 6
  GatedDeltaNet components composition-safe, +0.13% gap. 7 seeds.
- **Tiny Aya** (Cohere, 2026): 3.35B model using parallel transformer blocks.

## Experimental Design

2x2 factorial with 5 seeds per condition:

| | Hybrid 3:1 | Pure-Linear 4:0 |
|---|---|---|
| **Sequential** | A (baseline) | B |
| **Parallel** | C | D (test) |

Protocol (identical to parent experiments):
1. Pretrain shared base on all data (300 steps)
2. Fine-tune capsule groups per domain (300 steps, attention frozen)
3. Compose by concatenating domain groups, double top-k
4. Calibrate router on mixed data (100 steps)
5. Evaluate on per-domain val sets

## Empirical Results

### Main Results (5 seeds: 0-4)

| Condition | Composed (mean) | Joint (mean) | Gap mean | Gap median | Gap std |
|-----------|----------------|-------------|----------|------------|---------|
| seq_hybrid (A) | 0.5047 | 0.5072 | -0.50% | -0.35% | 1.29% |
| seq_pure_linear (B) | 0.5102 | 0.5073 | +0.58% | +0.52% | 0.56% |
| par_hybrid (C) | 0.5050 | 0.5057 | -0.13% | -0.31% | 0.73% |
| par_pure_linear (D) | 0.5121 | 0.5073 | +0.96% | +0.80% | 0.69% |

### Kill Criterion

    Degradation (par_pure_linear vs seq_hybrid): +1.48%
    Threshold: >5%
    Result: PASS

### Factorial Analysis

| Effect | Magnitude |
|--------|-----------|
| Parallel blocks (with hybrid attn) | +0.07% |
| Parallel blocks (with linear attn) | +0.37% |
| Pure-linear attn (with sequential) | +1.10% |
| Pure-linear attn (with parallel) | +1.41% |
| Predicted additive | +1.17% |
| Actual combined | +1.48% |
| **Interaction term** | **+0.31%** |

The interaction term (+0.31%) is small (< 1%), indicating the two
modifications are approximately additive. There is a slight negative
interaction (combined is marginally worse than sum of parts), but
well within noise for 5 seeds.

### Per-Seed Composition Gaps

| Seed | seq_hybrid | seq_pure_linear | par_hybrid | par_pure_linear |
|------|-----------|----------------|-----------|----------------|
| 0 | -1.15% | +0.07% | -0.32% | +0.54% |
| 1 | +0.40% | +0.77% | -1.05% | +2.14% |
| 2 | -2.33% | +1.42% | +0.07% | +0.91% |
| 3 | +0.94% | +0.10% | +0.95% | +0.80% |
| 4 | -0.35% | +0.52% | -0.31% | +0.41% |

Zero catastrophic failures (gap > 20%) across all 20 runs.

### Throughput Observations

From the training logs (tok/s during fine-tuning):

| Block type | Attention type | Fine-tune tok/s (approx) |
|-----------|---------------|------------------------|
| Sequential | Hybrid 3:1 | ~29K-45K |
| Sequential | Pure-linear | ~23K-35K |
| Parallel | Hybrid 3:1 | ~45K-48K |
| Parallel | Pure-linear | ~37K |

Parallel blocks are consistently faster (~30-50% speedup over sequential
with same attention type). Pure-linear is slightly slower than hybrid
at micro scale due to the sequential recurrence in GatedDeltaNet replacing
the batched matmul of full attention (this relationship inverts at macro
scale where O(T) vs O(T^2) dominates).

## Key Findings

1. **Modifications compose additively**: The combined effect (+1.48%) is
   well-predicted by the sum of individual effects (+1.17%), with a small
   interaction term (+0.31%). No destructive interference between parallel
   execution and linear attention.

2. **Composition gap dominated by attention type, not block type**: The
   parallel vs sequential choice barely affects composition (+0.07% to
   +0.37%), while hybrid vs pure-linear has a larger effect (+1.10% to
   +1.41%). The attention mechanism matters more than the block topology
   for composition quality.

3. **Combined architecture is viable**: At +1.48% degradation, the maximally-
   simplified architecture (parallel + all-linear) is well within the 5%
   tolerance. It provides the simplest possible composition-safe block.

4. **Variance structure**: Pure-linear conditions show lower gap variance
   (0.56%, 0.69%) than the hybrid conditions (1.29%, 0.73%). The
   GatedDeltaNet decay gate may regularize composition behavior.

## Implications

The combined parallel + pure-linear architecture is the **simplest
composition-safe transformer block** validated in this project:

```
x_{l+1} = x_l + GDN(Norm(x_l)) + CapsulePool(Norm(x_l))
```

Properties:
- One normalization per layer (vs two in sequential)
- All layers identical (vs mixed attention types in hybrid)
- Parallel branches (vs sequential dependency in standard)
- O(T) attention complexity (vs O(T^2) in full attention)

At macro scale, these simplifications compound:
- ~30% throughput gain from parallel blocks (measured at micro)
- O(T) scaling enables longer contexts without attention memory blowup
- Uniform layers simplify implementation and hardware mapping
- Composition protocol unchanged (same capsule pool API)

## Micro-Scale Limitations

1. **5 seeds per condition**: Adequate for directional signal but the
   interaction term (+0.31%) is smaller than per-seed noise (~1pp).
   More seeds would be needed to confirm true additivity.

2. **T=32 sequence length**: At this scale, neither the O(T) advantage
   of linear attention nor the parallel execution advantage is stressed.
   Both benefits compound at longer sequences.

3. **d=64, L=4**: GatedDeltaNet's recurrent state capacity (16x16 at
   micro) has not been stressed. At d=4096 with 256x256 states over
   24+ layers, cumulative state saturation without any full attention
   could be more damaging.

4. **No per-layer interference measurement**: Unlike the pure-linear
   experiment, this experiment does not decompose per-layer interference.
   The parallel block topology changes how interference propagates,
   making the existing sequential measurement invalid.

5. **MLX execution model**: The "parallel" branches execute sequentially
   on MLX. True hardware parallelism (GPU stream parallelism) at macro
   scale could show different characteristics.

## What Would Kill This

**At micro scale:**
- Combined gap exceeding 5% at N=5+ composed domains (more interference
  to manage without sequential normalization between branches)
- A deeper model (8+ layers) showing cumulative degradation from the
  absence of both sequential normalization and full attention

**At macro scale:**
- GatedDeltaNet state saturation at d_h=256+ causing quality collapse
  that a single full attention layer would prevent
- The 1.48% gap widening to >5% with real domain composition (code + math)
  where domains require more precise global context tracking
- Parallel block gradient flow instability at large scale (the lack of
  inter-branch normalization could cause activation magnitude divergence
  over 24+ layers)

## Summary

Combined parallel blocks + pure-linear attention compose well at micro scale.
The 2x2 factorial design shows approximately additive effects (+0.31%
interaction, within noise). The kill criterion (>5% degradation) is not
triggered at +1.48%. This validates the simplest possible composition-safe
architecture: one norm, two parallel branches (GatedDeltaNet + CapsulePool),
one residual. Zero catastrophic failures across 20 runs.
