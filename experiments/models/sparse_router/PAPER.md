# Sparse Routing: Top-k Sweep Results

## Summary

**Top-1 routing catastrophically fails at micro scale.** Selecting only the
single highest-scoring group per token degrades composition quality by ~200%
vs top-2 (3-seed mean). Two of four kill thresholds exceeded. However, k=2/4/8
are nearly identical (within 1.6%), meaning the current k=2 default is already
near-optimal — no compute savings from increasing k, and k=1 is unusable.

## Experiment Design

Shared-base composition protocol (validated in capsule_moe):
1. Pretrain shared base on all data (300 steps)
2. Fine-tune capsule groups per domain (300 steps each, attention frozen)
3. Compose 8 groups (4+4), calibrate fresh router per top_k (100 steps)
4. Sweep k in {1, 2, 4, 8}, 3 seeds (42, 123, 7)

Additional baselines: uncalibrated (random) router at each k.

## Results

### Composition Quality (val loss, 3-seed mean +/- std)

| Setting       | Mean   | Std    | vs Joint | vs k=2   |
|---------------|--------|--------|----------|----------|
| joint         | 0.5188 | 0.0064 | baseline | —        |
| learned k=1   | 1.5799 | 0.5246 | +204.5%  | +200.6%  |
| learned k=2   | 0.5256 | 0.0058 | +1.3%    | baseline |
| learned k=4   | 0.5184 | 0.0054 | -0.1%    | -1.4%    |
| learned k=8   | 0.5172 | 0.0049 | -0.3%    | -1.6%    |
| uniform k=1   | 3.9545 | 4.6414 | +662.2%  | —        |
| uniform k=2   | 1.1117 | 0.3567 | +114.3%  | —        |
| uniform k=8   | 0.8883 | 0.3503 | +71.2%   | —        |

### Router Analysis (mean across seeds, G=8)

| k | H(p)  | H/H_max | C_1   | Domain% |
|---|-------|---------|-------|---------|
| 1 | 1.791 | 0.861   | 0.285 | 50.4%   |
| 2 | 1.573 | 0.756   | 0.354 | 50.5%   |
| 4 | 1.627 | 0.782   | 0.318 | 50.7%   |
| 8 | 1.633 | 0.785   | 0.311 | 50.8%   |

### Kill Threshold Checks

| Criterion                       | Value    | Threshold | Result   |
|---------------------------------|----------|-----------|----------|
| Top-1 vs top-2 degradation      | +200.6%  | >10%      | **KILL** |
| Learned vs uniform at k=1       | Learned wins 2/3 seeds | Loses | PASS* |
| Router entropy ratio at k=1     | 0.861    | >0.9      | PASS     |
| Top-1 vs joint degradation      | +204.5%  | >15%      | **KILL** |

*Learned beats uniform on mean, but on seed=7 learned k=1 (1.87) is WORSE than
uniform k=1 (1.12). The win is inconsistent and dominated by uniform's catastrophic
failure on seed=42 (uniform=9.31 vs learned=1.89).

## Root Cause Analysis

### Why top-1 fails catastrophically

**The router probability distribution is too flat for hard selection.**

At k=1, the model selects argmax(scores) and assigns weight w=1.0 regardless
of the router's confidence. The router entropy at k=1 is 0.861 * H_max — nearly
uniform. The top-1 probability (C_1) is only 0.285, meaning the "best" group
captures less than 29% of the router's probability mass. The other 71% is spread
across 7 groups that are silenced.

This is MATH.md Assumption 3 materialized: at k=1, a token where p_{g*} = 0.15
still gets w_{g*} = 1.0. The hard selection amplifies routing noise that k=2's
soft averaging smooths over.

### Why k=2/4/8 are nearly identical

Information is distributed approximately uniformly across groups at micro scale
(d=64, character-level data). The softmax router's slightly non-uniform weights
help at k=2+ (vs random routing which is catastrophic), but adding more groups
beyond 2 provides diminishing returns. At k=2, two groups capture ~35% of the
probability mass — enough to reconstruct the signal. At k=4 or k=8, the
additional groups contribute marginally.

### Domain alignment is ~50% everywhere

The router does NOT learn domain-discriminative routing at any k. Domain
alignment is ~50% (random chance) across all settings. This is consistent with
the contrastive_router finding: at d=64 with a-m/n-z names, domains are
indistinguishable in hidden space.

The softmax router routes by per-token task quality (which group minimizes
reconstruction error), not by domain identity. But this task-quality signal
is too weak and distributed to concentrate into a single group.

## Key Insights

1. **Hard selection (k=1) is fundamentally different from soft selection (k>=2).**
   The w=1.0 renormalization at k=1 removes gradient information about router
   confidence. At k=2+, the relative weights between selected groups preserve
   confidence information. This creates a phase transition, not a gradual
   degradation.

2. **The quality-compute tradeoff is flat above k=2.** At micro scale,
   k=2 (25% active MLP params) achieves 98.7% of joint quality. Adding more
   groups (k=4: 50%, k=8: 100%) gains <1.6%. The "knee" is between k=1 and
   k=2, not between k=2 and k=4.

3. **Learned routing is essential but barely tested.** Learned routing massively
   beats random routing at all k values (random routing is catastrophic: +71%
   to +662% vs joint). But the learned router only slightly outperforms joint
   training (+1.3% at k=2). The router's value is in PREVENTING bad routing,
   not in achieving good routing.

4. **Micro scale cannot test sparse (k=1) routing.** With d=64 and 8 groups
   of 64 capsules each, a single group has only 8K active parameters — too few
   to represent the character-level language model. At macro scale with larger
   groups and stronger domain diversity, k=1 may become viable.

## Implications for VISION.md

- **Sparse routing (k=1) deferred to macro scale.** Like contrastive routing,
  this requires more capacity per group and stronger domain signal.
- **k=2 is validated as the optimal composition sparsity at micro scale.**
  No need to increase k for quality; no opportunity to decrease k for compute.
- **The phase transition between k=1 and k=2 suggests a minimum "routing
  bandwidth" below which quality collapses.** This could be formalized.
- **Next steps:** Procrustes decomposition (Exp 3) or scale to 5+ experts (Exp 4)
  where larger G may reveal more about the k=1 boundary.
