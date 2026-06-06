# LEARNINGS.md: exp_self_growing_toy

## Core Learning

**Sequential adapter promotion from random init produces catastrophic interference.
Each promotion overwrites 2-3 prior domains (sawtooth pattern). The grown base retains
only 26% of the knowledge that joint training captures. Self-growing from scratch is
NOT viable without structural interference guarantees during promotion.**

## Impossibility Structure

The failure is structural, not parametric:
1. Each promotion adds a rank-4 delta to the base weights
2. Sequential deltas with random A-matrices overlap in weight space
3. Later promotions partially undo earlier ones (last-writer-wins)
4. The sawtooth pattern: last-promoted domain retains 2.33 loss improvement,
   first-promoted retains only 0.29

This is fundamentally different from composition (Finding #225, #323):
- **Composition:** base stays FROZEN, adapters active simultaneously, Grassmannian
  A provides 17x interference decorrelation
- **Promotion:** base is MODIFIED, each promotion changes the optimization landscape
  for subsequent adapters

Even Grassmannian A would not fully solve promotion because:
- The base changes with each promotion
- Subsequent adapter training happens on a DIFFERENT landscape
- Orthogonal A-matrices are only orthogonal in the ORIGINAL weight space

## What Would Fix This

1. **Continual learning techniques:** Elastic Weight Consolidation (EWC), Progressive
   Neural Networks (PNP), PackNet — protect prior knowledge during promotion
2. **From pre-trained base, not random init:** A pre-trained base already has useful
   representations. Promotion adds domain expertise to existing knowledge rather than
   building from scratch in overlapping directions.
3. **Grassmannian + pre-trained:** Combine structural guarantees with a competent base.
   This is what the current Pierre architecture already does (pre-trained Qwen3-4B +
   Grassmannian LoRA adapters).

## What Worked (Partially)

1. **K842 PASS:** Training speed is constant across promotions (0.94x ratio).
   Early convergence IMPROVES (loss@50: 4.62 first vs 3.98 fifth). The promoted
   base does provide better features — the problem is interference, not utility.

2. **Individual promotions work:** Each domain improves during its own cycle
   (+0.71 to +2.00 loss reduction). The mechanism is sound per-cycle.

## Literature Validation

- **ReLoRA (2307.05695):** Proves gradient accumulation for periodic merge on
  SINGLE data stream. The analogy to MULTI-domain sequential training is structurally
  invalid (different optimization problems in different orders).
- **Continual learning literature:** This is catastrophic forgetting by another name.
  The field has extensive results showing sequential training on different distributions
  causes interference without explicit protection.

## Practical Implication

**Pierre's architecture is correct: pre-trained base + composable adapters.** The value
is in COMPOSITION (keeping the base frozen, using adapters for domain flexibility), not
in growing the base through promotion. Finding #330 (scale calibration) resolves the
last remaining issue (the scale catastrophe). The architecture is:
1. Pre-trained base (Qwen3-4B or BitNet-2B-4T)
2. Grassmannian LoRA adapters (structural interference guarantee)
3. Scale calibration at inference (scale=13 for domain + MMLU preservation)
4. Block-diagonal masking + per-token MLP routing for integrated serving (#323)
