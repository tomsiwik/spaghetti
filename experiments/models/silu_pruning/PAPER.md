# SiLU Pruning: Research Digest

## Hypothesis

Magnitude-threshold pruning on SiLU capsule MLPs can achieve meaningful
compression (>10% parameter reduction) without degrading quality more than
5% versus the unpruned model.

**Falsifiable.** Kill criterion: magnitude-threshold pruning degrades quality
>5% vs unpruned at any threshold that achieves >10% compression.

## What This Experiment Is

A controlled comparison of pruning viability between ReLU and SiLU capsule
MLPs at micro scale. Our prior experiments (Exp 9, Exp 10) proved that
ReLU dead capsule pruning achieves 57% compression with zero quality loss.
The macro experiment (Exp 5) then discovered that SiLU-based models have
0% dead capsules, blocking the transfer of this compression technique.

This experiment determines whether magnitude-based thresholds can substitute
for exact-zero detection, enabling pruning for SiLU activation functions
used in production models (Qwen, Llama, etc.).

**Protocol:**
1. Train ReLU and SiLU capsule models under identical conditions (d=64,
   P=128, 300 steps, single domain, 3 seeds)
2. Profile activation magnitudes: mean|activation| per capsule
3. Sweep pruning thresholds: tau in {0.001, 0.005, 0.01, 0.05, 0.1}
4. Compare pruned vs unpruned quality on joint validation set
5. Contrast with ReLU dead-capsule pruning baseline

## Lineage in the Arena

```
gpt (dense baseline)
  +-- relu_router (ReLU capsule MLP)
  |     +-- dead_capsule_pruning (Exp 9: 57% pruned, 0% loss)
  |     +-- pruning_controls (Exp 10: 54% single-domain death)
  |
  +-- silu_capsule (SiLU capsule MLP)
        +-- silu_pruning (this experiment)
```

## Key References

- Exp 9 (dead_capsule_pruning): ReLU pruning baseline, 57% dead, 0% quality loss
- Exp 10 (pruning_controls): 54% dead in single-domain ReLU, 87% training-induced
- Exp 5 (macro_match): 0% dead capsules in SiLU-based macro model
- ReDo (Klein et al., 2024): Activation-based dead neuron profiling
- Gurbuzbalaban et al. (2024): Non-monotonic death trajectories, cosine decay revival
- Qwen3.5 architecture (miniqwen.py): SiLU-gated MLP as the production standard

## Empirical Results

### Activation Distribution Comparison

| Property | ReLU (single-domain) | SiLU (single-domain) |
|----------|---------------------|---------------------|
| Exact zeros (mu=0) | 17.6% | 0.0% |
| Below tau=0.01 | 17.6% | 0.0% |
| Below tau=0.10 | 17.6% | 32.0% |
| Distribution shape | Bimodal (dead/alive) | Unimodal (all alive) |
| Min mean_abs | 0.000 | 0.046 |
| Median mean_abs | 0.04-0.50 | 0.083-0.128 |

### Pruning Threshold Sweep (3-seed mean)

| Method | Threshold | % Pruned | Quality Delta | Kill Gate |
|--------|-----------|----------|--------------|-----------|
| ReLU tau=0 | 0.000 | 17.6% | -0.00% | N/A |
| SiLU mean_abs | 0.001 | 0.0% | +0.00% | PASS |
| SiLU mean_abs | 0.005 | 0.0% | +0.00% | PASS |
| SiLU mean_abs | 0.010 | 0.0% | +0.00% | PASS |
| SiLU mean_abs | 0.050 | 0.1% | -0.01% | PASS |
| SiLU mean_abs | 0.100 | 32.0% | +1.01% | PASS |
| SiLU max_abs | 0.010 | 0.0% | +0.00% | PASS |
| SiLU max_abs | 0.100 | 0.0% | +0.00% | PASS |

### Kill Gate Analysis

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Quality degradation at tau=0.1 | +1.01% | >5% | **PASS** |
| Quality degradation at any tau | max +1.01% | >5% | **PASS** |

**VERDICT: PASS** -- No threshold degrades quality beyond 5%.

However, the practical interpretation is nuanced:

### The Threshold Gap Problem

The SiLU activation distribution has a **floor at ~0.05-0.09 mean_abs**.
This creates a gap:

```
tau <= 0.05:  0% pruned (safe but useless)
tau = 0.10:   32% pruned at +1.01% degradation (aggressive, lossy)
```

There is no "sweet spot" analogous to ReLU's tau=0 (57% pruned, 0% loss).
The transition from "prune nothing" to "prune something" is abrupt and
lossy, not gradual.

### Per-Seed Variation at tau=0.1

| Seed | % Pruned | Quality Delta |
|------|----------|--------------|
| 42 | 85.7% | +2.97% |
| 123 | 3.7% | +0.18% |
| 7 | 6.6% | -0.12% |
| Mean | 32.0% | +1.01% |

The high variance (3.7-85.7% pruned) reveals that seed 42 happened to
produce a model with many low-magnitude capsules (floor near 0.07) while
seeds 123 and 7 had higher floors (0.09-0.10). This seed sensitivity
makes SiLU pruning unreliable as a compression strategy.

### Baseline Quality Comparison

| Model | Val Loss | Std |
|-------|----------|-----|
| ReLU unpruned | 0.5562 | 0.0050 |
| SiLU unpruned | 0.5579 | 0.0133 |
| ReLU pruned (tau=0) | 0.5562 | 0.0050 |
| SiLU pruned (tau=0.1) | 0.5634 | varies |

SiLU and ReLU achieve comparable quality when unpruned. The difference
is entirely in pruning efficiency: ReLU allows free compression, SiLU
does not.

## The Core Finding

**SiLU's smooth activation prevents the formation of "dead" neurons at
any reasonable threshold.** The minimum mean absolute activation across
all capsules, layers, and seeds is ~0.046 -- nearly 5x above the strictest
safe threshold (tau=0.01). This is not a micro-scale artifact; it is an
inherent property of SiLU's gradient landscape:

1. **SiLU always provides gradient**: For any z != 0, SiLU'(z) != 0.
   This means gradient-based optimization never "kills" a neuron --
   every capsule remains sensitive to its inputs throughout training.

2. **ReLU kills neurons permanently**: For z < 0, ReLU'(z) = 0. Once
   a ReLU neuron's pre-activation becomes consistently negative, it
   receives no gradient and stays dead. This is the "dying ReLU" problem
   -- which is a bug for training but a feature for pruning.

3. **The floor effect**: SiLU's smooth gradient forces all capsules to
   maintain non-negligible activation magnitudes. The minimum mu_i is
   bounded below by ~0.2 * sigma (see MATH.md Section 4.3), where sigma
   is the standard deviation of pre-activations. At d=64, this produces
   a floor of ~0.05-0.10.

## Implications for Macro Scale

### What This Means for exp5_macro_match

The macro experiment found 0% dead capsules at d=896 with SiLU. This
micro experiment confirms: **SiLU pruning via magnitude thresholds does
not provide meaningful free compression.** The 57% dead-capsule pruning
from ReLU experiments fundamentally does not transfer to SiLU.

### Alternative Compression Paths for SiLU Models

Since magnitude-threshold pruning on SiLU is ineffective, other approaches
should be explored:

1. **Structured pruning by importance score**: Instead of activation
   magnitude, use gradient-based importance (Fisher information,
   Taylor expansion of loss change). This measures contribution to
   the loss, not just activation magnitude.

2. **SwiGLU-aware pruning**: The Qwen/Llama MLP uses SwiGLU:
   `out = fc3(silu(fc1(x)) * fc2(x))`. The gate `fc2(x)` can produce
   near-zero outputs even when `silu(fc1(x))` is nonzero. Pruning on
   the GATED output (silu(fc1) * fc2) may find more candidates.

3. **Low-rank factorization**: Instead of pruning individual capsules,
   decompose the (P, d) matrix into a lower-rank approximation using
   SVD. The effective rank may be much lower than P.

4. **Training with ReLU, deploying with SiLU**: If pruning is needed,
   train capsules with ReLU (getting free pruning), then distill into
   SiLU capsules for the final model. The pruning happens at the ReLU
   stage.

5. **Explicit sparsity training**: Add L0 or hard concrete regularization
   during SiLU training to force some capsules to become negligible.
   This trades training cost for post-hoc compression.

## Micro-Scale Limitations

1. **Small d=64 may not represent d=896+**: At larger dimensions, the
   activation distribution could be wider or narrower. The floor effect
   should persist (it's a property of SiLU, not scale) but the magnitude
   could shift.

2. **Single-domain training only**: The experiment does not test composed
   (multi-domain) models. Composition could create cross-domain capsules
   with lower activations, though the SiLU floor would still prevent
   exact-zero pruning.

3. **Short training (300 steps)**: Longer training might push some
   capsules closer to zero if they become redundant. However, SiLU's
   gradient signal would resist this -- there is always a force pushing
   capsules away from zero.

4. **Character-level names dataset**: Real code/text data at macro scale
   has different distribution properties that could affect activation
   magnitudes.

## What Would Kill This

The hypothesis PASSES the kill criterion (no >5% degradation), but the
practical finding is negative: **SiLU pruning provides no useful
compression at safe thresholds.**

What would change this conclusion:
- If macro-scale SiLU models (d=896+) show a wider activation distribution
  with some capsules below tau=0.01 (not predicted by the floor analysis)
- If SwiGLU-gated outputs (as opposed to raw SiLU outputs) show more
  sparsity (plausible -- the gate can push outputs toward zero)
- If explicit sparsity training produces capsules with mu_i ~ 0 despite
  SiLU activation (requires auxiliary loss engineering)

What would strengthen this conclusion:
- If macro-scale profiling confirms the same floor effect (mu_min >> 0.01)
- If SwiGLU-gated outputs also show no prunable capsules
- If the floor effect is demonstrated across multiple architectures
