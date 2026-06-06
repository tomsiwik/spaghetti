# Meta-Scaffold: MAML-Optimized Scaffold for Adapter Composition

## Hypothesis

A scaffold optimized via MAML (Model-Agnostic Meta-Learning) for adapter
composition quality will outperform a GaLore-grown scaffold, producing lower
composition interference and better individual adapter quality.

## What This Experiment Is

We test whether bilevel optimization (FOMAML) can tune a pretrained scaffold's
weights specifically for adapter composability. The inner loop trains fresh LoRA
adapters per domain; the outer loop updates the scaffold to minimize a meta-loss
combining average domain loss and composition penalty (pairwise cosine similarity).

This is the first experiment to directly optimize a base model for LoRA composition.
No prior published work exists on meta-learning scaffolds for multi-adapter composition.

## Key References

- **MAML** (Finn et al., 2017, ICML): Model-agnostic meta-learning via bilevel optimization.
- **Meta-LoRA** (arXiv 2510.11598): Two-stage MAML for LoRA task adaptation.
- **"Meta-Learning the Difference"** (TACL, doi:10.1162/tacl_a_00517): Bilevel optimization
  for base weights with low-rank task reparameterization.
- **GaLore Scaffold** (this project, exp_bitnet_galore_scaffold): Baseline scaffold
  grown via GaLore. PPL ratio 1.918x, composition ratio 1.045x.

## Design

- **Architecture**: TinyGPT, d=256, 6 layers, 4 heads, ~6.4M params
- **Data**: 5 domains (python, math, medical, legal, creative), character-level
- **Baseline**: Standard pretrained scaffold (2000 steps, Adam, full-rank)
- **Meta-scaffold**: Same pretrained init + 100 FOMAML meta-steps
  - Inner loop: 50 steps per domain, 3 domains sampled per meta-step
  - Meta-loss: avg(domain_val_loss) + 0.5 * pairwise_adapter_cosine
  - Outer loop: Adam, lr=1e-4, updates scaffold weights
- **Evaluation**: Ternary quantize both scaffolds, train 5 full adapters (400 steps),
  measure composition ratio and cosine similarity

## Empirical Results

### Meta-Training Dynamics

| Metric | Value |
|--------|-------|
| Meta-loss (first 10 avg) | 2.868 |
| Meta-loss (last 10 avg) | 2.835 |
| Reduction | 1.2% (below 5% convergence threshold) |
| Meta-training time | 176.7s |
| Total experiment time | ~5 min |

The outer loop did NOT converge. Meta-loss oscillated around 2.83 +/- 0.10 without
a clear downward trend across 100 meta-steps.

### Scaffold Quality Degradation

FOMAML outer loop updates destroyed scaffold quality:

| Metric | Standard | Meta | Ratio |
|--------|----------|------|-------|
| Base PPL (pre-quant, python) | 11.0 | 59.1 | 5.4x worse |
| Base PPL (pre-quant, mean) | 15.9 | 107.4 | 6.8x worse |
| Ternary PPL (mean) | 17.3 | 206.0 | 11.9x worse |

The FOMAML updates pushed weights into distributions that are catastrophically
degraded by ternary quantization.

### Adapter Quality (Surprising Resilience)

Despite the ruined scaffold, adapters trained on meta-scaffold achieve reasonable quality:

| Domain | Std Adapter PPL | Meta Adapter PPL | Ratio |
|--------|----------------|------------------|-------|
| python | 15.29 | 15.19 | 0.99 |
| math | 12.72 | 12.93 | 1.02 |
| medical | 14.06 | 14.71 | 1.05 |
| legal | 15.23 | 16.12 | 1.06 |
| creative | 10.67 | 11.18 | 1.05 |
| **Mean** | **13.59** | **14.03** | **1.03** |

Adapters only 3% worse despite scaffold being 12x worse. This confirms the
"adapter resilience" finding: LoRA can partially compensate for base quality loss.

### Composition Quality

| Metric | Standard | Meta | GaLore Baseline |
|--------|----------|------|-----------------|
| Composition ratio | 1.309 | 1.172 | 1.155 |
| Mean |cos| | 0.0022 | 0.0034 | 0.0027 |

Meta-scaffold shows LOWER composition ratio (better) than standard scaffold,
but does NOT beat the GaLore baseline (1.172 > 1.155).

### Kill Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1: Meta beats GaLore comp ratio | meta < 1.155 | 1.172 | **FAIL** |
| K1: Meta beats standard comp ratio | meta < standard | 1.172 < 1.309 | PASS |
| K2: MAML convergence | >5% reduction | 1.2% | **FAIL** |

**Overall: KILLED**

## Analysis: Why FOMAML Failed

### 1. Scaffold Destruction
The outer loop meta-gradient does not include any term preserving base language
modeling quality. It only optimizes for adapter endpoint loss + composition penalty.
Result: scaffold weights drift far from the pretrained distribution.

### 2. FOMAML Approximation Too Noisy
With K=50 inner steps, the first-order MAML approximation drops significant
second-order information. The true meta-gradient is:

```
dL/dW = partial L / partial W + (partial L / partial theta) * (d theta^K / dW)
```

FOMAML only captures the first term. The second term (how scaffold changes affect
the inner loop trajectory) is critical but dropped.

### 3. Ternary Quantization Incompatibility
FOMAML updates are continuous (Adam). The resulting weights have distributions
that are 12x worse after ternary quantization than standard pretrained weights.
A ternary-aware meta-training loop (STE in outer loop) might help but adds
significant complexity.

### 4. Composition Penalty Too Weak
Pairwise adapter cosine is already very low (~0.002) due to random LoRA initialization
at micro scale. The composition penalty term provides almost no gradient signal.

## Interesting Finding: Adapter Resilience

The most notable finding is that adapters trained on a MUCH WORSE scaffold (12x
higher base PPL) are only 3% worse in individual quality. This suggests:

1. LoRA adapters can compensate for significant scaffold degradation
2. The scaffold's absolute quality matters less than its "trainability" for adapters
3. Composition quality may depend more on adapter training dynamics than scaffold quality

This partially validates the base-free direction: even a degraded scaffold can
support useful adapter training, though composition quality does not improve.

## Limitations

1. Single seed (no multi-seed validation)
2. FOMAML only (full MAML or implicit differentiation might work better)
3. No STE in outer loop (continuous updates quantized post-hoc)
4. Small scale (d=256, character-level)
5. Meta-loss does not include a scaffold preservation term

## What Would Kill This (At Any Scale)

Already killed. Additional evidence that would further confirm:
- Full second-order MAML also fails to converge (would confirm FOMAML is not the issue)
- STE-aware outer loop also degrades scaffold (would confirm the approach is fundamentally flawed)
- Scaffold preservation term prevents meaningful composition improvement (would confirm
  the optimization objective is contradictory)

## What We Learned

1. **FOMAML is insufficient for scaffold optimization**: The first-order approximation
   with K=50 inner steps does not provide useful outer gradients.
2. **Unconstrained meta-updates destroy scaffolds**: Without explicit scaffold quality
   preservation, outer loop gradients wreck the pretrained weight distribution.
3. **Ternary quantization amplifies meta-learning damage**: Weights moved by FOMAML
   quantize 12x worse than standard pretrained weights.
4. **Adapter resilience is surprisingly strong**: Even on a degraded scaffold, adapters
   achieve near-baseline quality, suggesting the base-free direction may work with
   better scaffold construction methods.
5. **The GaLore scaffold baseline is hard to beat**: GaLore's gradient low-rank projection
   naturally produces composition-friendly weight distributions. Explicit meta-optimization
   does not improve on this.
