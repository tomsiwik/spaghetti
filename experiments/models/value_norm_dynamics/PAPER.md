# Value Norm Dynamics Under L2-Normalized QK Composition: Research Digest

## Hypothesis

Value norms remain bounded (growth <10x) during composition training with
L2-normalized QK products, and value norm growth does not correlate (r <0.5)
with composition quality degradation.

## Verdict: PASS (both kill criteria)

Kill criterion 1 (value norms grow >10x): **PASS**. The worst growth ratio
across 7 seeds is 1.09x. The mean growth ratio is 1.03x. Value norms are
essentially unchanged during composition.

Kill criterion 2 (growth-quality correlation >0.5): **PASS**. Pearson
correlations are r=0.275 (max growth vs gap) and r=0.323 (mean growth vs
gap). Neither exceeds the 0.5 threshold.

The state boundedness argument from the L2 QK normalization experiment is
empirically confirmed: the missing assumption (that value norms are
well-behaved) holds in practice. Value norms grow at most 1.09x during
the entire composition pipeline.

## What This Model Is

L2NormValueTrackingGPT extends the L2-normalized hybrid capsule MoE with
value norm instrumentation. The model is functionally identical to
L2NormHybridCapsuleMoEGPT -- the only addition is the ability to record
per-layer, per-head value vector norms during forward passes.

The experiment instruments the full composition pipeline:
- After pretraining (baseline norms)
- After domain fine-tuning
- During router calibration (every 10 steps)
- After calibration (final norms)

This answers the adversarial review concern that L2 QK normalization
bounds QK products but says nothing about value norms, which could
still grow and break state boundedness.

## Lineage in the Arena

```
gpt (dense baseline)
 |-- capsule_moe (routed capsule groups)
      |-- hybrid_capsule_moe (3:1 linear:full)
           |-- l2_norm_hybrid_capsule_moe (+ L2 QK norm)
                |-- value_norm_tracking_moe (+ norm instrumentation) <-- THIS
```

## Key References

- GatedDeltaNet (Yang et al., 2024): L2-normalized QK products, the
  mechanism that motivates the boundedness question.
- L2 norm composition stability experiment (this project): Proved 0/25
  catastrophic failures with L2 QK norm, but left the value norm
  assumption unverified.
- Adversarial review of exp_l2_norm_composition_stability: Identified
  value norm growth as a gap in the state boundedness argument.

## Protocol

Identical to the L2 norm composition experiment protocol, with added
value norm instrumentation:

1. Pretrain shared base on all data (300 steps)
2. Fine-tune capsule groups per domain (300 steps, attention frozen)
3. Compose by concatenating domain groups, double top-k
4. Calibrate router on mixed data (100 steps) -- **track value norms**
5. Evaluate on per-domain val sets

Value norms recorded at 5 phases: post-pretrain (baseline), post-finetune-A,
post-finetune-B, pre-calibration, and every 10 steps during calibration.

7 seeds (0-6).

Growth ratio = max(value_norm_during_composition) / baseline_value_norm.

## Empirical Results

### Kill Criterion Results

| Criterion | Metric | Value | Threshold | Verdict |
|-----------|--------|-------|-----------|---------|
| K1: Growth | Worst max growth ratio | 1.09x | 10x | PASS |
| K1: Growth | Mean max growth ratio | 1.03x | 10x | PASS |
| K2: Correlation | r(max_growth, gap) | 0.275 | 0.5 | PASS |
| K2: Correlation | r(mean_growth, gap) | 0.323 | 0.5 | PASS |

### Composition Quality (7 seeds)

| Metric | Value |
|--------|-------|
| Gap mean | -0.11% |
| Gap median | -0.47% |
| Gap std | 1.06% |
| Gap range | [-1.27%, +1.95%] |

### Value Norm Growth Ratios (7 seeds)

| Metric | Max Growth | Mean Growth |
|--------|-----------|-------------|
| Mean | 1.03x | 1.03x |
| Median | 1.01x | 1.03x |
| Worst | 1.09x | 1.04x |
| Range | [1.00x, 1.09x] | [1.01x, 1.04x] |

### Per-Layer Analysis (mean across seeds)

| Layer | Baseline Norm | Post-Calib Norm | Ratio |
|-------|--------------|-----------------|-------|
| Layer 0 (linear) | 7.25 | 7.25 | 1.00x |
| Layer 1 (linear) | 4.75 | 4.76 | 1.00x |
| Layer 2 (linear) | 3.98 | 4.07 | 1.02x |

Value norms are remarkably stable across layers. Layer 0 shows no change
at all; Layer 2 (deepest linear layer, closest to the full-attention
layer 3) shows the most growth at a negligible 1.02x.

### Calibration Trajectory (mean across seeds)

| Step | Max Norm | Mean Norm |
|------|---------|-----------|
| 1 | 9.81 | 5.41 |
| 10 | 9.69 | 5.38 |
| 20 | 9.74 | 5.38 |
| 50 | 9.63 | 5.36 |
| 100 | 9.59 | 5.36 |

Value norms actually **decrease slightly** during calibration (9.81 to
9.59 max). The router optimization does not amplify value norms -- if
anything, it regularizes them by finding better routing assignments.

### Per-Seed Summary

| Seed | Gap | Max Growth | Mean Growth | Base Max | Comp Max |
|------|-----|-----------|-------------|----------|----------|
| 0 | -1.27% | 1.08x | 1.04x | 6.77 | 7.32 |
| 1 | -0.47% | 1.01x | 1.01x | 7.13 | 7.19 |
| 2 | +1.95% | 1.09x | 1.04x | 12.10 | 13.14 |
| 3 | -0.78% | 1.02x | 1.03x | 8.88 | 9.05 |
| 4 | -0.47% | 1.01x | 1.03x | 10.35 | 10.46 |
| 5 | -0.28% | 1.01x | 1.03x | 12.08 | 12.20 |
| 6 | +0.56% | 1.00x | 1.02x | 10.23 | 10.27 |

Seed 2 shows both the worst growth (1.09x) and the worst gap (+1.95%).
This drives the positive correlation (r=0.275) but it is well below the
0.5 threshold. The growth is still tiny (1.09x) and the gap is within
the normal distribution.

## Why Value Norms Stay Bounded

The theoretical explanation is straightforward: during router calibration,
only router weights are trainable. The value projection W_v is frozen.
Value norms are:

    ||v_t||_2 = ||W_v @ RMSNorm(x_t)||_2

With W_v fixed and RMSNorm normalizing input magnitude, the value norm
is determined by the direction of x_t (which capsule groups contribute
to the residual stream), not its magnitude. Router changes alter which
groups contribute, but since all groups were trained from the same base
and have similar output magnitudes, the directional change produces
minimal norm variation.

The RMSNorm is the key mechanism: it absorbs magnitude changes in the
residual stream, ensuring that no matter how the routing changes the
hidden states, the input to the value projection has approximately
constant norm (sqrt(d_h)).

## Key Findings

1. **Value norms grow at most 1.09x during composition (threshold: 10x).**
   The state boundedness argument from the L2 QK norm experiment is
   confirmed: both QK products (bounded by L2 norm) and value norms
   (bounded by frozen W_v + RMSNorm) remain well-behaved.

2. **Value norm growth does not predict composition quality.** The
   correlations (r=0.275, r=0.323) are positive but weak and well below
   the 0.5 threshold. Value norm variation explains <10% of composition
   gap variance.

3. **Value norms decrease slightly during calibration.** The mean max
   norm drops from 9.81 to 9.59 over 100 calibration steps. Router
   optimization finds better assignments that happen to produce
   slightly lower value norms.

4. **Per-layer norms are stable.** Layer 0 shows zero growth; deeper
   layers show at most 1.02x. The lack of depth amplification confirms
   that RMSNorm effectively isolates layers.

## Micro-Scale Limitations

1. **d_h=16 head dimension.** At macro scale (d_h=128 or 256), value
   norms may follow different dynamics. However, the argument (frozen
   W_v + RMSNorm = bounded norms) is scale-independent.

2. **T=32 sequence length.** State accumulation over T=4096 tokens could
   amplify even small value norm growth. At 1.09x growth, the state
   bound increases by 9% -- negligible even at long sequences.

3. **Only router calibration tested.** If full fine-tuning (including
   attention weights) were used during composition, W_v would change
   and value norms could grow significantly. This experiment only
   validates the frozen-attention composition protocol.

4. **7 seeds.** Sufficient for the clear pass (1.09x vs 10x threshold),
   but weak for correlation analysis (r=0.275 with 7 points has a wide
   95% CI). More seeds would tighten the correlation estimate.

5. **Character-level toy data.** Subword-tokenized real text may produce
   different value norm distributions. The theoretical argument
   transfers, but empirical confirmation at macro scale would be
   valuable.

## What Would Kill This

**At micro scale:**
- Finding that full attention fine-tuning (not just router calibration)
  causes value norm explosion during composition
- Finding a configuration where RMSNorm is insufficient (e.g., very deep
  networks where residual stream magnitude grows despite normalization)

**At macro scale:**
- Value norms growing >10x with longer sequences (T=4096+) due to
  autoregressive state accumulation over many more steps
- Value norms growing >10x when composing many experts (N=20+) where
  the diversity of capsule group outputs is much larger
- The delta rule mechanism (v_t - S^T k_t) causing value norms to
  interact with state norms in a feedback loop (the correction term
  S^T k_t depends on accumulated state, which depends on previous values)
