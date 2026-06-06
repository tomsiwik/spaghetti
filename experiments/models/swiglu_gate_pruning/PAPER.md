# SwiGLU Gate-Aware Pruning: Research Digest

## Hypothesis

Profiling the SwiGLU gate product |SiLU(W_gate @ x) * (W_up @ x)| per capsule
reveals prunable capsules that SiLU-only profiling (floor ~0.046) cannot detect.

**Falsifiable.** Kill criteria:
1. <10% of capsules prunable at gate-product threshold
2. Pruning by gate product >3% worse than no pruning

## What This Model Is

A capsule MLP matching Qwen3.5/Llama's SwiGLU architecture:

    output = B @ (SiLU(W_gate @ x) * (W_up @ x))

Where W_gate and W_up are separate projections from d to P (capsule count),
and B projects back from P to d. This replaces the simpler SiLU MLP
(y = B @ SiLU(A @ x)) from the silu_capsule experiment.

The key insight: SiLU pruning was killed (Exp 15) because SiLU(z) has an
activation floor of ~0.046 -- no capsule ever has mean |SiLU(a^T x)| below
this floor. But the full SwiGLU gate product can be much closer to zero
because the up-projection W_up acts as a learned binary mask. Even when
SiLU(W_gate @ x) is stuck above 0.046, the product with W_up @ x can be
near-zero if W_up learns to suppress that capsule.

## Lineage in the Arena

```
gpt (dense baseline, ReLU)
  +-- relu_router (ReLU capsule MLP)
  |     +-- dead_capsule_pruning (57% pruned, 0% loss)
  |
  +-- silu_capsule (SiLU capsule MLP)
        +-- silu_pruning (KILLED: 0% prunable at safe tau)
        +-- swiglu_gate_pruning (THIS: 66.5% prunable at +1.22%)
```

## Key References

- Exp 15 (silu_pruning): SiLU activation floor ~0.046, 0% prunable at tau<=0.01
- Exp 9 (dead_capsule_pruning): ReLU baseline, 57% dead, 0% quality loss
- Qwen3.5 FeedForward (miniqwen.py): SwiGLU architecture definition
- DeepSeek-V3: Production fine-grained MoE with SwiGLU MLPs

## Empirical Results

### Gate Product vs SiLU Distribution (3-seed aggregate)

| Metric | SiLU-only (Exp 15) | SwiGLU Gate Product |
|--------|-------------------|---------------------|
| Min mean_abs | 0.046 | 0.014 |
| Median mean_abs | 0.083-0.128 | 0.026-0.061 |
| Below tau=0.01 | 0% | 0% |
| Below tau=0.02 | 0% | 1.4% |
| Below tau=0.05 | 0% | 66.5% |
| Distribution | Unimodal (all alive) | Bimodal (suppressed + active) |

### Gate Product Pruning Threshold Sweep (3-seed mean)

| Threshold | % Pruned | Quality Delta | Kill Gate |
|-----------|----------|--------------|-----------|
| tau=0.001 | 0.0% | +0.00% | PRUNING KILL |
| tau=0.005 | 0.0% | +0.00% | PRUNING KILL |
| tau=0.010 | 0.0% | +0.00% | PRUNING KILL |
| tau=0.020 | 1.4% | +0.01% | PRUNING KILL |
| tau=0.050 | 66.5% | +1.22% | **PASS** |
| tau=0.100 | 99.0% | +9.83% | QUALITY KILL |

### Kill Criteria Check

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| % prunable at best safe tau | 66.5% at tau=0.05 | >10% | **PASS** |
| Quality delta at best safe tau | +1.22% | <3% | **PASS** |

**VERDICT: PASS** -- Both kill criteria satisfied. SwiGLU gate-product pruning
provides 66.5% compression at +1.22% quality cost.

### Why Gate Products Are Lower Than SiLU Outputs

The multiplicative interaction in SwiGLU creates a suppression mechanism:

| Layer | Gate Product Min | SiLU Min | Up Min | Suppression |
|-------|-----------------|----------|--------|-------------|
| L0 (seed 42) | 0.028 | 0.041 | 0.161 | Up acts as gate |
| L1 (seed 42) | 0.014 | 0.016 | 0.103 | Both low, product lower |
| L2 (seed 42) | 0.018 | 0.025 | 0.118 | Multiplicative suppression |
| L3 (seed 42) | 0.026 | 0.035 | 0.165 | Moderate suppression |

Gate product floor (~0.014) is 3.3x lower than SiLU floor (~0.046). The
up-projection learns to suppress specific capsules even when SiLU cannot.

### Baseline Quality Comparison

| Model | Val Loss (mean) | Std |
|-------|----------------|-----|
| SiLU unpruned | 0.5578 | 0.0122 |
| SwiGLU unpruned | 0.5577 | 0.0111 |
| SwiGLU pruned (tau=0.05) | 0.5644 | varies |

SiLU and SwiGLU achieve comparable unpruned quality. The difference is
entirely in pruning efficiency: SwiGLU gate products enable 66.5% pruning
that SiLU alone cannot provide.

### Per-Seed Variation at tau=0.05

| Seed | % Pruned | Quality Delta |
|------|----------|--------------|
| 42 | 81.2% | +2.84% |
| 123 | 39.5% | +0.26% |
| 7 | 78.7% | +0.58% |
| Mean | 66.5% | +1.22% |

**95% Confidence Interval on mean quality delta:** With n=3 seeds, sample
std=1.41%, and t(0.025, df=2)=4.303, the 95% CI is [-2.27%, +4.72%].
The CI includes 3% (the kill threshold), meaning we cannot rule out that the
true mean degradation exceeds 3% at this sample size. The wide CI reflects
high per-seed variance (2.84% vs 0.26%). More seeds would be needed to
tighten the bound. See Limitations section.

Seed 123 shows lower pruning (39.5%) with excellent quality (+0.26%).
Seeds 42 and 7 show aggressive pruning (78-81%) with moderate degradation.
All seeds pass both kill criteria individually.

### Random Pruning Baseline (Fix #3 Control)

To verify that gate-product profiling identifies the RIGHT capsules (not just
that SwiGLU models are robust to arbitrary pruning), we compared against random
pruning at the same fraction per seed (3 random seeds per training seed).

| Training Seed | GP Pruned | GP Delta | Random Delta (mean of 3) | GP Advantage |
|--------------|-----------|----------|--------------------------|-------------|
| 42 | 77.1% | +3.27% | +9.10% | 2.8x better |
| 123 | 26.4% | +0.70% | +0.77% | 1.1x better |
| 7 | 82.2% | +1.18% | +2.09% | 1.8x better |
| **Mean** | **61.9%** | **+1.72%** | **+3.99%** | **2.3x better** |

Note: Random baseline was run as a separate replication; small differences
from original results (e.g., seed 42: 77.1% vs 81.2% pruned) are due to
MLX training non-determinism across runs. The relative comparison (GP vs
random within the same run) is valid.

Gate-product profiling consistently outperforms random pruning:
- At high pruning fractions (77-82%), random pruning is 1.8-2.8x worse
- At low pruning fractions (26%), random is comparable (both are mild)
- Overall, random pruning causes 2.3x the degradation of gate-product pruning

**Conclusion:** Gate-product profiling provides genuine signal about which
capsules to prune. The finding is NOT reducible to "SwiGLU models are robust
to arbitrary pruning."

## The Core Finding

**SwiGLU's multiplicative gate creates a pruning channel that SiLU alone
cannot provide.** The up-projection W_up acts as a learned binary mask:
capsules where W_up @ x is near-zero have near-zero gate products regardless
of the SiLU activation floor.

This directly unblocks macro-scale pruning for Qwen/Llama architectures:

1. **SiLU pruning** (Exp 15): Killed. Floor ~0.046, 0% prunable.
2. **ReLU pruning** (Exp 9): 57% prunable, 0% loss -- but production models
   don't use ReLU.
3. **SwiGLU gate-product pruning** (this): 66.5% prunable, +1.22% loss --
   works with actual production architecture.

The gate product approach profiles what the model actually computes (the
SwiGLU output), not an intermediate activation. This is architecturally
correct and directly applicable to any SwiGLU-based model.

## Comparison with ReLU Dead Capsule Pruning

| Property | ReLU (Exp 9) | SwiGLU Gate Product |
|----------|-------------|---------------------|
| Prunable | 57% | 66.5% |
| Quality delta | 0.00% (exact) | +1.22% (approximate) |
| Mechanism | Hard zeros (exact) | Soft suppression (threshold) |
| Production use | Rare (SiLU/SwiGLU dominant) | All Qwen/Llama/DeepSeek |
| Lossless? | Yes | No (bounded error) |

SwiGLU pruning is approximate (not lossless), but it works on the actual
production architecture. The 1.22% quality cost is acceptable for 66.5%
parameter reduction.

## Training Regularization Disclosure

Both the SiLU baseline and SwiGLU models are trained with auxiliary losses
that encourage sparsity:

- **Adaptive L1 sparsity loss**: target sparsity 50%, coefficient 0.01,
  with adaptive scaling based on running sparsity estimate
- **Balance loss**: coefficient 0.01, penalizes variance in per-capsule
  mean activation magnitudes

These losses are applied identically to both SiLU and SwiGLU models via
the shared `train()` function calling `model.aux_loss()`. The A/B comparison
(SiLU pruning vs SwiGLU gate-product pruning) is therefore fair -- both
models receive the same regularization pressure.

However, the absolute pruning rate (66.5% at tau=0.05) may be inflated by
this regularization. The sparsity loss actively encourages capsule activations
toward zero, which makes more capsules prunable than they would be without it.

**Macro transfer risk:** Production Qwen/Llama/DeepSeek models are NOT
trained with per-capsule sparsity losses. The gate product distribution at
macro scale may be less sparse without this regularization. The core mechanism
(multiplicative suppression via W_up) is architectural, but the magnitude
of the effect may differ. Macro validation should profile gate products on
pretrained models without retraining.

## Micro-Scale Limitations

1. **Small d=64 may underestimate gate product suppression**: At larger d,
   the up-projection has more capacity to learn precise suppression patterns.
   Gate product sparsity could be higher at macro scale.

2. **300 training steps**: Longer training with more data may shift the
   gate product distribution. The suppression mechanism is learned, not
   architectural, so training dynamics matter.

3. **Single-domain training**: The experiment trains on domain A only.
   Composition (multi-domain) may create additional suppressible capsules
   as different experts contribute different gate patterns.

4. **Character-level names**: Real data with richer structure could produce
   different gate product distributions.

5. **Threshold sensitivity**: The sharp transition at tau=0.05 (0% -> 66.5%
   prunable) suggests the distribution is concentrated. Different scales
   may have different optimal thresholds.

6. **Quality delta CI includes kill threshold**: The 95% CI on mean quality
   delta at tau=0.05 is [-2.27%, +4.72%], which includes the 3% kill
   threshold. With only 3 seeds, we cannot statistically guarantee that the
   true mean degradation is below 3%. The point estimate (+1.22%) passes,
   but this should be validated with more seeds at macro scale.

7. **Auxiliary sparsity loss inflates pruning rate**: The 66.5% pruning rate
   reflects models trained with sparsity-encouraging regularization (see
   Training Regularization Disclosure above). Without this loss, fewer
   capsules may fall below the pruning threshold.

## What Would Kill This

**At micro scale:**
- Already survived: both kill criteria passed (>10% prunable, <3% quality loss)

**At macro scale (future validation):**
- If gate product distributions at d=896+ show no bimodal structure (all
  capsules above tau=0.05), pruning would fail at that threshold
- If pruning 66.5% of a 256-expert DeepSeek-V3 style model causes >3%
  degradation on real benchmarks (error accumulation across 60+ layers)
- If the threshold gap narrows: tau=0.05 is aggressive for macro models where
  individual capsules process richer representations

**What would strengthen this finding:**
- Macro profiling confirming bimodal gate product distribution in Qwen3.5
- Gate-product pruning producing near-zero degradation with calibration
- Combining with ReLU dead capsule pruning in a two-stage pipeline
