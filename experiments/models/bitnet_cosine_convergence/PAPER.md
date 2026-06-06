# Cosine Convergence Trajectory on BitNet-2B: Research Digest

## Hypothesis

LoRA adapter pairwise cosine similarity on BitNet-2B-4T remains below 0.05 at full training convergence (2000 steps), disproving the claim that the low |cos|=0.001 measured at 400 steps was an under-training artifact.

## What This Experiment Is

This experiment trains 5 domain-specialized LoRA adapters (medical, code, math, legal, creative) on the ternary BitNet-2B-4T base model for 2000 steps each -- 5x longer than the original 400-step measurement. Every 100 steps, all 10 pairwise cosine similarities are computed, producing a 20-point trajectory of orthogonality over training. Additionally, composition PPL is measured at 6 checkpoints to track whether longer training degrades composability.

The motivation is a critical gap: macro-scale Qwen2.5 FP16 converged adapters showed cos=0.142 (142x higher than our 400-step BitNet measurement). If the same inflation occurred on BitNet with more training, the structural orthogonality advantage of ternary bases would collapse.

## Key References

- LoRA (Hu et al., 2021): Low-rank adaptation framework
- "LoRA vs Full Fine-tuning" (2410.21228): Tracks intruder dimensions emerging during LoRA training -- adapter subspace changes as training progresses
- "Subspace Geometry Governs Catastrophic Forgetting" (2603.02224): Principal angles between task gradient subspaces predict interference
- "OPLoRA" (2510.13003): Orthogonal projection prevents catastrophic forgetting in sequential LoRA training
- "Pause Recycling LoRAs" (2506.13479): Orthogonality alone is insufficient for semantic composability
- Prior SOLE findings: |cos|=0.001 at 400 steps (bitnet_2b_real_composition), cos=0.142 at macro Qwen FP16

## Empirical Results

### Cosine Trajectory: Plateau, Not Explosion

| Step | Mean |cos| | Max |cos| | Ratio to Step 100 |
|------|-----------|-----------|-------------------|
| 100  | 0.000942  | 0.003504  | 1.00x             |
| 200  | 0.000877  | 0.003345  | 0.93x             |
| 400  | 0.000875  | 0.003047  | 0.93x             |
| 800  | 0.000927  | 0.002839  | 0.98x             |
| 1200 | 0.001002  | 0.002749  | 1.06x             |
| 1600 | 0.001112  | 0.002824  | 1.18x             |
| 1800 | 0.001254  | 0.002763  | 1.33x             |
| 2000 | 0.001249  | 0.002581  | 1.33x             |

**Key finding**: Over 2000 steps (5x the original measurement), mean |cos| increased from 0.000877 (step 200) to 0.001249 (step 2000) -- a 1.42x increase that PLATEAUS at ~0.00125 by step 1800. This is 40x below the 0.05 kill threshold and 114x below the 0.142 observed on FP16 Qwen adapters.

The trajectory shows:
- **Steps 100-800**: Flat phase (mean |cos| oscillates 0.0009-0.0009)
- **Steps 800-1800**: Gentle rise phase (0.0009 -> 0.00125, +39%)
- **Steps 1800-2000**: Plateau (0.00125 -> 0.00125, <1% change)

The max |cos| (math-legal pair) actually DECREASES from 0.003504 to 0.002581 over training, even as the mean increases slightly. This suggests the slight mean increase comes from previously near-zero pairs developing small but stable overlap.

### Composition PPL: Stable Across Training

| Step | Avg Composed PPL | Ratio vs Base |
|------|-----------------|---------------|
| 200  | 10.05           | 0.871         |
| 400  | 9.96            | 0.862         |
| 800  | 9.89            | 0.856         |
| 1200 | 9.89            | 0.856         |
| 1600 | 9.88            | 0.855         |
| 2000 | 9.91            | 0.858         |

Composition PPL is virtually flat from step 400 onward. The composed model consistently beats the base model (ratio < 1.0) across all checkpoints. There is NO degradation of composability even as training continues 5x past the original stopping point.

### Convergence Status

| Domain   | Converged? | Plateau Step | Final Loss |
|----------|-----------|--------------|------------|
| Medical  | No*       | --           | 0.2715     |
| Code     | Yes       | ~1800        | 0.3008     |
| Math     | Yes       | ~1800        | 1.2266     |
| Legal    | Yes       | ~1800        | 2.5781     |
| Creative | Yes       | ~1800        | 1.5000     |

*Medical continues improving but at diminishing rate. Loss dropped from 3.23 to 0.27 (91.6% reduction), suggesting near-convergence.

4/5 domains converged by step 1800. Medical is still improving but the cosine trajectory has already plateaued, meaning even if medical trains further, it won't drive |cos| above the threshold.

### Kill Criteria Assessment

| Criterion | Threshold | Measured | Margin | Result |
|-----------|-----------|----------|--------|--------|
| K1: Mean |cos| at convergence | < 0.05 | 0.00125 | 40x margin | **PASS** |
| K2: Monotonic increase fraction | > 0.80 | 0.579 | -- | **PASS** |
| K2: Second-half CV (no plateau) | > 0.30 | 0.093 | -- | **PASS** |

**Verdict: SUPPORTED**

### Per-Pair Analysis

The highest-cosine pair throughout training is consistently math-legal (0.0025-0.0035). The lowest is code-legal (0.00001-0.0007). This pair structure is stable -- the relative ordering of which domains are most/least similar does not change during training, only the magnitudes shift slightly.

## Why BitNet Stays Orthogonal But FP16 Does Not

The 114x gap between BitNet |cos|=0.00125 and Qwen FP16 |cos|=0.142 demands explanation. We hypothesize three mechanisms:

1. **Discrete routing constraint**: Ternary base weights {-1, 0, 1} create a piecewise-constant gradient landscape. The shared component of the gradient (from base weight directions) is weaker because {-1, 0, 1} provides less directional information than learned FP16 weights.

2. **Sparsity**: ~33% of BitNet weights are exactly zero. These zero entries create "dead channels" that reduce the effective dimensionality of adapter overlap. For the same $r$ and $d$, ternary adapters share fewer active dimensions.

3. **Magnitude normalization**: FP16 weights have varying magnitudes across layers. Adapters trained on FP16 bases develop magnitude-correlated structure (larger updates where base weights are larger). Ternary weights have no magnitude variation, eliminating this correlation channel.

## Limitations

1. **Single seed**: Justified by multiseed CV=0.5% from bitnet_multiseed_validation, but a single-seed trajectory is noisier than the mean.

2. **seq_len=128**: Shorter than production (2048+). Longer sequences might change convergence dynamics, though unlikely to affect orthogonality since cosine is computed over all LoRA parameters, not per-token.

3. **FP16 LoRA on ternary base**: We tested standard FP16 LoRA adapters, not ternary QAT+STE adapters. Ternary adapters might show even lower cosine (prior finding: ternary |cos| 19.3% lower than FP16).

4. **5 domains only**: The trajectory might differ with 50+ adapters, though N=25 scaling showed |cos| actually DECREASES with more adapters (dilution effect).

5. **No FP16 control**: We did not run the same experiment on an FP16 base for direct comparison. The 0.142 comparison comes from a different model (Qwen2.5-7B), different domains, and different training setup.

6. **PPL-only evaluation**: Task-based evaluation was killed at 2B scale (NTP does not produce task-capable adapters). Orthogonality is a necessary but not sufficient condition for functional composability.

## What Would Kill This

**At micro scale**: If ternary QAT+STE adapters show significantly higher cosine trajectories than FP16 LoRA (suggesting the ternary adapter quantization, not the ternary base, drives orthogonality).

**At macro scale**: If BitNet-2B adapters trained on 10,000+ steps with full-length sequences (seq_len=2048) show cosine inflation above 0.05. This would suggest our convergence at 2000 steps was premature.

**Conceptually**: If composition quality degrades despite low cosine (as "Pause Recycling LoRAs" warns), then the entire cosine-based orthogonality framework is insufficient regardless of the trajectory.

## Summary

The cosine convergence trajectory experiment decisively answers the P0 adversarial question: **low cosine on BitNet-2B is NOT an under-training artifact**. Over 2000 steps with 4/5 domains converging, mean |cos| plateaus at 0.00125 -- 40x below the kill threshold and 114x below FP16 Qwen adapters. Composition PPL remains stable throughout. The structural orthogonality claim for ternary bases is solid at this scale.

**Runtime**: 72.4 minutes, $0 (Apple Silicon MLX).
