# Attention Self-Repair After Expert Removal: Research Digest

## Hypothesis

Frozen multi-head self-attention provides self-repair after LoRA expert
removal, reducing output deviation by >30% compared to MLP-only
architectures at the same depth and dimension.

## What This Experiment Tests

Three parent experiments (residual_layernorm_error_dynamics,
attention_layer_removal_safety, correlated_layer_errors) established that
pre-RMSNorm residual architectures are safe for expert removal but tested
only MLP-only models. Attention was flagged as an untested mechanism.

McGill et al. (2024) documented "self-repair" in transformers: when
attention heads are ablated, remaining heads compensate. This experiment
tests whether a similar effect occurs for LoRA expert removal in SOLE's
frozen-attention architecture.

We build two models sharing identical MLP weights:
1. **MLP-only**: Pre-RMSNorm residual MLP blocks (parent baseline)
2. **Transformer**: Pre-RMSNorm residual attention + MLP blocks

Both receive the same LoRA expert perturbations on MLP weights. We remove
one expert and compare output deviation between the two architectures.

## Key References

- McGill et al. (2024): "Self-repair" in transformers -- ablated heads are
  compensated by remaining heads. However, this requires trained redundancy.
- Parent: residual_layernorm_error_dynamics (amp_ratio=0.022 for Pre-RMSNorm)
- Parent: attention_layer_removal_safety (GS+naive hybrid strategy)
- Parent: correlated_layer_errors (correlation reduces amplification)

## Empirical Results

### K1: Self-Repair Exists (>30% lower deviation)

| Depth (L) | MLP-only Dev% | Transformer Dev% | Repair Ratio | K1 |
|-----------|---------------|-------------------|-------------|-----|
| 1         | 1.020         | 1.009             | 1.6%        | FAIL |
| 2         | 0.673         | 0.651             | 4.5%        | FAIL |
| 4         | 0.772         | 0.768             | 0.2%        | FAIL |
| 8         | 0.756         | 0.769             | -1.8%       | FAIL |
| 12        | 0.575         | 0.543             | 4.5%        | FAIL |
| 16        | 0.539         | 0.514             | 3.6%        | FAIL |

**Overall mean repair ratio: 2.1% (threshold: >30%). K1 FAIL.**

The effect is within noise. At L=8, the transformer actually shows
*higher* deviation than MLP-only (repair ratio -1.8%).

### K2: Self-Repair Increases with Depth

Layer-by-layer analysis at L=16 (3 seeds):

| Layer | MLP Dev% | TF Dev% | Repair |
|-------|----------|---------|--------|
| 0     | 0.692    | 0.654   | 5.6%   |
| 3     | 0.915    | 0.809   | 11.6%  |
| 7     | 0.707    | 0.624   | 11.7%  |
| 11    | 0.614    | 0.579   | 5.8%   |
| 15    | 0.539    | 0.514   | 4.6%   |

Regression: repair_ratio = -0.0029 * layer + 0.104 (R^2=0.286, p=0.033)

**Self-repair DECREASES with depth (significant negative slope). K2 FAIL.**

The repair ratio peaks in early-to-mid layers (~11.7% at layers 3-7)
then monotonically decreases. This is the opposite of the hypothesized
behavior.

### Amplification Ratios

| L  | MLP Amp Ratio | TF Amp Ratio | Reduction |
|----|---------------|--------------|-----------|
| 1  | 0.960         | 0.941        | 2.0%      |
| 4  | 0.227         | 0.226        | 0.4%      |
| 8  | 0.113         | 0.115        | -1.7%     |
| 16 | 0.037         | 0.035        | 4.1%      |

Both architectures show nearly identical amplification scaling. The
presence of attention changes the amplification ratio by at most ~5%.

### Dimension Scaling (L=12)

| d   | MLP Dev% | TF Dev% | Repair |
|-----|----------|---------|--------|
| 32  | 1.570    | 1.508   | 4.4%   |
| 64  | 0.575    | 0.543   | 4.5%   |
| 128 | 0.214    | 0.203   | 4.3%   |

Repair ratio is flat across dimensions (~4.4%), indicating no dimension
dependence.

## KILLED

**Both kill criteria fail. Hypothesis is KILLED.**

Frozen random attention provides ~2-4% reduction in output deviation,
far below the 30% threshold. The effect does not increase with depth;
it actually decreases (negative slope, p=0.033).

## Why It Failed

1. **Frozen attention lacks trained redundancy.** McGill et al.'s
   self-repair requires heads that have learned overlapping representations.
   Our frozen random attention weights have no such redundancy -- they
   cannot selectively suppress perturbations.

2. **Perturbation propagates linearly through V.** Attention is
   Attn(h) = softmax(QK^T/sqrt(d)) V. The perturbation epsilon enters
   through V linearly. Softmax only affects attention weights (the routing
   pattern), not the value space where the perturbation lives.

3. **Scale factor trade-off.** Transformer uses 1/sqrt(2L) per sub-layer
   vs MLP's 1/sqrt(L), meaning each MLP sub-layer contributes less.
   But attention sub-layers add their own variability. Net effect: roughly
   neutral.

## What Was Learned

1. **The residual_layernorm bound is tight.** Pre-RMSNorm amp_ratio=0.022
   is the correct safety bound for SOLE. Attention does not significantly
   improve it.

2. **Frozen attention is neutral.** It neither helps nor hurts expert
   removal safety. This validates the SOLE architecture where attention
   is frozen and only MLP weights carry expert deltas.

3. **Self-repair requires training.** The mechanism documented by McGill
   et al. is an emergent property of trained transformers, not an inherent
   architectural feature. Testing with trained (not random) base models
   at macro scale could yield different results, but this is a different
   hypothesis.

## Limitations

- Micro scale only (d=32-128, L<=16). Production transformers have d=4096+.
- Random (untrained) weights. Trained base models may exhibit self-repair
  through learned redundancy in attention heads.
- Expert deltas on MLP only. Attention-layer LoRA deltas are not tested
  (covered by attention_layer_removal_safety).
- Batch treated as sequence for attention (no causal mask, no positional
  encoding). This simplification makes attention fully symmetric.

## What Would Kill This at Macro Scale

If a macro experiment with a trained base model (Qwen2.5-7B) showed >30%
lower output deviation with attention-equipped experts vs attention-ablated
experts after expert removal, the micro result would be overturned. The
hypothesis would then be: "trained attention provides self-repair" (a
strictly weaker claim than "attention architecture provides self-repair").
