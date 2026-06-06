# BitNet Orthogonality Trained: Research Digest

## Hypothesis

Trained LoRA adapters on a ternary (BitNet-style) base model show lower
pairwise functional cosine similarity than the same adapters trained on
an FP16 base, because ternary weights create more separable feature channels.

**Result: KILLED (K1).** Ternary base does NOT improve adapter orthogonality.
Mean |cos| is 5.9% WORSE on ternary (0.276 vs 0.260). K2 passes trivially
because cosines are low at micro scale regardless of base type.

## What This Experiment Is

A diagnostic micro-scale experiment measuring pairwise cosine similarity
between trained LoRA adapter weight deltas on two base models:

1. **FP16 base:** Standard continuous weights (trained from random init)
2. **Ternary base:** Post-quantized to {-1, 0, 1} via BitNet absmean recipe

Both bases use the same architecture (d=64, r=4, L=2 transformer, 5 toy
domains). Same LoRA initialization, same training data, same hyperparameters.
Full 10-pair cosine matrix computed for each of 3 seeds.

## Key References

- BitNet b1.58 (Ma et al., 2402.17764): Ternary weight quantization recipe
- exp_bitnet_composition_stability: Ternary composition ratio 0.63 (SUPPORTED)
- exp_bitnet_ternary_adapter_composition: Ternary adapters decorrelate by -19.3% (SUPPORTED)
- exp_structural_orthogonality_characterization: FP16 adapter cosines at scale

## Empirical Results

### Kill Criteria Assessment

| Criterion | Metric | Threshold | Observed | Verdict |
|-----------|--------|-----------|----------|---------|
| K1 | mean |cos| ternary < FP16 | ternary < FP16 | 0.276 >= 0.260 | **KILLED** |
| K2 | arith-sort cos on ternary | < 0.5 | 0.121 | PASS (trivial) |

### Aggregate Results (3 seeds)

| Metric | FP16 | Ternary | Diff |
|--------|------|---------|------|
| Mean |cos| | 0.260 +/- 0.016 | 0.276 +/- 0.028 | +0.015 (worse) |
| Max |cos| | 0.767 +/- 0.040 | 0.827 +/- 0.009 | +0.060 (worse) |
| Arith-sort pair | 0.125 +/- 0.027 | 0.121 +/- 0.017 | -0.004 (neutral) |

Paired t-test on mean |cos| difference: t(2) = 0.643, not significant
(critical value at alpha=0.05 is 4.303). But direction is consistently
ternary-worse in 2/3 seeds and in aggregate.

### Per-Pair Breakdown (3-seed mean)

| Pair | FP16 |cos| | Ternary |cos| | Delta | Winner |
|------|-------------|---------------|-------|--------|
| reverse-sort | 0.767 | 0.827 | +0.060 | FP16 |
| reverse-repeat | 0.508 | 0.581 | +0.073 | FP16 |
| repeat-sort | 0.519 | 0.539 | +0.020 | FP16 |
| sort-parity | 0.153 | 0.185 | +0.032 | FP16 |
| repeat-parity | 0.062 | 0.079 | +0.017 | FP16 |
| arith-reverse | 0.100 | 0.129 | +0.029 | FP16 |
| arith-sort | 0.125 | 0.121 | -0.004 | Ternary |
| arith-repeat | 0.125 | 0.095 | -0.030 | Ternary |
| arith-parity | 0.078 | 0.036 | -0.043 | Ternary |
| reverse-parity | 0.164 | 0.164 | -0.000 | Tie |

**Pattern:** FP16 wins 6/10 pairs. Ternary wins 4/10 pairs. Critically,
ternary WORSENS the highest-overlap pairs (reverse-sort, reverse-repeat)
while slightly improving already-low-overlap pairs (arithmetic-parity).

### Interpretation

The ternary base concentrates adapter interference. Domains that naturally
overlap (reverse/repeat/sort all involve character-level permutation) become
MORE correlated on ternary because the reduced base capacity forces them
to share a narrower set of effective features. Domains that are already
dissimilar (arithmetic vs. parity) become slightly more orthogonal because
the sparse ternary structure provides cleaner separation for unrelated tasks.

This means ternary base is WORSE for the hard composition cases (high-overlap
domain pairs) while marginally better for the easy cases (already-orthogonal
pairs). This is the opposite of what SOLE needs.

## Connection to Prior Results

This result resolves a mystery from the BitNet-SOLE track:

1. **exp_bitnet_composition_stability** found composition ratio 0.63 on ternary
   base but noted "mechanism is quantization recovery, not interference reduction."
   **This experiment confirms:** the composition benefit is NOT from orthogonality.

2. **exp_bitnet_ternary_adapter_composition** found -19.3% decorrelation from
   ternary ADAPTERS. **This experiment shows:** the decorrelation comes from
   adapter quantization (discrete adapter weights), NOT from base quantization
   (discrete base weights).

3. Together these results mean: for SOLE, the ternary base helps via
   magnitude bounding (preventing logit explosion), while ternary adapters
   help via decorrelation (more orthogonal weight deltas). These are
   independent mechanisms with independent value.

## Limitations

1. **Micro scale only (d=64, r=4).** At this scale, mean |cos| ~ 0.26 is
   close to the random subspace bound sqrt(r/d) = 0.25. The effect of base
   type may be larger at production scale where trained cosines diverge more
   from the random baseline.

2. **Post-quantized ternary, not natively-trained.** A BitNet model trained
   from scratch with ternary weights throughout may develop different gradient
   dynamics than a post-quantized FP16 model.

3. **5 toy domains only.** Real domain overlap structure (math-medical
   cos=0.703 at macro) may interact differently with ternary base.

4. **Paired t-test underpowered.** With n=3 seeds, we can only detect large
   effects (Cohen's d > 2.5 at alpha=0.05). The observed d=0.37 is a small
   effect that would require ~60 seeds to detect.

5. **K2 is trivially satisfied.** The arithmetic-sort pair has low cosine
   regardless of base type at d=64 (0.12-0.13). This pair does not probe
   the math-medical overlap that motivates K2.

## What Would Kill This (at Macro Scale)

This experiment is already killed at K1 (micro). For completeness:
- If replicated at d=4096 with real adapters on BitNet-2B and Qwen2.5-0.5B:
  ternary mean |cos| > FP16 mean |cos| would confirm the kill.
- If math-medical pair cos on BitNet-2B < 0.5 (vs 0.703 on FP16 Qwen):
  that would rescue the hypothesis via scale-dependent effects.
