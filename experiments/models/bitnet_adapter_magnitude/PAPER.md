# BitNet Adapter Magnitude Analysis: Research Digest

## Hypothesis

LoRA adapter weight magnitudes are bounded on ternary (BitNet) base vs unbounded
on FP16, explaining why composition works despite worse orthogonality.

**Verdict: KILLED (K1).** Ternary base does NOT bound adapter magnitudes. Norm
variance is 2.6x HIGHER on ternary (3.37 vs 1.29). The composition stability
benefit comes from activation compression, not magnitude bounding.

## What This Experiment Is

A diagnostic experiment measuring six facets of adapter magnitude on FP16 vs
ternary base models, using the same 5-domain micro transformer (d=64, r=4, L=2)
from the bitnet_composition_stability experiment. Trains 5 LoRA adapters on
each base type across 3 seeds and measures: delta norms, per-layer norms,
activation magnitudes, logit-scale distributions, composition norms, and
adapter signal strength.

## Key References

- BitNet b1.58 (arxiv 2402.17764): ternary weight quantization
- exp_bitnet_composition_stability: ternary composed PPL ratio 0.63 vs FP16 1.01
- exp_bitnet_orthogonality_trained: ternary orthogonality 5.9% WORSE (mean |cos| 0.276 vs 0.260)

## Empirical Results

### Configuration

| Parameter | Value |
|-----------|-------|
| d (embed dim) | 64 |
| r (LoRA rank) | 4 |
| L (layers) | 2 |
| N (adapters) | 5 |
| Seeds | 42, 123, 314 |
| Runtime | 404.4s |

### Kill Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1: ternary norm var < FP16 norm var | var_t < var_f | FP16=1.29, Ternary=3.37 (2.6x HIGHER) | **KILLED (0/3 seeds)** |
| K2: ternary max/min ratio <= 10x | 10.0 | 1.71 (well under threshold) | **PASS (3/3 seeds)** |

### Delta Norm Statistics (aggregated across 3 seeds)

| Metric | FP16 | Ternary | Direction |
|--------|------|---------|-----------|
| Mean norm | 10.34 | 9.24 | Ternary 10.7% smaller |
| Norm variance | 1.29 | 3.37 | Ternary 2.6x HIGHER |
| Norm CV | 0.107 | 0.199 | Ternary 1.9x HIGHER |
| Max/min ratio | 1.36 | 1.71 | Ternary 1.3x HIGHER |

All 3 seeds show the same pattern: ternary has LOWER mean norms but HIGHER
variance. The magnitude bounding hypothesis is definitively killed.

### Where The Difference Actually Is

| Metric | FP16 | Ternary | Interpretation |
|--------|------|---------|----------------|
| Post-FFN activation (L0) | 0.85 | 0.47 | Ternary 1.8x smaller |
| Post-FFN activation (L1) | 2.23 | 1.13 | Ternary 2.0x smaller |
| Logit cross-domain CV | 0.129 | 0.109 | Ternary 15% more uniform |
| Signal strength (||delta||/||base||) | 0.48 | 0.63 | Ternary adapters 31% stronger relative to base |
| Composition efficiency | 1.32 | 1.38 | Similar (slight constructive interference) |

### Per-Layer Analysis

Ternary adapters show a striking pattern:
- **FFN layers (W1, W2)**: 20-35% SMALLER norms (base constraints reduce FFN delta)
- **Attention QKV**: 33-97% LARGER norms (compensating for reduced base expressivity)
- **Head**: 10-17% smaller norms
- **Attention output (Wo)**: Mixed (some layers larger, some smaller)

This suggests ternary base forces adapters to redistribute learning from FFN
to attention, consistent with the attention amplification finding from
ffn_only_vs_all_modules.

## The Real Mechanism: Activation Compression

The ternary base compresses the activation dynamic range by approximately 2x.
This means:

1. Each adapter's contribution to logit space is proportionally smaller in
   absolute terms, even though relative signal strength is 31% higher
2. The logit scale is more uniform across domains (CV 0.109 vs 0.129),
   reducing the "logit-scale mismatch" that causes composition catastrophe
3. Under 1/N composition, the absolute perturbation per adapter is smaller,
   making the sum more stable

This is NOT "magnitude bounding" (the deltas themselves are more variable).
It is "activation compression" -- the ternary base creates a lower-amplitude
signal pathway where adapter interference has less absolute impact.

## What This Means for SOLE

1. **The composition benefit of ternary base is real but the mechanism is
   different than hypothesized.** It comes from activation compression, not
   from constraining adapter learning.

2. **Activation normalization on FP16 base might achieve the same effect.**
   If the benefit is compressed activations, techniques like weight
   normalization, activation scaling, or temperature scaling at composition
   time could replicate this on FP16 bases.

3. **The 2x activation compression is a double-edged sword.** It makes
   composition more stable but also means the ternary base has less
   representational capacity per layer -- the adapters must work harder
   (31% higher signal strength) to achieve the same functional effect.

## Limitations

1. **Micro scale only** (d=64, r=4). Activation compression ratio may differ
   at d=4096 with real BitNet models.
2. **Post-hoc quantization**, not trained-from-scratch BitNet. Real BitNet
   b1.58 learns ternary weights during training, which may produce different
   activation dynamics.
3. **5 toy domains**. Real domain adapters may show different norm distributions.
4. **No causal analysis**. We show correlation between activation compression
   and composition stability, but have not proven causation.

## What Would Kill This

The activation compression hypothesis would be killed if:
- Artificially scaling FP16 activations to match ternary scale does NOT improve
  composition stability (would prove compression is not the mechanism)
- Real BitNet b1.58 models show similar activation magnitudes to FP16
  (would mean post-hoc quantization is not representative)
