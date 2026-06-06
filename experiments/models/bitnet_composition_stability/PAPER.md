# BitNet Composition Stability: Research Digest

## Hypothesis

Equal-weight LoRA composition on a ternary {-1, 0, 1} base model does not produce
catastrophic PPL explosion, because the discrete weight structure bounds adapter
interference and prevents the logit-scale mismatch observed on FP16 bases.

## What This Experiment Is

A controlled micro-scale comparison of LoRA composition stability on two base model
types: standard FP16 weights and ternary-quantized weights (BitNet absmean recipe).
Five domain LoRA adapters (arithmetic, reverse, repeat, sort, parity) are trained on
each base and composed at equal weight (1/N averaging). The composition quality is
measured as the ratio of composed PPL to base PPL on each domain's evaluation set.

The experiment uses the same toy character-level transformer (d=64, L=2, H=2, r=4)
as other SOLE micro experiments, enabling direct comparison with prior results
(cross_domain_composition, structural_orthogonality_proof, etc.).

## Key References

- BitNet b1.58 (arxiv 2402.17764): ternary weight architecture, absmean quantization
- LoTA-QAF (arxiv 2505.18724): lossless ternary adapter merging
- MoTE (arxiv 2506.14435): Mixture of Ternary Experts
- Rethinking Inter-LoRA Orthogonality (arxiv 2510.03262): orthogonality alone insufficient

## Empirical Results

### Configuration

| Parameter | Value |
|-----------|-------|
| d (embed dim) | 64 |
| r (LoRA rank) | 4 |
| L (layers) | 2 |
| N (adapters) | 5 |
| Seeds | 42, 123, 314 |
| Base training | 30 epochs on mixed data |
| LoRA training | 30 epochs per domain |
| Runtime | 332.5s total |

### Kill Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1: composed PPL > 100x base | 100.0 | max ratio = 0.72 | **PASS (3/3 seeds)** |
| K2: composed PPL > 10x single on >50% domains | 2.5 domains | max 0 domains | **PASS (3/3 seeds)** |
| K3: adapters converge on ternary base | convergence | all 5 domains converge | **PASS (3/3 seeds)** |

### Composition Ratios (PPL_composed / PPL_base, mean across 3 seeds)

| Domain | FP16 Ratio | Ternary Ratio | Ternary Advantage |
|--------|-----------|---------------|-------------------|
| arithmetic | 1.04 | 0.83 | 1.26x |
| reverse | 0.86 | 0.53 | 1.62x |
| repeat | 1.28 | 0.60 | 2.13x |
| sort | 0.85 | 0.48 | 1.76x |
| parity | 1.03 | 0.73 | 1.41x |
| **Mean** | **1.01 +/- 0.01** | **0.63 +/- 0.01** | **1.60x*** |

*The 1.60x ratio improvement is driven by the ternary base having ~1.6x higher
PPL (the denominator), not by the composed model having lower absolute PPL. In
absolute terms, FP16 composed outperforms ternary composed on 3/5 domains (see
Absolute Composed PPL table below).

### Absolute Composed PPL Comparison (seed 42)

| Domain | FP16 composed PPL | Ternary composed PPL | Winner |
|--------|-------------------|---------------------|--------|
| arithmetic | 6.59 | 6.35 | Ternary (barely) |
| reverse | 4.05 | 4.39 | FP16 |
| repeat | 2.75 | 4.42 | FP16 |
| sort | 4.23 | 3.76 | Ternary |
| parity | 1.75 | 2.74 | FP16 |

**FP16 wins 3/5 domains in absolute composed PPL.** The ternary ratio advantage
(0.63 vs 1.01) is driven primarily by the higher ternary base PPL denominator,
not by ternary composed models having lower absolute perplexity.

### Key Observations

1. **Ternary composition is stable.** All 5 domains on all 3 seeds show
   composed PPL ratio < 1.0, meaning composition IMPROVES over the ternary base.

2. **FP16 composition is neutral at micro scale.** Mean ratio = 1.01, no
   catastrophic failure. This contrasts with macro results (PPL in trillions),
   suggesting the catastrophe is a scale-dependent phenomenon.

3. **Sum composition (no averaging) is catastrophic for BOTH bases.**
   FP16 sum PPL: 370-2,070. Ternary sum PPL: 4,350-46,000. The 1/N averaging
   is essential; without it, both bases fail.

4. **Ternary adapters converge to comparable quality.** Single-adapter PPL is
   similar on both bases (e.g., repeat: FP16 single=1.36, ternary single=1.37),
   showing LoRA adapters fully recover the quantization gap.

### Diagnostic Metrics

| Metric | FP16 | Ternary | Interpretation |
|--------|------|---------|----------------|
| Mean adapter \|cos\| | 0.260 +/- 0.016 | 0.275 +/- 0.028 | Comparable (not improved) |
| Delta norm CV | 0.107 +/- 0.030 | 0.199 +/- 0.013 | Ternary has HIGHER variance |
| Base PPL range | 1.8 - 6.6 | 3.0 - 9.2 | Ternary ~1.6x worse (quantization) |

Note: the ternary max pairwise cosine similarity reaches 0.83, higher than the
FP16 max of 0.71-0.80, suggesting ternary adapters may actually have MORE
interference on some domain pairs despite comparable mean cosines.

### Mechanism Analysis

The original hypothesis was that ternary weights bound adapter magnitudes,
reducing interference. **This is not the mechanism observed.** Instead:

1. **Quantization recovery effect:** The ternary base has higher PPL (quantization
   loss). Each LoRA adapter partially recovers this loss. Equal-weight composition
   retains partial recovery, yielding ratio < 1.0.

2. **Lower denominator:** The ternary base PPL (denominator) is higher, making
   the ratio mechanically smaller even when composed PPL is similar in absolute terms.

3. **No orthogonality improvement:** Adapter cosine similarities are comparable
   between FP16 and ternary bases (0.260 vs 0.275). The ternary base does NOT
   create more separable feature channels at this scale.

4. **No magnitude bounding:** Ternary adapter delta norm CV is actually HIGHER
   (0.199 vs 0.107), contradicting the magnitude bounding hypothesis.

## What This Actually Shows

1. **Ternary base composition does not catastrophically fail at micro scale.**
   All kill criteria pass comfortably (K1/K2/K3) across 3 seeds. This is a
   non-catastrophe result, not a superiority result.

2. **The mechanism is quantization recovery, not interference reduction.**
   The ternary base has ~1.6x higher PPL than FP16. LoRA adapters partially
   recover this quantization loss. The favorable ratio (0.63) reflects adapters
   repairing quantization damage, not reduced inter-adapter interference.

3. **This result does NOT predict behavior on natively-trained BitNet bases.**
   A model trained from scratch with ternary constraints has no quantization gap
   to recover. The primary mechanism observed here would not apply.

4. **The FP16 baseline also does not catastrophically fail at micro.** FP16
   composition ratio is 1.01 (neutral), in stark contrast to the macro result
   (PPL in trillions). Since the FP16 catastrophe does not reproduce at micro
   scale, comparing FP16 vs ternary at micro has limited value for resolving
   the macro composition crisis.

## Limitations

1. **Post-training quantization, not native BitNet.** Real BitNet b1.58 is trained
   from scratch with ternary constraints during forward pass. Our experiment
   quantizes a converged FP16 model. A natively-trained ternary base has no
   "quantization gap" to recover, so the primary observed mechanism would not apply.

2. **Micro scale does not reproduce FP16 catastrophe.** The FP16 composition
   ratio is 1.01 at micro (d=64), but macro experiments showed PPL in trillions.
   Therefore, the comparison between FP16 and ternary at micro scale may not
   predict behavior at macro scale.

3. **Toy data and tiny model.** Character-level tasks with d=64, r=4. The
   dynamics of ternary weights on real language at d=2048+ may differ fundamentally.

4. **Equal-weight only.** Did not test PPL-probe weighting or top-k routing.

5. **3 seeds, 5 domains.** Limited statistical power (15 data points total).

## What Would Kill This

- **At micro:** If a natively-trained ternary model (trained from scratch with
  ternary forward pass, not post-quantized) shows the same ratio > 100x base PPL.
  This would confirm the effect is purely from quantization recovery, not from
  any fundamental ternary advantage.

- **At macro:** If BitNet-2B base + 5 LoRA adapters composed at equal weight
  produces PPL > 100x the base model's PPL on held-out data. Given that the
  micro effect is driven by quantization recovery (which would not apply to
  natively-trained BitNet), this is a real risk.

- **Alternative kill:** If the ternary base PPL is so high that even with
  R < 1, the absolute composed PPL is worse than FP16 composed PPL on every domain.
  (Partially observed: ternary composed arithmetic PPL = 6.35 vs FP16 = 6.59,
  but ternary composed repeat PPL = 4.42 vs FP16 = 2.75.)

## Verdict

**All three kill criteria PASS.** Equal-weight LoRA composition on a ternary base
is stable (R = 0.63, well below the 100x threshold). However, the mechanism is
**quantization recovery, not the hypothesized interference reduction**. This means
the result for natively-trained BitNet models is uncertain and requires macro
validation. The experiment provides directional evidence that ternary bases do not
*worsen* composition, but the *improvement* may be an artifact of post-quantization.

**Status recommendation: SUPPORTED** (not proven, because the mechanism differs
from hypothesis and the micro FP16 baseline does not reproduce the macro catastrophe).
