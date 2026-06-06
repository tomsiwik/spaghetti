# LEARNINGS.md: exp_rank_sensitivity_per_domain

## Core Learning

**The SVD improvement from Finding #325 is magnitude reduction, not directional
regularization.** Full-rank adapters at matched Frobenius norm (scale ~13 instead of
20) beat SVD truncation in 4/5 domains. This means SVD extraction is UNNECESSARY for
quality improvement — simple scale reduction achieves the same effect more cheaply.

**CAVEAT (from review):** The scale-control uses uniform scaling across all 252 modules
while SVD truncation retains different energy fractions per module. The H1/H2
discrimination shows uniform scaling beats heterogeneous truncation, but doesn't
definitively rule out directional effects within individual modules.

## Key Numbers

| Config | Mean PPL Ratio | Mean Behavioral |
|--------|---------------|----------------|
| Raw LoRA s=20 | 1.000 | 0.193 |
| SVD rank=1 | 0.710 | 0.455 |
| SVD rank=4 | 0.766 | 0.248 |
| SVD rank=16 | 0.999 | 0.201 |
| Scale control (~13) | ~0.72 | not measured |
| Base (no adapter) | N/A | 0.425 |

## What This Means for Pierre

1. **Scale=20 is destructive.** Raw LoRA at scale=20 degrades behavioral by 55% vs
   base (0.193 vs 0.425). The adapters are actively harmful at their training scale.

2. **Scale reduction is the fix, not SVD.** Reducing scale from 20 to ~13 improves
   both PPL and behavioral. No expensive SVD computation needed.

3. **Rank-16 LoRA has insufficient spectral diversity.** All domains show similar
   singular value profiles (SV ratio ~3x). FlexMoRE's knowledge-vs-reasoning pattern
   (knowledge peaks at r=4, reasoning at r=2896) doesn't replicate because our adapters
   are too low-rank to develop domain-specific spectral signatures.

4. **Behavioral > PPL.** SVD rank=1 has the best behavioral (0.455, above base) but
   rank=4 is better on PPL. PPL and behavioral diverge at low ranks. Behavioral is
   the right metric (per project guidelines).

## Impossibility Structure

The scale dilemma (Finding #320, #324) has a simpler explanation than previously thought:
it's not about specific destructive directions in the adapter — it's about MAGNITUDE.
Scale=20 is too strong. The solution is trivial: use a lower scale. The question becomes:
what scale gives the best tradeoff between domain PPL improvement and behavioral preservation?

## Recommended Follow-ups

1. **exp_scale_sweep_behavioral (P0):** Sweep scale {1, 2, 5, 8, 13, 15, 20} measuring
   BOTH PPL and behavioral. Find the scale sweet spot where domain improvement and
   behavioral preservation are both good.

2. **exp_solidified_composition_mmlu (P0):** Still needed — test whether lower-scale
   or SVD-truncated adapters fix the -60pp MMLU catastrophe under composition.

3. **exp_train_at_low_scale (P1):** Train NEW adapters at scale=8-13 instead of 20.
   Adapters trained at lower scale may learn different representations optimized for
   that scale, rather than post-hoc scaling a scale=20 adapter.
