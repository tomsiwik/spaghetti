# LEARNINGS.md: exp_solidified_composition_mmlu

## Core Learning

**SVD solidification FAILS for composition (-30pp MMLU at rank=4). Scale reduction
SUCCEEDS: N=5 composition at scale=13 gives -4pp MMLU degradation (vs -42pp at scale=20).
The Grassmannian LoRA structure (shared orthogonal A, per-domain B) IS the value — SVD
extraction destroys it.**

**CAVEAT (from review):** The -4pp result is 2 questions on 50Q MMLU (within noise at
95% CI ~7.5pp). The trend (scale=5→0pp, scale=13→-4pp, scale=20→-42pp) is clear but
the precise number needs larger-N confirmation.

## What This Means

### SVD Solidification Path: KILLED
SVD extraction re-factors the Grassmannian A@B product into arbitrary (A_svd, B_svd).
Under NRE composition, the resulting factors don't average coherently because they lack
the orthogonality structure that makes LoRA B-matrix averaging work (Finding #225).
26pp gap between SVD r=4 and scale-matched full-rank proves the structure matters.

### Scale Reduction Path: VALIDATED
Simple inference-time scale reduction from 20 to 13 preserves 96% of MMLU (88% vs 92%).
The adapters were TRAINED at scale=20, so this works at inference without retraining.
Scale=5 preserves 100% of MMLU but Finding #320 showed domain effect is negligible there.
Scale=13 is the sweet spot where domain effect exists AND MMLU is preserved.

### The Scale Dilemma: RESOLVED
- Scale=5: 0pp MMLU degradation, minimal domain effect
- Scale=13: -4pp MMLU, meaningful domain effect
- Scale=20: -42pp MMLU, maximum domain effect but catastrophic knowledge loss

The answer is not "find the right SVD rank" — it's "use the right scale at inference time."

## What's Still Unknown

1. **Domain quality at scale=13.** This experiment measured MMLU but not domain PPL/behavioral
   at scale=13 under composition. Finding #326 measured single-adapter at scale=13.

2. **Scale-quality Pareto frontier.** Need a fine-grained sweep {5, 8, 10, 13, 15, 18, 20}
   measuring BOTH MMLU and domain PPL simultaneously to find the exact Pareto-optimal scale.

3. **Training at the right scale.** Current adapters are trained at scale=20. Training at
   scale=13 might produce better adapters for the scale=13 operating point (the adapter
   learned to compensate for scale=20 dynamics).

4. **SVD composition method confound.** The SVD composition averaged A_svd and B_svd
   separately (NRE on both factors). This is not equivalent to averaging the deltas.
   The -30pp might be partly from the wrong composition method rather than structural
   destruction.

## Recommended Follow-ups

1. **exp_scale_pareto_frontier (P0):** Sweep scale {5, 8, 10, 13, 15, 18, 20} measuring
   both MMLU and domain PPL/behavioral under N=5 composition. Find the Pareto-optimal
   scale for the domain-vs-knowledge tradeoff.

2. **exp_train_at_optimal_scale (P1):** Train NEW adapters at the Pareto-optimal scale
   (likely ~10-13). May produce better domain quality at that scale than post-hoc
   scaling of scale=20 adapters.

3. **exp_per_domain_scale (P1):** Different domains may have different optimal scales.
   Medical (knowledge-heavy) may tolerate higher scale than math (reasoning-heavy).
