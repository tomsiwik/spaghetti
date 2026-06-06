# LEARNINGS.md: exp_svd_extraction_quality

## Core Learning

**SVD truncation of rank-16 LoRA deltas at rank=4 achieves 23.4% BETTER domain PPL
than raw LoRA AND halves MMLU degradation (-30pp vs -60pp) on Qwen3-4B-4bit. Whether
this is directional regularization (removing destructive SVD components) or simple
magnitude reduction (equivalent to lowering scale from 20 to ~13) is NOT YET DETERMINED.**

## What Worked

1. **Eckart-Young verification.** Rank >= 16 is exactly lossless (ratio = 0.999,
   reconstruction error = 0.0). Ranks 32/64/128 identical to rank 16. The LoRA delta
   is exactly rank 16 as expected.

2. **Rank=4 sweet spot.** All 5 domains show the same pattern: rank 4 IMPROVES over
   raw LoRA. Medical: 9.36 vs 10.55 (ratio 0.887), Code: 6.83 vs 9.55 (0.715),
   Math: 4.27 vs 5.27 (0.811), Legal: 23.13 vs 32.76 (0.706), Finance: 22.82 vs
   32.08 (0.711).

3. **MMLU damage reduction.** SVD rank=4 medical adapter: 62% MMLU (base: 92%,
   -30pp degradation). Raw LoRA at scale=20: 32% MMLU (-60pp). Truncation halves
   the knowledge destruction.

4. **SVD extraction pipeline.** The factored QR approach (never materializing full
   delta matrices) works within 2.5 GB memory. Practical for M5 Pro.

## What's Uncertain

1. **Mechanism: regularization vs magnitude reduction.** SVD rank=4 keeps ~44% of
   Frobenius energy. This is equivalent to reducing scale from 20 to ~13. The
   improvement could be ENTIRELY from reducing perturbation magnitude. Three
   controls needed to distinguish:
   - Random rank-4 projection (same dims, different subspace)
   - Scale reduction to ~13 (same magnitude, full rank)
   - Bottom-4 SVD (keep smallest SVs)

2. **Optimal rank generalizability.** Rank=4 was optimal on these specific adapters
   at scale=20. Different scales, models, or training regimes may have different optima.
   FlexMoRE found knowledge tasks peak at r=4 but reasoning at r=2896.

3. **Composition of SVD experts.** Single-expert quality is verified. Multi-expert
   composition could re-introduce interference. Deferred to exp_solidified_composition_mmlu.

## Impossibility Structure

The scale=20 MMLU catastrophe (-60pp, Finding #320) has a structural cause:
the LoRA perturbation at scale=20 is ~100x larger (in Frobenius norm relative to base
weight norm) than the perturbation at scale=1. This overwhelms the base model's
knowledge representations. SVD truncation reduces this perturbation, bringing it
closer to the safe regime. The remaining -30pp degradation suggests rank=4 still
has too much perturbation for full MMLU preservation.

## Literature Connections

- **FlexMoRE (2312.15007):** Independently observed SVD extraction improves quality
  in 5/6 experts. Our result confirms this in the LoRA/Grassmannian context.
- **Eckart-Young-Mirsky (1936):** Truncated SVD is optimal rank-r approximation.
  Verified: reconstruction error is monotonically non-decreasing as rank decreases.
- **Finding #320:** Scale=20 destroys MMLU by -60pp. SVD rank=4 halves this to -30pp.
- **Finding #228 (killed):** Bridge extraction (partially undoing top SVD components)
  hurts quality. Consistent with Hypothesis A (top components carry useful signal).

## Recommended Follow-ups

1. **exp_solidified_composition_mmlu (P0):** Does SVD extraction fix the -60pp MMLU
   catastrophe under composition? This is THE key question.

2. **exp_svd_vs_scale_control (P1):** Distinguish regularization from magnitude
   reduction. Compare: SVD rank=4, random rank=4, scale=13.3, bottom-4 SVD.

3. **exp_rank_sensitivity_per_domain (P1):** Optimal rank per domain. Medical vs
   code vs math may have different rank requirements.
