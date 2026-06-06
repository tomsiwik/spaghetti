# LEARNINGS: Frobenius-Norm Equalized Composition

## Core Finding

Per-domain adapter scales encode a **dual nature** — part training artifact (causing 21.6:1 spectral imbalance that drowns minority domains) and part genuine capability signal (encoding how much each domain needs to perturb the base model). Full equalization fixes spectral disease but kills capability; partial equalization (50% log-compression) threads the needle. This completes a three-experiment arc proving that **cross-domain scale imbalance, not within-domain spectral shape, is the root cause** of composed adapter spectral pathology.

## Why This Happened

The 21.6:1 Frobenius norm ratio between medical (scale=20) and finance (scale=1) adapters dominates the composed spectrum because energy in orthogonal sums is additive (Pythagorean property). B-matrix norms are nearly identical (29.1–31.5), proving the scale factor `s_i` is the sole driver of energy imbalance.

Full equalization fails because the scale factors are not pure artifacts. Medical and math adapters (trained at scale=20) genuinely need larger perturbations — removing this signal degrades their PPL by 16–18%. This is consistent with **DeLoRA** (arXiv:2503.18225, ICLR 2025), which found that decoupling magnitude from direction in LoRA is fundamental: the "strength" (magnitude) of adaptation carries task-specific information distinct from the "angle" (direction). Our finding independently confirms this for multi-adapter composition: you can partially compress magnitude but cannot eliminate it.

## Confirming Evidence

1. **FroM** (arXiv:2506.02478) — Frobenius-norm adaptive merging. Uses per-task Frobenius norms as merging coefficients. Our experiment validates their core premise (Frobenius norm matters for merging quality) but shows uniform normalization is too aggressive — adaptive/partial is needed.

2. **DO-Merging** (arXiv:2505.15875) — Magnitude-direction decoupling confirms that LoRA magnitude variance breaks spectral methods. Our B-norm similarity (8% spread) vs scale factor spread (21.6:1) is a concrete instance of their insight.

3. **DeLoRA** (arXiv:2503.18225) — Decouples angles and strength in LoRA via Frobenius-norm bounding. Shows magnitude carries task-specific information that cannot be discarded. Our "dual nature" finding is the multi-adapter composition analogue.

4. **OSRM** (arXiv:2505.22934, ACL 2025) — Orthogonal subspaces for robust model merging. Constrains LoRA subspaces to be orthogonal *before* fine-tuning to prevent interference. Validates our Grassmannian approach from a complementary angle (they enforce orthogonality to data distributions; we enforce it geometrically on A-matrices).

## Contradicting Evidence

1. **Scale factor as pure signal:** LoRA+ (arXiv:2402.12354) argues that different learning rates for A and B matrices improve performance, implying the scale ratio may be a training artifact fixable by better optimization, not a genuine capability requirement. If true, equalization should work better than we observed. However, LoRA+ addresses within-adapter optimization, not cross-adapter composition.

2. **NB-LoRA** (arXiv:2501.19050) — Norm-bounded low-rank adaptation provides explicit singular value control *during training*, avoiding post-hoc scale problems entirely. This suggests our post-hoc equalization approach may be treating symptoms — the better solution is to control norms during adapter training. This would make the 50% compression factor unnecessary.

## Alternative Approaches (paper-backed only)

1. **Norm-bounded training** (NB-LoRA, arXiv:2501.19050) — Train adapters with explicit Frobenius norm bounds so all domains produce similar-scale deltas by construction. Eliminates the artifact component of scale at the source. Would require retraining all 5 adapters with matched norm budgets.

2. **Fisher-weighted composition** — Weight each adapter's contribution by its Fisher information (task-specific importance) rather than by raw norm. This would preserve capability signal while correcting artifact: Fisher weighting would naturally up-weight finance/legal (high Fisher, low norm) and down-weight medical/math where the norm already handles capability. Referenced in model merging literature (Fisher Merging, Matena & Raffel 2022).

3. **Adaptive per-layer equalization** — The review noted that full equalization has 3x the Gini variability of raw sum (0.037 vs 0.022 std), suggesting different layers need different compression factors. Per-layer adaptive scaling (informed by layer-wise B-matrix Gini) could outperform global 50% compression.

4. **Training-time orthogonal subspace constraints** (OSRM, arXiv:2505.22934) — Instead of post-hoc composition, constrain adapter subspaces during fine-tuning to be orthogonal to other domains' data distributions. Would improve composition quality at the source. Complementary to our Grassmannian A-matrix orthogonality.

## Implications for Next Experiments

1. **The spectral composition arc is resolved.** Three experiments (DC-Merge #277, surgery #278, Frobenius #279) converge: the disease is cross-domain scale, not within-domain spectral shape. Post-hoc spectral methods are either wrong-variable or structurally inverted. Partial equalization is the practical fix for uniform composition.

2. **Routing may make equalization moot.** Per-token routing selects top-k adapters (typically 1–2), so uniform all-5 summation is not the production path. The 21.6:1 scale imbalance only matters when summing all adapters. With routing, the research priority shifts to routing quality, not composition spectral health.

3. **The 50% compression factor is unprincipled.** It works for this setup (N=5, these domains, r=16) but has no theoretical backing. If equalization matters for production, the compression factor should either be: (a) derived from Fisher information, (b) learned per-layer, or (c) avoided entirely by norm-bounded training.

4. **Scale threshold from spectral surgery (Finding #278) remains.** At r=128, N=25, B-matrices must overlap (125% span). The current findings hold for r=16, N=5 where B-overlap is negligible. Higher scale may require different treatment.

## Recommended Follow-Up

**If continuing the composition quality line:** Fisher-weighted composition (motivation: Finding #279 shows raw norms mix signal+artifact; Fisher information separates them). However, this requires per-adapter Fisher computation, adding complexity.

**If pivoting to production priorities:** Route quality experiments. The spectral composition arc showed that uniform summation has treatable but real pathology. Per-token routing sidesteps the problem entirely by activating 1–2 adapters. Research priority should be routing accuracy and latency, not composition spectral health.

**Not recommended:** Further spectral methods (DC-Merge, surgery, SVD-based approaches). The three-experiment arc definitively shows these treat the wrong variable or are structurally inverted for Grassmannian compositions.
