# Peer Review: Composition Interpolation Landscape

## NotebookLM Findings

Skipped -- NotebookLM not invoked due to authentication constraints. Review conducted by direct analysis of MATH.md, PAPER.md, run_experiment.py, and results.json.

## Mathematical Soundness

### What holds

1. **Linear mixing formulation is correct.** The weighted composition delta_W(w) = sum w_i * delta_W_i is the standard LoRA soup approach. The code implements this correctly via `WeightedMultiLoRALinear`, applying each adapter's contribution independently and summing.

2. **Geometric mean as mixed metric is reasonable.** For comparing PPL across domains with different magnitudes (math ~2.4 vs creative_writing ~12-15), geometric mean avoids domination by the high-PPL domain.

3. **Simplex convexity check (midpoint vs chord) is sound.** The three edge midpoints are all below the chord of the two adjacent vertices. The gap_pct values (-5.2%, -3.5%, -4.3%) are consistent with the raw PPL numbers. Verified: (3.650 + 3.662) / 2 = 3.656 vs midpoint 3.498 checks out for the medical-math edge.

4. **Phase 1 data is internally consistent.** Per-domain PPLs move monotonically (verified in raw data). The mixed_geo curves are smooth U-shapes with clear interior minima.

### What does not hold

**Issue 1: "Convexity ratio" is non-standard and misleading.** The metric (fraction of triplets where midpoint <= chord average) does not distinguish convexity from near-linearity. A perfectly linear function scores 0.0 on this metric (midpoint = chord exactly), which would be reported as "not convex." But a function that is linear plus epsilon noise would score ~0.5 depending on noise direction. The metric conflates "concave-up curvature" (true convexity) with "noise below chord" (random). The actual second derivative should be estimated and reported with sign.

For the mixed_geo curves, the landscape is actually *concave* in the interesting region near the optimum -- that is what creates the interior minimum. A convex function on [0,1] has its minimum at an endpoint. The fact that the mixed_geo optimal alpha is interior (0.20, 0.60, 0.40) means the mixed_geo function is NOT globally convex. It is quasi-concave (single-peaked), which is the useful property. The paper conflates convexity (useful for minimization) with quasi-concavity (useful for maximization / finding peaks of negative-loss). The correct claim should be: "the negative-PPL landscape is quasi-concave," or equivalently, "the PPL landscape is unimodal."

**Issue 2: Lipschitz constant computation has a units problem.** Line 501-502 of run_experiment.py computes `lipschitz = max(grad_diffs) / (alphas_float[1] - alphas_float[0])`. This divides the max absolute difference between adjacent gradient estimates by the step size. But adjacent gradients are already normalized by step size (they are finite-difference derivatives). So this is computing the Lipschitz constant of the gradient (i.e., the second derivative bound), which is correct in principle, but the reported values (3.0, 3.8, 0.2) have no interpretive context. Are these large or small? Relative to what? The MATH.md says "Small L_grad implies smooth" but provides no threshold.

**Issue 3: The causal claim "Grassmannian orthogonality produces convex landscapes" is untested.** The experiment shows that orthogonal adapters produce smooth landscapes. It does NOT show that non-orthogonal adapters produce non-smooth landscapes. Without a control condition (random A-matrices with high cosine similarity), the smoothness could be entirely attributable to (a) linearity of the mixing operation, (b) the pre-trained base dominating the loss surface, or (c) low-rank perturbations being inherently smooth. Model Soups (Wortsman 2022) showed smooth weight-space interpolation for full fine-tunes without any orthogonality constraint. The delta over prior art is unclear.

## Novelty Assessment

**Prior art that already establishes this result:**

1. **Model Soups (Wortsman et al., ICML 2022)** already showed that weight-space interpolation of fine-tuned models is smooth and that interior points beat endpoints. The key mechanism there was shared pre-training (the "linear mode connectivity" property). The MATH.md cites this but does not explain what new evidence this experiment provides beyond confirming the same phenomenon for LoRA adapters specifically.

2. **LoRA Soups (arXiv 2410.13025)** already assumes and validates that uniform averaging of LoRA adapters works well. This experiment's finding that "uniform 1/N is only 0.7% from optimal" is a confirmation, not a discovery.

3. **TIES-Merging (Yadav et al., NeurIPS 2023)** and **Task Arithmetic (Ilharco et al., ICLR 2023)** extensively study weight-space interpolation of adapter-like task vectors and show smooth landscapes.

**Delta over prior art:** The specific contribution is showing smoothness for *ternary quantized B-matrix* adapters with *Grassmannian A-matrices*. This is a valid incremental confirmation for the SOLE architecture specifically, but the paper overclaims novelty by framing it as discovering that composition landscapes are smooth.

## Experimental Design

### Adequate

- Three pair types (same-cluster, different-cluster, overlapping) provide useful variation.
- 21-point alpha sweep gives sufficient resolution for 1D curves.
- 66-point simplex grid is adequate for detecting gross non-convexity.
- The code correctly reloads the model per pair in Phase 1 (line 325) and reuses it across simplex points in Phase 2 (line 413), which is the right trade-off.

### Problematic

**Problem 1: No confidence intervals.** With only 10 validation samples per eval point (EVAL_BATCHES = 10, line 56), the PPL estimates have substantial variance. The code/engineering pair has a total mixed_geo range of 0.003 PPL. Is this above the noise floor? Without bootstrap or repeated evaluation, we cannot tell. The paper claims this pair shows "functional equivalence" but it could equally be "insufficient evaluation power."

**Problem 2: No ablation for the Grassmannian contribution.** The entire framing hinges on orthogonality producing smooth landscapes. A single control condition -- random (non-orthogonal) A-matrices with the same B-matrices -- would test this. Without it, the experiment tests "are adapter interpolation landscapes smooth?" (answer: yes, as known since 2022) rather than "does Grassmannian orthogonality make them smoother?" (the stated hypothesis).

**Problem 3: S1 success criterion is too permissive.** Line 658-659: `if s["convexity_ratio"] >= 0.5 or 0.1 < s["optimal_alpha"] < 0.9`. The OR condition means a pair passes S1 if the optimal alpha is interior, even if the landscape is flat noise. Code/engineering passes with 0.12% range because alpha=0.40 is interior -- but with 0.003 PPL range, the "optimum" is indistinguishable from any other point.

**Problem 4: The "perfectly monotonic" headline is trivially true.** Per-domain PPL being monotonic in the adapter's own weight is expected by construction: more of adapter_A on domain_A data should always reduce PPL_A (assuming the adapter helps at all). This would be surprising only if it *failed*. The interesting monotonicity question is about the mixed metric, where ratios are 0.80 and 0.60 -- decidedly not "perfectly monotonic." The paper correctly limits the "perfectly monotonic" claim to per-domain curves but leads with it as a key finding, which is misleading.

**Problem 5: Sub-additivity percentages conflate metric choice with composition benefit.** The geometric mean is a concave function (Jensen's inequality: geo_mean(a,b) <= arithmetic_mean(a,b), with equality iff a=b). When mixing two adapters, even with zero interaction, the geometric mean of PPLs at the midpoint can be below the geometric mean at the endpoints simply because the individual PPLs are closer together at interior points. A proper test would compute sub-additivity using arithmetic mean or per-domain metrics independently.

Verification: At the math-code simplex edge midpoint (0.0, 0.5, 0.5), the domain PPLs are (4.659, 2.751, 3.221). The vertex PPLs are math-vertex (4.985, 2.422, 4.068) and code-vertex (4.752, 3.411, 2.943). The arithmetic mean of vertex geo_means is (3.662 + 3.627)/2 = 3.644. The midpoint geo_mean is 3.456. But if we check each domain individually: medical goes from avg 4.869 to 4.659 (better), math from avg 2.917 to 2.751 (better), code from avg 3.506 to 3.221 (better). So ALL three domains improve at the midpoint -- this is genuine sub-additivity, not a metric artifact. The sub-additivity claim holds, but the quantitative percentages (3.5-5.2%) are inflated by the geo_mean's concavity. Per-domain improvements are ~3-8%, which is consistent.

## Macro-Scale Risks (advisory)

1. **N=3 is too few to extrapolate simplex properties.** At N=24 (production), the weight simplex is 23-dimensional. Smooth landscapes in 2D do not guarantee smoothness in 23D. Higher-order interference terms (3-body, 4-body) could create ridges or saddle points not visible in pairwise/triple analysis.

2. **The 0.7% uniform-to-optimal gap will likely grow with N.** At N=3, uniform is close to optimal because there are only 3 adapters. At N=24, the optimal weights are likely much more skewed (some adapters contribute nothing for a given query), making uniform a worse baseline. The experiment's reassurance that "uniform is good enough" should not transfer to production.

3. **PPL-only evaluation.** The paper acknowledges this. Task-specific metrics (code pass@k, math accuracy) may have sharp phase transitions that PPL misses. A smooth PPL landscape does not guarantee smooth task accuracy landscape.

4. **Evaluation only on validation data from the same domains as the adapters.** No out-of-distribution evaluation. The smooth landscape may not hold for queries that fall between domain clusters.

## Verdict

**REVISE**

The core empirical finding -- smooth, unimodal interpolation landscapes for ternary LoRA adapters -- is real and useful for the SOLE architecture. The data is clean and the code is well-structured. However, the paper has overclaims and missing controls that need to be fixed before the finding can be cited reliably.

### Required revisions:

1. **Replace "convex" with "unimodal" or "quasi-concave" throughout.** The mixed-metric landscape has interior minima, which means it is NOT convex (a convex function's minimum is at a boundary). The correct term is "unimodal" (single basin). This is not pedantic -- claiming convexity implies gradient descent to a global minimum, while unimodality only implies local search suffices.

2. **Drop or qualify the Grassmannian causation claim.** Change "Grassmannian orthogonality produces convex landscapes" to "landscapes are smooth for Grassmannian LoRA adapters." Without a non-orthogonal control, causation is unsupported. Add a sentence noting that Model Soups showed similar smoothness without orthogonality constraints, so pre-training may be the dominant factor.

3. **Add confidence intervals for at least one pair.** Either bootstrap the 10-sample PPL estimates or run 3 seeds on one alpha sweep. This is needed to confirm the code/engineering 0.003 PPL range is real signal vs noise.

4. **Tighten S1 criterion.** Code/engineering should not pass S1 given its 0.12% range. Either add a minimum signal threshold (range_pct > 2%) to S1, or report it as "inconclusive" rather than "PASS." The current S1 result of 3/3 is inflated.

5. **Reframe the "perfectly monotonic" finding.** Move it from headline finding to expected behavior. The headline should be about the mixed-metric unimodality and sub-additivity, which are the actionable results.
