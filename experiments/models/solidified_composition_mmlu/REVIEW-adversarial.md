# Peer Review: solidified_composition_mmlu

## Experiment Type
Frontier extension (Type 3) -- extending Finding #326 (single-adapter magnitude
reduction) to multi-adapter composition setting.

## Hack Detector
- Fix count: 1 (SVD truncation under composition). No stacking. CLEAN.
- Is MATH.md a proof or a description? **Mixed.** Theorem 1 is a description
  dressed in proof language ("proof sketch" with no QED). Theorem 2 is
  "immediate from Eckart-Young and Davis-Kahan" -- valid but trivial. Theorem 3
  is the interesting claim and was FALSIFIED by the experiment.
- Metric used as evidence: MMLU accuracy (50Q logit-based). Reasonable proxy
  for knowledge preservation but has statistical issues (see below).
- Kill criteria source: K837 is partially derived from proof (composition should
  not amplify beyond single-adapter degradation). K838 is reasonable (domain
  quality preservation). Both are sensible.

## Self-Test Audit

1. **One-sentence impossibility property:** "Magnitude reduction via SVD
   truncation or scale reduction tightens the Davis-Kahan bound." This is
   a property of TWO mechanisms (SVD and scale), not one. FLAG: the experiment
   went in expecting both to work equally (Theorem 3) and the whole point
   was that they do NOT work equally. The self-test did not identify the
   critical single property.

2. **Cited theorems:** Davis-Kahan (1970) and Eckart-Young-Mirsky (1936).
   Both are real and well-established. However, Davis-Kahan applies to
   SYMMETRIC matrices and eigenspaces. Transformer weight matrices are
   rectangular, not symmetric. The paper applies Davis-Kahan to singular
   subspaces of W^T W or W W^T implicitly, but never states this.
   FLAG: precondition not verified for the non-symmetric setting.

3. **Predicted numbers:** Specific and falsifiable. SVD r=4 composed: -25 to
   -35pp. Scale=13 composed: -25 to -35pp. These are testable. PASS.

4. **Falsification condition:** "If SVD composition is WORSE than raw
   composition" and "if scale-reduced composition is >10pp different from SVD
   at energy-matched scale." The second condition was met (26pp gap), which
   should have falsified Theorem 3. The paper acknowledges this. PASS
   (the falsification worked as designed).

5. **Hyperparameter count:** 2 (SVD rank, scale). Correctly acknowledged. PASS.

6. **Hack check:** "This experiment TESTS whether a single operation solves
   the problem." Clean. PASS.

## Mathematical Soundness

### Theorem 1 (Composition magnitude bound)
**Status: Description, not proof.**

The "proof sketch" argues that NRE rescaling targets mean(||B_i||) = sigma.
This is correct for the B-matrices, but the theorem statement is about
||delta_composed||_F where delta = scale * B^T @ A^T. The proof conflates
||B_composed|| with ||delta_composed||_F. Since A is shared (Grassmannian),
||delta_composed||_F = scale * ||B_composed^T @ A^T||_F, which depends on the
structure of B_composed relative to A, not just ||B_composed||. The NRE
rescaling of B preserves ||B||, but ||B^T @ A^T|| is NOT simply proportional
to ||B|| when A has specific structure.

Verdict: The conclusion (NRE does not amplify beyond individual norm) is
empirically supported (-42pp composed vs -60pp single, i.e., composition
HELPS). But the proof is informal and the bound ||delta_composed||_F <= sigma
is not rigorously derived.

### Theorem 2 (SVD truncation tightens Davis-Kahan)
**Status: Correct but trivially true.**

The observation that SVD truncation reduces ||delta||_F but not ||delta||_op
is correct and important. The implication (scale reduction is better because
it reduces both) follows immediately. This is the strongest part of the math.

One issue: the paper applies Davis-Kahan using ||delta||_op / delta_gap, but
never estimates delta_gap for Qwen3-4B. Without a spectral gap estimate,
the bound is qualitative only. The prediction is directional ("scale reduction
should be better than SVD truncation") but not quantitative.

### Theorem 3 (Scale-equivalent SVD truncation under composition)
**Status: FALSIFIED.**

The theorem predicted SVD r=4 and scale=13 would give the same MMLU
degradation (both -25 to -35pp). Measured: SVD r=4 = -30pp, scale=13 = -4pp.
The 26pp gap is far outside any noise margin.

The paper correctly identifies the root cause: SVD destroys Grassmannian
A-orthogonality, so NRE averaging of SVD factors produces unstructured
interference that Grassmannian composition avoids. This is a genuine insight.

However, the Theorem 3 falsification reveals a deeper issue: the entire
Davis-Kahan framework treats perturbations as unstructured. The STRUCTURE
of the perturbation (Grassmannian vs arbitrary) matters far more than
its NORM. The Frobenius norm alone is an insufficient predictor of MMLU
degradation under composition. This undermines the quantitative
predictions in Section D.

## Prediction vs Measurement

The paper contains the required table. Assessment:

| Configuration | Predicted | Measured | Match? | Notes |
|---|---|---|---|---|
| Base | 0pp | 0pp | YES | Trivial |
| Raw scale=20 N=5 | -44pp | -42pp | YES | Replication |
| SVD r=4 N=5 | -25 to -35pp | -30pp | YES | Within range |
| SVD r=1 N=5 | -17 to -27pp | -8pp | NO | 2x better than predicted |
| Scale=13 N=5 | -25 to -35pp | -4pp | NO | 6-9x better than predicted |
| Scale=5 N=5 | 0 to -2pp | 0pp | YES | Trivial (replication) |

Score: 4/6 matched (counting 2 trivial replications). The 2 misses are
the most interesting conditions. The paper honestly reports these.

**Critical statistical caveat the paper underplays:** With N=50 binary
questions, the 95% confidence interval is approximately +/-7.5pp (Wilson
interval around p=0.88 for scale=13). The measured -4pp is NOT
statistically distinguishable from 0pp. The difference between scale=5
(46/50) and scale=13 (44/50) is 2 questions. This could easily be noise.

The claimed "near-solution" at scale=13 rests on the difference between
getting 44 vs 46 questions right out of 50, which is p=0.50 by Fisher's
exact test (not significant by any standard).

## SVD Composition Method: Major Confound

The `compose_svd_experts` function averages A_svd and B_svd factors
SEPARATELY with NRE norm rescaling on each factor. This means:

    delta_composed = mean_A_rescaled @ mean_B_rescaled

This is NOT equivalent to averaging the deltas:

    delta_avg = (1/N) * sum(A_svd_i @ B_svd_i)

For the raw LoRA composition, NRE averages only B (A is shared via
Grassmannian skeleton), so the composed delta is:

    delta_composed = A_shared @ B_composed

This is mathematically principled because A is fixed.

For SVD composition, averaging both factors separately is not mathematically
justified. The composed delta from factor-wise averaging depends on the
alignment of A_svd factors across domains, which is arbitrary (each domain's
SVD produces its own rotation). This composition method may be creating
additional interference beyond what the SVD truncation itself causes.

The paper identifies this in Limitation #3 ("Averaging SVD factors separately
may not be the best way...delta-space averaging could perform differently")
but does not test the alternative. The "Grassmannian structure destruction"
claim may be partly or largely an artifact of factor-wise averaging being
the wrong composition method for SVD experts.

This does NOT invalidate the scale=13 result (which uses the proven
Grassmannian composition path), but it DOES weaken the "SVD solidification
is killed" conclusion.

## Energy-Matching Inconsistency

MATH.md claims scale=13 is "energy-matched to SVD rank=4" and cites
E(4) ~= 0.52 from Finding #327. But:

    scale_eff = 20 * sqrt(0.52) = 20 * 0.72 = 14.4

So the energy-matched scale should be ~14.4, not 13. Using scale=13
gives a Frobenius norm ratio of 13/20 = 0.65, i.e., E_equiv = 0.42,
which is LOWER than E(4) = 0.52. Scale=13 removes MORE energy than
SVD rank=4. This makes the 26pp gap even more striking (the lower-energy
method performs much better), but the "energy-matched" framing is
numerically inaccurate.

## Novelty Assessment

The finding that Grassmannian structure matters more than norm reduction
under composition is novel within this project's context. It advances
the understanding of why the Grassmannian skeleton is valuable.

Scale reduction as the solution to the MMLU catastrophe is incremental
rather than novel -- it is the obvious first thing to try. Finding #326
already showed scale reduction beats SVD for single adapters.

The SVD composition failure is interesting but may be confounded by the
composition method (see above).

## Macro-Scale Risks (advisory)

1. The scale=13 result is not statistically significant at N=50. Needs
   validation on full MMLU (14K questions) before any architectural
   decisions are made.

2. Domain quality at scale=13 was not measured. The tradeoff curve
   (MMLU preservation vs domain specialization) at scale=13 is unknown.
   Scale=5 may be too weak for domain utility; scale=13 may be the
   sweet spot or may not be -- we do not know.

3. The MMLU questions used are trivially easy (high school factual recall).
   Base model gets 92%. A more discriminating evaluation would use
   MMLU-Pro or questions where base accuracy is 50-70%.

4. The "scale as a routing weight" suggestion in the implications section
   is interesting but unexplored.

## Verdict

**PROCEED** (as a provisional finding, status capped per Type 3 rules)

The experiment is well-designed, honestly reported, and the key discovery
(Theorem 3 falsification revealing the importance of Grassmannian structure
under composition) is genuine and valuable for the project. The experiment
correctly identifies that SVD solidification is not the path forward.

However, the following caveats must be attached to the finding:

1. The scale=13 result (-4pp) is NOT statistically significant with N=50
   questions (within the 7.5pp CI). The claim should be "scale=13 shows
   promising MMLU preservation, requiring validation at larger N" not
   "near-solved."

2. The SVD composition failure may be partly an artifact of factor-wise
   averaging. The "Grassmannian destruction" claim is plausible but not
   proven -- an alternative SVD composition method (delta-space averaging)
   should be tested before concluding SVD is fundamentally incompatible
   with composition.

3. The mathematical framework (Davis-Kahan applied to non-symmetric
   matrices via implicit symmetrization) should be stated more carefully.
   The proofs are closer to mechanism descriptions than formal guarantees.

4. Domain quality at scale<=13 under composition is unmeasured and is
   the critical missing piece for any practical recommendation.

Finding status recommendation: **provisional** (frontier extension with
partial prediction failure; the Theorem 3 falsification is the real result).
