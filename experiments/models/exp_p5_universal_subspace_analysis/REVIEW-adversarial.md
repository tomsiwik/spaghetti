# Adversarial Review: exp_p5_universal_subspace_analysis

## Verdict: PROCEED

## Strengths

1. **Honest confound analysis.** The researcher identified the dataset confound (shared B-matrices,
   mixed init methods) and correctly reinterpreted all results through that lens. No evidence of
   cherry-picking or result fabrication.

2. **Sound theorems.** Theorems 1-4 in MATH.md are mathematically correct. The predictions failed
   because the dataset violated the theorem assumptions (all Grassmannian), not because the proofs
   were wrong.

3. **Genuine structural findings.** B-matrix rank-5 domain structure and A-matrix init clustering
   are real observations confirmed by singular value analysis. The B-matrix identity across
   standard/ortho pairs (verified in PCA coordinates) is a clean empirical result.

## Issues (non-blocking)

1. **K1282 should be reported as FAIL, not PASS (degenerate).** With N=11 and K evaluated at 8-11,
   the criterion is trivially satisfied. The real test is K=4 (70.7% < 80% = FAIL). Calling it
   "PASS (degenerate)" is technically correct but overly generous. The finding summary should
   emphasize K=4 FAIL.

2. **K1284 PASS obscures composition damage.** cos=0.959 passes the quality threshold, but
   orthogonality degrades from max_cos=0.60 to max_cos=0.96 (proj_max_cos in results.json).
   Compression preserves individual adapter fidelity but destroys the Grassmannian composition
   guarantee. The finding should note: "compression preserves adapter quality but kills composition."

3. **"All 5 predictions missed" undersells results.** The theorems correctly predict behavior
   for Grassmannian-only subsets. The mismatch is in dataset composition, not mathematical error.
   PAPER.md acknowledges this in confound analysis but the prediction table framing suggests
   theoretical failure where there is only data confounding.

4. **Missing: Grassmannian-only subset analysis.** The experiment could have computed PCA on just
   the 5 ortho adapters (or just the 6 standard adapters) to test the theorems within each
   population. This would have directly verified Theorem 1 predictions. Not blocking — would
   strengthen a future follow-up.

## Kill Criteria Assessment

| ID | Reported | Reviewer Assessment |
|----|----------|-------------------|
| K1282 | PASS (degenerate) | Should be FAIL — K=4 is 70.7%, K≥N is trivial |
| K1283 | FAIL (2/5) | Agree FAIL — deltas <0.02, effectively noise |
| K1284 | PASS (weak) | Agree PASS for quality, but composition destroyed |

## Status Assessment

SUPPORTED is appropriate for guided exploration. The experiment answered a different question
than originally posed: instead of "does Universal Subspace apply to our adapters?" it answered
"what happens when you mix Grassmannian and standard adapters in subspace analysis?" The answer
is structurally informative:

- B-matrices converge to domain subspace regardless of A-init (rank-5 for 5 domains)
- A-matrices cluster by initialization method, not by domain
- Grassmannian init empirically confirmed: cos ≈ 0 (ortho) vs cos ≈ 0.82 (standard)
- Universal subspace compression incompatible with Grassmannian composition (trade-off confirmed)

These findings confirm Finding #65 on Gemma 4 and add the B-matrix domain structure observation.
