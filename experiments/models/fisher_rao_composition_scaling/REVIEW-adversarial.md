# Peer Review: Fisher-Rao Composition Scaling (POST-REVISION)

## Experiment Type
Verification (Type 1)

## Revision Audit: Were the 5 Required Fixes Addressed?

| Fix | Required | Done? | Quality |
|-----|----------|-------|---------|
| 1. Downgrade Theorems 2-3 to Conjectures | Yes | YES | Good. MATH.md labels them "Conjecture 2" and "Conjecture 3" with explicit notes that predictions were wrong in direction. Honest about the linear response model failure. |
| 2. Norm-rescaled Euclidean baseline | Yes | YES | Good. Full implementation in `compose_b_norm_rescaled_euclidean()`, measured at all N values, compared fairly against FR and raw Euclidean. |
| 3. N=5 ceiling acknowledged | Yes | YES | Good. PAPER.md Limitations section states "All metrics plateau at N=5. Scaling claims are restricted to N=5." Prediction table footnotes the synthetic adapter issue. |
| 4. N=1 scale confound fixed | Yes | YES | Good. Code uses `scale = np.mean(list(OPTIMAL_SCALES.values()))` = 12.8 for ALL N values (line 698). Comment explicitly references Fix #4. |
| 5. Prediction table updated honestly | Yes | YES | Good. PAPER.md has separate tables for Theorem 1 (verified) and Conjectures 2-3 (wrong direction). "Honest Assessment" section enumerates all 5 things the original got wrong. |

**All 5 fixes are properly implemented.** The revision is substantive, not cosmetic.

## Hack Detector
- Fix count: 1 (norm preservation via either Karcher mean or rescaling). CLEAN.
- Is MATH.md a proof or a description? **Proof with QED** for Theorem 1. Conjectures 2-3 are correctly labeled as conjectures (not theorems). No equations-dressed-as-proof issue.
- Metric used as evidence: Norm shrinkage ratio (directly from Theorem 1), PPL (behavioral proxy). Both are appropriate.
- Kill criteria source: K690/K691 thresholds remain somewhat arbitrary (not derived from proof), but K692 is directly from Theorem 1. Acceptable given the conjectures were downgraded.

## Self-Test Audit

1. **One-sentence impossibility property:** "Norm preservation after averaging prevents 1/sqrt(N) shrinkage." Single property, correctly identified. Now honestly includes that both Karcher mean and rescaling achieve it. PASS.

2. **Cited theorems:** Karcher (1977), Jang et al. (2024). Real, conditions verified. PASS.

3. **Predicted numbers:** Norm shrinkage 1/sqrt(N) vs 1.0. Specific, falsifiable, and verified. Conjectures correctly noted as having wrong predictions. PASS.

4. **Falsification condition:** Two conditions given: (a) norm preservation doesn't improve PPL (falsified: it does), (b) Karcher mean outperforms norm-rescaled Euclidean (tested: it doesn't). Both are genuine tests. Improvement from prior review which flagged the tautological kill criterion. PASS.

5. **Hyperparameter count:** 0. Correct. PASS.

6. **Hack check:** "The experiment reveals that the Riemannian manifold machinery is overkill." This is an admirably honest conclusion that simplifies rather than complexifies. PASS.

## Mathematical Soundness

### Theorem 1 (Norm Preservation) -- SOUND

The proof remains correct and structurally simple:
- Part (a): By construction, u_FR is on S^(d-1), so ||u_FR|| = 1. r_FR = mean(r_i). Product = mean(r_i). Ratio = 1. QED. Verified.
- Part (b): Triangle inequality application is correct. Equality condition (all u_i identical) is correct.
- Corollary: 1/sqrt(N) for orthogonal vectors follows from ||sum of N orthogonal unit vectors|| = sqrt(N). Verified at N=3,5.

**Remaining issue (non-blocking):** Theorem 1 is still somewhat tautological -- it proves that "if you define the merge to preserve norms, norms are preserved." The real content is the behavioral consequence (PPL improvement), which is empirical, not proven. But this is acceptable for a Type 1 verification: the theorem predicts a specific measurable quantity (norm ratio), the experiment measures it, and the behavioral consequence is observed.

### Conjectures 2-3 -- APPROPRIATELY HANDLED

The downgrade is done correctly. The MATH.md:
- Explains what the original linear response model predicted
- Documents what actually happened (wrong sign for both)
- Offers revised conjectures with correct directional predictions
- Does NOT claim these revised conjectures are proven

This is the right way to handle failed predictions: acknowledge, explain, revise.

## Prediction vs Measurement

PAPER.md contains the required table. Assessment:

| Prediction | Measured | Verdict |
|-----------|----------|---------|
| FR norm shrinkage = 1.0 | 1.0000 at all N | PASS (by construction) |
| NRE norm shrinkage = 1.0 | 1.0000 at all N | PASS (by construction) |
| Euc shrinkage = 0.577 at N=3 | 0.5750 | PASS (0.3% error) |
| Euc shrinkage = 0.447 at N=5 | 0.4482 | PASS (0.2% error) |
| Euc shrinkage at N=10,15 | Plateaus at 0.447 | Correctly noted as synthetic adapter ceiling |
| Norm-preserved PPL < raw Euc | 9.17/9.20 vs 10.44 at N=5 | PASS |
| Conjectures 2-3 | Wrong direction | Honestly reported as wrong |

The prediction-vs-measurement table is honest and complete. The revision correctly separates proven results from failed conjectures.

## Key Finding Assessment: NRE Matches FR

The central finding of this revision is that norm-rescaled Euclidean (NRE) matches Fisher-Rao (FR) on all metrics while being 10x faster. This is a clean, well-controlled comparison:

- Same scale (12.8) for all methods at all N
- Same B-matrices as input
- NRE implementation is straightforward (lines 234-277): compute Euclidean mean, measure ratio of mean-source-norm to mean-norm, rescale
- FR implementation uses the full Karcher mean machinery (lines 280-350)
- Results: PPL 9.17 (NRE) vs 9.20 (FR) at N=5; identical activation variance and effective rank

**Is the comparison fair?** Yes. Both methods receive identical inputs. The only difference is directional averaging strategy. The conclusion -- that direction quality from Karcher mean adds nothing measurable over normalized Euclidean mean -- is well-supported.

**One subtlety:** For nearly orthogonal unit vectors, the Karcher mean and the normalized Euclidean mean converge to the same direction (as noted in MATH.md Section F worked example). So the equivalence is expected from the math. With highly non-orthogonal adapters, the Karcher mean might differ. But that regime is not the one this architecture targets (Grassmannian A-matrices enforce near-orthogonality).

## NotebookLM Findings

Skipped -- the mathematical and experimental analysis is sufficient for this revision review.

## Novelty Assessment

The Fisher-Rao Karcher mean approach is from arXiv:2603.04972. The novel contribution of this experiment is the **negative result**: the Riemannian manifold machinery is unnecessary when adapters are near-orthogonal. Norm rescaling alone captures the entire benefit. This is a useful simplification finding for the architecture.

## Macro-Scale Risks (advisory)

1. **The N=5 ceiling is the real limitation.** All claims stop at 5 truly independent adapters. Macro needs N=15-25. The mechanism (norm preservation) should generalize, but this is untested.
2. **With truly independent adapters at scale, Karcher mean vs NRE might diverge.** When adapters are NOT near-orthogonal (e.g., overlapping domain knowledge), the directional averaging quality could matter. This warrants a macro-scale retest.
3. **The "moderate norm shrinkage as regularization" hypothesis** (listed in "What Would Kill This") is worth investigating. At macro scale with well-trained adapters, some shrinkage might be beneficial.

## Verdict

**PROCEED**

The revision addresses all 5 required fixes substantively:

1. Theorems 2-3 properly downgraded to Conjectures with honest documentation of prediction failures
2. Norm-rescaled Euclidean baseline added, fairly compared, and found equivalent to Fisher-Rao
3. N=5 ceiling acknowledged throughout with claims properly scoped
4. N=1 scale confound fixed (consistent scale=12.8 for all N)
5. Prediction table honestly separates verified Theorem 1 from failed Conjectures 2-3

**What this experiment proves:** Theorem 1 (norm preservation) is verified exactly. Norm preservation via any method (Karcher mean or simple rescaling) improves PPL by ~12% over raw Euclidean averaging at N=5. The Riemannian manifold structure adds no measurable benefit beyond norm preservation.

**What this experiment does not prove:** Anything about activation variance or effective rank mechanisms (Conjectures 2-3 failed). Scaling beyond N=5 (synthetic adapter ceiling). Whether norm preservation matters with truly diverse, independently trained adapters.

**Finding status recommendation:** `supported` (Theorem 1 verified, but Conjectures 2-3 wrong; practical implication is a simplification -- use NRE instead of FR).
