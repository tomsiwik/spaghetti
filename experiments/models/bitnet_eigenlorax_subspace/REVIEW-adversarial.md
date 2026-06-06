# Peer Review: EigenLoRAx Subspace Extraction

## NotebookLM Findings

Skipped (experiment already KILLED by ideator; review is retrospective confirmation).

## Mathematical Soundness

**Step-by-step verification:**

1. **SVD setup (MATH.md Step 1-3):** Correct. Stacking N flattened adapter vectors into (N, mn), centering, and performing SVD is standard PCA. The implementation in `run_experiment.py` (lines 547-574) correctly uses the flattened approach for training, matching the math.

2. **Variance explanation formula (Step 4):** Correct. rho(K) = sum s_k^2 / sum s_k^2 is the standard cumulative explained variance ratio.

3. **Grassmannian prediction (lines 89-95 of MATH.md):** The claim that orthogonal adapters yield near-uniform singular values is directionally correct but the math is imprecise. The paper states "s_k ~ s_1 / sqrt(1 + epsilon) for all k" which would imply all singular values are approximately equal. The actual prediction should be: if N vectors are uniformly spread on a sphere in R^d (d >> N), the (N-1) nonzero singular values of the centered matrix are approximately equal, giving rho(K) ~ K/(N-1). This is correct in principle.

4. **Observed vs predicted discrepancy (lines 99-106):** The predicted rho_A(16) ~ 16/24 = 0.667 assumes random uniform spreading on the sphere. The observed 0.313 is lower, which the paper attributes to Grassmannian packing being "maximally spread" beyond random. This explanation is plausible but not rigorously derived. An alternative explanation: the frozen A matrices from Grassmannian packing are not just orthogonal to each other -- they may also have structured (non-random) orientations that concentrate variance in dimensions the PCA does NOT select. The paper should note this is a hypothesis, not a derivation.

5. **B-matrix trivial recovery (lines 108-111):** Correct. B has shape (r, d_out) with r=16. Each flattened B vector lives in R^(16*d_out). With N=25 > 16, the rank of B vectors is at most 16, so K=16 PCs span the entire B-space. The 100% variance recovery is mathematically guaranteed, not an empirical finding. Well identified.

6. **Reconstruction error analysis (lines 116-122):** The claim that holdout A is "ALSO orthogonal to the 24-adapter subspace" is an important logical step. This holds because the Grassmannian packing ensures the 25th adapter's A is orthogonal to all 24 others. Since the PCs are derived from those 24, the holdout projects to near-zero. The observed 1.008 error (slightly above 1.0) is consistent -- the mean component adds a small nonzero projection but the centered component contributes nothing. Sound.

7. **Implementation concern -- Phase 2 SVD mismatch:** The initial `extract_subspace_per_key` (lines 101-141) uses column-concatenation (axis=1), not row-stacking of flattened vectors. This is a DIFFERENT decomposition than what MATH.md describes. The training phase (lines 547-574) correctly re-extracts using flattened vectors. The K1 variance numbers come from the column-concatenation approach (Phase 2), while K2 training uses the flattened approach (Phase 4). These are different decompositions and their variance numbers are not directly comparable. **However**, this is a minor concern because: (a) the K1 metrics are reported as a diagnostic, not used downstream, and (b) the flattened re-extraction in Phase 4 is what actually drives the training result.

**Hidden assumptions:**
- Float32 SVD precision: valid at these dimensions, no numerical concern.
- 400 training steps: acknowledged limitation. Does not affect the orthogonality argument since A matrices are frozen (training duration is irrelevant to A-space structure).
- Wikitext as sole holdout: single domain, but the mechanism (A orthogonality) is domain-independent.

**Verdict on math:** Sound overall. The column-concat vs flatten mismatch in Phase 2 vs Phase 4 is sloppy but not result-invalidating. The Grassmannian prediction (31% vs 67%) is observational rather than derived.

## Novelty Assessment

**Prior art:** EigenLoRAx (Ramezani et al., 2025, arXiv 2502.04700) is properly cited. The experiment's contribution is not a new algorithm but a **negative result**: demonstrating the incompatibility between Grassmannian-packed adapters and subspace transfer.

**Delta over existing work:** The EigenLoRAx paper assumes standard LoRA with correlated A matrices (random init from same seed family). No prior work tests EigenLoRAx on orthogonally-constrained adapters. This negative result is genuinely informative for the project and fills a gap in the design space analysis.

**Reinvention check:** No reinvention. The algorithm is faithfully implemented from the paper. The `references/eigenlorax-subspace` directory is registered.

## Experimental Design

**Does this test what it claims?** Yes. The experiment cleanly tests whether EigenLoRAx works on Grassmannian-packed adapters. The hypothesis is falsifiable, the kill criteria are reasonable, and the results are unambiguous.

**Controls:**
- From-scratch LoRA baseline: appropriate control with identical training setup (same data, steps, learning rate, STE ternary quantization).
- Base PPL (no adapter): good reference point.
- Holdout reconstruction error: excellent diagnostic that directly confirms the mechanism of failure.

**Could a positive result be explained by a simpler mechanism?** N/A (negative result). But could the negative result be explained by a simpler mechanism than Grassmannian orthogonality? Potentially -- the short training (400 steps) and ternary quantization together could limit learned structure. However, the A matrices are frozen, so training duration is irrelevant to A-space variance. The paper correctly identifies that A-space is the bottleneck, and A-space structure is entirely determined by the Grassmannian skeleton, not training.

**One genuine weakness:** The subspace adapter starts from alpha=0 (the mean adapter), which is a reasonable initialization. But the from-scratch baseline starts from Kaiming-uniform A + zero B, which is a DIFFERENT initialization. The subspace method initializes at the mean of 24 trained adapters (a non-trivial starting point) while the baseline starts from the standard LoRA init. This asymmetry slightly favors the subspace method, making the 80.8% gap even MORE damning.

**K1 assessment concern:** K1 technically passes at 65.6% but this is driven by B's trivial 100%. The paper correctly flags this as misleading. The HYPOTHESES.yml kill criterion says "<50% variance" without specifying A-only vs combined. The combined metric technically passes. This is an honest reporting of a poorly-specified kill criterion, not gaming. The paper's self-correction ("practically misleading") is appropriate.

## Hypothesis Graph Consistency

- Node `exp_bitnet_eigenlorax_subspace` status: killed. Correct.
- Kill criteria in HYPOTHESES.yml match what was tested.
- K2 is the binding kill: +80.8% PPL gap vs 20% threshold. Clear.
- Blocks: none. The node doesn't block anything, so the kill has no cascading impact on the roadmap.
- Consistent with spectral surgery kill (same root cause: orthogonal adapters resist cross-adapter manipulation).

## Macro-Scale Risks (advisory)

Not applicable -- the experiment is killed. No macro follow-up needed.

The one actionable insight for macro: **the Evolve track cannot use subspace transfer**. New adapters must be trained independently from scratch. The B-matrix mean initialization idea (PAPER.md line 130) is a modest follow-up worth a separate micro experiment, but expectations should be low (B alone provides output-space structure without input-space selection).

## Verdict

**PROCEED** (as a completed, killed experiment -- the kill is correct and well-documented)

The ideator's KILL decision is sound. The experiment was well-designed, the implementation is correct (with the minor Phase 2/4 SVD mismatch noted above), and the negative result is genuinely informative. The fundamental tradeoff identified -- orthogonality for composition vs shared structure for evolve -- is an important architectural insight.

No revisions needed. The MATH.md, PAPER.md, and results.json are consistent and honest. The K1 "technically passes" observation is properly caveated. The connection to the spectral surgery kill reinforces the finding.

This experiment should remain in its current KILLED state. The finding should be referenced when designing the Evolve track: independent training is the only viable path for Grassmannian-packed adapters.
