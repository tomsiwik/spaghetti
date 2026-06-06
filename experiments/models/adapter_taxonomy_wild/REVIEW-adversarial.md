# Peer Review: adapter_taxonomy_wild (Revision)

## Previous Review Status

The previous review requested 4 required fixes and 3 advisory items. All have been addressed.

### Required Fix Verification

**Fix 1: Interference bound formula corrected to r/sqrt(D).**
VERIFIED. MATH.md Section 5.1 now states `E[|cos(dW_i, dW_j)|] ~ r / sqrt(D)` with the correct numerical evaluation: 16 / sqrt(67,879,424) ~ 1.9 * 10^{-3}. The previous erroneous formula `sqrt(2/(pi*D)) * r^2 / D` has been removed. The `r/sqrt(D)` scaling is a reasonable heuristic for random rank-r matrices -- not a tight bound, but the paper uses "~" notation appropriately and relies on the empirical cos ~ 0.0002 as the actual evidence. Acceptable.

**Fix 2: Full-rank Class A caveat added (single-adapter only).**
VERIFIED. PAPER.md Section "Key Findings, 1. Three Composition Classes" now contains a clear note that full-rank adapters are Class A only in the single-adapter case, with the explanation that `E[|cos|] ~ 1/sqrt(d)` (not ~ 10^{-4}) for full-rank deltas. MATH.md Section 6 contains the same caveat. Both correctly state that the O(N^2 * epsilon) bound requires epsilon << 1, which does not hold for full-rank.

**Fix 3: He et al. 2022 + Lialin et al. 2023 cited.**
VERIFIED. PAPER.md "Key References" section now includes both:
- He et al. 2022, "Towards a Unified View of Parameter-Efficient Transfer Learning"
- Lialin et al. 2023, "Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning"

**Fix 4: effective_dims_per_param labeled as heuristic.**
VERIFIED. Every instance in `adapter_taxonomy_wild.py` now carries the comment `HEURISTIC ESTIMATE -- no formal derivation` with a brief explanation of the intuition behind each value. The values remain in the code (they are not used in the FIT score), but the labeling prevents misinterpretation.

### Advisory Item Verification

**Advisory 5: FIT circularity acknowledged.**
VERIFIED. PAPER.md line 84-85 now states: "The FIT weights were chosen to reflect our architecture's priorities; they confirm our existing choice of LoRA rather than discovering it."

**Advisory 6: exp_relora_composition_test added as next step.**
VERIFIED. PAPER.md "Recommended Next Steps" section now includes item 4: `exp_relora_composition_test` with a clear description of what it tests and why it matters for the base-freedom path.

**Advisory 7: Survey/literature-review nature noted.**
VERIFIED. PAPER.md "What This Experiment Is" section now includes: "This experiment is a literature review and analytical survey, not empirical validation." The Limitations section reinforces this. FINDINGS.md carries the caveat: "Survey/analytical experiment, no adapters trained."

## Mathematical Soundness

All mathematical content that was correct in the first review remains correct. The corrected interference bound formula `r/sqrt(D)` is a reasonable scaling heuristic for the expected absolute cosine between random rank-r matrices. The numerical evaluation is arithmetically correct. The empirical measurement (cos ~ 0.0002) remains the primary evidence, with the formula serving as a theoretical reference point.

No new mathematical errors introduced by the revision.

## Novelty Assessment

Unchanged from previous review. This is a survey that synthesizes existing literature into a project-specific classification framework. The FIT scoring is internal tooling, not a generalizable contribution. The novel claim -- "the base could be expressed as a composable adapter" -- is correctly attributed as the project's extrapolation from ReLoRA, not as prior work's claim.

Prior art coverage is now adequate with the addition of He et al. 2022 and Lialin et al. 2023.

## Experimental Design

The fundamental nature has not changed: this is a literature review with analytical scoring, not an empirical experiment. The revision is transparent about this throughout. The status "supported" in HYPOTHESES.yml is slightly generous for a pure survey (the previous review suggested "documented"), but the evidence field accurately describes what was done. This is a minor bookkeeping preference, not a blocking issue.

## Hypothesis Graph Consistency

- Kill criteria match what the survey evaluates: YES
- Status "supported" with clear caveats about lack of empirical validation: ACCEPTABLE
- Downstream blockers correctly identified (exp_base_free_composition, exp_adapter_as_base): YES
- exp_relora_composition_test added as the empirical gate for the base-freedom path: YES

## Macro-Scale Risks (advisory)

Unchanged from previous review:

1. ReLoRA-as-base is the real bet and remains untested. The survey correctly identifies this and the new exp_relora_composition_test is the right next step.

2. The full-rank Class A caveat (now addressed) eliminates the previous risk of misinterpreting the taxonomy as endorsing full-rank multi-adapter composition.

3. Storage scaling at 5,000+ LoRA experts (300 GB at 60 MB/expert) may motivate investigating compressed variants (LoRA-XS, VeRA) earlier than planned.

## Verdict

**PROCEED**

All 4 required fixes have been properly implemented. All 3 advisory items have been addressed. The mathematical content is sound. The survey is transparent about its nature and limitations. The downstream path (exp_relora_composition_test) is correctly identified as the empirical gate for the base-freedom hypothesis.

This experiment serves its purpose as a structured design rationale document and literature foundation for the composable architecture. It correctly unblocks exp_base_free_composition and exp_adapter_as_base for further investigation, while clearly noting that empirical validation of ReLoRA-based composition is still required.
