# Peer Review: FFN-only vs All-Modules LoRA Composition (Re-Review)

## Previous Review Summary

The initial review identified 5 issues. This re-review assesses whether the revisions adequately address them.

## Fix Verification

### Fix 1: Acknowledge retroactive-subset confound
**Status: ADEQUATELY ADDRESSED.**

PAPER.md Limitation #1 now explicitly states: "The 'FFN-only' measurements are the FFN subset of jointly-trained adapters, not adapters trained with only FFN targets." It explains three specific ways independently trained FFN-only adapters may differ (compensation for absent attention, changed optimization landscape, different gradient flow). The limitation is no longer buried -- it appears prominently in the Limitations section and is referenced in the Kill Threshold Checks and Verdict sections.

### Fix 2: Report median and outlier-excluded results as co-equal
**Status: ADEQUATELY ADDRESSED.**

The revised PAPER.md presents three views of the data:
- Mean with outlier (0.0605 vs 0.0711) -- still reported but explicitly flagged as outlier-driven
- Mean without outlier (0.0017 vs 0.0009) -- honestly shows FFN is LESS orthogonal
- Median (0.0017 vs 0.0010) -- confirms the without-outlier story

The paper correctly reframes the insight: "The critical insight is not about average orthogonality but about tail behavior: when domains genuinely overlap, attention amplifies the correlation far more than FFN (0.85 vs 0.59)." This is a more defensible and actually more interesting claim than the original.

### Fix 3: Downgrade raw-parameter cosine to approximate proxy
**Status: ADEQUATELY ADDRESSED.**

MATH.md Assumption 2 now reads "Cosine similarity of raw parameters is an approximate proxy for delta cosine" and includes a concrete counterexample (A1=A2, B1 orthogonal to B2 yields raw cosine ~0.5 but arbitrary delta cosine). It states the directional claim "likely holds" but is "an assumption, not a proven fact." The limitation is also restated in PAPER.md Limitation #3.

### Fix 4: Reconcile N_max formulas
**Status: ADEQUATELY ADDRESSED.**

MATH.md Section 7.1 now contains an explicit note distinguishing D/r^2 (full flattened delta, upper bound, used in this paper) from d^2/r^2 (per-layer per-module, more conservative, used in VISION.md). States the per-layer formula is "more conservative and practically relevant" and recommends it for architectural decisions.

### Fix 5: Change status from "proven" to "supported"
**Status: ADEQUATELY ADDRESSED.**

HYPOTHESES.yml shows `status: supported`. PAPER.md verdict reads "SUPPORTED (not proven)."

## Mathematical Soundness (Re-Check)

All parameter counting, dimensionality analysis, and expected cosine calculations verified in the initial review remain correct. No new mathematical claims were introduced in the revision.

Minor data note: results.json reports ffn_delta_dim=5,703,204,864 while MATH.md calculates 5,702,452,224 (a 0.01% difference attributed to "dimension ordering conventions" in the code). This does not affect any conclusions.

## New Issues Introduced by Revision

**None identified.** The revisions are conservative -- they add caveats and nuance without introducing new claims or changing the data analysis.

## Remaining Known Limitations (Not Blocking)

These were identified in the original review and are now properly acknowledged in the paper:

1. **Retroactive subset analysis** -- acknowledged in Limitations #1, flagged as needing macro validation
2. **No quality comparison at matched rank** -- acknowledged in Limitations #2 and Kill Threshold Checks
3. **Raw parameter vs expanded delta cosine** -- acknowledged in Limitations #3 and MATH.md Assumption 2
4. **N=5 domains** -- acknowledged in Limitations #4
5. **No composition quality measurement** -- acknowledged in Limitations #6

These are all appropriate scope limitations for a micro experiment. None are blocking.

## HYPOTHESES.yml Consistency

One minor observation: the evidence entries in HYPOTHESES.yml still lead with "FFN-only mean |cos|=0.0605 < all-modules mean |cos|=0.0711" without noting this is outlier-driven. The paper itself handles this honestly. The HYPOTHESES.yml entries could be updated for consistency but this is not blocking.

## Macro-Scale Risks (advisory, unchanged from initial review)

1. FFN-only may underperform on code tasks requiring domain-specific attention patterns
2. The co-adaptation confound may matter more at scale with independently trained FFN-only adapters
3. Composition quality at cos~0.06 is untested (prior validation was at cos~0.0002)

## Verdict

**PROCEED**

All 5 issues from the initial review have been adequately addressed. The revised paper presents a more honest, nuanced, and ultimately more interesting claim: FFN-only's advantage is not about average orthogonality (where attention actually wins) but about tail behavior -- when domains overlap, attention amplifies correlation far more than FFN (0.85 vs 0.59), making it the dominant risk factor for composition safety. The limitations are properly scoped and the status is correctly set to "supported" rather than "proven."

The experiment provides sufficient directional evidence to inform the architectural decision (default to FFN-only targets) while correctly identifying that macro validation with matched-rank FFN-only training is required before the decision is final.
