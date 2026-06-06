# Peer Review: Knowledge Region Overlap (Revision 2)

## Experiment Type
Guided exploration (Type 2)

## Prior Review Fix Verification

The prior adversarial review required 5 specific fixes. Status of each:

| Fix | Required | Status | Evidence |
|-----|----------|--------|----------|
| 1. K1 reclassified as INCONCLUSIVE | K1 was vacuously PASS | ADDRESSED | PAPER.md table row: "INCONCLUSIVE"; Kill Criteria section: "K1 cannot be evaluated with the original definition because the PPL improvement threshold is too weak for ternary base models" |
| 2. K2 recorded as FAIL | Post-hoc L2 redefinition not allowed | ADDRESSED | PAPER.md table row: "FAIL"; Kill Criteria section: "The pre-registered kill criterion is not met. The post-hoc observation... does NOT satisfy the original K2 definition" |
| 3. Finance confound bounded | Finance at 1/20th scale | ADDRESSED | PAPER.md has explicit "IMPORTANT: Finance adapter confound" callout, excludes finance from structural conclusions, repeats in Limitations |
| 4. Sheaf language trimmed | Framework never informatively tested | ADDRESSED | MATH.md Section C adds note: "The sheaf-theoretic application below was not successfully tested"; PAPER.md adds "Status of Sheaf-Theoretic Analysis" section stating the analysis could not be performed |
| 5. Status downgraded to PROVISIONAL | Original predictions all failed | ADDRESSED | PAPER.md verdict: "PROVISIONAL. The original hypothesis is REFUTED as stated -- all pre-registered predictions failed" |

**All 5 fixes are substantively addressed.** The revisions are not cosmetic -- they change the actual assessments and conclusions.

## Hack Detector
- Fix count: 0 (measurement experiment, no mechanisms/losses/tricks)
- Is MATH.md a proof or a description? Description of a framework with the unknown precisely identified. Acceptable for Type 2 guided exploration.
- Metric used as evidence: Specialization sets (argmin PPL), L2 relative difference, PPL disagreement ratios. PPL improvement acknowledged as miscalibrated instrument.
- Kill criteria source: Derived from sheaf framework prerequisites (non-trivial cover, variable compatibility). K1 INCONCLUSIVE, K2 FAIL -- both honestly reported.

## Self-Test Audit

1. **One-sentence impossibility property:** States this is a measurement experiment, not a fix. Appropriate for guided exploration. PASS.
2. **Cited theorems:** Nerve theorem (Leray/Borsuk), sheaf cohomology (Hansen & Ghrist). Real theorems, real references. The gap between continuous topology and discrete point sets is acknowledged. PASS.
3. **Predicted numbers:** Specific ranges for overlap sizes, cosine, std. All turned out wrong -- but they were falsifiable, which is the point. PASS.
4. **Falsification condition:** Targets the framework (contractibility condition). Could be sharper but acceptable. PASS.
5. **Hyperparameter count:** 1 (layer 15). Correctly identified. PASS.
6. **Hack check:** "No. Measurement experiment." Correct. PASS.

## Mathematical Soundness

The MATH.md is well-structured for a Type 2 exploration. The sheaf-theoretic framework is correctly described at a conceptual level, and the revision adds an honest caveat that the framework was not successfully tested (Section C, lines 74-79).

**Remaining concern (non-blocking):** The restriction maps in Section C are still described in a way that collapses to pointwise comparison (h_i(x) = h_j(x)), which means the "sheaf" apparatus is doing no more analytical work than pairwise similarity measurement. This was noted in the prior review and is acknowledged by the "FRAMEWORK FOR FUTURE WORK" label. Acceptable.

**The guided-exploration framing is correct.** The proven framework is sheaf theory on covers; the unknown is whether domain adapters induce a cover satisfying the framework's prerequisites. The experiment measured this and found the prerequisites are NOT met with the original definitions (improvement sets are degenerate, cosine is saturated).

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table (lines 12-22). Every prediction was wrong, and this is stated without evasion. The table correctly records K1 as INCONCLUSIVE and K2 as FAIL.

The secondary analysis (specialization sets, L2 norms, PPL disagreement) is clearly labeled as post-hoc and not pre-registered. The paper does not claim the secondary analysis satisfies the original kill criteria. This is the right framing.

## New Issue: Legal Adapter Scale Discrepancy

**results.json shows legal at scale=4.0, not scale=20.0.** PAPER.md line 75 states: "The finance adapter operates at scale=1.0 while all others are at scale=20.0." This is factually incorrect -- the domain_scales in results.json are:

- medical: 20.0
- code: 20.0
- math: 20.0
- legal: 4.0
- finance: 1.0

Legal is at 1/5th the scale of the top three adapters. This matters because:
1. Legal's "partial specialization" (34/50 own-domain, reported as genuine weakness) could be partially a scale artifact, similar to finance.
2. The paper treats only finance as confounded, but legal should carry a similar (though less severe) caveat.
3. The "3 strong specialists + 1 partial specialist + 1 dominated" narrative in the Key Findings may overstate the structural interpretation -- it could be "3 high-scale specialists + 2 lower-scale adapters."

This is a factual error that needs correction. It does not invalidate the experiment but changes the interpretation of legal adapter performance.

## Novelty Assessment

The genuine novel findings are:
1. PPL improvement sets are degenerate for ternary base models (any adapter helps)
2. Specialization sets (argmin) are the correct cover definition
3. Cosine similarity saturates at layer 15 for LoRA on BitNet-2B; L2 norm is informative
4. Adapter landscape structure (strong/partial/dominated specialization)

These are useful empirical observations for the project's sheaf-theoretic roadmap. They do not require sheaf theory (as the prior review noted), but they correctly inform what a future sheaf experiment should measure.

## Macro-Scale Risks (advisory)

1. At FP16 scale, improvement sets may not be degenerate. The "specialization sets" finding may be ternary-specific.
2. With equal adapter scales, the landscape structure will likely change substantially.
3. Layer 15 extraction is a single-point measurement; multi-layer analysis may reveal richer structure.

## Verdict

**PROCEED** (with one minor fix)

All 5 required fixes from the prior review are substantively addressed. The paper is now intellectually honest about its failures: K1 INCONCLUSIVE, K2 FAIL, status PROVISIONAL, sheaf framework not tested, finance confound bounded. The secondary findings (specialization sets, L2 norms, PPL disagreement) are genuinely useful for the project roadmap and correctly labeled as post-hoc.

**One minor fix required before recording the finding:**

1. **Correct the legal adapter scale claim.** results.json shows legal at scale=4.0, not 20.0. PAPER.md line 75 ("all others are at scale=20.0") is factually wrong. Update to acknowledge that legal is also at a reduced scale (4.0 vs 20.0), and add a caveat that legal's partial specialization (34/50) may be partially a scale artifact. This does not require re-running the experiment -- just correcting the text and softening the "partial specialist" conclusion for legal.

This is a minor documentation fix, not a structural revision. The experiment's conclusions and PROVISIONAL status are appropriate.
