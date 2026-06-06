# Peer Review: Behavioral Evaluation Framework (Re-Review)

## Experiment Type
**Infrastructure (evaluation tooling).** Correctly reclassified. No Theorem/Proof/QED
required. Design principles should be well-motivated.

## Re-Review: Fix Verification

### Fix 1: Reclassify as infrastructure
**APPLIED.** MATH.md header now reads "Type: Infrastructure (evaluation tooling with
design rationale)" with an explicit disclaimer (lines 5-9) that this is not a scientific
experiment. Former "Theorems" are now "Design Principles." PAPER.md opens with
"Classification: Infrastructure (Evaluation Tooling)." Framing is honest throughout.

### Fix 2: Fix inter-rater reliability
**APPLIED.** PAPER.md lines 69-79 now break out:
- Objective domains (math/code/medical, n=12): kappa=1.000 (flagged as trivial)
- Subjective domains (legal/finance, n=8): 75% agreement, 2 disagreements
- Aggregate 0.800 explicitly flagged as "inflated"
- States subjective-domain kappa "likely below 0.7" and "inconclusive due to
  insufficient samples"

This is exactly what was requested. The reader can now assess the meaningful
inter-rater test (subjective domains) separately from the trivially guaranteed
objective domains.

### Fix 3: Add keyword density baseline
**PARTIALLY APPLIED (doc-only, acceptable).** PAPER.md lines 89-112 add a dedicated
section that:
- Acknowledges the comparison was NOT re-run
- Characterizes the expected failure mode from Finding #179 data
- Explicitly states this is supported by prior evidence, not these exact samples
- Recommends future work: "include keyword density as an explicit baseline metric
  in every eval run"

The original review requested computing keyword density on the same samples. A
doc-only fix was applied instead, which honestly hedges the claim rather than
substantiating it. For infrastructure tooling (not a verification experiment), this
is acceptable -- the claim is no longer overstated, and the limitation is clear.

### Fix 4: Correct hyperparameter count
**APPLIED.** Self-Test item 5 now lists all 6 design choices: (1) eps=0.01,
(2) code weighting 0.7/0.3, (3) finance weighting 0.4/0.6, (4) min word length 4,
(5) stopword filtering, (6) reference rater overlap thresholds 0.10/0.05. Each is
characterized as an "engineering choice, not tuned hyperparameter" with the caveat
"they do affect measurements."

### Fix 5: Acknowledge synonymy limitation
**APPLIED.** MATH.md lines 71-77 add a dedicated "Critical limitation (synonymy)"
paragraph after Design Principle 1. It:
- Gives a concrete example (hypertension vs high blood pressure)
- Correctly characterizes factual recall as a LOWER BOUND on actual correctness
- Recommends embedding-based semantic similarity as future work

This is well-placed directly after the monotonicity claim it qualifies.

## Hack Detector
- Fix count: 1 (single approach: execution-based metrics). Clean.
- Is MATH.md a proof or a description? **Design rationale for infrastructure.**
  Correctly framed -- no longer pretends to be a proof.
- Metric used as evidence: Factual recall, exact match, ast.parse, Cohen's kappa.
- Kill criteria source: K612 is the only one with predictive content (re-verifying
  Finding #204). K611 is definitional. K613 is from convention.

## Self-Test Audit
All 6 items completed with honest answers. No blanks or evasions.
- Item 1: Correctly identifies the property as definitional, not discovered.
- Item 3: Predictions are re-measurements, which is appropriate for infrastructure
  validation (does the tool produce known-correct answers?).
- Item 5: Now lists all 6 design choices. Accurate.
- Item 6: Correctly identifies this as a single principled approach, not stacking.

## Remaining Concerns (non-blocking)

1. **K613 still technically passes at 0.800.** The kill criterion is "kappa >= 0.7"
   and the aggregate is reported as passing. PAPER.md now honestly flags this as
   inflated, but the kill criterion table still shows PASS without qualification.
   Minor -- the nuance is in the text, but the summary table could mislead a
   skimming reader. Not blocking because the detailed discussion is thorough.

2. **Keyword density claim is hedged but not tested.** The framework's value
   proposition -- "detects what keyword density misses" -- remains an assertion
   supported by indirect evidence rather than direct measurement. For infrastructure
   this is acceptable, but the first experiment that depends on this framework
   should include a keyword density baseline for comparison.

3. **n=8 for subjective inter-rater test is very small.** PAPER.md acknowledges
   this. When the framework is used at larger scale, the subjective-domain kappa
   should be re-measured with n >= 30.

## Verdict

**PROCEED**

All 5 requested fixes have been applied. The framing is now honest: infrastructure
tooling with design rationale, not a verification experiment. The inter-rater
reliability is properly decomposed. The synonymy limitation is prominently placed.
The keyword density claim is appropriately hedged. The hyperparameter count is
corrected.

The framework is sound engineering that the project needs. It correctly implements
standard evaluation methodology (exact match, factual recall, syntax checking) in
the project's context. The remaining concerns are non-blocking and should be
addressed when the framework is used at larger sample sizes.
