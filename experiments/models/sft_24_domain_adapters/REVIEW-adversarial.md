# Adversarial Review -- exp_sft_24_domain_adapters (Post-Revision)

## Verdict: PROCEED

All three blocking issues from the original review have been correctly addressed.
The experiment is a clean guided-exploration that narrows a well-defined unknown
(SFT recipe generalization from N=5 to N=24) within a proven framework
(Grassmannian orthogonality). Results are strong (24/24 converge, mean 17.3%
improvement). Status of SUPPORTED is warranted.

## Fix Verification

### Fix #1: False QED on non-proof -- CORRECTLY APPLIED

MATH.md line 3 now reads "Experiment Type: Guided Exploration" (was: verification).
The QED that previously appeared at line 104 has been removed. Proposition 1 is
now framed as an "empirical scaling argument" with an "*Argument.*" header instead
of "*Proof.*" The only occurrence of "proof" in the entire MATH.md is the
disclaimer: "This is not a formal convergence theorem but an empirical scaling
argument, not a formal convergence proof" (line 108). This is honest and appropriate.

### Fix #2: K752 FAIL reported honestly -- CORRECTLY APPLIED

PAPER.md lines 109-125 now clearly report K752 as **FAIL** (line 111). The
explanation of why the automated check is flawed (comparing val loss against
first training sample) is present and technically correct. Crucially, PAPER.md
includes a process note (lines 124-125) explicitly acknowledging that
retroactively redefining a failed kill criterion is a process violation. This
is the right way to handle a flawed metric: report the FAIL, explain the flaw,
provide a separate convergence assessment, but do not override the coded result.

current_direction.md line 13 now accurately reads: "K752 FAIL (code check
flawed: compared val vs first train sample; all 24 domains improve over base)".
This matches results.json where `K752.pass: false`.

### Fix #3: Self-Test #1 vacuous impossibility -- CORRECTLY APPLIED

Self-Test #1 (MATH.md lines 193-197) now reads: "No impossibility property
claimed. This is a guided-exploration experiment operating within the proven
Grassmannian orthogonality framework (Finding #54)." It further acknowledges
that non-zero gradient is "necessary but not sufficient for convergence -- it
does not prevent oscillation, saddle points, or capacity saturation." This is
a significant improvement over the original evasive claim and is fully honest.

Self-Test #2 was also improved: Finding #206 and LIMA are now honestly labeled
as "empirical findings and hypotheses, not formal convergence theorems" rather
than being mischaracterized as theorems. This addresses a secondary concern
from the original review.

## New Issues (if any)

No new blocking issues introduced by the revision.

One minor observation: MATH.md Proposition 1 (lines 80-110) still carries the
"Proposition" label despite being explicitly disclaimed as not a formal result.
This is cosmetic -- the disclaimer at line 93 and the "*Argument.*" framing make
the status clear, and "Proposition" is acceptable in mathematics for conjectured
or partially-supported claims. Not blocking.

## Notes

1. **K752 code fix for future experiments:** The flawed K752 check
   (`final_val_loss > 2 * initial_train_loss`) remains in the code but is now
   honestly reported. For any future experiment reusing this training script,
   the check should be fixed to compare `final_val_loss > base_val_loss` (same
   distribution) before running.

2. **Advisory #4 (B-matrix inter-cosine) remains open:** Format dominance
   (Finding #216) was not measured in this experiment. This is correctly deferred
   to the composition experiment (exp_n24_composition_proof), where it matters
   for routing. Not blocking for this convergence-focused experiment.

3. **Strong empirical result:** 24/24 domains converging with the same recipe
   (0 new hyperparameters, no per-domain tuning) is a robust guided-exploration
   outcome. The predicted range of 22-24 was confirmed at the upper bound,
   with the weakest domain (finance, 5.9%) exceeding the predicted floor (3%).
   All five quantitative predictions (P1-P5) match measurements.

4. **Process integrity:** The revision demonstrates good research hygiene. The
   researcher accepted the critique that K752 failed as coded, reported it
   honestly rather than retroactively redefining success, and correctly
   reclassified the experiment type to match its actual mathematical content.
   This is what revision is supposed to look like.
