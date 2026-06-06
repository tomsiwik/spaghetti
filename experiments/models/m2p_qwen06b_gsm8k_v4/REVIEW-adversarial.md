# Peer Review: m2p_qwen06b_gsm8k_v4 (RE-REVIEW after REVISE)

## Experiment Type
Guided exploration (Type 2) -- correctly self-identified.

Proven framework: Theorem 5 from v3 (functional tensor argument flow guarantees
non-zero gradients). Unknown: quality_ratio at n=500 after 1000 steps. Legitimate
Type 2 exploration within a verified framework.

## Hack Detector
- Fix count: 0 new mechanisms vs v3. Pure compute-budget extension. CLEAN.
- Is MATH.md a proof or a description? Informal convergence/statistics reasoning
  within a proven framework (Theorem 5). Appropriate for Type 2 -- the proof
  obligation was on v3, not v4.
- Metric used as evidence: quality_ratio + two-proportion z-tests. Both are
  standard statistical tools, correctly applied.
- Kill criteria source: K916/K917 derive from Theorem 5 and Adam convergence.
  K918 derives from quality_ratio algebra in MATH.md Theorem 9. Reasonable.

## Self-Test Audit

1. One-sentence impossibility property: "functional tensor argument flow makes
   gradients non-zero with probability 1." Single property, clearly stated. PASS.

2. Cited theorems:
   - Theorem 5 (v3 MATH.md): Real, verified. Applicable. PASS.
   - Kingma & Ba (arXiv:1412.6980) Theorem 4.1: Real. Applied outside
     preconditions (non-convex loss), but labeled "informal." NOT BLOCKING.
   - Wilson (1927): Real. Correctly applied. PASS.
   - Cobbe et al. (arXiv:2110.14168): Real. Cited for protocol, not theorem. PASS.

3. Predicted numbers: All 5 predictions are specific and falsifiable. PASS.

4. Falsification condition: "K916 FAIL would mean Theorem 5 has a bug." Targets
   the proof, not just the experiment. PASS.

5. Hyperparameter count: 0 new vs v3. PASS.

6. Hack check: No fix-stacking. PASS.

Self-Test: COMPLETE, no blanks, no evasions.

## Previous REVISE Fixes -- Verification

### Fix 1: "M2P exceeds SFT" claim must be qualified (p=0.36)

**APPLIED.**

Section 3 (K918 result, line 89): "M2P point estimate (28.6%) exceeded SFT
(26.0%) by 2.6pp absolute at n=500, but this difference is not statistically
significant (p=0.36, two-proportion z-test -- see Section 4)."

Section 6 (Comparison vs v3, lines 177-180): "The point estimate nominally crosses
the SFT baseline of 26.0%), but this difference is not statistically significant
(p=0.36)." Properly qualified with redirect to M2P-vs-base significance.

Section 7 (Conclusion, lines 200-207): "The M2P point estimate (28.6%) nominally
exceeds SFT (26.0%), but this difference is NOT statistically significant (p=0.36,
two-proportion z-test)." Clear, unambiguous.

### Fix 2: SFT baseline uncertainty in quality_ratio CI

**APPLIED.**

Section 4 (lines 122-129): Full paragraph acknowledging SFT Wilson CI [0.204, 0.323],
denominator uncertainty, delta method would widen CI, CI_lower=0.773 labeled
"optimistic," Fieller's method suggested. Thorough and honest.

### Fix 3: Finding text must not overstate

**APPLIED.**

Section 7 (lines 204-207): "The correct conclusion: M2P training works (gradient
flow confirmed), M2P accuracy significantly exceeds base, and M2P accuracy is
comparable to SFT within binomial noise at current sample sizes." Nearly verbatim
the requested language.

Experiment status is "SUPPORTED" throughout. No "BREAKTHROUGH" or "conclusive" claims.

## Mathematical Soundness

The mathematical content is unchanged from the previous review. Summary:

- Theorem 7 (convergence): Informal, correctly labeled. The "monotone improvement"
  phrase (MATH.md line 52) is technically incorrect (Adam does not guarantee this),
  and the observed loss curve confirms non-monotonicity. This was flagged in the
  previous review as non-blocking and remains so.

- Theorem 8 (Wilson CI): Correctly stated and implemented. Worked example in
  MATH.md Section F is algebraically correct.

- Theorem 9 (K918 sufficient conditions): Algebraically correct. The self-defeating
  analysis showing CI_lower >= 0.60 requires quality_ratio >> 1.0 is honest math.

- K918 CI_lower = 0.773: This result was a genuine surprise (MATH.md predicted
  0.10-0.40). The surprise came from quality_ratio = 1.43 >> 0.83 (v3). The math
  explaining why CI_lower exceeded 0.60 is correct post-hoc.

## Prediction vs Measurement

PAPER.md Section 2 contains the table. Six predictions, five matched, one surprise
(CI_lower much higher than predicted due to unexpectedly high quality_ratio). The
surprise is honestly reported and explained. PASS.

## New Issues Found in RE-REVIEW

### Advisory 1 (NON-BLOCKING): Two-proportion z-test uses wrong n for SFT

PAPER.md Section 4, line 133:
```
z = (0.286 - 0.260) / sqrt(0.286*0.714/500 + 0.260*0.740/500)
```

Both terms use n=500, but SFT was measured at n=200. The correct formula uses
n_SFT=200 in the SFT variance term:
```
z = 0.026 / sqrt(0.000408 + 0.000962) = 0.026 / 0.0370 = 0.702 (p ~ 0.48)
```

The reported z=0.923, p=0.36 overstates significance slightly. The true p-value
is ~0.48, even LESS significant. This error is conservative (works against the
paper's interest), so it does not invalidate the conclusion. However, it should
be corrected for accuracy.

### Advisory 2 (NON-BLOCKING): M2P-vs-base z-test treats base as known constant

Line 140: `z = (0.286 - 0.200) / 0.02018 = 4.26 (p < 0.0001)` uses only M2P
variance, treating base_acc=0.200 as a known constant. MATH.md explicitly adopts
this assumption (line 186: "carried forward as known constants"). If base
uncertainty (n=200) is included: z ~ 2.47, p ~ 0.014. The conclusion (M2P
significantly exceeds base) holds at p=0.014, but "p<0.0001" is overstated by
~2 orders of magnitude.

This is internally consistent with MATH.md's stated assumptions and the Fix 2
caveat already warns about denominator uncertainty. Non-blocking because the
qualitative conclusion is unchanged.

### Advisory 3 (NON-BLOCKING): Typo in Section 6

Line 177: "crosses the SFT baseline of 26.0%)" has a stray closing parenthesis
with no matching opener.

### Advisory 4 (NON-BLOCKING, carried from previous review): Parameter asymmetry

M2P uses ~357M learnable params vs SFT's ~100K. PAPER.md does not contextualize
this 1000x parameter advantage. The comparison "M2P matches SFT" understates what
SFT achieves per parameter. This is relevant for the broader M2P narrative (the
deployment-time argument) but does not affect the experiment's kill criteria.

## Novelty Assessment

Unchanged from previous review. v4 adds no novelty over v3; it is a compute-budget
extension for statistical closure. The finding that M2P can match SFT quality on
GSM8K is useful for the project roadmap but not novel research.

## Macro-Scale Risks (advisory)

1. M2P parameter scaling: 357M for a 0.6B base is disproportionate. Needs redesign
   for larger base models.
2. SFT baseline fidelity: Re-measure SFT at n=500 before making macro claims.
3. Training efficiency: 1000 steps of M2P training vs SFT training -- deployment-time
   advantage needs latency measurements to justify the training cost.

## Verdict

**PROCEED**

All three blocking fixes from the previous REVISE have been applied correctly:

1. "M2P exceeds SFT" claims are now properly qualified with p=0.36 in all relevant
   sections (3, 6, and 7). The conclusion correctly centers on M2P-vs-base significance.

2. SFT baseline uncertainty is thoroughly acknowledged in Section 4 with explicit
   Wilson CI, delta method warning, and "optimistic" label on CI_lower.

3. Finding text uses the correct framing: "M2P accuracy significantly exceeds base,
   and M2P accuracy is comparable to SFT within binomial noise at current sample sizes."

The experiment status is correctly "SUPPORTED." The core result is sound: M2P
learns from GSM8K data, significantly exceeds base accuracy (z=4.26 under the
known-constant assumption, z~2.47 with full uncertainty; either way p<0.05), and
retains at least 77% of SFT's improvement (CI_lower=0.773, noting this is optimistic
per the SFT uncertainty caveat).

The four new advisories are non-blocking and can be addressed in future work or
a follow-up cleanup pass.
