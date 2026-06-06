# Peer Review: m2p_sft_n500_baseline

## Experiment Type

Verification (Type 1). MATH.md contains two Theorem/Proof/QED blocks building on
established statistical theory (Wilson 1927, Newcombe 1998, Casella & Berger 2002).
The experiment measures SFT accuracy at n=500 and verifies the proof's predictions
about CI calibration.

## Hack Detector

- Fix count: 0. This is a pure measurement experiment. No mechanisms, losses, or tricks.
- Is MATH.md a proof or a description? **Proof with QED.** Theorem 1 proves calibration
  of Wilson CI; Theorem 2 proves CI_lower=0.773 is upward-biased when denominator
  variance is propagated. Both are genuine derivations, not mechanism descriptions.
- Metric used as evidence: quality_ratio CI bounds and two-proportion z-test p-value.
  These are standard frequentist statistics, not proxy metrics.
- Kill criteria source: K919 and K921 are unconditional measurements (derived from
  "what must be computed"). K920 threshold (p<0.05) is conventional statistical
  significance. All derived from the proof framework.

## Self-Test Audit

1. **One-sentence impossibility property:** Correct. Identifies the specific omission
   (Var(SFT_acc) in the ratio denominator) as the root cause of bias. Single property,
   not a list. PASS.

2. **Cited theorems:** Wilson (1927), Newcombe (1998), Casella & Berger (2002 section 5.5.4),
   Brown/Cai/DasGupta (2001). All real, all correctly cited. Conditions checked:
   - Wilson CI: n=500, binomial, p not extreme -- conditions met.
   - Two-proportion z-test: both n*p >= 5, n*(1-p) >= 5 -- conditions met (500*0.3=150).
   - Delta method: requires Var(Y) small relative to (E[Y]-base)^2 -- at n=500, 
     Var(SFT) ~ 0.00043 vs (SFT-base)^2 = 0.013; ratio ~0.033, so delta method is
     in its valid regime.
   PASS.

3. **Predicted numbers:** Three quantitative predictions: SFT accuracy ~0.24-0.30,
   p-value ~0.35, Fieller CI_lower ~0.30 (if SFT=0.26). These are specific and
   falsifiable. PASS.

4. **Falsification condition:** If SFT accuracy falls below base (0.200), the ratio
   denominator goes to zero and the framework breaks. This targets the proof, not just
   the experiment. PASS.

5. **Hyperparameter count:** Zero. All parameters fixed by prior experiments or
   statistical convention. PASS.

6. **Hack check:** Not adding a fix. This is a measurement. PASS.

All six self-test items answered correctly.

## Mathematical Soundness

### Theorem 1 (Wilson CI computation)

The Wilson formula is correctly transcribed from Wilson (1927). The worked example
in section F computes hat_p=0.260 at n=500 and derives CI=[0.224, 0.300]. I verified
the arithmetic step-by-step; it is correct.

The actual computation in run_experiment.py (wilson_ci function, lines 146-158)
correctly implements the formula. The results.json values [0.274867, 0.355969]
for the actual k=157, n=500 case are numerically correct (verified independently).

### Theorem 2 (CI_lower bias)

The proof structure is sound: it shows that the correct variance has two additive
terms (numerator and denominator uncertainty), and the v4 calculation omitted the
second term, making CI_lower necessarily too high.

**Minor issue in the proof exposition (not a mathematical error):** Lines 181-185
contain an intermediate calculation that produces Term2=110.4 (with n_sft=200),
which the author catches and corrects ("Wait -- this gives se_true >> se_v4").
The corrected calculation on lines 196-201 is correct. The "Wait" moment is
unusual in a proof document -- it would be cleaner to present only the correct
derivation. However, this is an expository issue, not a mathematical one. The
final result (CI_lower drops from 0.773 to ~0.302 at SFT=0.26) is correctly
derived.

**The corrected numerics at SFT=0.26 (lines 196-201):**
- Term1 = 0.000408 / 0.0036 = 0.1135 -- correct
- Term2 = 0.000385 * 0.007396 / 1.296e-5 = 0.2197 -- correct (verified: 2.847e-6 / 1.296e-5 = 0.2197)
- se_true = sqrt(0.3332) = 0.577 -- correct
- CI_lower = 1.433 - 1.96*0.577 = 0.302 -- correct

### Delta method application (Theorem B3)

The delta method is applied correctly. The covariance term is correctly set to
zero (M2P and SFT evaluated on independent model instances). The partial
derivatives are correctly computed:
- df/dX = 1/(Y-c) -- correct for the numerator
- df/dY = -(X-c)/(Y-c)^2 -- correct for the denominator
- These produce the two-term variance formula as stated. Verified.

### Fieller CI implementation (run_experiment.py)

The fieller_quality_ratio_ci function (lines 181-216) correctly implements:
- var_m2p = p*(1-p)/n for binomial variance
- var_sft = p*(1-p)/n for binomial variance
- term1 = var_m2p / denom^2
- term2 = var_sft * numer^2 / denom^4
- se = sqrt(term1 + term2)
- CI = ratio +/- z*se

This matches the delta-method formula derived in MATH.md. No code-math mismatch.

### Two-proportion z-test implementation

The two_proportion_ztest function (lines 161-178) uses math.erfc for the p-value:
p_value = erfc(|z| / sqrt(2)). This is mathematically equivalent to
2*(1 - Phi(|z|)) for the standard normal CDF. Verified correct.

### Numerical verification of results.json

I independently computed all key outputs:
- Wilson CI [0.2749, 0.3560] for k=157, n=500: MATCHES
- z = -0.966, p = 0.334: MATCHES
- quality_ratio = 0.7544, CI = [0.3148, 1.1939]: MATCHES
- term1 = 0.0314, term2 = 0.0189: MATCHES
- se = 0.2243: MATCHES

No numerical errors found anywhere.

## Prediction vs Measurement

PAPER.md section 2 contains the prediction-vs-measurement table. Assessment:

| Prediction | Match? | Notes |
|---|---|---|
| SFT ~0.24-0.30 | Slightly outside range (0.314) | Directionally correct; 0.314 is within v2's prior Wilson CI [0.204, 0.323] |
| p-value ~0.35 | YES (0.334) | Excellent match |
| Fieller CI_lower ~0.30 | CONDITIONAL MATCH | Prediction was conditional on SFT=0.26; actual SFT=0.314 changes the input. At SFT=0.314, the prediction P3 table gives delta=0.114, which is between the 0.28 and 0.30 scenarios. The measured CI_lower=0.315 is consistent. |
| CI_lower bias ~0.47 | NO (0.092) | Honestly reported. The bias calculation assumed SFT=0.26; actual SFT=0.314 made the denominator larger and the bias smaller. |

The PAPER.md honestly identifies the discrepancy: the large predicted bias (0.47) was
conditional on SFT=0.26. The actual SFT=0.314 makes delta=0.114 instead of 0.060,
which shrinks the denominator variance term dramatically. This is not a failure of the
proof -- the proof correctly showed that CI_lower is monotonically sensitive to SFT,
and the predictions in section D gave the SFT=0.28 and SFT=0.24 scenarios as well.
The paper correctly attributes the dominant source of v4's optimism: the n=200 SFT
point estimate was too low (26.0% vs true 31.4%), not just the missing variance term.

## NotebookLM Findings

Skipped -- not running NotebookLM for this review as the mathematical framework is
sufficiently tractable for manual verification.

## Novelty Assessment

This is not a novel contribution -- it is a measurement experiment applying standard
statistics (Wilson CI, two-proportion z-test, delta method) to correct a known bias
in a prior experiment's reporting. No novelty claim is made, and none is needed.

The value is in the honest calibration: v4's quality_ratio=1.433 and CI_lower=0.773
are replaced with quality_ratio=0.754 and Fieller CI_lower=0.315. This is the kind
of statistical rigor that should have been in v4 from the start, and it is commendable
that the researchers chose to run it explicitly rather than hand-waving.

## Additional Issues Found

### 1. base_acc treatment is inconsistent with the proof's own logic

MATH.md correctly identifies that treating SFT_acc as a known constant biases CI_lower
upward. By the same logic, base_acc=0.200 (measured at n=200) should also not be
treated as a known constant. The delta method for a ratio (M2P-base)/(SFT-base) with
three uncertain quantities has three variance terms, not two. The third term
(Var(base_acc)) contributes to both numerator and denominator uncertainty.

At n_base=200, Var(base_acc) = 0.200*0.800/200 = 0.0008. This is non-negligible.
The Wilson CI for base_acc at n=200 is approximately [0.150, 0.259].

However, MATH.md Assumption A3 explicitly states "base_acc = 0.200 is fixed (not
remeasured)" and acknowledges the consequence: "If base accuracy differs,
quality_ratio denominators shift." This is acknowledged but not addressed.

**Severity: LOW.** The base_acc uncertainty affects both numerator and denominator
in correlated ways (both M2P and SFT are measured relative to the same base). A full
three-variable delta method would produce wider CIs, but the qualitative conclusion
(M2P and SFT are indistinguishable) would not change. The finding status "supported"
is appropriate given this caveat.

### 2. M2P and SFT may not be evaluated on the same 500 test examples

The code uses SEED=42 to shuffle the GSM8K test set and takes the first 500 examples.
This matches the claim. However, M2P's accuracy (0.286, 143/500) is taken from v4's
results.json as a fixed constant -- it was measured in a separate experiment run. If v4
used a different data loading path, dataset version, or shuffling implementation, the
500 examples may differ.

**Severity: LOW.** Assumption A1 explicitly addresses this. The code reuses the
identical SEED, dataset, and shuffling logic. The risk is if the datasets library
version changed between v4 and this experiment, altering the test split order. This
is a reproducibility concern, not a fundamental flaw.

### 3. The z-test uses pooled proportion, but M2P accuracy is not from this run

The two-proportion z-test assumes two independent samples from the same population
under H0. The SFT sample (n=500) is from this run, but the M2P sample (n=500) is
from a prior run. The pooled proportion p_pool=(143+157)/1000=0.300 treats them as
two independent draws, which is correct only if the same test set was used. Per issue
2 above, this is acknowledged and likely valid, but not verified within this experiment.

**Severity: LOW.** Same as above.

### 4. PAPER.md section 3b row 2 clarity

The table shows "v4 method (SFT as constant, actual 31.4%)" with CI_lower=0.407.
This is the v4 formula (ignoring SFT variance) but using the new SFT=0.314 point
estimate. This is a useful comparison, but the row label could be clearer -- it is
a hypothetical (what v4's formula would produce with the corrected SFT estimate),
not a measurement from v4.

**Severity: COSMETIC.**

## Macro-Scale Risks (advisory)

1. At macro scale, the quality_ratio < 1 finding (M2P slightly below SFT) may persist
   or worsen. The M2P value proposition is composability, not single-domain quality.
   Future macro experiments should set expectations accordingly.

2. The delta method becomes exact as n grows (by CLT). At macro n (e.g., n=5000),
   variance terms shrink and the z-test gains power. The current p=0.334 is consistent
   with either no real gap or a small real gap masked by noise. Macro experiments
   will resolve this.

## Verdict

**PROCEED**

Justification:

1. Mathematical framework is sound. Both theorems are correctly stated, correctly
   proved, and correctly implemented in code. No numerical errors found.

2. Self-test is complete and honest. All six items answered correctly.

3. Prediction-vs-measurement table is present and honest. The main discrepancy
   (SFT higher than predicted) is correctly identified and explained. The dominant
   source of v4's optimism is clearly attributed: the n=200 SFT point estimate was
   too low, not just the missing variance term.

4. The qualitative conclusion is appropriately modest: M2P and SFT are statistically
   indistinguishable at n=500. The v4 "breakthrough" (quality_ratio=1.433) was noise.
   This is exactly the kind of honest self-correction that builds credibility.

5. Finding status "supported" is appropriate: the measurement was made correctly,
   the statistical framework is standard and verified, and the conclusion (no
   significant M2P-SFT gap) is well-supported.

6. The MATH.md "Wait" passage (lines 181-188) where the author catches their own
   arithmetic error mid-derivation is mildly unprofessional in a formal document
   but does not affect correctness. A minor cleanup would be nice but is not blocking.

7. The base_acc=0.200 treatment as fixed (not propagating its uncertainty) is a
   known limitation acknowledged in assumptions. It does not change the qualitative
   conclusion.

No blocking issues found.
