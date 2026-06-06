# Peer Review: Per-Domain Module Selection (RE-REVIEW)

## Re-Review Context

This is a re-review following 5 revision requests. Each fix is verified below.

## Experiment Type
Guided exploration (Type 2). Framework: module separability (Finding #300). Unknown: optimal module partition per domain. Type is correctly identified and appropriate.

## Hack Detector
- Fix count: 1 (per-domain module bitmask). No stacking. CLEAN.
- Is MATH.md a proof or a description? Description with worked numerical analysis. Acceptable for Type 2 (guided exploration requires a proven framework + unknown, not a new theorem).
- Metric used as evidence: PPL (established proxy), behavioral recall (crude word overlap, adequate for directional signal), MMLU accuracy (n=15 per domain, correctly marked as suggestive only).
- Kill criteria source: K766 derived from Finding #263. K767 pragmatic threshold. K768 from separability proof. Mixed provenance, adequately justified.

## Self-Test Audit

1. **Impossibility property:** "Removing MLP modules makes it structurally impossible for the adapter to perturb MLP weights." The tautological part (not applying MLP deltas = no MLP perturbation) is trivially true. The useful claim (this preserves knowledge) is empirical. Previous review flagged this as PARTIAL. Still PARTIAL but acceptable for guided exploration -- the experiment is discovering whether this tautological structural property produces the behavioral outcome.

2. **Cited theorems:** Frobenius norm analysis (used), module separability Finding #300 (used), LoRA attention sufficiency Hu et al. 2021 (used), MLP-as-memory Geva et al. 2021 (used). **Weyl's inequality REMOVED per Fix 3.** All remaining citations are actually applied. PASS.

3. **Predicted numbers:** Four predictions listed, all specific and falsifiable. The MMLU prediction (~1.4pp degradation) is still listed as a live prediction in MATH.md, which is appropriate since MATH.md is the pre-experiment framework. PAPER.md correctly marks it WRONG SIGN. PASS.

4. **Falsification condition:** Three conditions, one was falsified (interaction > 10%, K768 FAIL). Honestly reported. PASS.

5. **Hyperparameter count:** 0 (discrete domain-module mapping, not continuous tuning). PASS.

6. **Hack check:** First attempt at per-domain module selection. No stacking. PASS.

## Fix Verification

### Fix 1: MMLU claims downgraded to "suggestive, not statistically significant"
**APPLIED.** PAPER.md line 29 states "suggestive but not statistically significant at n=15 per domain." Line 119 labels per-domain analysis as "(suggestive, not statistically significant at n=15)." Lines 127-129 add explicit statistical caveat: "95% CI is +/-24pp. None of the per-domain MMLU differences are statistically significant." The Key Insight section (lines 209-220) focuses on the PPL subadditivity finding rather than MMLU numbers. Adequately addressed.

### Fix 2: Finding status changed to PROVISIONAL
**APPLIED.** PAPER.md line 27: "Status: PROVISIONAL." Lines 28-33 provide explicit justification: "Two of three kill criteria failed (K767, K768). The sole pass (K766) rests on MMLU results that are suggestive but not statistically significant." Correctly calibrated.

### Fix 3: Weyl's inequality removed
**APPLIED.** Grep for "Weyl", "Stewart", "spectral", "eigenvalue" across MATH.md returns zero hits. Self-test item 2 now lists only theorems actually used in the analysis. Clean.

### Fix 4: 2240% math interaction -- absolute values reported
**APPLIED.** PAPER.md line 139 now reads: "2240% (misleading: denominator -0.02 ~ 0; absolute interaction = 0.45 PPL points)." Verified the arithmetic: attn improve = 0.327, MLP improve = 0.101, sum = 0.428, full improve = -0.020, absolute interaction = 0.428 - (-0.020) = 0.448, rounds to 0.45. Correct.

### Fix 5: Prediction 5 marked WRONG SIGN
**APPLIED.** PAPER.md prediction table line 19 now reads: "WRONG SIGN (predicted -1.4pp degradation, measured +1.4pp improvement; directionally useful but linear model A2 is wrong)." Clear and honest.

All 5 fixes correctly applied.

## Mathematical Soundness

No new issues. The analysis is parameter counting leading to perturbation fraction estimates. The 28% prediction vs 31.5% measurement is a clean confirmation that parameter count ratio approximates perturbation norm ratio (both use Grassmannian A-matrices, so per-parameter perturbation is approximately uniform).

The module separability framework (Finding #300) is correctly cited as the proven foundation. The K768 FAIL (interaction effects 13-151% for domains at scale=20) correctly identifies where the framework's assumptions break down: nonlinear compounding through LayerNorm/SiLU/softmax at high adapter scale.

One observation on the separability failure: the subadditivity is consistent -- combined effects are always LESS than sum of parts. This means module separability overestimates the benefit of adding modules, not underestimates it. The individual module evaluations (attn-only, MLP-only) are therefore valid lower bounds on what jointly-trained single-module adapters could achieve. This is noted in the paper's Limitation 2 (jointly-trained B-matrices confound) but could be stated more explicitly as a methodological strength.

## Prediction vs Measurement

PAPER.md contains a clear 5-row prediction-vs-measurement table. Assessment:

| # | Prediction | Status | Notes |
|---|-----------|--------|-------|
| 1 | Attn perturbation ~28% | MATCH | Measured 31.5%, within 15% |
| 2 | Code behavioral drops with attn-only | MATCH | -67%, large effect |
| 3 | Medical/math behavioral maintained | MATCH | +7% / +0.5% |
| 4 | Module interaction < 10% | FAIL | 13-151% at scale=20 |
| 5 | MMLU degradation ~1.4pp | WRONG SIGN | +1.4pp improvement vs -1.4pp predicted |

Score: 2 clean matches, 1 qualitative match (code drops), 1 wrong sign (MMLU), 1 fail (interaction). For a guided exploration, 3/5 predictions confirmed (including the most actionable ones: perturbation ratio, code-needs-MLP, prose-attn-sufficient) is adequate for PROVISIONAL status.

Numerical consistency verified against results.json: perturbation fractions, PPL values, MMLU counts, and interaction calculations all match within rounding.

## New Observations (not in previous review)

### Math MMLU paradox (advisory, not blocking)
The proposed optimal config for math is attn-only (better PPL: 3.43 vs 3.78, better behavioral: 0.665 vs 0.662). However, attn-only math MMLU drops from 40% to 20% (6/15 to 3/15). The hybrid config also shows math MMLU at 20%. At n=15 this is 3 questions and could be noise, but it creates a tension: the config that improves domain-specific quality may degrade domain-specific MMLU. This should be flagged as a concern for follow-up with larger MMLU samples.

### Code MMLU paradox (advisory, not blocking)
Attn-only code MMLU improves from 40% to 60% (6/15 to 9/15), while code behavioral collapses to 0.281. The adapter that destroys code generation simultaneously improves code knowledge recall. This is actually consistent with the MLP-as-memory thesis: attention-only preserves MLP knowledge stores (MMLU) while failing to provide MLP-mediated syntax generation (behavioral). But at n=15, this is 3 questions.

### Subadditivity is the real finding
The paper correctly identifies this in the Key Insight section. The subadditive pattern (applying all modules yields less improvement than the sum of individual module improvements) is measured with PPL, which has more statistical power than MMLU. This finding does not require MMLU to be significant. It is the strongest and most actionable result.

## Novelty Assessment

Prior art acknowledged: LoRA attention sufficiency (Hu et al. 2021), AdaLoRA importance allocation (Zhang et al. 2023), MLP-as-memory (Geva et al. 2021). The novel contribution within the project is the empirical discovery that at ternary LoRA behavioral scale (s=20), subadditive module interference means fewer modules can be better. The per-domain module selection table is a practical output. The perturbation ratio prediction/confirmation is a clean quantitative result.

## Macro-Scale Risks (advisory)

1. Static per-domain config requires empirical determination for each new domain.
2. Subadditivity may compound with multi-adapter composition (untested).
3. The 43% fewer adapter operations for prose domains is a clean speed win.
4. Jointly-trained adapters evaluated in isolation is a confound; purpose-trained attn-only adapters may perform differently (possibly better).

## Verdict

**PROCEED**

All 5 revision requests have been correctly applied:
1. MMLU claims appropriately downgraded to "suggestive"
2. Finding status correctly set to PROVISIONAL
3. Weyl's inequality removed (no phantom citations remain)
4. Math interaction reported with absolute values and misleading-percentage caveat
5. MMLU prediction correctly marked WRONG SIGN

The experiment is a well-executed guided exploration that produces three actionable results:
- Perturbation fraction prediction confirmed (28% predicted, 31.5% measured)
- Per-domain module selection table derived (attn-only for prose, full for code)
- Subadditive module interference discovered at behavioral scale

The PROVISIONAL status is correctly calibrated: useful directional findings with two kill criteria failed and MMLU claims appropriately caveated. The experiment advances the project's knowledge base without overclaiming.

No blocking issues remain.
