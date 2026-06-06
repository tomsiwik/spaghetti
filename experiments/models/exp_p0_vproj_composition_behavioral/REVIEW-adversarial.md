# Adversarial Review: exp_p0_vproj_composition_behavioral

## Verdict: PROCEED (with caveats noted below)

## What's Solid

1. **Data integrity confirmed.** PAPER.md numbers match results.json exactly. No fabrication.
2. **Structural finding is genuine.** 4/5 domains show retention >= 100% under 5-way parameter merging. PPL max 2.1%. Composition is demonstrably not the bottleneck.
3. **Honest prediction-vs-measurement table.** PAPER.md correctly flags two predictions as WRONG/EXCEEDED rather than hiding the mismatch. Good scientific practice.
4. **Status "SUPPORTED" is appropriate** for guided exploration where the mechanism question is answered even though absolute thresholds were calibrated to a different baseline.

## Issues (Non-Blocking)

### 1. Theorem 3 predicted degradation; observed improvement

Theorem 3 predicts R(d) in [1/N, 1/sqrt(N)] — i.e., composition always hurts. Measured R(d) mean = 1.67x at N=5. The ensemble improvement effect (consistent with Finding #496) is entirely outside the theorem's prediction range. The theorem's CLT noise argument assumes adapter outputs are uncorrelated noise for off-domain inputs — but adapters apparently contribute useful signal cross-domain. This doesn't invalidate the experiment, but the theorem should not be cited as a predictive model in future work. It predicted the wrong direction.

### 2. Low statistical power on routing comparison

Equal-weight (0.23) vs peaked (0.22) with n_eval=20 per domain. Each improvement rate is a binomial proportion from 20 trials — 95% CI width is roughly +/-0.20. The claim "routing doesn't help at N=5" is within noise. The correct statement is: "no statistically significant difference detected at n=20." Future work at N=25 should use larger n_eval if routing benefit is the question.

### 3. Solo baseline 3x discrepancy unexplained

Solo rates differ 3x from P8 (17% vs 52%). The paper attributes this to "measurement variance" and "stochastic generation," which is plausible for a vocabulary-count metric on 20 samples. But 3x is large. Possible confounds: different prompt templates, different temperature/sampling params, different base model state. This doesn't affect the retention ratios (same metric, same conditions), but it does mean the absolute kill criteria (K1316-K1318) were miscalibrated. Future experiments need either a more stable metric or kill criteria defined as retention ratios, not absolute scores.

### 4. Theorem 2 bound is loose

The interference bound in Theorem 2 is an argument sketch, not a tight bound. The "worst case is when all off-domain contributions point opposite" claim is correct directionally but the bound is never compared to measurement. Since this is guided exploration (not verification), this is acceptable — but upgrading to "conclusive" in future would require tighter math.

## Recommendation

PROCEED to finding. The structural question — "does composition preserve behavioral quality?" — is answered affirmatively. The caveats (weak theorem predictions, low statistical power on routing, metric variance) are real but don't undermine the core conclusion. They should inform future experiment design:
- Use retention ratios for kill criteria, not absolute scores
- Increase n_eval for statistical comparisons
- Develop a lower-variance behavioral metric
