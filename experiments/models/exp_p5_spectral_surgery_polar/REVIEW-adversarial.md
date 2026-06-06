# Adversarial Review: exp_p5_spectral_surgery_polar

## Verdict: PROCEED (as KILL)

## Strengths

1. **Three independent impossibility theorems** — Theorems 5-7 are mathematically clean and each independently closes the direction. The flat spectrum result (Thm 5) is elegant: Stiefel retraction forces sv = [1,...,1], leaving nothing for surgery to differentiate.

2. **Prediction-measurement alignment** — All 7 predictions in the table match. The flat spectrum CoV = 1.5e-9 confirms sv = 1 to numerical precision. Basis dependence cosine = 0.619 is decisively << 1. GSM8K 0.0pp delta is exactly predicted.

3. **LoRA control validates the experimental apparatus** — LoRA has non-flat spectrum (CoV = 0.58, sr = 1.88) as expected, confirming the surgery procedure itself is functional. The failure is PoLAR-specific, not an implementation bug.

## Issues (non-blocking)

1. **K1272 prediction was wrong** — MATH.md predicted "PASS: ~10s" for surgery speed, but measured 182.6s. The error: Theorem 1 only accounts for the r x r SVD cost, not the finite-difference sensitivity estimation (42 layers x 6 components x forward passes). This doesn't affect the kill verdict but the prediction table should honestly label this as a prediction miss, not just an explanation. PAPER.md handles it adequately by noting "sensitivity estimation dominates."

2. **LoRA surgery also degraded quality (-6.7pp)** — This raises whether the reweighting formula (Theorem 3) is itself flawed for short-trained adapters, or whether the finite-difference sensitivity estimate is noisy with only 10 calibration examples. The paper attributes this to "too aggressive reweighting for short training" which is plausible but unproven. Not blocking since the PoLAR impossibility is structural and independent of the formula.

3. **Sample size** — 30 GSM8K examples for evaluation. At n=30, the 95% CI for 43.3% is roughly +/-18pp, so the 0.0pp delta could be anywhere in [-18, +18]. However, the structural impossibility (flat spectrum, basis non-uniqueness, Stiefel violation) makes the statistical argument moot — the theorems predict no systematic improvement regardless of sample size.

## Finding Recommendation

Status: **killed** — three structural impossibility theorems confirmed, permanently closes spectral surgery for PoLAR.

Impossibility structure: Stiefel retraction -> flat spectrum -> surgery is basis-dependent noise -> any non-trivial surgery breaks Stiefel constraint. This is a mathematical invariant, not a hyperparameter issue.
