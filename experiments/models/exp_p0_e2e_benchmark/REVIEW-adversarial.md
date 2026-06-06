# REVIEW: exp_p0_e2e_benchmark

**Verdict: PROCEED**

## Integrity Check

- Prediction-vs-measurement table: Present, complete.
- Kill criteria match evidence: All 5 K-values in results.json match PAPER.md claims exactly.
- Code matches claims: Eval uses proper holdout splits (GSM8K test, HumanEval test, MedMCQA validation). Routing uses separate seed (SEED+1). No data leakage detected.
- Finding status: SUPPORTED is appropriate. Quantitative predictions were 2-4x conservative, so not "conclusive" (predictions didn't match), but the directional claim is strongly verified.

## Issues Found

### 1. Theorem 1 is tautological (non-blocking)
"Training on distribution D reduces loss on benchmark B drawn from the same distribution" is a restatement of what SGD does. The "QED" overstates the formalism. This doesn't invalidate results — it's a framing issue, not a math error. The real contribution is the empirical verification, not the theorem.

### 2. Predictions were 2-4x off due to wrong base estimate (non-blocking)
MATH.md predicted base GSM8K at 40-60% (from Finding #421 caveat), actual was 17%. The prediction framework works (direction correct, all pass) but the base model performance estimation was poor. PAPER.md explains this honestly. For future experiments, measure base first before predicting deltas.

### 3. Latency test metformin misroute (non-blocking)
The latency phase uses a 9-sample router that misroutes the metformin query to "math". This is independent of the routing evaluation (which uses 200+100 samples and scores 98.3%). The latency measurement itself is valid — it measures route+load+generate time regardless of correctness. Noted in PAPER.md caveat.

### 4. Medical train loss anomaly explained by format (non-blocking)
Medical train loss 0.07 vs math 0.40 / code 0.52 initially looks suspicious. But the training format is "Reply with only the letter" -> "A: {text}" — very short outputs explain low loss. MedMCQA accuracy of 50% (lowest delta) confirms no overfitting artifact.

### 5. HumanEval code extraction fragility (acknowledged, non-blocking)
The regex ````python\n(.*?)``` ` may miss completions without code fences. Fallback uses raw response, which could cause syntax errors when appended to the function prompt. PAPER.md caveat 3 addresses this: even at 30% true base, +33pp still passes K1329.

## What This Proves

The full Pierre pipeline (train adapter + TF-IDF route + generate) works on standard benchmarks with large effect sizes (19-56pp). The training cost claim ($2 / 20 min per domain) is verified: 13-30 min, 21.8 MB per adapter. This is the first end-to-end validation of the complete system.

## What It Doesn't Prove

1. Composition under benchmark evaluation (solo adapters only; #505 proved composition works on PPL but not on accuracy benchmarks)
2. Scaling beyond N=3 domains on accuracy benchmarks
3. Behavioral quality (this is a benchmark-accuracy experiment, not a behavioral one)

## Recommendation

PROCEED to finding. Status: **supported** (verification experiment, all kill criteria pass, predictions directionally correct but quantitatively conservative).
