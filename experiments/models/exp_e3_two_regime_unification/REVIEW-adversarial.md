# E3: Two-Regime Unification — Adversarial Review

## Verdict: KILL

Override: `is_smoke=true` → KILL (not PROVISIONAL) because the failure is method-level, not sample-level. 1.8% vs 20% threshold is a 10× structural miss; ρ=-0.19 is opposite-sign. No amount of additional data changes these conclusions. Same precedent as E1 (#801), E2 (#802), E6 (#804).

## Kill Rationale

1. **K2022 FAIL (structural)**: 6/336 heads (1.8%) show sigmoid response with inflection in [3,7] vs ≥20% threshold. 10× below minimum. The behavioral phase transition at s∈[4,6] cannot be explained by attention head cascading when <2% of heads respond at those scales.

2. **K2023 FAIL (correlation)**: ρ=-0.19 (p=0.58) vs ≥0.7 threshold. Opposite sign — more flipped heads = worse accuracy. Head-flip count tracks adapter perturbation magnitude, which is destructive above optimal scale. Head flipping is symptom, not cause.

3. **Target-gated (F#666)**: Both structural (K2022) and behavioral (K2023 correlation) KCs failed. K_target marginal PASS confirms the phase transition is real (F#250) but the proposed mechanism (head cascade) is falsified. Clean kill on mechanism.

## Adversarial Checklist

| Item | Result | Notes |
|------|--------|-------|
| (a) verdict consistency | ✓ | results.json=KILLED, DB→killed |
| (b) all_pass vs claim | ✓ | all_pass=false, no supported claim |
| (c) PAPER.md verdict | ✓ | "KILLED" |
| (d) is_smoke override | ✓ | Method-level failure, override valid |
| (e) KC pre-reg integrity | ✓ | K2022, K2023, K_target match MATH.md |
| (f) tautology | ✓ | KCs measure real quantities |
| (g) KC code↔math | ✓ | Consistent |
| (h-k) code antipatterns | ✓ | No adapter summing, shutil.copy, etc. |
| (i) unsafe scale | ✓ | Measurement sweep, not application |
| (l) hardcoded pass | ✓ | None |
| (m) model match | ✓ | Gemma 4 E4B in both |
| (m2) skill invocation | ✓ | /mlx-dev, /fast-mlx invoked; forward-pass only |
| (n) base truncation | ✓ | Base=40%, not truncated |
| (o) sample size | ✓ | N=5/scale, structural claim not statistical |
| (r) pred-vs-meas table | ✓ | Present, 5 rows |
| (s) math correctness | ✓ | Softmax margin theorem correct |
| (t) target-gated kill | ✓ | Structural + correlation both FAIL |

## Key Finding for Downstream

The behavioral phase transition at s∈[4,6] is **real** (GSM8K peak at s=6, confirming F#250) but its mechanism is **not** attention head activation cascading. Only 1.8% of heads respond at the transition scale. The mechanism must be downstream: FFN gating, output logit softmax, or format template activation. Q-proj perturbation grows linearly with scale; the threshold behavior comes from nonlinear downstream processing.

## Assumptions
- Override of smoke→KILL is justified by 10× structural miss (not a borderline case).
- GSM8K N=5/scale is sufficient for the smoke claim because the structural KC (head fits over 336 heads) is the primary kill signal.
