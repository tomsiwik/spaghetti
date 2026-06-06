# Adversarial Review: exp_p9_ttlora_polar_hybrid

## Verdict: PROCEED (with caveats)

## What works

1. **Honest prediction-vs-measurement table.** Two clear MISSes (sr ratio, retraction time) are documented transparently. No cherry-picking.
2. **Post-hoc mechanism is well-supported.** The norm regularization explanation (||DW||_F drops 2-3x) is directly visible in per-layer data and doesn't require speculation.
3. **Theorem 1 (norm concentration) validated.** The Stiefel constraint does control ||DW||_F exactly as Oseledets predicts. The math framework held even though the predicted *consequence* (sr improvement) was wrong.
4. **Experiment type correct.** Guided-exploration with honest MISS on the unknown (mechanism is norm, not spectral). SUPPORTED is appropriate.

## Caveats (non-blocking)

### 1. GSM8K +8pp is not statistically significant
31/50 vs 35/50 = 4 questions. Fisher's exact test: p ~ 0.42 (two-sided). This is well within chance variation. The finding should note that the +8pp requires replication at larger N before being treated as reliable. The norm regularization story stands on the ||DW||_F data regardless.

### 2. K1363 is a marginal PASS
sr ratio 1.03x (1.94 vs 1.88 on trained layers) is within measurement noise. The MATH.md predicted 1.5-3x. Calling this "PASS" is technically correct (1.94 > 1.88) but the criterion was motivated by an expected large effect. The paper correctly identifies this as a MISS on the prediction, which is the right framing.

### 3. Only 3/42 layers have non-zero DW
500 steps of v_proj-only training doesn't propagate gradients to layers 30+. The experiment measures Stiefel effects on ~7% of the model. This is a valid micro-experiment scope, but composition implications (interference bounds) should be stated cautiously — deeper layers may behave differently.

## Not issues

- **K1365 FAIL (9ms retraction)**: Correctly attributed to numpy data-transfer overhead, not fundamental math cost. The FLOP analysis (210 SVDs of 48x6) is sound. Not a kill criterion for the finding itself.
- **Mechanism pivot**: Changing from "spectral spreading" to "norm regularization" mid-experiment is exactly what guided-exploration is for. The framework (Stiefel helps) held; only the specific mechanism was discovered.

## Recommendations for finding

- Note GSM8K result requires N>200 replication for significance
- Frame sr PASS as "marginal/noise-level" rather than confirmed
- The durable finding is: Stiefel on TT cores = norm regularizer (||DW||_F controlled to sqrt(r) per core), which may improve generalization and reduce interference
