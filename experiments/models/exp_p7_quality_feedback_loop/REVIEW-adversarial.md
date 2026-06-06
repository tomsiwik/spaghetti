# Adversarial Review: exp_p7_quality_feedback_loop

## Verdict: PROCEED

## Summary

Clean verification-of-impossibility experiment. MATH.md proves null-space projection
magnitude carries zero quality information (3 theorems). PAPER.md confirms all 3 kill
criteria FAIL as predicted. Data in results.json is internally consistent. KILLED status
is correct.

## Checklist

| Check | Status |
|-------|--------|
| Prediction-vs-measurement table | Present, complete |
| Kill criteria match evidence | All 3 verified against results.json |
| Finding status appropriate | KILLED correct for verified impossibility |
| Math errors | None found |
| Fabricated evidence | None detected |

## Strengths

1. **Projection constancy confirmed empirically**: results.json shows proj_magnitude
   is identical per adapter across all 10 texts (e.g., medical=47.57 for all inputs).
   This directly validates Theorem 2 (norm-dominated) — the "signal" has zero
   input-dependent variation.

2. **Feedback loop correctly diagnosed**: K1306 shows 0.00pp improvement because
   the feedback signal converges to exactly 0.0 for all adapters. The mechanism
   is clearly explained (static = feedback when there's no information to learn from).

3. **Impossibility structure well-derived**: The chain null(W_v) ⊥ range(W_v^T) →
   domain info stripped → I(proj, quality) ≈ 0 → no learnable function is sound
   and cleanly closes the geometry-as-reward line.

4. **Good connection to prior findings**: Extends Finding #495 (r=-0.19 for routing)
   to quality prediction (r=-0.224, AUC=0.43). Same structural cause, different
   downstream application. Also correctly explains why LoRAHub uses task loss, not
   geometric features.

## Minor Issues (non-blocking)

1. **Per-adapter AUC values misleading**: PAPER.md states "within-adapter AUC is
   degenerate" but results.json shows per_adapter_auc = 1.0 for code/legal/finance.
   These are artifacts of having only 0-2 negative samples per adapter (n_negative=4
   total across 50 pairs), making within-adapter AUC trivially 0 or 1. Not wrong,
   but "degenerate" should note it means "undefined due to class imbalance" rather
   than implying AUC=0.5.

## What This Closes

P7 null-space line is now complete:
- Null-space isolation: works (Finding #494)
- Null-space dimensionality: characterized (Finding #496)
- Null-space routing: killed (Finding #495)
- Null-space quality prediction: killed (this experiment)
- Null-space feedback: killed (this experiment)

**Bottom line**: Null-space geometry is a construction tool (isolation), not an
information source (routing/quality/feedback). Future routing work must use
range(W_v) features or external signals.
