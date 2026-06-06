# Adversarial Review: exp_p0_e2e_n10_routing_quality

## Verdict: PROCEED (SUPPORTED)

## Summary

Clean verification experiment. Theorem 1 (linear quality degradation) validated
within 1.2pp on all 3 benchmarks. All 4 kill criteria PASS. 8/11 predictions
within 1pp of measured values. SUPPORTED status is appropriate for verification type.

## What Checks Out

1. **Prediction accuracy is excellent.** 8 of 11 metrics within 1pp. No prediction
   off by more than 2pp. This is unusually tight for an LLM experiment.

2. **Kill criteria verified against results.json.** K1482 (GSM8K 77% >= 70%),
   K1483 (HumanEval 56% >= 48%), K1484 (MedMCQA 54% >= 45%), K1485 (max loss 4pp <= 8pp).
   All consistent between results.json, PAPER.md, and experiment evidence.

3. **Theorem 1 math is correct.** Cross-checked:
   - GSM8K: 0.98*77 + 0.02*15 = 75.8%, actual 77%, +1.2pp (correctly explained: base solves easy questions)
   - HumanEval: 0.97*57 + 0.03*18 = 55.8%, actual 56%, +0.2pp
   - MedMCQA: 0.86*58 + 0.14*28 = 53.8%, actual 54%, +0.2pp

4. **Distribution shift result is strong and non-obvious.** HumanEval routes
   BETTER from benchmark text (97%) than MMLU training data (94%). Good explanation:
   pure code has cleaner signal than CS MCQ.

5. **Oracle values consistent with prior findings.** GSM8K 77%, HumanEval 57%,
   MedMCQA 58% all match Finding #508 adapter deltas (+62, +39, +30 from base).

## Non-blocking Issues

1. **Theorem 1 is an upper bound, not exact.** The proof assumes misrouted queries
   perform at base level. Actual performance can be BETTER (base solves some problems)
   or WORSE (wrong-adapter routing). Currently safe because confusing domains lack
   adapters, but PAPER correctly flags this risk for N=25+ with more adapters.

2. **N=100 sample sizes.** Statistical confidence at N=100 is moderate. MedMCQA 54%
   has 95% CI of roughly [44%, 64%]. The 4pp routing loss is within noise. However,
   the PATTERN across all 3 benchmarks is consistent with Theorem 1, making this
   a systematic validation rather than a single noisy measurement.

3. **Routed deltas in PAPER line 117.** States "+62/+38/+26pp" — these are routed
   (not oracle) deltas from base. Correct: 77-15=62, 56-18=38, 54-28=26.

## Conclusion

This is one of the tighter prediction-vs-measurement results in the project.
Theorem 1 provides a reliable model for predicting quality at any N given routing
accuracy. The distribution shift non-issue is a genuine surprise worth highlighting.
Ready for finding registration.
