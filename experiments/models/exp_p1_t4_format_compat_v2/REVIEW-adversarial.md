# REVIEW-adversarial.md — T4.5v2: Loophole Fix

**Verdict: PROCEED**

## Summary

All three loopholes from T4.5 are closed with hard-fail tests. Results are clean, evidence matches predictions, and the math is sound. No fabrication or weak claims detected.

## Checks

**Prediction-vs-measurement table:** Present and complete in PAPER.md. ✓

**Kill criteria vs evidence:**
- K1243 (PeftModel CPU load): results.json confirms `loaded in 5.9s, logits shape [1, 4, 50272]`. PASS ✓
- K1244 (Grassmannian honesty): QR deviation 1.19e-07 < 1e-4 → tag written; random deviation 814.8 >> 1e-4 → no tag. 8-order separation. PASS ✓
- K1245 (round-trip max_diff): 0.0 exactly. PASS ✓

**Math review:**
- Theorem 1 (bijection): Correct. Transpose is an involution; round-trip exact in float32. Prediction (0.0) confirmed.
- Theorem 2 (Grassmannian reporting): Correct. QR gives < 1e-7 residual; training destroys Stiefel constraint (deviation 0.579 measured in T4.5). The 1e-4 threshold separates the two regimes by 7+ orders.
- Theorem 3 (CPU loadability): Correct. The three necessary conditions (schema, key matching, shapes) are sufficient for PeftModel loading; confirmed empirically.

**Concerns (non-blocking):**
1. **OPT-125m proxy**: Acknowledged in PAPER.md caveat. The bijection proof is model-agnostic (transpose + key rename). Gemma 4 would require a separate test but the mechanism is identical. This is appropriate scope for a micro-experiment.
2. **QR deviation 1.19e-07 vs predicted 1e-12**: Noted in PAPER.md. float32 QR has ~1e-7 residual due to finite-precision Householder reflections. The threshold (1e-4) remains valid; no impact on K2 verdict.
3. **Partial adapter warning (layers 3-11)**: UserWarning only, not a RuntimeError. Valid PEFT behavior for partial coverage. Production adapters should cover all target layers (noted as caveat).

## Finding Status

SUPPORTED is correct. All 3 loopholes are closed. The format compatibility claim is now backed by actual PeftModel loading, honest Grassmannian reporting, and an exact round-trip bijection.
