# Adversarial Review: exp_p0_polar_premerge_composition

## Verdict: PROCEED (KILLED)

Finding #527 KILLED status is fully justified.

## Review

**Prediction-vs-measurement table:** Present and honest. Four of six predictions match; the critical one (pre-merged GSM8K >= 40%) catastrophically fails at 0.0%. The paper correctly marks Theorem 3's bound as "IRRELEVANT" rather than hiding the failure.

**Kill criteria vs evidence:** All four kill criteria match results.json exactly:
- K1451 FAIL: pre-merged GSM8K = 0.0% (needed >= 50%) — primary kill
- K1452 PASS: sr = 6.0000 (all 3 adapters, min 5.9999999)
- K1453 PASS: max cross-cosine = 0.0011 (100x below threshold)
- K1454 PASS: solo GSM8K = 62%

**Finding status:** KILLED is correct. K1451 is the primary behavioral criterion and it failed catastrophically.

## Strengths

1. **Honest failure analysis.** MATH.md explicitly listed "pre-merge fails despite low cosine" as kill mechanism #3. The paper confirms exactly this mechanism operated. This is textbook proof-first research: the proof predicted the failure mode even when the theorem predicted success.

2. **Impossibility structure is strong.** Four independent experiments (std LoRA #510, TT-LoRA #526, PoLAR #527, and earlier work) all show 0% pre-merge GSM8K across different orthogonality strategies. The conclusion that pre-merge is structurally impossible for independently-trained adapters is well-supported.

3. **Clean separation of weight-space vs functional-space.** The key insight — that weight-space orthogonality (cosine 0.001) is irrelevant when perturbations compound through 42 nonlinear layers — is precisely identified and will prevent future wasted effort.

## Minor Issues (non-blocking)

1. **Compounding error heuristic is rough.** The (1+0.0043)^42 calculation is a hand-wave — it assumes layer-wise independence and additive composition, neither of which holds. The actual failure mechanism is attention's multiplicative Q/K/V interaction, which the paper mentions but doesn't formalize. Not blocking because the conclusion stands regardless of the precise mechanism.

2. **Training "skipped" in results.json.** All three adapters show `"skipped": true`, meaning pre-trained adapters were loaded. This is fine if they were trained in a prior step of the same experiment, but the provenance could be documented more explicitly.

3. **Sample size (N=50).** GSM8K evaluation on 50 samples. For 0% vs 62%, significance is unambiguous. But if the result had been borderline, this would be insufficient.

## Conclusion

Clean kill. The impossibility structure is the most valuable output: pre-merge of independently-trained adapters fails not because of any fixable property (magnitude, compression, spectral regularity, weight-space alignment) but because nonlinear functional interference through deep networks is fundamentally different from linear weight-space interference. Routing is confirmed as the only viable composition method.
