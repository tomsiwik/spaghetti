# REVIEW-adversarial.md — exp_p1_p1_ridge_routing_n25

**Verdict: PROCEED**
**Round:** 1 (final)
**Reviewer:** adversarial-reviewer

---

## Kill Criteria Verification

All 4 criteria PASS. results.json is authoritative:

| Criterion | Predicted | Measured | Verdict |
|-----------|-----------|---------|---------|
| K1158: N=25 acc ≥ 90% | 91–94% | 98.76% | PASS (4.76pp above ceiling prediction) |
| K1159: N=5 acc ≥ 96% | 97–99% | 99.0% | PASS |
| K1160: p99 ≤ 2ms | ~0.2ms | 0.40ms | PASS |
| K1161: train ≤ 1s | ~0.1s | 0.567s | PASS |

Baseline replication confirmed: centroid N=5=96.6%, N=25=86.08% match Finding #431 exactly.

---

## Math Review

**Theorem 1 (Ridge Accuracy Bound):** The proof sketch is directionally correct — ridge regression jointly uses all training examples, suppressing shared vocabulary and amplifying discriminative terms. The intermediate claim "IDF weighting ensures full column rank when d > N" is slightly confused (λ_min(Φ^T Φ) is near zero for sparse TF-IDF; invertibility comes from the λI term, not column rank). This does not affect the conclusion and is non-blocking.

**Theorem 2 (Computational Complexity):** The honest dual vs. primal discussion is correct. sklearn's conjugate gradient implementation handles sparse Φ efficiently; the empirical 0.567s confirms the prediction.

**Centroid comparison:** Claim that W_centroid = Φ^T Y is a simplification (actual centroid = Φ^T Y / N_per_class), but the conceptual point — no cross-domain adjustment — is valid.

---

## Concerns (non-blocking)

1. **Rounding:** PAPER.md reports "98.8%" but results.json shows 0.9876 = 98.76%. Off by 0.04pp. Acceptable rounding; not fabrication.

2. **Intercept warning:** `RuntimeWarning: divide by zero in intercept computation` already documented in PAPER.md caveats. The effect is confirmed zero-impact on accuracy. Non-blocking.

3. **MMLU-synthetic N=25:** The 20 extra domains are MMLU subjects, not real user adapter domains. Real domains could be more confusable, especially for adjacent technical fields (e.g., astrophysics vs. physics vs. chemistry). The 93% floor for the hardest domain (legal, medical) gives reasonable headroom. Non-blocking; documented in caveats.

4. **No adversarial queries:** Routing accuracy was evaluated on in-distribution MMLU MCQ test queries. Adversarial queries (e.g., domain-mixing questions like "explain the physics of compound interest rates") were not tested. This is a known frontier gap for all routing experiments; deferred to production validation.

---

## Finding Status Assessment

**SUPPORTED** is correct for a guided exploration that:
- Made quantitative predictions (91–94%)
- Got results that substantially exceeded predictions (98.8%)
- Replicated the baseline exactly
- Provided mechanistic explanation consistent with theory

Finding #458 is appropriately recorded.

---

## Impact Assessment

This experiment directly unblocks the P1 production target: routing was the last bottleneck
(86.1% insufficient for 25-domain deployment). 98.8% accuracy with 0.57s training and 0.40ms
inference makes ridge routing a drop-in replacement for centroid routing with no latency cost.

The finance domain recovery (74% → 93%) specifically validates the impossibility structure
derived after Finding #456: ridge's discriminative boundary suppresses shared calculation
vocabulary that TF-IDF centroid cannot separate.

## Verdict: PROCEED
