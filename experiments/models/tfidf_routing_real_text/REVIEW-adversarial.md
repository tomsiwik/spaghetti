# Adversarial Review: exp_tfidf_routing_real_text

**Verdict: PROCEED**

---

## Checklist

- [x] Prediction-vs-measurement table present
- [x] Kill criteria results match results.json
- [x] Finding status (supported) appropriate for measurement type
- [x] No fabricated results

---

## Issues Found

### Non-blocking: Theorem 1 cosine prediction violated for math_text

Theorem 1 predicts `cos(math, text) < 0.30`. Measured value = 0.504 — substantially above.

PAPER.md correctly flags this as a "theoretical surprise" and explains the mechanism:
math word problems share narrative prose with news text, inflating centroid similarity.
The proof's claim rests on high-IDF vocabulary suppression, but the *centroid* cosine
is driven by moderate-IDF shared story words ("he", "was", "has") which aren't fully
suppressed. Theorem 2 (IDF suppression of shared vocab) holds asymptotically but the
finite-corpus behavior differs.

**Resolution:** The kill criterion K950 (routing accuracy ≥80%) still passes at 100%
because discriminating n-grams ("how many", "in python") create a clean decision
boundary even when centroids are moderately similar (0.504). The prediction failure
is acknowledged. No structural issue with the finding.

**Recommendation for future work:** Tighten Theorem 1 to predict routing accuracy directly
(not centroid cosines), since centroid cosines are an intermediate proxy that can fail
while routing still succeeds.

---

## Strengths

1. K950 PASS at 100% vs 80% threshold — large margin, no cherry-picking concern.
2. Zero errors in both NC and LR classifiers across 300 test examples.
3. Top discriminating terms directly confirm the theorem's mechanism (question-structure
   n-grams for math, Python vocabulary for code).
4. 10s runtime — negligible overhead, confirms pre-forward routing claim.
5. math_text = 0.504 is an honest surprise, not swept under the rug.

---

## What This Closes

- TF-IDF routing is production-ready for real 3-domain NLP (math/code/text).
- The Q_wrong bottleneck (Finding #386, -58% relative harm) is fully addressable by routing.
- Routing is NOT the bottleneck. Next bottleneck: M2P adapter quality (d_M2P=64 insufficient,
  per Finding #387). exp_m2p_vera_bottleneck is the correct next step.
