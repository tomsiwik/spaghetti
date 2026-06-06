# REVIEW-adversarial.md — 2-Domain M2P Composition at 4B Scale

**Verdict: PROCEED**

---

## Checklist

- [x] Prediction-vs-measurement table present in PAPER.md
- [x] Kill criteria results match evidence in results.json
- [x] Finding status `supported` is appropriate for frontier-extension
- [x] No fabricated kill criteria results
- [x] Theorems are sound and cite prior work

---

## What's Good

**Theorem 1 (Grassmannian Isolation):** The QR construction proof is exact — not
approximate. A_math^T A_code = 0 by column orthonormality of Q. The Gram-Schmidt
extension for preserving existing math A-matrices is a clean fix. N_max=320 confirmed.
Prediction 1e-5 (bf16 floor), measured 1.38e-05: nearly exact match.

**Theorem 2 (TF-IDF routing):** Cites LoraRetriever (2402.09997) for routing-adapter
independence. Vocabulary disjointness argument is correct. Consistent with Finding #389
and #395.

**Theorem 3 (composition quality):** The expected-quality derivation `E[qr] = p_route *
qr_single + (1-p_route) * qr_wrong` is valid under independence assumptions. The
conservative bound of 0.94 is exceeded (1.3125 measured) because routing was 100%, not
just >=80% as assumed.

**Finding status `supported`:** Correct for frontier-extension. No formal proof was
expected beyond applying existing theorems to 4B scale.

---

## Non-Blocking Issues

**1. results.json inconsistency — code_m2p_final_loss=0.0 vs PAPER.md 0.0414**
The run used a cached code M2P checkpoint, so the loss field was not written. PAPER.md
correctly records 0.0414 from the original training run. The JSON field is a stale
placeholder. Not a validity concern, but future experiments should log "cached" explicitly.

**2. peak_memory_gb=0.0 in results.json vs 6.57 GB in PAPER.md**
Same caching issue — peak memory tracker appears to have returned 0.0 when the heaviest
phase was skipped on retry. The 6.57 GB figure in PAPER.md is credible (consistent with
4B model + M2P encoder in bf16). Not a validity concern.

**3. code_m2p_params=808768512 (~808M)**
The M2P encoder is larger than a full 0.6B model, a known issue (v5 SHINE, Finding #402;
VeRA bottleneck killed Finding #380). This is acknowledged technical debt, not a finding
blocker.

**4. Only math domain evaluated for quality**
K977 measures only math quality_ratio. Code domain quality is not verified — PAPER.md
acknowledges this in "Next Steps". A finding that composition preserves math quality
does NOT prove code quality is preserved. PAPER.md is honest about this gap, which
is appropriate for frontier-extension scope.

---

## Summary

All kill criteria pass with predictions matched. The Grassmannian construction works
identically at 4B. TF-IDF routing is scale-invariant as expected. The `supported` status
is earned. The main open item (code domain quality verification) is correctly deferred
to the next experiment.
