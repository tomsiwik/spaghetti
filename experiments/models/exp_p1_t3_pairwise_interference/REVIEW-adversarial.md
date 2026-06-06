# REVIEW-adversarial.md — T3.1: Pairwise Interference

**Reviewer verdict: PROCEED**  
**Date:** 2026-04-10

---

## Summary

Solid killed experiment. The reversed failure pattern (low-cosine math/code collapse, high-cosine MMLU cluster survives) is honestly reported and correctly analyzed. Kill criteria are consistent with results.json. The structural fix (routing) follows directly from the impossibility structure.

---

## Checklist

- [x] Prediction-vs-measurement table present and complete
- [x] Kill criteria results match results.json exactly (K1050/K1051/K1052 all FAIL)
- [x] Finding status = killed — appropriate (all 3 criteria failed)
- [x] No fabricated data — all numbers traceable to results.json
- [x] The reversed prediction is acknowledged, not buried

---

## Issues

### Non-blocking (note for T3.2)

**1. n=25 is too small for MMLU cluster claims.**  
Medical/legal/finance ratios (0.815–0.867) are 1–2 sigma below 0.90 at n=25 (SE ≈ 8–10%). The math/code collapse (0.098, 0.121) is unambiguous at any sample size, but "MMLU cluster degrades only moderately" should be treated as provisional at n=25. T3.2 should use n≥50 per domain.

**2. The exponential bound `acc ≤ acc_single × (1 - (N-1)×ε)^L` is empirical, not a theorem.**  
PAPER.md correctly notes "ε >> 0.01 for these adapters" when the formula underpredicts collapse. Label this as a heuristic model in any future citation, not a formal bound (it doesn't derive ε from first principles).

**3. Finding #225 citation needs care.**  
PAPER.md claims Finding #225 used "Grassmannian (QR-init) adapters with routing." If Finding #225 used a different composition method, this attribution is incorrect. Non-blocking since it's supporting context, but T3.2 should verify before inheriting this claim.

---

## What's Good

- Theorem 1 (trace trick for Frobenius cosine) is mathematically correct. Cyclic property of trace properly applied.
- The core finding — weight-space cosine is NOT the interference predictor — is well-supported and honestly contrasted against the original hypothesis.
- The impossibility structure is clean: simultaneous N-adapter activation = O(N-1) additive noise terms. Routing makes interference = 0 by construction. This is correct.
- The finding motivates T3.2 (PLE-M2P routing) as a structural necessity, not a heuristic improvement. This framing is appropriate.

---

## Verdict

**PROCEED.** The killed status is correct. The analysis is honest. The structural fix is sound. Ship it.
