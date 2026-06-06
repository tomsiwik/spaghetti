# Adversarial Review: exp_p1_t0_grassmannian_gemma4 (Round 2 — Post-REVISE)

**Verdict: PROCEED**

---

## Fix Verification (Round 2)

All 3 blocking fixes from Round 1 correctly applied:

**Fix 1 — Kill criteria relabeled:** MATH.md Predictions table now shows actual tested dims
(d=512/d=1024, rank=4). Gemma 4 dims correctly labeled "analytical corollaries". ✓

**Fix 2 — PAPER.md shows real values:** Cross-checked against results.json:
- K990: PAPER 1.696e-16 = results.json 1.6958e-16 ✓
- K991: PAPER 1.799e-16 = results.json 1.7993e-16 ✓
- K992: N_max=128/256 = results.json n_max_small/large ✓
- K993: 0.00112s = results.json gpu_construction_time_s ✓

**Fix 3 — Rank corrected:** Both MATH.md and PAPER.md explicitly state rank=4 for smoke.
N_max=176/336 at r=16 correctly labeled as analytical corollaries. ✓

Theorem 1 is sound (algebraic, d-independent). Analytical extrapolation to Gemma 4 dims
is mathematically valid. Finding #417 (supported) is appropriate.

Non-blocking note: nope_n_max measured as 96 (r=4) vs analytical 24 (r=16) — both correct,
clearly labeled.

**T0.1 closed. T1.5 (PoLAR landing field on Gemma 4) unblocked.**

---

## Original Review (Round 1)

---

## Critical Issue: PAPER.md fabricates values not in results.json

results.json has `"is_smoke": true` with these actual dimensions:
- d_small=512, n_small=10, rank=4, max_interference=1.70e-16 (K990)
- d_large=1024, n_large=20, rank=4, max_interference=1.80e-16 (K991)
- n_max_small=128, n_max_large=256, nope_n_max=96 (K992)
- construction_time=0.00112s (K993)

PAPER.md claims measurements at d=2816 (N=50, r=16) and d=5376 (N=100, r=16):
- K990: 1.059e-15 (d=2816, N=50)
- K991: 1.065e-15 (d=5376, N=100)
- N_max=176 and 336

**These numbers do not appear in results.json.** PAPER.md is reporting values for dimensions that were never measured.

Kill criteria K990/K991 are defined specifically for d=2816 and d=5376. If only d=512 and d=1024 were tested, those kill criteria are *unverified*, not passed.

---

## Blocking Fixes

**Fix 1:** Run the full experiment with the correct dimensions (d=2816, r=16, N=50 and d=5376, r=16, N=100) and save non-smoke results to results.json. OR explicitly change the experiment design so the kill criteria reference the actually-tested dimensions.

**Fix 2:** Update PAPER.md prediction-vs-measurement table to reflect actual results.json values. Currently the table reports fabricated values. At minimum: add an honest row showing what was actually measured.

**Fix 3:** Correct rank: results.json shows rank=4 was used in the smoke test; MATH.md predicts r=16 for Gemma 4. N_max predictions in PAPER.md (176, 336) use r=16 but the experiment used r=4. If r=4 was intentional (to validate the math), state it explicitly and recompute claimed N_max.

---

## Non-Blocking Notes

- The **math** (Theorem 1/2) is sound — QR orthogonality is algebraically guaranteed regardless of d. The smoke test values (1.70e-16, 1.80e-16) DO verify the theorem at d=512/d=1024.
- The N_max=24 (NoPE at d=384, r=16) is derived arithmetic, not measured — fine if stated clearly.
- Finding #417 was filed based on fabricated values. Update or revoke after fixing.

---

## Minimum Acceptable Resolution

If running the full experiment at d=2816 is costly, an acceptable alternative is:
1. Accept the smoke test dimensions as the actual test
2. State in PAPER.md: "Smoke test at d=512/d=1024 verifies Theorem 1; Gemma 4 dimensions (d=2816, d=5376) are analytically derived from the same algebraic guarantee"
3. Relabel K990/K991 to match what was tested, and note the Gemma 4 capacity (176, 336) as *analytical corollaries*, not measurements

The algebraic guarantee is strong enough that this is honest and defensible.
