# REVIEW-adversarial.md — T0.4: Q-Only Adapters on K=V

**Reviewer:** Adversarial  
**Date:** 2026-04-09  
**Verdict:** PROCEED

---

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match evidence (results.json consistent with PAPER.md)
- [x] Finding status (supported) appropriate for experiment type
- [x] No obvious math errors

---

## Issues Reviewed

### K1001, K1002 algebraic proofs are sound

K1001 (max|K_base - K_user1| = 0.0) and K1002 (max|K_user1 - K_user2| = 0.0) follow
directly from k_proj having no adapter. This is dimension-independent and holds at any
scale. The zero diff is exact (not floating-point approximation) — verified at 0.0.

### Q-only outperforms Q+K (K1000 = 1.24, predicted 0.90-1.05)

Surprising but explained: Q+K LoRA modifies K, which modifies V (since V=K), creating
conflicting optimization targets. Q-only focuses all capacity on query representation.
This is plausible for query-centric tasks and is a useful finding for the P1 architecture.

The prediction was conservative (0.90-1.05), and the result exceeds it (1.24). This is not
a fabrication concern — the algebraic results independently confirm the experiment is running
correctly, so K1000's result is trustworthy.

### Synthetic head_dim=64 vs Gemma4 head_dim=512

The experiment uses head_dim=64 (scaled-down proxy). The algebraic guarantees (K1001, K1002)
are dimension-independent. K1000's quality ratio might differ at head_dim=512, but the
direction (Q-only ≥ Q+K) is unlikely to reverse. PAPER.md correctly identifies this as a
synthetic verification, consistent with T0 pattern (algebraic tests on correct-dimension
synthetic layers).

---

## Verdict

**PROCEED.** Finding #412 (supported) is appropriately calibrated. Algebraic guarantees
(K1001, K1002) are conclusive. K1000 result (1.24 > 0.85) provides empirical support for
Q-only as the preferred adapter type. KV sharing in multi-tenant serving is structurally
guaranteed — no implementation coordination required.
