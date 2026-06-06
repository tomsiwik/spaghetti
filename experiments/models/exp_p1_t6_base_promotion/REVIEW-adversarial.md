# REVIEW-adversarial.md — T6.3: Base Promotion

## Verdict: PROCEED

All four kill criteria pass. Math is sound. Caveats are clearly stated. SUPPORTED status is appropriate.

---

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match evidence (verified against results.json)
- [x] Finding status (SUPPORTED) appropriate for experiment type (verification on synthetic data)
- [x] No fabricated evidence

---

## Issues Found

### Non-blocking: K1125 threshold inconsistency

MATH.md states: "Prediction K1125: max_layer(ε_layer) < 0.05 implies MMLU change < 1pp" (5% threshold).
PAPER.md and results.json use threshold = 10%.

Measured max_ε = 4.78%, which passes **both** thresholds. Not blocking.
Recommendation: align MATH.md and PAPER.md to use the same threshold (10%) in T6.4.

### Non-blocking: K1127 trainability evidence is weak

Loss ratio = 0.9994 over 5 steps (0.06% decrease). This is essentially flat — the gradient
is non-zero but barely. The theorem's claim (gradient magnitude unchanged) is correct in
principle but the 5-step test on a single layer with random init is too short to be
behaviorally meaningful.

T6.4 should include 50+ steps of real adapter training on the promoted base to confirm
convergence speed is similar to training on the original base.

### Non-blocking: Synthetic base weights

W_base uses std=0.05 (synthetic). Real Gemma 4 weights have larger ||W||_F → lower ε
(more favorable). Caveat clearly stated in PAPER.md. T6.4 addresses this.

### Non-blocking: A-matrix handling

T6.2 crystallized B-matrices only; canonical A was used for promotion. In practice with
multiple users having different A-matrices, the promoted ΔW = A_canonical @ B_crystal
may not represent the true domain centroid. This is future work correctly flagged.

---

## Mathematical Soundness

- **Theorem 1** (formula exact): Correct. Standard linear algebra. Verified by experiment (cos=0.99999988).
- **Theorem 2** (Davis-Kahan): Valid application. ε < threshold → spectral gap preserved. Finding #333 provides reasonable empirical backing.
- **Theorem 3** (slot liberation): Trivially correct by construction. Verified.
- **Theorem 4** (trainability): Proof is correct — adapter gradient path is through A_new, not through W_promoted. The 5-step test confirms non-zero gradients.

---

## Flywheel Structural Assessment

The T6.1→T6.2→T6.3 chain is structurally complete for single-cycle flywheel:
- T6.1: Cluster adapters into domains (silhouette=0.82) ✓
- T6.2: Crystallize domain (cos=0.9806, +6.5pp, 80% compression) ✓
- T6.3: Promote crystal → base (exact, ε=3.6%, slot freed, trainable) ✓

T6.4 (sequential cascade on real model) is the right next step. Cascade failure risk
is real: N sequential promotions each add ε; after K promotions, cumulative sin(θ) ≤ 1
is not guaranteed. This is correctly identified as the outstanding risk.

---

## Decision

**PROCEED** — SUPPORTED status confirmed. T6.4 is the correct next experiment.
