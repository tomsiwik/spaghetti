# REVIEW-adversarial.md — T6.5: Dynamic Adapter Registry

**Verdict: PROCEED (with caveats)**

---

## What This Experiment Actually Tests

This is a pure algebraic integration test — no model inference. It verifies four registry
operations (register, remove, promote, crystallize) complete within time bounds and that
the *final* adapter set maintains Grassmannian consistency (max pairwise cos < 0.15).

All 5 kill criteria pass. MATH.md theorems are formally correct. PAPER.md prediction table
is complete and accurate.

---

## Non-Blocking Issues

### 1. Kill thresholds are trivially easy
K1132 threshold is 5s; measured 1.2ms (4167× margin). K1133 is 1s; measured 1.5μs (667K× margin).
These bounds cannot distinguish a working implementation from a trivially fast one.
The experiment tests *existence of correctness*, not *tight performance bounds*.
This is acceptable for a verification experiment, but the thresholds should not be cited
as performance guarantees.

### 2. K1136 "throughout" claim overstated
MATH.md claims "all operations maintain max cos < 0.15 throughout." In practice,
max_cos reached 0.9580 during registration of user variants (same-domain adapters).
The code notes "Register regardless (for testing — real system would reject)."
The K1136 pass is on the *final state only*, not intermediate states.
This is not a flaw in the math (same-domain variants are expected to be similar),
but the "throughout" language should be clarified: the invariant holds *across domains*,
not within a cluster slot before crystallization.

### 3. Design couples T6.2/T6.3 results without new evidence
The promote ε results (3.63%/4.78%) are identical to Finding #452 because the same
synthetic weights (std=0.05) and same formula are used. This is a registry integration
test, not a new measurement of promote correctness. The value here is showing that
the four operations compose correctly in sequence.

---

## No Blocking Issues

- MATH.md theorems are formally correct with proper citations
- PAPER.md has prediction-vs-measurement table
- is_smoke: false (full run complete)
- all_pass: true, results consistent with prior findings

---

## Verdict: PROCEED

Status: **SUPPORTED** — algebraic registry operations verified, lifecycle composition
correct (8→remove→promote→crystallize→4 adapters, final max_cos=0.1221 < τ).
