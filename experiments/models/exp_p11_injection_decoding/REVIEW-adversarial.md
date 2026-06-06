# REVIEW-adversarial.md — P11.Z1: Injection Decoding

**Verdict: PROCEED** — All blocking fixes applied

Round 1 had 2 blocking fixes. Both confirmed resolved:

---

## Fix 1 (RESOLVED): PAPER.md written

PAPER.md now contains:
- Prediction-vs-measurement table with TBD for full run ✓
- Smoke test findings (N=6, injection never triggered at 500 threshold) ✓
- Honest documentation of ceiling effect in smoke test ✓
- Non-blocking issues noted (PS conflict, Theorem 1 informality) ✓

---

## Fix 2 (RESOLVED): MIN_THINKING_CHARS = 500 → 1500

Confirmed at `run_experiment.py:39`:
```
MIN_THINKING_CHARS = 1500  # Threshold: below this, inject "Wait" (raised from 500: Gemma 4 mean=1641, 500 never triggers)
```

At mean=1641 chars, raising to 1500 gives ~30-50% trigger rate. This makes
K1533 (injection >= base + 1pp) testable rather than definitionally null.

---

## Non-Blocking Issues (documented in PAPER.md, no action needed)

1. **PS_PREFIX vs "Do not explain" conflict**: Question body says letter-only answer, PS
   prefix asks for planning steps. May cause PS conditions to show degradation. Noted in
   PAPER.md — let the data decide.

2. **Theorem 1 monotonicity informal**: Wei et al. 2022 proves CoT > no-CoT but not
   within-CoT monotonic scaling. Noted in PAPER.md as assumption, not derived result.
   Acceptable for Type 2 exploration.

---

## Pending (full run results)

The experiment is queued (pueue task 13). PAPER.md will need TBD rows filled in
after full run. Expected paths:
- K1533 FAIL if injection still rarely fires or adds noise at 1500 threshold
- K1534 PASS (degenerate loops < 5% — max 2 injections per question)
- K1532 uncertain — depends on PS prompt performance alone if injection rarely fires

Core scientific value: establishing that Gemma 4 E4B does NOT under-think (2614 chars
mean) is itself a finding worth documenting, regardless of injection results.

---

## Summary

| Issue | Severity | Status |
|-------|----------|--------|
| PAPER.md missing | Blocking | RESOLVED ✓ |
| 500-char threshold | Blocking | RESOLVED ✓ (→1500) |
| PS/instruction conflict | Non-blocking | Documented in PAPER.md |
| Theorem 1 informality | Non-blocking | Documented in PAPER.md |
