# Adversarial Review: exp_p9_memoryllm_pool_gemma4

## Verdict: PROCEED (KILLED, Finding #512)

### Post-REVISE Verification

Both blocking fixes from Round 1 applied correctly:

1. **PAPER.md status SUPPORTED → KILLED** ✓ (line 123: "## Status: KILLED")
2. **results.json K1366 pass:true → pass:false** ✓ (line 9: `"pass": false` with explanatory note)

### Kill Criteria vs Evidence

| Kill | results.json | PAPER.md | Consistent? |
|------|-------------|----------|-------------|
| K1366 recall >50% | HS rate 0.0, pass:false | 0% HS / 100% context, FAIL | ✓ |
| K1367 latency <5ms | avg 2.82ms, pass:true | 2.82ms avg, PASS | ✓ |
| K1368 quality ±2pp | delta +30pp, pass:false | INCONCLUSIVE (prompt bug) | ✓ |

### Non-blocking Issues (carried from Round 1)

1. **Theorem 3 identified position risk as secondary.** The "Failure mode" paragraph correctly flagged position encoding mismatch but treated it as a secondary risk. In hindsight, it should have been the PRIMARY analysis. Not blocking — MATH.md at least included it.

2. **K1368 test invalid.** Base accuracy 0% due to prompt formatting. PAPER.md honestly labels INCONCLUSIVE. Acceptable.

3. **Write latency prediction 2x off.** Predicted <1ms, measured 2.82ms. Still passes. mx.eval sync overhead explained.

4. **Impossibility structure is the key deliverable.** Three independent reasons (RoPE corruption, untrained attention, per-layer inconsistency) are well-derived and actionable.

### What's Good

- MATH.md has 4 theorems, all with testable predictions
- Impossibility structure clearly identifies 3 independent failure mechanisms
- Alternative paths (KV-cache prefilling, PLE injection) properly identified
- KILLED status is correct — 0% vs predicted 70% is decisive refutation
- Finding #512 already recorded with proper metadata
