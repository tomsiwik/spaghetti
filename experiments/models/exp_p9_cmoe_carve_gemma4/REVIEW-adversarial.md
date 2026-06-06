# Adversarial Review: exp_p9_cmoe_carve_gemma4 (Post-REVISE)

## Verdict: PROCEED (KILLED)

All 3 REVISE fixes applied and verified. Re-run confirms structural failure.

## Verification

1. **Prediction-vs-measurement table**: Present in PAPER.md. All 3 kill criteria
   listed with predictions from MATH.md and actual measurements from results.json.

2. **Kill criteria match evidence**:
   - K1342: results.json shows `ppl_degradation_pct: 219.48` — matches PAPER.md "+219.5%". FAIL.
   - K1343: results.json shows `total_carve_time_s: 12.96` — matches PAPER.md "13s". PASS.
   - K1344: results.json shows `speedup: 0.42` — matches PAPER.md "0.42x". FAIL.
   - Decomposition exactness verified: max_diff 2.86e-6 (layer 0). Theorem 1 confirmed.

3. **Finding status**: KILLED is correct. 2/3 kill criteria fail with structural
   impossibility identified (MLX dispatch barrier for N small matmuls).

4. **Math review**: Theorem 1 (exact decomposition) is trivially correct and
   empirically verified. Theorem 3 (speedup prediction of 1.4-1.8x) was wrong —
   it assumed O(k) compute, but implementation is O(N) and MLX dispatch overhead
   dominates. PAPER.md correctly identifies this.

## Non-Blocking Issues (acknowledged in PAPER.md)

1. **Tautological PPL test**: Base PPL 37K (IT model on raw text) makes the PPL
   measurement uninterpretable. PAPER.md acknowledges this as "Root Cause #1."

2. **Dequantization confound**: No control for dequantized-not-carved baseline.
   Noted in PAPER.md resurrection criteria.

3. **Theorem 2 bound is qualitative**: The 1-3% PPL prediction was not derived
   from the bound — it was a guess. Moot since the experiment is killed.

4. **O(N) compute implementation**: CMoELayer runs all experts then masks.
   PAPER.md correctly attributes 0.42x speed to this + MLX dispatch overhead.

5. **Impossibility structure is sound**: N small matmuls < 1 fused matmul on MLX
   is a platform-level constraint, not a hyperparameter issue. Resurrection
   requires O(k) conditional compute which MLX doesn't efficiently support.

## Conclusion

Clean kill. Decomposition math is correct (verified). Failure is structural:
MLX dispatch model makes CMoE speed gains impossible at current expert granularity,
and quality measurement was confounded by eval distribution mismatch. PAPER.md
provides clear resurrection criteria for future work.
