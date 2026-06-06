# Adversarial Review: exp_p10_budget_forcing

## Verdict: REVISE (1 blocking fix)

## Summary

Budget forcing experiment completed. Results clearly KILL the hypothesis:
no budget < 2048 achieves 90% retention (K1464 FAIL), no token savings possible
(K1465 FAIL). The Gamma CDF model from MATH.md is catastrophically wrong —
all 5 predictions off by 15-43pp. But PAPER.md is missing entirely.

## Blocking Fixes

**1. Write PAPER.md with prediction-vs-measurement table and analysis.**
Must include:
- Prediction vs. actual table (all 5 budgets)
- The cliff effect: B=128-512 all ~10-12% (WORSE than base no-thinking 41.7%)
- The threshold: massive jump from 11.4% (B=512) to 34.3% (B=1024)
- Why the Gamma CDF model fails: thinking isn't gradual degradation, it's binary (coherent chain or garbage)
- B=2048 = 46.7% vs Finding #530 = 62.1% discrepancy: likely N=15 vs N=20 sample variance, but must acknowledge
- Kill criteria table with evidence strings

## Non-Blocking Observations

1. **Gamma CDF model is structurally wrong.** The model assumes truncated thinking
   reverts to base accuracy. Reality: truncated thinking is WORSE than no thinking
   (10.5-11.9% vs 41.7% base). The model produces incomplete reasoning that
   actively misleads its own answer generation.

2. **Thinking chars confirm the mechanism:** B=128 generates 0 thinking chars,
   B=256 generates 6.6 chars/question, B=512 generates 73 chars/question.
   The model never enters a useful thinking mode at low budgets — it burns tokens
   on format/preamble and never reasons.

3. **Non-monotonicity at B=512 < B=256** (11.4% vs 11.9%) is consistent with
   "partial thinking is worse than no thinking" — at B=512 the model starts
   thinking but can't finish, while at B=256 it barely tries and sometimes
   guesses correctly.

4. **The real finding:** There exists a critical threshold (~1024 tokens) below
   which thinking is actively harmful. This is a strong result that extends
   Finding #530 and has implications for adaptive compute allocation.

## Status Assessment

KILLED is the correct status. Budget forcing with fixed budgets does not work
for 4-bit quantized models. The overhead cannot be reduced without catastrophic
accuracy loss. Best strategy remains: full thinking (B=2048+) or no thinking,
with no useful middle ground.
