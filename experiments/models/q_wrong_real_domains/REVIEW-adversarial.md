# REVIEW-adversarial.md — exp_q_wrong_real_domains

**Verdict: PROCEED**

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match evidence (results.json confirms K944 PASS, 3 pairs)
- [x] Finding status (supported) is appropriate for measurement type
- [x] No fabricated numbers — results.json matches PAPER.md exactly

## What's Strong

- Large, directional effects (-57%, -58%, 0%) that don't require statistics to interpret
- Mechanism is well-supported: generation output example shows "#### N" injection pattern
- Structural explanation (format compatibility determines Q_wrong sign) is coherent
- Connection to TF-IDF routing (Finding #354) closes the loop on why routing matters

## Non-Blocking Issues

1. **Missing confidence intervals**: At n=50, Wilson 95% CIs for sort (6/50: [0.055, 0.243]) and reverse (13/50: [0.154, 0.396]) would formally bound the estimates. With |Q_wrong| ≈ 0.57, the effect is large enough that CIs don't change the conclusion. Fine for a measurement experiment.

2. **Finding number inconsistency**: MATH.md cites "Finding #354" for TF-IDF routing; PAPER.md says "Finding #381". One of these is wrong. Non-blocking — the routing result is real regardless.

3. **count_even exact tie (18/18)**: Same examples correct under base and adapted — plausible because numeric "Answer: N" format is compatible with math adapter's "#### N" format, so the adapter produces small perturbation. Deserves a parenthetical note but doesn't affect K944.

## Summary

Clear structural result: format injection explains Q_wrong sign. Math adapter is harmful on language tasks (-57–58%), neutral on numeric tasks (0%). Routing is confirmed non-optional. Status=supported is correct. No revisions required.
