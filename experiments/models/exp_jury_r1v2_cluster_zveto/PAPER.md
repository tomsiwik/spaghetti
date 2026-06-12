# PAPER — Jury R1v2: per-question z-scored verifier as cluster veto+reweight

## Question
Can GRPO-style within-question z-normalization of the R1 verifier scores (F#877 data) convert
pass@8 headroom into accuracy over self-consistency, via cluster veto (maxz < −1) + reweight
(alpha·vote + (1−alpha)·maxz) with a guess gate (σ_q < τ falls back to SC)?

## Method
Pure reanalysis of cached R1 data (`exp_bet_jury_r1_verifier_gain/results.json`, 200 questions ×
8 candidates, ZERO new tokens). Even half (100 q) used to tune alpha and τ over a 21×8 grid;
odd half (100 q) held out. Controls: self-consistency, raw-vscore BoN, z-only top-1.

## Prediction vs measurement (held-out, odd half, n=100)

| Quantity | Predicted | Measured |
|---|---|---|
| Jury accuracy | ≥ SC + 3pp (≥ 0.86) | **0.84** |
| Win-flips | > 5 | **3** (vs 2 loss-flips) |
| Gain over SC | ≥ +3pp | **+1pp** (0.84 vs 0.83) |

Controls (held-out): SC@8 = 0.83, raw-vscore BoN@8 = 0.80, z-top1 = 0.80.
Tuned config: alpha = 0.8, τ = 2.0; tune-half jury = 0.85 vs SC 0.81 (+4pp did not transfer).

## Analysis
- Z-normalization fixes the BoN pathology: jury (0.84) no longer loses to SC, unlike raw BoN
  (0.80). The within-question relative-score claim is directionally right.
- But the net effect is +1pp from 3 win-flips minus 2 loss-flips — binomial noise at n=100.
- The grid plateau (alpha=1.0 rows all at 0.84 tune acc, best 0.85 at alpha=0.8) shows the
  verifier signal adds at most one question over pure voting; the tune-half +4pp was overfitting
  to the grid.

## Verdict
**KILLED** (kill #2332): held-out 0.84 < SC+3pp gate (0.86) AND win-flips 3 ≤ 5. The z-scored
cluster jury does not convert pass@8 headroom (0.935) into accuracy beyond noise; it only repairs
raw-BoN degradation back to SC parity.
