# MATH.md — DARE vs M2P-gated Re-run (with Finding #831 fix applied)

## Problem
Original `exp_pierre_m2p_gated_composition` killed at humaneval=20% — false kill from `__call__` override bug (Finding #831). Need head-to-head with the bug fixed.

## Test
Same 7 PoLAR adapters, same eval slice. Both methods use `_FusedDeltaLinear`.
- M2P-gated: continuous learned gate weights (peaked via entropy penalty)
- DARE: drop 90% + rescale 10× + linear average

## Decision tree
- M2P-gated > DARE → ship M2P-gated (richer calibration story)
- M2P-gated ~ DARE → ship DARE (simpler) + use M2P confidence as separate signal
- M2P-gated < DARE → ship DARE alone, document M2P-gated as not worth complexity

## References
- Finding #831 (composition fix)
- Tier 1 of Pierre Phase 1 ablation
