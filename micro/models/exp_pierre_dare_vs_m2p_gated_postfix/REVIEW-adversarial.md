# Adversarial Review — exp_pierre_dare_vs_m2p_gated_postfix

## Verdict: KILL (confirmed)

## Checklist

| # | Check | Result |
|---|-------|--------|
| a | results.json verdict matches DB status | PASS |
| b | all_pass matches claim | PASS |
| c | PAPER.md verdict matches | PASS |
| d | smoke → provisional guard | N/A (not smoke) |
| e | KC not modified post-run | PASS |
| f | No tautological KC | PASS |
| g | Code measures what MATH.md describes | PASS |
| h | Composition: Σ(A_i @ B_i) not (ΣA)@(ΣB) | PASS |
| i | SCALE < 12 | PASS (imported from polar_train) |
| j | Routing per-sample not global | PASS (per-prompt, bucketed top-2) |
| k | No shutil.copy adapter duplication | PASS |
| l | No hardcoded pass | PASS |
| m | Model in MATH.md = model in code | PASS |

## Non-blocking flags

- n=30 per benchmark (adequate)
- All 3 KCs are target-metric (task accuracy)

## Notes

Clean experiment. Pre-registered decision tree followed correctly. Gate routing works (99.6% holdout) but continuous composition underperforms DARE's stochastic pruning by 10pp. Calibration signal absent (ρ=0.097). Decision to ship DARE alone is well-supported.
