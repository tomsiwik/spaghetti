# REVIEW-adversarial.md — exp_pierre_pico_rescues_dare_b

## Verdict: PROCEED (kill confirmed)

## Adversarial Checklist

| # | Check | Result |
|---|-------|--------|
| a | results.json verdict matches DB status | PASS — both KILLED |
| b | all_pass matches claim | PASS — 3/4 KC FAIL, correctly reported |
| c | PAPER.md verdict matches DB status | PASS — KILLED |
| d | is_smoke → provisional | N/A |
| e | KC not modified after first run | PASS — thresholds match MATH.md |
| f | No tautological KC | PASS — all KCs test real composition quality |
| g | Code measures what MATH.md describes | PASS — Pico SVD → DARE dropout → mean → norm-rescale |
| h | No independent A/B summation | PASS — B-only method, A unused |
| i | LORA_SCALE < 12 | PASS — scale=6.0 |
| j | No single-sample routing leak | PASS — no routing |
| k | No shutil.copy adapter cloning | PASS |
| l | No hardcoded pass | PASS |
| m | Model consistency | PASS — gemma-4-e4b-it-4bit throughout |
| n | Base accuracy nonzero | PASS — single_best 62.0% |
| o | n ≥ 15 | PASS — n=50 |
| p | Target-metric KC present | PASS — K1/K2 avg accuracy, K4 MedQA |

## Minor Note

K4 `k4_value: null` in results.json (not programmatically extracted), but MedQA=20% is present in `methods.pico_then_dare_b.medqa`. Non-blocking for a kill.

## Scientific Value

Clean negative result with strong interpretive power: Pico anti-synergy with DARE proves the multiplicative-interaction hypothesis is dominant over concentrated-B. This closes the B-space DARE rescue line definitively.
