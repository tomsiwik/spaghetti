# REVIEW-adversarial.md — exp_pierre_kan_adapter_lagrangian

**Verdict: KILL (confirmed)**

## Adversarial Checklist

| # | Check | Result |
|---|-------|--------|
| a | results.json verdict matches DB | N/A — script killed mid-Q3, no verdict field. Kill justified from K1+K2 data alone. |
| b | all_pass matches claim | N/A — same as (a). |
| c | PAPER.md verdict matches DB | PASS — both say KILLED |
| d | is_smoke → provisional | N/A |
| e | KC not modified post-run | PASS — untracked files, KCs in MATH.md match code exactly |
| f | No tautological KC | PASS — K1 tests against real std baseline, K2 against best-single avg, K3 against std K=2 |
| g | Code measures what MATH.md describes | PASS — GSM8K for K1, 3-bench avg for K2/K3 |
| h | No independent lora_A/lora_B sum | PASS — composition via spline coefficient addition (correct for KAN) |
| i | LORA_SCALE < 12 | PASS — scale=6.0 |
| j | No single-sample routing | N/A |
| k | No shutil.copy adapter cloning | PASS |
| l | No hardcoded pass | PASS |
| m | Model consistency | PASS — gemma-4-e4b-it-4bit throughout |
| n | Base accuracy > 0% | PASS — M0=66% |
| o | n ≥ 15 | PASS — N=50 |
| p | Target-metric KC present | PASS — GSM8K, HumanEval, MedQA |

## Notes

- **results.json incomplete**: Script was killed during Q3 (stuck 5+ hours), so verdict/kill_criteria fields were never written. Non-blocking — K1 FAIL (58 < 61) and K2 FAIL (39.3 << 64) are sufficient for kill from available data.
- **Mechanism analysis in PAPER.md is strong**: Correctly identifies that spline coefficient addition is formally valid (weighted B-spline average) but practically catastrophic because averaging task-specialized activation patterns produces a "nothing" function. MedQA 14% < random 25% confirms destructive interference.
- **Connection to Finding #842** (Pico) is well-drawn: nonlinear transformations of B-signal make composition harder, not easier.
- **Code quality**: Clean MLX, no float64, proper dtype casting, correct warm-start initialization.
