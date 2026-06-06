# REVIEW-adversarial.md — Pierre vs Raw Gemma 4 baseline

## Adversarial Checklist

| # | Check | Result | Notes |
|---|-------|--------|-------|
| (a) | results.json verdict matches DB status | PASS | INCONCLUSIVE in results.json, killed in DB (correct mapping) |
| (b) | results.json all_pass matches claim | PASS | No all_pass field; verdict=INCONCLUSIVE with K4 fail — consistent |
| (c) | PAPER.md verdict matches DB status | PASS | PAPER.md says INCONCLUSIVE |
| (d) | is_smoke → provisional | N/A | Not a smoke run |
| (e) | KC not modified after first run | PASS | MATH.md thresholds match results.json exactly |
| (f) | No tautological KC | PASS | All 4 KCs test non-trivial claims |
| (g) | Code measures what MATH.md describes | PASS | 3 configs (raw/FR/oracle), 3 benchmarks, 4 KCs as specified |
| (h) | No independent lora_A/lora_B summation | N/A | Fisher-Rao on B-dicts, shared A installed separately — correct |
| (i) | LORA_SCALE < 12 | PASS | SCALE=6.0 |
| (j) | No single-sample routing applied to all | N/A | Oracle is per-benchmark, not per-sample routing |
| (k) | No shutil.copy of sibling adapter | PASS | Each adapter loaded from its own state |
| (l) | No hardcoded pass/True | PASS | KC booleans computed from measured values |
| (m) | Model in MATH.md = model in code | PASS | Both use mlx-community/gemma-4-e4b-it-4bit |

**Non-blocking flags:**
- (n) Raw MedQA=6% is below chance (25%) — strong evidence of eval format dependency. Already captured as K4 failure.
- (o) N=50 — above threshold. Noise ±7pp on HumanEval pass@1.
- (p) Target-metric KCs — gsm8k/humaneval/medqa are behavioral. PASS.

## Verdict: KILL confirmed

All blocking checks pass. K4 failure is genuine: raw HumanEval=16% and MedQA=6% are far below expected floors, indicating the eval pipeline is adapter-dependent. The INCONCLUSIVE verdict is the correct outcome per pre-registered criteria.

Pierre's absolute scores (64.7% avg) are credible but the +36.7pp delta cannot be trusted as a clean measurement of adapter value.
