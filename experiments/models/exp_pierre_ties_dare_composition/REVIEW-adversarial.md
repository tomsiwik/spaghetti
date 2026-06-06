# REVIEW-adversarial.md — exp_pierre_ties_dare_composition

**Verdict: KILL confirmed**

## Adversarial Checklist

| # | Check | Result |
|---|-------|--------|
| (a) | results.json verdict matches DB status | PASS — both KILLED |
| (b) | all_pass matches claim | PASS — false, 3/5 fail |
| (c) | PAPER.md verdict matches | PASS — "KILLED" |
| (d) | smoke → provisional | N/A — is_smoke=false |
| (e) | KC not modified after run | PASS — untracked files, no prior version |
| (f) | No tautological KC | PASS — all KCs test real behavioral/structural outcomes |
| (g) | Code measures what MATH.md says | PASS — K2141 swapped PPL for acc ratio (documented, transparent) |
| (h) | Composition math correct | PASS — `scale * (a @ b)` per adapter, then merge on full delta |
| (i) | LORA_SCALE < 12 | PASS — SCALE=6.0 from polar_train.py |
| (j) | No single-sample routing | PASS — weight-space merging, no routing |
| (k) | No shutil.copy fake | PASS |
| (l) | No hardcoded pass:True | PASS |
| (m) | Model match | PASS — gemma-4-e4b-it-4bit in both |
| (n) | Base accuracy 0% | FLAG — strategy_full medqa=0.0, but other adapters non-zero; not truncation |
| (o) | n >= 15 | PASS — n=30 |
| (p) | Target-metric KC present | PASS — K2140 (beats single) and K2141 (acc ratio) are target metrics |

## Key Assessment

The KILL is correct. K2138/K2139 fail by 1.7pp on gsm8k only (6.7pp vs 5pp threshold). K2142 sparsity is genuinely below 70%. The target-metric KCs (K2140, K2141) pass — DARE composition works (+4.4pp avg), just not within the strict 5pp-per-bench threshold.

The transformative finding is the `__call__` bug fix: proper `_FusedDeltaLinear(nn.Module)` replacement rescues uniform 1/N from catastrophic collapse (HumanEval 20% -> 90%). Three prior composition KILLs were implementation bugs, not theoretical failures.

Finding #829 added. Experiment completed as killed.
