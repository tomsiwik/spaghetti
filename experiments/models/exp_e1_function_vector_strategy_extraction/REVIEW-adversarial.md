# REVIEW: E1 Function Vector Strategy Extraction

## Verdict: KILL

## Adversarial Checklist

| Check | Result | Note |
|---|---|---|
| (a) results.json verdict vs DB | MISMATCH | results.json=KILLED, DB=provisional, PAPER=PROVISIONAL. Resolving to KILL. |
| (b) all_pass vs claim | OK | K2018 failed; results.json correctly says KILLED |
| (c) PAPER.md verdict | INCONSISTENT | Says PROVISIONAL; results.json says KILLED. Resolving to KILL. |
| (d) is_smoke vs claim | N/A | Not claiming supported |
| (e) KC pre-reg mutation | OK | First run, no mutation |
| (f) Tautology | FLAG | K2017/K2019 pass vacuously (cross-strategy cos=0.99). Researcher correctly identified this. |
| (g) Code vs MATH.md | OK | Measurements match descriptions |
| (h) Buggy composition | OK | No independent A/B summing |
| (i) LORA_SCALE | OK | 6.0 (safe) |
| (j) Single-sample routing | N/A | No routing |
| (k) shutil.copy | OK | Not present |
| (l) Hardcoded pass | OK | Not present |
| (m) Model match | OK | Both MATH.md and code use gemma-4-e4b-it-4bit |
| (m2) Skill invocation | MINOR | Skills not cited in MATH.md, but code is idiomatic MLX (mx.eval, mx.clear_cache, LoRALinear.from_base) |
| (n) Base=0% truncation | OK | Base=30% (3/10), not truncated |
| (o) N<15 stats | FLAG | N=10 GSM8K, but structural evidence (cos=0.99) is not sample-dependent |
| (r) Prediction table | OK | Present in PAPER.md |
| (t) Target-gated kill | PASS | K2018 is target metric, K2018 FAILED. Kill is on target, not proxy. |

## Kill Rationale

The mechanism failure is structural, not statistical:

1. **Cross-strategy cos > 0.98** at the best extraction layer. The mean-difference method captures "system prompt present vs absent" — format signal dominates strategy signal. This is an algebraic property of the extraction method, not a sampling artifact.

2. **Target KC K2018 = 0pp** (30% → 30%) with identical per-problem answers baseline vs adapted. The injected adapter produced zero behavioral change.

3. Per F#666: proxy-PASS (K2017, K2019) + target-FAIL (K2018) = "tautological proxy, kill on target." The proxies passed because they measured format consistency, not strategy consistency.

4. Full-N cannot fix this: cos=0.99 cross-strategy is a ceiling on the method, not a finite-sample effect. A full run would waste compute confirming what's already structurally determined.

## Smoke Caveat Override

`is_smoke: true` normally triggers PROVISIONAL. Overriding because:
- The kill signal is structural (method-level), not statistical
- The PAPER.md itself states "A full run is unlikely to change the 0pp delta or the 0.99 cross-strategy similarity"
- Spending 10x compute to confirm cos=0.99 → still cos=0.99 is not informative

## Assumptions

- Judged that cos=0.99 cross-strategy is conclusive even at N=10, because it reflects method design (mean-difference conflates format and strategy), not sampling variance.

## Finding

The mean-difference activation extraction method (Function Vectors 2310.15213) fails for strategy vectors because problem-solving strategies are not narrow input-output functions — they are broad behavioral modes where format signal (instruction-following mode) dominates strategy signal by ~100x in activation magnitude. The PAPER.md's proposed fixes (contrastive extraction, residual subtraction, head-level selection) are plausible but would constitute a different experiment.
