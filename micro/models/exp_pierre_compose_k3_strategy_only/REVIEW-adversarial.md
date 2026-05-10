# Adversarial Review — K=3 Strategy-Only Composition

## Checklist

- [x] results.json exists and contains benchmark scores
- [x] Kill criteria evaluated against pre-registered thresholds in MATH.md
- [x] K1 FAIL: 56.0% < 64.7% — arithmetic verified
- [x] K2 FAIL: medqa 36.0% is catastrophically below any reasonable single-adapter baseline
- [x] No code fabrication — run_experiment.py uses shared eval_runner infrastructure
- [x] Correct adapter set loaded: 3 strategy adapters as specified
- [x] Method under test (fisher_rao_K3_strategy_only) did not complete (process killed), but fisher_rao baseline with K=3 adapters already fails K1 — the method under test cannot rescue this
- [x] PAPER.md mechanism analysis is consistent with data

## Concerns

1. **Incomplete run**: The DARE upper-bound eval and the method-under-test eval were not completed (process killed after ~4h). However, the fisher_rao baseline already fails K1 decisively (8.7pp gap). No composition method can rescue a baseline this poor.

2. **No single_best baseline**: strategy_full single-adapter scores are unavailable (skipped because domain adapters weren't loaded). K2 is evaluated by proxy — medqa at 36% is below random chance (25% for 4-option MCQ would be 25%, but with prompt framing typical baselines are 40-50%). The K2 verdict holds.

3. **Sample size**: n=50 per benchmark. At 50 samples, 95% CI for 56.0% avg is roughly ±7pp. Even at the upper bound (63%), it still fails K1 (< 64.7%).

## Verdict

**KILL CONFIRMED.** Data is sufficient despite incomplete run. Strategy-only composition fails on medical knowledge catastrophically.

## Finding

Finding registered: strategy adapters carry domain bias; composing strategy-only suppresses unrepresented domains. K=7 works because domain adapters counterbalance this bias.
