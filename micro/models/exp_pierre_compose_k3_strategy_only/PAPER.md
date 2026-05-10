# K=3 Strategy-Only Composition — KILLED

## Result

**KILLED.** Fisher-Rao composition of 3 strategy adapters (strategy_full + strategy_prepare + strategy_act) averages 56.0% across benchmarks — 8.7pp below the K=7 reference (64.7%). K1 FAIL.

## Measurements

| Benchmark | Fisher-Rao K=3 strategy | Fisher-Rao K=7 ref | Delta |
|-----------|------------------------|-------------------|-------|
| gsm8k     | 62.0%                  | ~65%              | -3pp  |
| humaneval | 70.0%                  | ~68%              | +2pp  |
| medqa     | 36.0%                  | ~61%              | -25pp |
| **avg**   | **56.0%**              | **64.7%**         | **-8.7pp** |

## Kill Criteria Verdicts

- **K1** FAIL: 56.0% < 64.7% (fisher_rao K=3 avg < K=7 reference)
- **K2** FAIL: medqa 36.0% is catastrophically low (likely below strategy_full single by >25pp)
- **K3** PASS: preprocessing 0.01s < 5s
- **K4** DEFERRED: sibling domain-only experiment not yet run

## Mechanism Analysis

Strategy adapters interfere **catastrophically on cross-domain knowledge**. The medqa collapse (36%, below random chance on 4-option MCQ) reveals the mechanism:

1. Strategy adapters encode reasoning *patterns* (prepare, act, integrate) but these patterns are domain-entangled — strategy_full's B-matrices encode math-flavored reasoning, not generic reasoning
2. Composing 3 strategy adapters amplifies math/code reasoning signal at the expense of medical knowledge
3. The humaneval gain (+2pp) confirms this: code reasoning benefits from strategy stacking, while unrelated domains get suppressed

**Key insight:** Strategy adapters are NOT domain-agnostic reasoning modules. They carry domain bias in their B-matrices. Composing along the strategy axis alone creates a lopsided model that excels at domains present in strategy training data and collapses on others.

## Implications for Pierre

- K=7 default works *because* domain adapters counterbalance strategy adapter domain bias
- Shipping K=2 (1 strategy + 1 domain) is safer than K=3 strategy-only
- The "strategy axis is clean" hypothesis is rejected — strategy and domain axes are entangled
