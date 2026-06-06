# Learnings — K=2 Strategy x Domain Composition

## Finding #839: Strategy adapter destructively interferes with code specialization

K=2 (strategy_full + domain_code) gets 66% HumanEval vs 78% domain_code alone (-12pp).
Strategy adapter overwrites code-specific features. Not a composition method failure —
Fisher-Rao, DARE both show it. The adapters themselves interfere.

## Finding #840: K=7 outperforms K=2 by 10.7pp — more adapters help, not hurt

K=2 FR avg=54.0% vs K=7 FR=64.7%. The "less interference at K=2" hypothesis is wrong.
Additional adapters at K=7 provide complementary signal that compensates for pairwise interference.
This closes the K=2-is-cleaner theory.

## Finding #841: DARE fails at K=2 but Fisher-Rao survives

DARE (drop_rate=0.9) is too aggressive with only 2 adapters. Math pair: DARE=49.3% vs FR=50.7%.
Code pair: DARE=42.7% vs FR=57.3%. Exception: medical pair DARE=62% > FR=54% — pruning
removes strategy interference for medical domain.

## Eval runner bug fixes (reusable)

1. `adapter_names_override` caused KeyError in single_best loop — fixed with skip check.
2. `compose_fisher_rao` signature mismatch with b_only call convention — added `A_dict=None` param.
Both fixes in `_pierre_shared/eval_runner.py`, tested across 3 pairs.
