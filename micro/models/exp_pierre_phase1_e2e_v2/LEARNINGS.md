# Learnings — exp_pierre_phase1_e2e_v2

## Core Finding
The original Phase 1 kill (avg 26.7%) was a false-kill caused by the `m.__call__` infrastructure bug (Finding #831). Re-run with the fix produces K=6 Fisher-Rao avg 61.3% — composition works, no benchmark collapses.

## Why
The bug prevented adapter weights from being applied during inference. With the fix, all benchmarks score ≥56%. Composition even lifts MedQA from 42% (single_best) to 56%, confirming cross-domain transfer through strategy adapters.

## Implication for Next Experiment
K=6 still trails K=7 reference by 3.4pp — dropping `strategy_integrate` loses signal. Ship all 7 adapters. The 10.7pp gap to DARE (72.0%) remains the ceiling to chase; composition method improvements (routing, weighting) are the path, not adapter pruning.
