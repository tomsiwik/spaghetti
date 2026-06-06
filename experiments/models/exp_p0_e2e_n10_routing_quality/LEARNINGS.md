# LEARNINGS: exp_p0_e2e_n10_routing_quality

## Core Finding
E2E pipeline scales from N=3 to N=10 with only 4pp max quality loss. Combined logistic
routing at N=10 achieves 90.7% accuracy, and Theorem 1 (quality_loss = (1-α)*(Q_oracle - Q_base))
is validated within 1.2pp on all 3 benchmarks.

## Why
Routing errors overwhelmingly fall to semantically adjacent non-adapter domains (medical→psychology,
code→engineering), producing base-model fallback rather than wrong-adapter routing. This keeps
observed loss below the Theorem 1 upper bound. Distribution shift is a non-issue: pure benchmark
text (code, word problems) routes as well or better than MMLU training data.

## Key Numbers
- GSM8K: 77% routed (0pp loss from oracle), HumanEval: 56% routed (1pp loss), MedMCQA: 54% routed (4pp loss)
- N=3→N=10: routing accuracy drops 9pp but max quality loss only grows from 0pp to 4pp (sub-linear)
- Theorem 1 prediction at N=25 (Finding #531: 88.8% routing): ~5pp max loss expected

## Implications for Next Experiment
The weakest link is medical routing (86% due to medical↔psychology semantic overlap). Improving
routing accuracy in overlapping domains would directly reduce quality loss. At N=25+, wrong-adapter
routing risk grows as more adapters cover adjacent domains — the Theorem 1 upper-bound assumption
(misrouted → base) may break down and actual loss could exceed predictions.
