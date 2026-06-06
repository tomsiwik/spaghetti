# LEARNINGS.md — exp_pierre_adapter_cross_matrix

## Core Finding
Every Pierre adapter is a general capability booster, not a domain specialist. All 7 adapters lift HumanEval from 22% to 34–78%. strategy_full is pathological: it zeros MedQA (−6pp), while its decomposed sub-adapters (act, integrate) achieve 30–32% without catastrophic interference.

## Why
The adapters primarily teach instruction-following and reasoning structure, not domain-specific knowledge. strategy_full over-fits to GSM8K/HumanEval patterns at the expense of medical reasoning, but the decomposed sub-strategies avoid this by not co-optimizing all three capabilities in one weight matrix.

## Implication for Next Experiment
1. Drop strategy_full permanently — decomposed sub-adapters strictly dominate.
2. Domain adapters are interchangeable enough that routing can be relaxed (domain_medical gets 68 on GSM8K, matching strategy_act).
3. K=2 routing should pair one strategy sub-adapter + one domain adapter, not two domain adapters. Best safe pairs: act+medical for GSM8K, code+prepare for HumanEval, medical+math for MedQA.
