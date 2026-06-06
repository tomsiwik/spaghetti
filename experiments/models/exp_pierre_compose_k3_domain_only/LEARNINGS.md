# Learnings — exp_pierre_compose_k3_domain_only

## Core Finding

K=3 domain-only composition (domain_math + domain_code + domain_medical) averages 46.0% — 16pp below single-best and 10pp worse than K=3 strategy-only. Domain adapters are MORE interference-prone than strategy adapters, not less.

## Why

Domain adapters share structural weight-space patterns (attention heads, MLP projections) despite targeting different knowledge domains. Without strategy adapters to mediate, these overlapping activation patterns destructively interfere. The humaneval collapse (34% vs 78% single-best) shows domain_code is completely drowned by domain_math and domain_medical signal.

## Implication for Next Experiment

K=7 is the minimum viable adapter set — neither axis can be pruned. Reducing adapter count requires architectural changes (per-token routing, gating) rather than subset selection. The results.json verdict bug (reports SUPPORTED when K1 passes but K2 fails) should be fixed in the shared eval_runner for future experiments.
