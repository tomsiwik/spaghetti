# LEARNINGS — Per-prompt routed K=2 composition

## Core Finding

Per-prompt routing (TF-IDF classifier → K=2 composition) averages 58.0% vs K=7 static 64.7%. Routing accuracy is not the bottleneck — composition quality at K=2 is. medqa collapses to 30% because strategy_full carries cross-domain bias that destructively interferes with domain_medical.

## Why It Matters

This closes the sub-K=7 routing investigation. Three attempts failed:
- K=3 strategy-only: 56% (medqa 36%)
- K=3 domain-only: 46% (humaneval 34%)
- K=2 routed: 58% (medqa 30%)

All sub-K=7 approaches collapse on at least one domain. The adapters are cross-entangled — each carries signal from other domains that interferes at low K.

## Implication

K=7 static Fisher-Rao is the shipping path. Reducing adapter count requires architectural changes (domain-aware strategy adapters or learned gating), not post-hoc routing or pruning.
