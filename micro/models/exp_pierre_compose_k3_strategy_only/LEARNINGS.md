# Learnings — K=3 Strategy-Only Composition

## What we learned

Strategy adapters are **not domain-agnostic**. Composing 3 strategy adapters without domain adapters causes catastrophic failure on medical QA (36%) while slightly improving code generation (70%). This proves strategy B-matrices carry domain-specific signal, not pure reasoning patterns.

## Why it matters

This kills the hypothesis that Pierre could ship a lightweight "strategy stack" without domain adapters. The K=7 default works precisely because domain adapters counterbalance the domain bias baked into strategy adapters.

## Connection to prior work

- Confirms Finding #843 (KAN): B-matrix signal structure is load-bearing. Strategy adapters that look domain-agnostic in isolation become domain-biased when composed.
- Informs sibling experiment `compose_k3_domain_only`: if domain-only composition scores higher than 56%, it proves domain axis is lower-interference than strategy axis.

## Design implication

Pierre's minimum viable composition is K=2 (1 strategy + 1 domain), not K=N strategy-only. Any product path that lets users select "reasoning style" without domain grounding will produce medqa-like collapses on out-of-distribution domains.
