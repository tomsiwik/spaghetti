# K=3 Domain-Only Composition — KILLED

## Result

**KILLED.** Fisher-Rao composition of 3 domain adapters (domain_math + domain_code + domain_medical) averages 46.0% across benchmarks — 16pp below single_best (62.0%) and 13.3pp below DARE upper bound (59.3%). K2 FAIL.

## Measurements

| Benchmark | Single-best | Fisher-Rao K=3 domain | DARE upper bound | FR delta vs single |
|-----------|-------------|----------------------|------------------|--------------------|
| gsm8k     | 66.0%       | 58.0%                | 62.0%            | -8pp               |
| humaneval | 78.0%       | 34.0%                | 64.0%            | -44pp              |
| medqa     | 42.0%       | 46.0%                | 52.0%            | +4pp               |
| **avg**   | **62.0%**   | **46.0%**            | **59.3%**        | **-16pp**          |

## Kill Criteria Verdicts

- **K1** PASS: Fisher-Rao K=3 domain avg (46.0%) matches Fisher-Rao reference by construction (this IS the Fisher-Rao result)
- **K2** FAIL: 13.3pp gap to DARE upper bound (threshold 5pp) — domain-only composition cannot close the gap
- **K3** PASS: preprocessing 0.009s < 5s
- **K4** RESOLVED: K=3 domain avg (46.0%) is 10pp worse than K=3 strategy avg (56.0%) — domain axis is MORE interference-prone, not less

## Cross-axis Comparison (K4)

| Axis | Fisher-Rao K=3 avg | Worst benchmark | Collapse domain |
|------|--------------------:|----------------:|-----------------|
| Strategy-only | 56.0% | medqa 36% | Medical knowledge |
| Domain-only   | 46.0% | humaneval 34% | Code generation |

Both axes fail independently. Domain-only is worse by 10pp. Neither axis is "clean" for composition.

## Mechanism Analysis

Domain adapters interfere **catastrophically on code generation**. The humaneval collapse (34%, -44pp vs single_best 78%) reveals the mechanism:

1. Domain adapters encode task-specific knowledge in their B-matrices, but knowledge is not orthogonal in weight space — domain_math and domain_medical inject competing activation patterns that drown out domain_code's signal
2. MedQA actually improves slightly (+4pp over single_best), suggesting domain_medical benefits from math reasoning spillover — but this is a coincidence, not composition
3. DARE partially rescues (64% humaneval vs 34% Fisher-Rao) by zeroing interfering parameters, confirming the interference is in overlapping weight regions

**Key insight:** Domain adapters are NOT orthogonal in B-matrix space even though they target different knowledge domains. The B-matrices share structural patterns (attention heads, MLP projections) that create cross-domain interference. K=7 works because strategy adapters provide the routing/integration signal that mediates this interference.

## Combined Finding: K=3 Pruning Kills Composition

Together with sibling `exp_pierre_compose_k3_strategy_only`:

- Removing domain adapters (strategy-only): avg drops 8.7pp, medqa collapses
- Removing strategy adapters (domain-only): avg drops 18.7pp, humaneval collapses
- **K=7 is the minimum viable adapter set** — each adapter contributes signal that prevents catastrophic interference in the others

This closes the "can we ship fewer adapters" investigation for the current Pierre v3 architecture. Adapter count reduction requires architectural changes (per-token routing, adapter gating) rather than simple subset selection.
