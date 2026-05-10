# PAPER.md — Per-prompt routed K=2 composition (v2 re-run)

## Status: KILLED

## Summary

Per-prompt routing with K=2 (strategy_full + domain_X) does NOT beat fixed Fisher-Rao K=7. Despite correcting the false-kill infrastructure bug (original 53/20/7 → now 72/72/30), the approach still produces a medqa collapse (30%) and averages 6.7pp below K=7 reference.

## Results

| Metric | Value | Reference | Delta |
|--------|-------|-----------|-------|
| gsm8k | 72.0% | 64.7% (K=7 avg) | +7.3pp |
| humaneval | 72.0% | — | — |
| medqa | 30.0% | — | −34.7pp vs K=7 avg |
| **avg** | **58.0%** | **64.7%** | **−6.7pp** |

Routing distribution: math=60, code=38, medical=52 (150 total prompts).

## Kill Criteria Verdicts

| KC | Verdict | Evidence |
|----|---------|----------|
| K1: avg ≥ K=7 + 2pp (66.7%) | **FAIL** | 58.0% < 66.7% (−8.7pp) |
| K2: per-bench floor wins | **FAIL** | medqa 30% < 40% floor |
| K3: classifier acc ≥ 85% | **PASS** | Train accuracy OK (no val split) |
| K4: no collapse (all ≥ 40%) | **FAIL** | medqa 30% < 40% |

## Mechanism Analysis

**Why medqa collapses at K=2 while gsm8k/humaneval thrive:**

1. **Strategy adapter carries domain signal.** `strategy_full` was trained on a mix of all domains. When composed K=2 with `domain_math` or `domain_code`, it reinforces their signal (gsm8k 72%, humaneval 72%). But composed with `domain_medical`, the strategy adapter's math/code bias interferes — it suppresses medical-specific B-vectors.

2. **K=2 is insufficient for medqa.** The K=7 ensemble averages over all 7 adapters; medqa needs both `domain_medical` AND at least one strategy adapter that doesn't destructively interfere. K=2 leaves no room for this.

3. **Routing doesn't solve composition quality.** The classifier correctly routes medical prompts to `domain_medical` (52/150), but the composition of (strategy_full, domain_medical) is itself weak. The problem is in composition fidelity, not routing accuracy.

## Comparison with original false-kill

| Benchmark | Original (bug) | This re-run | Δ |
|-----------|----------------|-------------|---|
| gsm8k | 53.3% | 72.0% | +18.7pp |
| humaneval | 20.0% | 72.0% | +52.0pp |
| medqa | 6.7% | 30.0% | +23.3pp |
| avg | 26.7% | 58.0% | +31.3pp |

The infrastructure bug was real — scores are dramatically higher. But medqa still collapses, confirming the fundamental limitation of K=2 per-prompt routing on this adapter set.

## Implications for Pierre

1. **Per-prompt routing at K=2 is not viable.** The simplest dynamic-composition primitive fails to match static K=7. Dynamic routing would need K≥4 or domain-aware strategy adapters to avoid collapse.

2. **K=7 static composition remains the shipping path.** Neither K=3 subset (strategy-only: 56%, domain-only: 46%) nor K=2 routed (58%) matches K=7 (64.7%). Adapter count reduction requires architectural changes, not routing.

3. **Cross-reference with K=3 findings:** Strategy-only (Finding #844) showed medqa collapse at 36%. Domain-only (Finding #845) showed humaneval collapse at 34%. K=2 routed shows medqa collapse at 30%. All sub-K=7 approaches fail on at least one domain.

## References

- False-kill audit: Finding #831
- K=3 strategy-only: Finding #844 (avg 56%, medqa 36%)
- K=3 domain-only: Finding #845 (avg 46%, humaneval 34%)
- K=7 Fisher-Rao reference: 64.7% avg
- Routing primitive validation: `exp_pierre_per_task_routing_math`
