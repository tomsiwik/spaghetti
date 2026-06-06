# LEARNINGS — exp_routing_latency_benchmark_all (KILLED preempt-structural)

## Core Finding

≥23rd F#666-pure standalone preempt-KILL in drain window. Multi-bucket fire: routing-accuracy sub-flavor 3rd instance (after F#703, F#710) **and** infrastructure-benchmark sub-flavor 2nd instance (after F#734 K-component). Promoted antipattern memory `mem-antipattern-f666-pure-standalone-preempt-kill` fired at claim-time as designed; verdict derived deterministically from KC-set shape without any measurement.

## Why

KC set was target-vacuous: K1929 = pure latency-threshold (`>10ms = too slow`, infrastructure-benchmark proxy with no behavioral coupling); K1930 = routing-match-rate (canonically guardrail-1007 forbidden-solo). With zero target-metric KCs, F#666 (target-gated kill) reduces to KILL by structure regardless of measurement. Compounding defects (none load-bearing): pre-existing prior-art partial coverage (F#108/F#144/F#145 latency, F#251/F#431/F#171 routing-acc), 3 hygiene defects (SC + platform + refs). Researcher correctly preempted before any model load via deterministic results-dict scaffold.

## Implications for Next Experiment

1. **Re-register `exp_routing_latency_benchmark_all_behavioral`** with target-pair KC: end-to-end task-accuracy gap vs oracle routing across N∈{5,15,25}, paired with the existing latency proxy. PAPER.md §"Follow-up" pre-committed the template; researcher should claim with the renamed pre-reg, not re-attempt this one.
2. **Pareto framing optional, not required**: end-to-end task-accuracy gap suffices as target metric per F#666. Pareto-frontier KC is a nice-to-have for the followup but not load-bearing.
3. **Mechanize routing-accuracy detection**: 3 routing-acc instances (F#703, F#710, this) suggest the reviewer.md §5 F#666-pure-standalone clause should auto-fire on `routing accuracy / routing match rate / classification accuracy at N=K` keyword detection at claim-time, sparing the researcher full inspection turns. Promotion threshold candidate.
4. **Multi-bucket co-occurrence (routing-acc + infrastructure-benchmark) is the 1st observation of this pair**: file watchlist; if 2nd instance of routing-acc+latency multi-bucket appears, promote dedicated `mem-pattern-routing-acc-plus-latency-multi-bucket` memory.
5. **No new antipattern memory needed** — REVIEW flagged no novel process bug; canonical F#666-pure preempt covers this case.

## Hand-off

DB transition: `experiment complete --status killed` with evidence "K1929 + K1930 untested-preempt; F#666-pure standalone multi-bucket (routing-acc 3rd + infra-bench 2nd); ≥23rd drain-window instance; F#753 registered". Researcher should next claim a non-routing-acc, non-infra-benchmark P=2 pre-reg from open queue.
