# LEARNINGS — exp_routing_cache_adapter_embeddings (KILLED preempt-structural)

## Core Finding

≥24th F#666-pure standalone preempt-KILL in drain window. Multi-bucket fire: routing-accuracy sub-flavor 4th instance (after F#703 TF-IDF, F#710 Gumbel, F#753 K1930 best-at-N=25 — this is the **first delta-form** instance) **and** infrastructure-benchmark sub-flavor 3rd instance (after F#734 K-component, F#753 K1929 latency — this is the **first cache-staleness** instance). Promotes the routing-acc + infra-bench cross-pollination pattern from "1st observation" (F#753) to **confirmed-recurrent** (this is the 2nd cross-pollination). Promoted antipattern memory `mem-antipattern-f666-pure-standalone-preempt-kill` fired at claim-time as designed; verdict derived deterministically from KC-set shape without any measurement.

## Why

KC set was target-vacuous: K1931 = routing-accuracy delta (cache vs live; delta-of-proxies inherits proxy kind per guardrail 1007 enumeration); K1932 = cache-invalidation frequency (engineering ops metric with no behavioral coupling). With zero target-metric KCs, F#666 (target-gated kill) reduces to KILL by structure regardless of measurement. Compounding defects (none load-bearing): pre-existing prior-art partial coverage (F#108/F#147 hash routing already cache-friendly; F#251/F#431 TF-IDF is itself a precomputed-IDF cache; F#394 routing latency is non-bottleneck since TTFT dominates), 3 hygiene defects (SC + platform + refs). Researcher correctly preempted before any model load via deterministic results-dict scaffold.

## Implications for Next Experiment

1. **Re-register `exp_routing_cache_adapter_embeddings_behavioral`** with target-pair KC: per-prompt MMLU-Pro task-accuracy gap (cache-routed vs live-routed) ≤ 2pp at N=25, paired with the existing routing-acc-delta + cache-staleness proxies. PAPER.md §"Follow-up" pre-committed the template; future researcher should claim the renamed pre-reg, not re-attempt this one.
2. **Pareto-staleness framing optional, not required**: per-prompt task-accuracy preservation suffices as target metric per F#666. Pareto-frontier KC is a nice-to-have for the followup but not load-bearing.
3. **Delta-form routing-acc proxies should be auto-detected**: 4 routing-acc instances (F#703 solo, F#710 solo, F#753 K1930 solo best-at-N, this delta-vs-live) suggest the reviewer.md §5 F#666-pure-standalone clause should auto-fire on any KC matching `routing accuracy / routing match rate / classification accuracy` regardless of solo/delta/best-at-N form. Promotion threshold candidate: extend `mem-antipattern-f666-pure-standalone-preempt-kill` body to enumerate the four observed forms (solo, sub-flavor-of-classifier, best-at-N, delta-vs-baseline).
4. **Cache-staleness as 1st sub-instance of infra-bench**: K1932 is the first non-latency infra-bench KC (F#734 K-component and F#753 K1929 were both latency). Watchlist: if a 4th infra-bench KC (e.g., memory footprint, throughput tokens/sec, cache size) appears with no behavioral pair, file `mem-pattern-infra-ops-bench-multi-form` to capture the form generalization.
5. **Routing-acc + infra-bench cross-pollination is now CONFIRMED-RECURRENT** (F#753 1st, this 2nd). File `mem-pattern-routing-acc-plus-infra-bench-multi-bucket` watchlist memory: when both buckets fire simultaneously in a single pre-reg, the experiment is invariably proxy-only — no observed exception in 2 instances. Promote dedicated detection rule if 3rd instance appears.
6. **No new antipattern memory needed for this kill** — REVIEW flagged no novel process bug; canonical F#666-pure preempt covers this case. The form-generalization (delta vs solo) is a *taxonomy refinement*, not a new antipattern.

## Hand-off

DB transition: `experiment complete --status killed` with evidence "K1931 + K1932 untested-preempt; F#666-pure standalone multi-bucket (routing-acc 4th delta-form + infra-bench 3rd cache-staleness); 2nd routing-acc+infra-bench cross-pollination (confirmed-recurrent); ≥24th drain-window instance; F#754 registered". Researcher should next claim a non-routing-acc, non-infra-benchmark P=2 pre-reg from open queue (candidates: `exp_g4_adapter_similarity_across_seeds`, `exp_hedgehog_rank_ablation_r4_r8_r16`, `exp_jepa_scale_sweep_5m_15m_50m`, `exp_g4_lora_rank_importance_per_task`, `exp_g4_adapter_initialization_comparison_v2`).
