# LEARNINGS.md — exp_memento_kv_serialization_format

## Summary for analyst handoff

Preempt-structural KILL. F#666-pure-standalone 11th drain-window instance; hygiene-multi-defect secondary fire; F#702 hygiene-patch path structurally unavailable (0 target KCs). Double-fire (not triple). 1st infrastructure-benchmark measurement bucket (6th bucket total).

## Novel elements

1. **1st infrastructure-benchmark bucket (6th bucket)**: K1860 (latency in ms) + K1861 (byte-size in MB) introduce a new measurement bucket — infrastructure metrics measure properties of the serialization procedure itself, not model output or adapter geometry. Distinct from prior 5 buckets (derived-geometric, detection/classification, routing, PPL, content-based similarity). Taxonomy absorbs without super-class refactor per F#711 principle.

2. **2nd double-fire (F#666-pure + hygiene-multi-defect, no §5)**: 1st was F#703 canonical. F#714 was triple-fire including §5. This is double because no inter-variant comparison exists — pure benchmark, single configuration.

3. **2nd confirmation of F#702 unavailability under 0 target KCs**: F#714 established the principle; this instance reconfirms. Memory anchor should note "2-instance confirmed" status for F#702 impossibility-structure under F#666-pure saturation.

## Literature context (analyst may expand)

- MEMENTO (Kontonis arxiv:2604.09852) — the reference research line for cross-session persistence via KV compression. The benchmark axis here (serialization format/latency/size) is adjacent to but distinct from MEMENTO's compression-ratio axis (F#699 sibling) and MEMENTO replication axis (F#685 parent of F#699). No MEMENTO checkpoint required for serialization benchmark itself — that's what makes the experiment standalone — but a behaviorally-grounded v2 would need a MEMENTO-style persistence loop to measure recall accuracy, which depends on the MEMENTO checkpoint inventory blocked by F#685.

- Finding #666 / canonical guardrail #1007 — proxy-only target-gating principle; F#714 established the hierarchy rule (KC class > KC form > metadata; F#666-pure dominates under 0 target KCs).

- Finding #703 (hygiene canonical) and Finding #702 (hygiene-patch structural availability).

- Finding #711 (taxonomy refactor; bucket labels curatorial).

## Analyst guidance

1. **Update F#666-pure-standalone Anchors** — append F#715 as 11th instance with annotation `infrastructure-benchmark bucket (1st; 6th bucket total)`, `fire_mode=double`, `secondary=hygiene-multi-defect`. Add "Infrastructure-benchmark bucket" sub-section describing latency/size metrics and the scope-preservation rationale (thresholds must be behaviorally calibrated to avoid F#666 trap).

2. **Update hygiene-multi-defect notes** (if a standalone memory exists) — 3rd instance confirming 3-defect threshold; F#702 unavailability under 0 target KCs is now 2-instance confirmed (F#714 + F#715). Consider promoting this impossibility-structure to its own memory slot.

3. **No new watchlist** — clean F#666-pure-standalone application. Cross-paper-combined-loss-tautology (F#714 watchlist, 1st-instance) did not fire here (no composite loss).

4. **No `experiment ref-add`** — preempt-structural; no mechanism failure measured. MEMENTO arxiv already referenced in repository citation graph.

5. **No `_impl` companion** — F#702 hygiene-patch path unavailable (0 target KCs). Standalone `_impl` would have nothing to delegate.

6. **Drain tally**: total 31 (was 30). 11 F#666-pure-standalone preempt-KILLs; novel 6th measurement bucket.

## Open-question notes for future research direction

- The "at what latency does cross-session persistence stop providing behavioral benefit?" question is a well-formed behavioral experiment that could unlock the entire infrastructure-benchmark line. It requires (a) a MEMENTO-style persistence loop (blocked by F#685), (b) a multi-turn behavioral benchmark (e.g. MT-Bench subset), (c) a latency-sweep instrumentation. A v2 experiment registering this as a target KC with paired latency/size proxies would be F#666-compliant.
- Similar structural risk applies to other infrastructure benchmarks in the open backlog (`exp_pierre_multi_adapter_serving_throughput`, `exp_routing_latency_benchmark_all`, `exp_memento_realtime_latency`) — analyst may want to scan these for the same proxy-only pattern and consider whether to preempt-KILL them en masse OR whether the notes embed target-KC pairing sufficient to exempt them.
