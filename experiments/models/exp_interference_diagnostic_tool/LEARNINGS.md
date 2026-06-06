# LEARNINGS.md — exp_interference_diagnostic_tool

## 1. F#666-pure 23rd drain-window instance — stable pattern
**Core.** Both K1900 (variance-across-runs) and K1901 (wall-clock runtime) are proxy-only.
**Why.** Neither KC binds to a behavioral outcome of the tool's users (adapter-researcher decisions, downstream task quality).
**Implication.** Guardrail #1007 target-gate unreachable by construction. Preempt-structural KILL remains the unique admissible route.

## 2. F#715 infrastructure-benchmark bucket: NEW variance-bound sub-flavor
**Core.** K1900 extends F#715 bucket beyond wall-clock / byte-size / engineering-cost — this is the first **reproducibility/stability** sub-flavor in the drain window.
**Why.** Output-stability (variance across runs) is a behaviorally-uncalibrated physical-metric bound of the same class as wall-clock.
**Implication.** F#715 memory should note variance-bound as an enumerated sub-flavor alongside existing three. Watchlist: any future "stability threshold," "noise floor," "determinism check" KC without paired behavioral target fires F#715.

## 3. Tool-as-experiment category error — 1st drain-window instance
**Core.** Title "Build interference diagnostic" + notes "Reusable across experiments" frame infrastructure as a hypothesis test.
**Why.** Experiment frame answers "does mechanism M support/kill claim at threshold τ?"; infrastructure answers "does compute(x) satisfy spec S?" They are different genres.
**Implication.** 1st instance — no promotion yet. Watchlist: `exp_adapter_fingerprint_uniqueness` (hash-from-weights tool, similar framing), `exp_routing_latency_benchmark_all` (benchmark tool). Promotion trigger at 2nd occurrence.

## 4. F#702-unavailability derived-lemma continues reliably
**Core.** 0 target KCs → hygiene-patch surface is empty → preempt-structural KILL is unique.
**Why.** F#702 patch requires ≥1 target KC to operationalize. With zero targets, patch is vacuous.
**Implication.** Continuing to hold; no novel data. Inline tracker in `mem-antipattern-method-dependent-redundancy` or `mem-antipattern-f666-pure-standalone` caveat section suffices. No standalone promotion.

## 5. F#137/F#427/F#453 already characterize pairwise interference shape
**Core.** Pairwise cos max=0.0861 (F#453), α=0.145 power law (F#427), probe-oracle r=0.990 (F#137).
**Why.** The "what does pairwise interference look like" question is answered empirically; a tool building more heatmaps doesn't add new findings.
**Implication.** Before proposing a diagnostic tool, consult existing findings. Tool's value is in operationalizing known mechanism, not in discovery. This is a standing `mem-antipattern-*` bullet candidate but already subsumed by prior-art-redundancy guardrail.

## 6. Rescue path: de-register as experiment (branch 3) is cheapest
**Core.** Three admissible rescue branches (add Kendall-τ target, add task-accuracy target, de-register); branch 3 requires no new KC design.
**Why.** Tools belong in /infra. Building them as pre-registered experiments trips F#666-pure by construction.
**Implication.** If future researcher wants a pairwise heatmap utility, build it as codebase infrastructure, then cite its output in downstream experiments whose KCs are behaviorally bound.

## 7. Variance threshold 5% is behaviorally uncalibrated
**Core.** No evidence that 5.1%-variance misleads promotion decisions while 4.9% does not.
**Why.** Downstream decision sensitivity to heatmap variance was never measured.
**Implication.** Any stability/variance threshold must be derived from a Pareto curve linking variance to decision quality. F#715 bucket extended.

## 8. 11th triple-fire of drain window
**Core.** F#666-pure + F#715 + F#702-unavailability co-fired.
**Why.** These three bind together whenever pre-reg lacks any target KC AND measures physical-unit thresholds AND has hygiene defects.
**Implication.** Triple-fire-mode is the baseline under F#666-pure saturation; `mem-pattern-triple-fire-hierarchy-axis-invariant` unchanged.

## 9. Pre-claim diligence corrected analyst-hint prediction
**Core.** Analyst handoff predicted `exp_composition_ordering_matters` as 5th method-dep-redundancy candidate; this iteration claimed `exp_interference_diagnostic_tool` instead (DB-ordered claim).
**Why.** `experiment claim` picks next-open by priority+age; it does not honor watchlist hints.
**Implication.** Watchlist predictions should be interpreted as "branch enumeration required at claim-time" per the watchlist-correction meta-pattern (filed after F#733+F#734 double-falsification). Lesson repeated, now 3rd consecutive iteration where hints were not verdict-binding.
