# MATH.md — exp_interference_diagnostic_tool (PREEMPT-KILL triple-fire)

## Verdict (pre-run, structural)

**KILLED** — preempt-structural. No runnable path under F#666 target-gating. Triple-fire:
1. **F#666-pure 23rd drain-window instance** — both KCs proxy-only with zero target-behavioral KC
2. **F#715 infrastructure-benchmark bucket 4th drain-window instance** (post-promotion anchor-append) + NEW sub-flavor: reproducibility/variance-bound (distinct from wall-clock and byte-size)
3. **F#702 hygiene-patch unavailable** — derived lemma (F#714/F#715): 0 target KCs ⇒ hygiene-patch surface is empty

## KCs (as registered)

- K1900: "Diagnostic tool produces inconsistent results across runs (> 5% variance)"
- K1901: "Diagnostic runtime > 5 min for N=25 adapters"

## Hygiene defects (pre-claim)

- `success_criteria: []` — missing
- `platform: null` — missing
- `experiment_dir: ~` — missing
- `references: []` — missing
- Title frames this as a TOOL ("heatmap generator"), notes explicitly say "Reusable across experiments" — an infrastructure artifact, not a hypothesis.

## Theorem 1 — F#666-pure structural impossibility (both KCs proxy-only)

**Statement.** For K1900 and K1901 under F#666/guardrail #1007 (target-gated kill), no behavioral target KC is paired with either proxy, so every admissible verdict collapses:
- PASS(K1900) ∧ PASS(K1901) ⇒ tautological-support: "tool is fast and reproducible" is a definition of functional software, not a finding about adapter geometry.
- FAIL(K1900) ∨ FAIL(K1901) ⇒ finding-about-thresholds: a 5.1% variance or a 5-min-1-sec runtime says nothing about whether the *interference scores the tool outputs* are load-bearing for any downstream decision.

**Proof.** F#666 requires every proxy-metric KC to be paired with a target-metric KC such that KILL = (proxy FAIL ∧ target FAIL) and SUPPORTED = (proxy PASS ∧ target PASS). K1900 and K1901 are both measurement artifacts of the tool's implementation:
- K1900: variance of tool output across runs (tool-internal reproducibility)
- K1901: wall-clock runtime of tool on N=25 adapters (tool-internal cost)

Neither is downstream of a behavioral claim. No KC asks "does the heatmap the tool produces enable a promotion/demotion decision that matches oracle interference ranking?" or "does a user acting on the heatmap reach higher task accuracy than acting on a uniform-random baseline?" With zero target KCs, F#666 target-gate cannot be instantiated, and the pre-reg is structurally unverifiable. QED.

**Prior anchors (F#666-pure drain-window chain).** F#687, F#689, F#691, F#695, F#697, F#698, F#699, F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#715, F#716, F#720, F#721, F#727, F#728, F#729, F#730, F#734. 22 prior instances. This is instance #23.

## Theorem 2 — F#715 infrastructure-benchmark bucket 4th anchor-append (post-promotion)

**Statement.** K1901 (wall-clock runtime bound) and K1900 (variance bound) are both "infrastructure-benchmark" measurements per the promoted memory `mem-antipattern-infrastructure-benchmark-bucket-f715`. The promotion rule fires on 3+ drain-window instances; this is the 4th. Anchor-append.

**Proof of K1901 match.**
- Metric class: wall-clock physical unit (seconds).
- Threshold: 5 min = 300 s. Bare, behaviorally uncalibrated.
- Decision surface: no evidence that 5-min-1-sec degrades user workflow vs 4-min-59-sec at any deployment constraint. No gain-vs-cost Pareto anchor. No comparator to a baseline tool or no-tool counterfactual.
- Identical shape to F#715/K1860 (100 ms round-trip), F#732/K1894 (wall-clock), F#734/K1899 (cluster overhead >10 ms). QED.

**Proof of K1900 match — NEW sub-flavor: reproducibility/variance-bound.**
- Metric class: variance across runs (unitless ratio, %).
- Threshold: 5% variance. Bare, behaviorally uncalibrated.
- Decision surface: no evidence that 5.1%-variance misleads promotion decisions while 4.9% does not. No linkage to a downstream outcome whose sensitivity to heatmap variance has been measured.
- Distinct from prior F#715 sub-flavors (wall-clock, byte-size-on-disk, engineering-cost-per-gain) — this is the first **output-stability** flavor in the drain window. Add to F#715 bucket as new sub-flavor. QED.

**Drain-window instance count (F#715 bucket after this experiment):**
| # | Finding | Experiment | KCs | Sub-flavor |
|---|---------|------------|-----|------------|
| 1 | F#715 | exp_memento_kv_serialization_format | K1860, K1861 | wall-clock + byte-size |
| 2 | F#721 | exp_hedgehog_layer_selection_top6 | K1874 | engineering-cost-per-gain |
| 3 | F#732 | exp_composition_runtime_vs_merge_n10 | K1894 | wall-clock |
| 4 | F#734 | exp_composition_clustering_group | K1899 | wall-clock |
| 5 | THIS | exp_interference_diagnostic_tool | K1900, K1901 | **NEW: variance-bound** + wall-clock |

## Theorem 3 — F#702 hygiene-patch unavailability (derived-lemma reuse)

**Statement.** F#702 hygiene-patch is restricted to pre-regs with ≥1 target-metric KC (F#702 impossibility-structure: "≥1 target-KC + hygiene defects ⇒ hygiene-patch + _impl"). F#714 and F#715 established the derived-lemma: "F#666-pure saturation (0 target KCs) implies F#702 patch unavailability." Under Theorem 1 (Thm 1 proved 0 target KCs), F#702 patch surface is empty. Preempt-structural KILL is the unique admissible route.

**Proof.** Let T = { KCs with target-metric status } ⊆ { K1900, K1901 }. By Theorem 1, T = ∅. F#702 patch requires |T| ≥ 1 to bind the operationalization surface. With |T| = 0, patch is vacuous — no KC to operationalize, rescue, or defer to `_impl`. The companion `_impl` pattern (F#702's execution-deferral) collapses: there is nothing to execute, because there is no measurement path. QED.

**Drain-window F#702-unavailability chain.** F#714 (1st), F#715 (2nd), F#716 (3rd, promotion-threshold per `mem-antipattern-method-dependent-redundancy` escalation tracking), F#720 (4th), and continuing. This is the post-promotion N-th confirmation; no separate memory action beyond inline tracking.

## Theorem 4 (non-promoting) — Tool-as-experiment category error

**Statement.** The experiment title "Build interference diagnostic: pairwise adapter interference heatmap generator" and notes "Reusable across experiments" frame the work as **infrastructure**, not a **hypothesis test**. Infrastructure belongs under `/infra` or a codebase utility, not as a pre-registered experiment with KCs.

**Proof.** An experiment pre-reg answers a question of the form "is mechanism M supported or killed on metric M_target at threshold τ?" (F#666 target-gated). An infrastructure artifact answers the form "does the function compute(x) satisfy spec S?" The latter reduces to a unit/integration test, not a scientific experiment. The KC structure "tool produces inconsistent results" and "tool runtime > 5 min" are test assertions, not proxy metrics paired with behavioral targets.

**Axiomatic reduction.** If the goal is "reusable pairwise interference score," the existing findings F#137 (PPL-probe weighting, probe-oracle r=0.990), F#453 (pairwise cos=0.0861 max, 0.0187 mean), F#269 (direction-interference persists), F#427 (adapter power law c=0.061, α=0.145, R²=0.94), F#498 (subspace destroys composition max_cos 0.60→0.96) already specify the interference-score mathematical surface. The tool would be a wrapper around existing mechanisms; no new finding is derivable from its construction.

**Non-promoting.** 1st instance of "tool-as-experiment" in drain window. Tracked inline; promote to antipattern at 2nd occurrence. Open-queue watchlist candidates: `exp_adapter_fingerprint_uniqueness` (hash-from-weights tool, 0 behavioral KC likely), `exp_routing_latency_benchmark_all` (latency benchmark, infrastructure-bucket primary).

## 5-branch enumeration of "runnable if patched" alternatives

| # | Branch | Why it would bind F#666 | Verdict |
|---|--------|-------------------------|---------|
| 1 | Add target KC: "heatmap-guided promotion decision matches oracle ranking at Kendall-τ > 0.7" | Pairs K1900/K1901 with behavioral decision outcome | **Admissible** (would lift F#666-pure + F#715 simultaneously) |
| 2 | Add target KC: "user acting on heatmap reaches downstream task accuracy ≥ baseline+5pp on held-out task" | Pairs tool output with end-task behavioral metric | Admissible but requires user-in-loop protocol (operational complexity) |
| 3 | Remove K1900 + K1901, reframe as pure infrastructure PR (no pre-reg) | Exits experiment frame entirely | Admissible (preferred for tool builds) |
| 4 | Register K1900 variance threshold as a *Pareto anchor* derived from known sensitivity of downstream decision to variance | Would bind variance-bound to behavioral outcome via measured sensitivity curve | Admissible but requires prior sensitivity study |
| 5 | Keep as-is | F#666-pure, F#715, F#702-unavailable triple-fire | **Inadmissible** — current state |

Only 3 of 5 branches (1, 2, 4) would admit the experiment back into the drain queue. Branch 3 (de-register as experiment) is the cheapest rescue.

## Prior-art citations (load-bearing)

- **F#137** — PPL-probe interference-measurement (probe-oracle r=0.990). Tool would wrap this mechanism.
- **F#269** — direction-interference impossibility (MMLU degrades regardless of sparsity). Theoretical interference floor.
- **F#427** — adapter power law (α=0.145, R²=0.94, max_cos N=50 = 0.107). Interference scaling already characterized.
- **F#453** — pairwise cos_max=0.0861, mean=0.0187 (3-promotion flywheel). Empirical interference distribution already measured.
- **F#498** — subspace destroys composition (max_cos 0.60→0.96). Pathological interference regime known.
- **F#666** — proxy-target gating (guardrail #1007). Binds KC admissibility.
- **F#702** — hygiene-patch availability depends on ≥1 target KC.
- **F#714, F#715, F#716, F#720** — F#666-pure + F#702-unavailability chain.
- **F#721** — infrastructure-benchmark engineering-cost-per-gain sub-flavor.
- **F#732, F#734** — infrastructure-benchmark wall-clock sub-flavor.

## Antipattern audit (self-check)

- [x] (a) Cosine-similarity-as-KC — N/A (no cosine KC here, but interference-heatmap tool would output cos-sims).
- [x] (b) No baseline — YES (no random-heatmap or no-tool baseline).
- [x] (c) No target metric — YES (F#666-pure; primary fire).
- [x] (d) Proxy without target pair — YES (K1900, K1901 both proxy).
- [x] (e) Tautological KC — YES (K1900 PASS defines "reproducible"; K1901 PASS defines "fast").
- [x] (f) Threshold uncalibrated — YES (5% variance, 5 min both bare).
- [x] (g) Behavioral claim unspecified — YES (no behavior claimed, only tool properties).
- [x] (h) Prior-art redundancy — YES (F#137/F#427/F#453 already characterize pairwise interference).
- [x] (i) LORA_SCALE=20 — N/A.
- [x] (j) Missing hygiene fields — YES (success_criteria, platform, dir, references all missing).

## Pre-run predictions (testable)

- **P1.** If run as-is, K1900 PASS (any deterministic implementation gives 0% variance) and K1901 PASS (N=25 pairwise cos-sims on LoRA-r=6 adapters takes ~O(N² · r²) = 25² · 6² = 22500 ops, well under 5 min on Apple Silicon). Both would trivially pass, confirming the tautological-support prediction.
- **P2.** K1900 FAIL is structurally unreachable without injecting deliberate nondeterminism (seed-dependent sampling inside the tool), which would itself be an implementation bug, not a finding.
- **P3.** No target behavioral outcome measured (by construction), so no claim about interference mechanism, adapter geometry, or routing quality is derivable regardless of K1900/K1901 verdict.

Under F#666 target-gate: P1 = tautological-support; P2 = finding-about-thresholds; P3 = null result. All three branches non-informative.

## Conclusion

Preempt-structural KILL. Triple-fire (F#666-pure #23 + F#715 #4 [variance sub-flavor new] + F#702-unavailability #N). No `_impl` companion (preempt-structural precludes execution-deferral per F#687/F#698/F#699/F#700/F#701/F#703/F#731/F#732/F#733/F#734 chain). Rescue path requires branch 1, 2, or 4 from the 5-branch table above — branch 3 (de-register) is cheapest.
