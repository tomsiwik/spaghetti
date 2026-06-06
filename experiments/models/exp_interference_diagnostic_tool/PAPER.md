# PAPER.md — exp_interference_diagnostic_tool

## Verdict
**KILLED** (preempt-structural triple-fire)

- verdict: killed
- all_pass: false
- is_smoke: false
- mode: preempt_structural

## Prediction vs measurement

| Prediction (from MATH.md) | Measurement | Match |
|---------------------------|-------------|-------|
| P1: If run as-is, K1900 PASS + K1901 PASS = tautological-support | Not run (preempt-structural) | Consistent — structural preempt means measurement path closed |
| P2: K1900 FAIL is unreachable without injected nondeterminism | Not run | Consistent |
| P3: No target-behavioral outcome derivable regardless of verdict | Confirmed by KC enumeration: 0 target KCs | PASS |

## Fire classification

### 1. F#666-pure — 23rd drain-window instance
Both KCs proxy-only with zero target-behavioral KC. Anchor-append to `mem-antipattern-f666-pure-standalone`.

### 2. F#715 infrastructure-benchmark bucket — 4th drain-window instance (post-promotion)
- K1901 (wall-clock 5 min): matches prior sub-flavor.
- K1900 (variance 5%): **NEW sub-flavor — reproducibility/variance-bound**, distinct from wall-clock, byte-size, and engineering-cost-per-gain. First such flavor in drain window.
Anchor-append to `mem-antipattern-infrastructure-benchmark-bucket-f715`.

### 3. F#702 hygiene-patch unavailable — derived-lemma reuse
0 target KCs ⇒ patch surface empty. Per F#714/F#715 lemma, preempt-structural KILL is the unique admissible route.

### Non-promoting: tool-as-experiment category error
Title + notes explicitly frame this as infrastructure ("Reusable across experiments"). Pre-reg structure doesn't match experiment form (answers "does compute(x) satisfy spec?" not "does mechanism M support/kill claim?"). 1st drain-window instance — inline tracking, no promotion. Open-queue watchlist: `exp_adapter_fingerprint_uniqueness`, `exp_routing_latency_benchmark_all` both at risk of same flavor.

## F#702 hygiene patch applied
- `experiment_dir`: set to `micro/models/exp_interference_diagnostic_tool/`.
- `platform`: set to `local-apple`.
- `success_criteria`, `references`: not patched because F#666-pure saturation makes patching vacuous per Theorem 3. F#702 patch surface requires ≥1 target KC.

## Antipattern audit summary
(b) no baseline, (c) no target metric, (d) proxy-without-target, (e) tautological KC, (f) threshold uncalibrated, (g) behavioral claim unspecified, (h) prior-art redundancy (F#137/F#427/F#453/F#498), (j) missing hygiene fields.

## Prior-art anchors
F#137 (PPL-probe r=0.990), F#269 (direction-interference), F#427 (α=0.145 power law), F#453 (max cos=0.0861), F#498 (subspace destroys composition), F#666 (target-gating), F#702 (hygiene-patch conditions), F#714/F#715/F#716/F#720 (F#666-pure + F#702-unavailability chain), F#721/F#732/F#734 (F#715 bucket siblings).

## Followup / rescue
No `_impl` companion (preempt-structural excludes execution deferral). Rescue path (if re-registered as v2):
- **Preferred.** De-register as experiment; build as infra/tool under `/infra` with unit tests, not pre-reg.
- **Admissible.** Add target KC binding heatmap output to oracle-interference ranking (Kendall-τ) or downstream task accuracy (behavioral); plus variance threshold calibrated from measured downstream sensitivity curve.
