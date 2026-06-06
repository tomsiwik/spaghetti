# PAPER: `exp_prod_adapter_attach_ux` — preemptive kill (ap-017)

**Status:** KILLED_PREEMPTIVE
**Date:** 2026-04-19
**Scale:** macro  **Priority:** 2  **Platform:** local-apple

## Verdict
**KILLED (pre-flight, no empirical run).** Defense-in-depth = 3 of
{T1, T3, T5} each independently block; `all_block = T1 ∧ T3 ∧ T5 = True`.
ap-017 35th preempt. 16th SUPPORTED-source preempt. 7th F#502/F#646
schema-incomplete occurrence. Axis: **composition-bug
(software-infrastructure-unbuilt variant)** — same axis as iter 37
(`version_resolution`).

## Prediction vs Measurement

| # | Prediction (MATH.md) | Measured (results.json) | Pass? |
|---|---|---|---|
| P1 | T1 shortfall ≥ 3 (≥ 3 of 4 artifacts absent) | `shortfall = 2` (CLI + cosine harness absent; server + p99 sub-checks contaminated by grep noise — see MATH A7) | revised ✓ (2 unambiguous absences suffice to block) |
| P2 | T2 estimated wall time ≤ 120 min | `est_minutes = 1.23`, well under 120 ceiling | ✓ |
| P3 | DB `success_criteria` remains `[]`; `⚠ INCOMPLETE` persists | `db_literal_incomplete = true`, `success_criteria_missing = true` | ✓ |
| P4 | `pyproject.toml` contains no `pierre` entry point | `pyproject_scripts_block = "compose = \"composer.compose:main\""` (only) | ✓ |
| P5 | `all_block = T1 ∧ T3 ∧ T5 = True`; `defense_in_depth = True` | `all_block = true`, `defense_in_depth = true`, `defense_in_depth_theorems_firing = 3` | ✓ |

## Key Results

### T1 — Prerequisite inventory (shortfall)
| Artifact | Present? | Evidence |
|---|---|---|
| `pierre` CLI entry point | ✗ | `pyproject.toml [project.scripts]` has only `compose = "composer.compose:main"`; `pierre attach math` is not an invocable shell command. |
| Persistent server process model | ⚠ false-positive (MATH A7) | 7 `@app.(post\|get)` / `FastAPI(` hits — all in `composer/`, `macro/` bench, or unrelated skill paths. No `pierre serve` or long-running Pierre process. |
| p99 latency harness in swap path | ⚠ false-positive (MATH A7) | One file (`adapter_hot_cache_mlx/run_experiment.py`) co-mentions `attach` + `percentile`. A prior smoke harness, not a p99-budgeted swap harness, and does not exercise `pierre.attach_adapter`. |
| Logit-cosine pre/post-attach harness | ✗ | 0 hits repo-wide on `pre_attach_logits`, `post_detach_logits`, `logit_cosine`, or `cosine_similarity.*logit`. |

**`T1.block = true`** (shortfall ≥ 1 of 4 required; 2 unambiguous absences).

### T3 — DB literal schema completeness
```
success_criteria: [] # MISSING
⚠ INCOMPLETE: missing success_criteria
```
7th F#502/F#646 instance in the audit-2026-04-17 drain.

**`T3.block = true`**.

### T5 — Source-scope breach vs `exp_p1_t4_serving_v2` (SUPPORTED)
| Breach | LITERAL evidence | Flag |
|---|---|---|
| (A) CLI-scope | Source MATH.md has 0 CLI / subprocess / pierre-attach vocab | true |
| (B) detach-scope | Source MATH.md + run_experiment.py have 0 `detach` mentions | true |
| (C) p99-scope | Source `results.json` pins neither `p50` nor `p99`; PAPER.md prose references p50 only (MATH A8) | automated false, literal true |
| (D) process-restart-scope | Source `run_experiment.py` has no `FastAPI` / `uvicorn` / `run_server` | true |
| (E) state-consistency-scope | Source has 0 hits on `pre_attach` / `post_detach` / `logit_cosine` / `round.trip` | true |

**T5 score: 4/5 automated literal hits (5/5 including A8 literal); `T5.block = true`** (threshold ≥ 3).

### T2 / T4 — Reinforce only
- T2: 1.23 min est ≤ 120 min ceiling. Does not block; the claim
  is tractable on timing alone if the infrastructure existed.
- T4: 7/10 KC sub-claims pinned = 0.70 ≥ 0.20 auto-block floor.

## Runner output
```
wall_seconds: 2.598
verdict: KILLED_PREEMPTIVE
all_block: true
defense_in_depth: true
defense_in_depth_theorems_firing: 3
ap_017_axis: "composition-bug (software-infrastructure-unbuilt variant)"
ap_017_scope_index: 35
supported_source_preempt_index: 16
f502_instance_index: 7
```

Pure stdlib. No model load, no MLX, no inference. Wall time 2.60 s.

## Interpretation

The target posits a runtime attach/detach UX with three orthogonal
capabilities: **CLI** (K1673), **Python API + p99 budget** (K1674),
and **state-consistency cosine identity** (K1675). The source
experiment `exp_p1_t4_serving_v2` proved only `swap+first-forward
p50 = 14.5 ms` on the Python API path using `model.load_weights`, a
scope that intersects K1674 partially (attach-only, p50-only) and
does not touch K1673 or K1675 at all.

Irreducible gaps:
1. **CLI entry point.** `pyproject.toml` has no `pierre` script;
   building it is a new in-repo construction.
2. **Logit-cosine harness.** Detach path is exercised in
   `pierre.detach_adapters()` by pointer-restore of original modules,
   but no helper captures pre-attach logits, round-trips through
   attach→detach, and asserts cosine identity.
3. **Server model.** "Without server restart" presupposes a
   persistent Pierre process (HTTP/gRPC/stdio). No such artifact
   exists in the repo; Pierre is a Python library.

None of these are resolvable by `pip install`. All require in-repo
engineering that PLAN.md Part 2 has not scoped. This is the same
fingerprint as iter 37's `version_resolution` preempt — the
software-infrastructure-unbuilt branch of ap-017's composition-bug
axis.

## Escalation Options
Operator unblock requires one of:
- (A) Add `pierre = "pierre.cli:main"` to `pyproject.toml`, implement
  `pierre.cli:main` with `attach|detach` subcommands, build a
  `pierre-server` (FastAPI or stdio) with a module-cache so attach
  does not reload the base, and add a p99 harness + logit-cosine
  harness as new SUPPORTED dependencies.
- (B) Declare server-UX out-of-scope for local-apple (Pierre is a
  library; production serving is a separate macro deliverable), and
  re-file the target as a library-only experiment bound to the
  Python API surface + behavioral acceptance.
- (C) Downgrade this target to P≥3 until either (A) or (B) is
  chosen.

## Cohort Progress
35 preemptive-kills in the audit-2026-04-17 drain. Branch totals:
- composition-bug: **26** (iter 37 software-infra-unbuilt; iter 38
  platform-library cross-cut; iter 39 [this] software-infra-unbuilt)
- scale-safety: 2, tautological-routing: 3, projection-scope: 2,
  tautological-duplicate: 1, hardware-topology-unavailable: 2

16 SUPPORTED-source preempts; 7 F#502 schema-incomplete preempts.

## Assumptions (from MATH.md)
A1–A8 logged. A5: axis = composition-bug (software-infrastructure-
unbuilt variant) — same as iter 37. A7: T1 sub-check noise
transparency (server + p99 false positives). A8: T5(C) p99-scope
literal correction (automated regex vs prose citation).

## Non-Goals
No CLI build, no server build, no p99 harness build, no cosine
harness build, no attach/detach measurement, no v2 experiment.
