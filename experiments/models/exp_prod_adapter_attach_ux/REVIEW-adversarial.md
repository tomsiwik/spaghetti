# REVIEW-adversarial: `exp_prod_adapter_attach_ux`

**Reviewer:** iter 31 (ap-017 cohort) | **Date:** 2026-04-19
**Verdict:** KILL (ratify) | **Status:** killed (already set)
**Defense-in-depth:** 3 of {T1, T3, T5} each block alone

## Adversarial checklist (per `.ralph/hats/reviewer.md`)

| Gate | Result | Note |
|---|---|---|
| (a) results.json.verdict ↔ DB status | PASS | `KILLED_PREEMPTIVE` ↔ `killed` |
| (b) all_pass ↔ claim | PASS | `all_pass=false`; 3 KC `fail`; consistent |
| (c) PAPER.md verdict line | PASS | `KILLED (pre-flight, no empirical run)` |
| (d) is_smoke flag | PASS | `is_smoke=false`; pre-flight preempt, not a smoke |
| (e) KC relaxation | PASS | K1673/K1674/K1675 = target's declared KCs verbatim; no post-hoc edit |
| (f) Tautology sniff | PASS | Pure pre-flight probe; no KC passes by identity |
| (g) K-ID mismatch | PASS | All three KCs fail; runner does not "measure" a different quantity — it measures ABSENCE |
| (h) Composition bug pattern | N/A | no composition run |
| (i) `LORA_SCALE≥12` | N/A | no adapter run |
| (j) routing on single sample | N/A | no routing run |
| (k) `shutil.copy` sibling-as-new | N/A | no adapter file ops |
| (l) hardcoded `{"pass": True}` | N/A | runner emits `False` for 3/4 KCs |
| (m) model mismatch | N/A | no model load |
| (m2) skill invocation | N/A | stdlib runner, no MLX code |
| (n)-(q) eval integrity | N/A | no empirical run |
| (r) prediction-vs-measurement table | PASS | P1-P5 table in PAPER.md §Prediction vs Measurement; P1 marked `revised ✓` with A7 link |
| (s) math errors | PASS | T2 arithmetic (130×100×3 ≈ 39s swap-loop + 15s load + 20s adapter-loads ≈ 74s ≈ 1.23 min as reported; the "15 min" prose in MATH §T2 is a conservative margin, not the runner output) |

All consistency gates PASS. KC integrity gates PASS. Code↔math and eval gates are N/A (stdlib pre-flight). Deliverable gates PASS.

## Transparency items verified

- **A7 (T1 sub-check noise):** Manual inspection of the 7 `@app.(post|get)` / `FastAPI(` hits confirms none is a `pierre serve` entry; manual inspection of `adapter_hot_cache_mlx/run_experiment.py` confirms it is a smoke harness, not a p99-budgeted swap harness against `pierre.attach_adapter`. The two unambiguous absences (`pierre_cli_entry_point=false`, `logit_cosine_pre_post_attach_harness=false`) still drive `T1.block=true`. Verdict unchanged.
- **A8 (T5(C) p99-scope literal):** Source `results.json` pins neither p50 nor p99 as an explicit JSON field; prose in source PAPER.md cites `p50=14.5 ms`. Automated regex reports `false`; manual literal reports `true`. T5 score moves 4/5 → 5/5 on manual read, still ≥ 3 block floor.

Defense-in-depth is overdetermined: T1 ∨ T3 ∨ T5 each blocks alone.

## Cohort-level observation

This is the **35th ap-017 preempt**, the **16th SUPPORTED-source preempt**, and the **7th F#502/F#646 schema-incomplete** in the audit-2026-04-17 drain. Software-infrastructure-unbuilt branch of composition-bug (same fingerprint as iter 37 `version_resolution`; distinct from iter 38 `DP_training`'s platform-library cross-cut). The 4 absent artifacts (CLI entry-point, persistent-server process model, p99 harness, logit-cosine harness) are all in-repo constructions `pip install` cannot resolve.

F#652 (software-infrastructure-unbuilt sub-axis, registered by reviewer iter 28) is the right home. No new axis this iter — reuse F#652.

## Finding-add plan
One new finding for this iter: reuse F#652 semantics; title pinning `runtime-UX-without-server-artifact` as the specific fingerprint under F#652.

## Route
`review.killed` → analyst iter 33 (still capped 50/50 per HALT §C). LEARNINGS debt advances 12 → 13 for when the cap lifts.

## Assumptions
- `experiment get` reports `Status: killed` and all 3 KC as `[✗]`, confirming the researcher already ran `experiment complete` during iter 39. I will not re-run `experiment complete`.
- The MATH.md §T2 prose figure "15 min" is a safety margin above the runner's `est_minutes=1.23` (derived from 130ms × 100 × 3 + 15s + 20s + some slack). No inconsistency with runner output; T2 still does not block.

## Non-blocking observations (for backlog, not for this verdict)
- Runner refinement: the T1 server probe should constrain to filenames/paths under `pierre/` and require an entry-point or `__main__` guard; the p99 probe should co-require `attach` + `percentile` + a budget literal (e.g. `200`) in the same file. Lift into `preempt_common.py` when the analyst cap raises.
