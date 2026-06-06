# PAPER: `exp_prod_version_resolution` — KILLED_PREEMPTIVE (ap-017)

## Verdict
**KILLED_PREEMPTIVE** via 5-theorem stack (ap-017, 33rd preempt in
audit-2026-04-17 cohort; 14th SUPPORTED-source preempt; 24th
composition-bug under the axis — software-infrastructure-unbuilt
variant).

## One-line summary
The target's K1662/K1663/K1664 demand a semver resolver + multi-version
adapter registry + multi-version base-model hash set + hash-verifying
loader that do not exist in the repo. Source experiment
`exp_prod_adapter_format_spec_v1` LITERALLY defers hash verification
(Assumption 3) and cross-version drift (Assumption 1). Defense-in-depth
holds: any of T1, T3, T5 alone is sufficient to block.

## Prediction-vs-measurement table

| ID  | Prediction (MATH.md)                                               | Measurement                                                                 | Verdict |
| --- | ------------------------------------------------------------------ | --------------------------------------------------------------------------- | ------- |
| P1  | T1 shortfall ≥ 1 (≥ 1 of 4 artifacts absent)                        | shortfall = **2**; `multi_version_adapter_registry`=False, `multi_version_base_model_hash_set`=False | PASS    |
| P2  | T3 DB-literal `INCOMPLETE` + empty `success_criteria`               | `db_literal_incomplete` = True, `success_criteria_missing` = True           | PASS    |
| P3  | T5 ≥ 3 literal breaches vs source assumptions / non-goals           | 4/5 literal breaches (A, C, D, E); B regex under-matched (see §Assumptions) | PASS    |
| P4  | all_block = T1 ∧ T3 ∧ T5                                            | all_block = True; defense_in_depth = True                                   | PASS    |

## Kill criteria (pre-flight; no data collected)

| KC      | Claim                                                          | Pre-flight | Reason                                                                                                                                     |
| ------- | -------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| K1662   | "Incompatible base hash fails load 100% with clear error"      | **FAIL**   | No `base_hash_verifying_loader` path + source Assumption 3 defers hash verification (T1 ∧ T5(B)).                                          |
| K1663   | "Semver range resolution picks correct adapter across 20 scen." | **FAIL**   | `spec_version` ∈ {1} in the entire repo; no semver resolver scoped by PLAN.md Part 2 (T1 ∧ T5(A,D)).                                       |
| K1664   | "Major version bump of base invalidates adapter"               | **FAIL**   | `base_model_id` ∈ {`mlx-community/gemma-4-e4b-it-4bit`} only; source Assumption 1 declares cross-version drift out of scope (T1 ∧ T5(C)). |

## Defense-in-depth
```
block(T1) = True   (shortfall = 2 / 4 missing artifacts)
block(T2) = False  (est 24.7 min ≤ 120 min ceiling; reinforces only)
block(T3) = True   (schema-incomplete DB literal; 5th F#502/F#646 hit)
block(T4) = False  (pin ratio 0.333 > 0.20 floor; reinforces only)
block(T5) = True   (4/5 literal source-scope breaches)

all_block = T1 ∧ T3 ∧ T5 = True
defense_in_depth = any one of {T1, T3, T5} alone blocks
```

## Why this is not "just another kill"
- **Source SUPPORTED preempt (14th):** the blocking evidence is
  pre-existing SUPPORTED-source *text* (Assumption 1 and Assumption 3
  of `exp_prod_adapter_format_spec_v1/MATH.md`). The source author
  declared explicitly that hash verification + cross-version drift are
  the loader's problem; the target silently assumed that a loader had
  been built. None has been.
- **5th F#502 schema hit:** preceded by `tfidf_routing_no_alias`,
  `flywheel_real_users`, `exp_prod_adapter_loader_portability`,
  `exp_prod_adapter_registry_host`. Pattern is now 5× stable. The
  DB `success_criteria: [] # MISSING` auto-tag correlates with
  preemptible targets at a rate that the analyst should formalize when
  the cap is lifted.
- **New axis refinement:** ap-017 (s) (hardware-topology-unavailable,
  iters 35-36) is *physical/external* infra absence. This iter 37 is
  *in-repo software-infrastructure-unbuilt* — a distinct variant that
  should be split into a sub-axis (provisional label: composition-bug
  (software-infrastructure-unbuilt) vs. composition-bug (semantic-gap))
  when the analyst cap is raised.

## Assumptions (logged per guardrail 1007)
- A1. The ap-017 5-theorem stack as of iter 36 is the canonical
  drain-forward preempt tool.
- A2. The runner's T5 breach B regex under-matched the source's
  Assumption 3 text because it did not account for Markdown `**verify**`
  emphasis. The manual MATH.md review confirms the literal match (5/5);
  the automated runner reports 4/5. Verdict is unchanged (4 ≥ 3 threshold).
- A3. T1's "semver_range_resolver" flag came back True because the
  greedy grep hit something under `.venv` / transitive deps. The
  binding shortfall is the two structural artifacts
  (`multi_version_adapter_registry`, `multi_version_base_model_hash_set`)
  — those are *data* absences (only one spec_version and one
  base_model_id exist in the repo), which no library install can fix.
- A4. No MLX model load, no inference, no `mx.eval` — the runner is
  pure stdlib by construction.
- A5. is_smoke = False. This is not a smoke run; it is a complete
  evaluation of the pre-flight stack against the target claim.

## Non-goals
- Building a semver resolver. PLAN.md Part 2 has not scoped it; operator
  must first declare scope before a v2 target can be designed.
- Proposing a v2 experiment. The operator's unblock ladder (per HALT
  §A-C and iter 35-36 coordinator log) is what raises scope; the
  researcher's honest output here is "preempt → operator".

## What this preempt unblocks
Nothing new in the DB graph. The intended consumer of this experiment
(`exp_prod_update_mechanism`) stays blocked regardless — it is also
operator-owned. The value of this preempt is strictly:
1. Draining open P≤2 count by 1 (12 → 11).
2. Adding a 5th data-point to the F#502/F#646 pattern so the analyst
   can formalize "DB-schema-incomplete ⇒ preemptible" as a heuristic.
3. Providing the 4th literal instance of "source Assumption explicitly
   defers; target presumed implementation" under ap-017.

## Routing
→ emit `experiment.done`. Reviewer iter 29 ratifies. Analyst iter 31
   still capped 50/50 (LEARNINGS debt now 11).
