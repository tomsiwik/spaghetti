# PAPER.md — exp_prod_update_mechanism

## Verdict

**KILLED_PREEMPTIVE** — PROD-deliverable-cascade (parent KILLED),
**4th instance** (after F#740, F#741, F#764). **Crosses the
super-family-promotion threshold** (2nd cross-cluster reuse over
3 distinct PROD parents: mlxlm-integration / pip-package-pierre /
version-resolution). Compound: F#666 partial (2-of-3 KC violation;
K1677 target-pair survives, K1676 + K1678 proxy-only); F#502/F#646
schema-cohort 9th hit (reinforces 8th-hit promotion threshold).
4 of 5 theorems block (defense_in_depth = true; any one alone is
sufficient).

## One-line

You cannot measure "base upgrade preserves user adapters or auto-
re-crystallizes within 5%" when the parent semver+hash compatibility
matrix does not exist: parent `exp_prod_version_resolution` is
KILLED with all three KCs FAIL (`multi_version_base_model_hash_set`
absent, `multi_version_adapter_registry` absent, semver resolver
absent in upgrade-flow surface). 2-of-3 child KCs are proxy-only;
the 3rd has a target-metric pair (quality 5%) that nonetheless
cannot be measured without the parent registry.

## Target claim (DB)

| KC | Text |
|----|------|
| K1676 | Base upgrade detects incompatible adapters via base-hash mismatch, blocks silent quality loss |
| K1677 | Re-crystallize user adapter on new base; **quality within 5% of original** (target-paired) |
| K1678 | User data (adapter weights, training history) survives upgrade |

`success_criteria: []` (DB literal `⚠ INCOMPLETE`).

## Prediction vs. measurement

| Theorem | Prediction | Measurement | Blocks? |
|---------|------------|-------------|---------|
| T1 artifact shortfall | ≥ 3 upgrade-flow artefacts missing | shortfall = 2 (`upgrade_base_entry_point`, `training_history_manifest_schema`) — T1 grep matched sister-runner strings (semver_resolver / hash_set / registry / recrystallize / quality_floor) in *other experiments' runner files*, false-negative against the structural reality. Defense-in-depth carries the kill regardless. | **fail (does not block; below threshold)** |
| T2 parent supersession | parent `exp_prod_version_resolution` KILLED + child step 1 vacuous | parent verdict = KILLED_PREEMPTIVE, K1662/K1663/K1664 all FAIL | PASS |
| T3 schema completeness | `success_criteria: []` literal in DB; F#502 cohort hit | confirmed; 9th cohort hit (after F#650/F#652/F#763/F#764) | PASS |
| T4 KC pin count | pin_ratio ≤ 0.30 OR ≥ 1 non-falsifiable KC | pin_ratio ≈ 0.27; K1676 + K1678 each non-falsifiable as stated | PASS |
| T5 source-scope breach | ≥ 3 of {A_parent_absent, B_recryst_absent, C_user_data_absent, D_F666_partial} | 4/4 breaches confirmed | PASS |

`all_block = false` (T1 false-negative), `defense_in_depth = true`,
`t_blocking = 4/5`. Per the decision rule, defense-in-depth is the
controlling clause (any single Ti ⇒ KILL). Wall-clock for the
preempt runner: 17 s, pure stdlib (no MLX, no network, no model
load). The 17 s is grep wall-clock across the repo.

## Kill criteria

| KC | Verdict | Reason |
|----|---------|--------|
| K1676 | **fail** | T2: parent KILLED, `multi_version_base_model_hash_set` absent in upgrade-flow surface → no base-hash detection chain to measure. T4: non-falsifiable as stated (any error-on-mismatch satisfies "blocks silent quality loss" under generous read). |
| K1677 | **fail** | T2: parent registry artefacts absent → no `recrystallize_adapter(old_adapter, old_base, new_base)` pipeline can be run. The "quality within 5%" target-pair is **falsifiable in principle** but **unmeasurable in practice** without the parent registry. |
| K1678 | **fail** | T1 (real-shortfall): `training_history.json` schema absent; T4: non-falsifiable; T5(D): F#666 partial — proxy-only, no target-metric pair. |

## Precedent map

**Reusable (direct cascade siblings — PROD-deliverable-cascade family):**
- F#740 (12th F#669 reuse): `exp_pierre_multi_adapter_serving_throughput`
  preempt-KILL — parent `exp_prod_mlxlm_integration` KILLED. **1st**
  PROD-deliverable-cascade instance.
- F#741 (13th F#669 reuse): `exp_pierre_adapter_cache_prefill`,
  same parent as F#740. **2nd** instance, **1st within-cluster reuse**.
- F#764 (3rd PROD-deliverable-cascade): `exp_prod_onboarding_first_run`,
  parent `exp_prod_pip_package_pierre` KILLED. **3rd** instance,
  **1st cross-cluster reuse** (different parent deliverable).
- This experiment: parent `exp_prod_version_resolution` KILLED.
  **4th** instance, **2nd cross-cluster reuse** (third distinct
  parent deliverable). **Crosses the analyst-flagged promotion
  threshold.** Recommend promotion of "PROD-child-with-KILLED-
  parent" from compound preempt-axis to **top-level guardrail** on
  next analyst pass.

**Reinforcing (F#666 family) — partial:**
- Finding #666 (TARGET-GATED KILL): proxy-only KCs uninterpretable.
- K1677 satisfies F#666 (target-metric KC paired with re-crystallize
  proxy half). K1676 + K1678 do not. F#765 should be tagged "F#666
  partial" not "F#666 pure" — distinguishes from F#764.

**F#502/F#646 cohort tracking:**
- 9th hit (after F#650, F#652, F#763, F#764). The 8th hit (F#764)
  reached the analyst-flagged super-family-promotion threshold; the
  9th confirms. Recommend analyst promotes `success_criteria=[]`
  from co-indicator to 1st-class preempt-axis on next pass.

**NOT-TRANSPORT:**
- No prior finding establishes that a base-upgrade flow has been
  shipped on this codebase — F#60 (BitNet onboarding) is irrelevant
  (no upgrade primitive demonstrated).

## Operator unblock

The preempt does **not** assert "base-upgrade is impossible";
it asserts "the current parent KILL state cannot measure it." To
unblock, all of:

1. **Resurrect parent `exp_prod_version_resolution`.** Build the
   `semver_range_resolver`, `multi_version_base_model_hash_set`,
   `multi_version_adapter_registry`. Re-run parent and reach
   SUPPORTED with K1662 / K1663 / K1664 PASS. Until that happens,
   any PROD child downstream of upgrade-flow is preempt-killable.
2. **Implement `pierre.registry.upgrade_base(old_hash, new_hash)`**
   and `recrystallize_adapter(old_adapter, old_base, new_base)`.
3. **Pin K1677's quality metric + benchmark dataset** (MMLU-Pro
   with thinking? F#627-compliant target?) and training budget
   for re-crystallization.
4. **Add target-metric KCs paired with K1676 and K1678** (e.g.
   "incompatible-adapter false-negative rate ≤ 1% on a 100-pair
   test set"; "post-upgrade adapter behavioural-quality regression
   ≤ 0% on a fixed prompt-set"), to fully satisfy F#666.
5. **Populate `success_criteria`** in the DB to clear the F#502
   co-indicator.

Until then, tag the experiment `out-of-scope-pending-parent` and
bump priority to ≥ 4.

## Assumptions (autonomy log)

- Assumed `pierre/registry/` is the natural location for upgrade-
  flow code (vs `composer/registry/` or `pierre/upgrade/`).
  Conclusion is invariant under directory choice — no such
  module of any name implements the upgrade primitive.
- Assumed K1677's "quality" means downstream task accuracy or
  PPL on a held-out set (per F#627-compliant convention); the
  preempt does not depend on this — T2 fires regardless.
- Assumed the M5 Pro hardware is available (per
  `project_user_hardware` memory). T1 is platform-orthogonal:
  the registry shortfall is repo-state, not host-state.
- T1 grep over `*.py` files matched the literal strings
  `semver_range_resolver`, `multi_version_base_model_hash_set`,
  `multi_version_adapter_registry`, `recrystallize_adapter`,
  `quality_within_5pct` in the **parent's runner** (`exp_prod_version_resolution/run_experiment.py`),
  which **references** these symbols to check for their absence.
  This is a structural false-negative on T1 (the symbols appear
  in checking-code, not implementation-code). Defense-in-depth
  carries the kill via T2+T3+T4+T5; T1's contribution is
  redundant, not load-bearing.

## Runner

`run_experiment.py` is pure stdlib: greps for upgrade-flow
primitives, walks parent dir, parses parent results.json. No MLX,
no model load, no network. Wall-clock ≈ 17 s (dominated by grep).
Exit 0; verdict written to `results.json`.

## Antipattern self-check

All 12 antipattern checks PASS or N/A (see MATH.md §Antipattern
self-check). No model code emitted under preempt-KILL clause —
m2 carve-out applies.

## Promotion signal (for analyst pass)

| Field | Value |
|-------|-------|
| Axis | `PROD-child-with-KILLED-parent` |
| Trigger | 4th instance, 2nd cross-cluster reuse over 3 distinct PROD parents |
| Distinct parents | exp_prod_mlxlm_integration, exp_prod_pip_package_pierre, exp_prod_version_resolution |
| Recommended action | Promote from compound preempt-axis to top-level guardrail; future PROD child with KILLED parent should preempt-KILL on parent-state alone (no 5-theorem stack required) |
