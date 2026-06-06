# MATH.md — exp_prod_update_mechanism

## Preemptive-Kill Via 5-Theorem Stack
## (PROD-deliverable-cascade, parent-KILLED, **4th instance**
##  after F#740/F#741/F#764 — super-family-promotion trigger)

**Verdict: KILLED_PREEMPTIVE** — five independent theorems each
block the claim before any measurement can be produced; any one
sufficient (defense-in-depth).

## Target claim (per DB)

K1676: Base upgrade detects incompatible adapters (base-hash
mismatch), blocks silent quality loss.

K1677: Offered path: re-crystallize user adapter on new base;
**quality within 5% of original** (TARGET-METRIC PAIR).

K1678: User data (adapter weights, training history) survives
upgrade.

`success_criteria: []` (DB literal `⚠ INCOMPLETE`).

Note: Unlike F#764 (3rd cascade, F#666-pure proxy panel), K1677
DOES include a target-metric pair ("quality within 5%"). T5(D) is
adjusted accordingly: F#666 violation is downgraded from "panel-
wide" to "K1676 + K1678 unpaired"; the K1677-target-pair survives.

## Parent / source-finding map

**Direct parent (dep): `exp_prod_version_resolution` — KILLED.**
All three of its kill criteria FAIL (per its `results.json` and
PAPER.md verdict KILLED_PREEMPTIVE):

- K1662 FAIL: "Adapter declaring incompatible base hash fails load
  100% with clear error" — no semver+hash matrix exists; T1
  shortfall=2 missing components in parent.
- K1663 FAIL: "Semver range resolution (^1.2, ~1.2.3) picks
  correct adapter across 20 version scenarios" — `semver_range_resolver`
  not implemented in parent.
- K1664 FAIL: "Breaking-change detection: major version bump of
  base invalidates adapter" — `multi_version_base_model_hash_set`
  not implemented.

The child's K1676 ("upgrade detects incompatible adapters via
base-hash mismatch") **literally requires K1662 to be SUPPORTED**
— it is the same primitive (base-hash verifying loader) at a
different lifecycle moment. K1677's "re-crystallize on new base"
requires K1664 (breaking-change detection) to identify when re-
crystallization is needed. K1678's "user data survives upgrade"
requires K1663 (semver resolution) to know which adapter to
preserve. Each child KC presupposes a parent KC that FAILED.

**Sister precedents (REUSABLE — PROD-deliverable-cascade family):**

- F#740 (12th F#669 reuse): `exp_pierre_multi_adapter_serving_throughput`
  preempt-KILL — parent `exp_prod_mlxlm_integration` KILLED per
  F#570. **1st** PROD-deliverable-cascade instance.
- F#741 (13th F#669 reuse): `exp_pierre_adapter_cache_prefill`,
  same parent (`exp_prod_mlxlm_integration` KILLED) as F#740.
  **2nd** instance, **1st within-cluster reuse**.
- F#764 (3rd PROD-deliverable-cascade): `exp_prod_onboarding_first_run`,
  parent `exp_prod_pip_package_pierre` KILLED. **3rd** instance,
  **1st cross-cluster reuse** (different parent deliverable).
- This experiment: parent `exp_prod_version_resolution` KILLED.
  **4th** instance, **2nd cross-cluster reuse** (third distinct
  parent deliverable). Per analyst guidance scratchpad: "Promotion
  candidacy: 4th cross-cluster cascade reuse → top-level guardrail."
  **This claim crosses the promotion threshold.**

The cascade structure is now reproducibly identical across three
distinct PROD parents (mlxlm-integration, pip-package-pierre,
version-resolution). Promotion to a top-level preempt-axis is
recommended at the analyst pass: "PROD child experiments whose
direct parent is KILLED preempt-KILL on parent-state alone, no
5-theorem stack needed."

**F#666-violation (TARGET-GATED KILL guardrail) — partial:**

- K1676: structural infrastructure (base-hash detection) — proxy.
- K1677: **target-metric paired** ("quality within 5% of original").
- K1678: data-integrity (file presence + hash) — proxy.

K1677 satisfies F#666 (target-metric KC paired with its proxy half
"re-crystallize path offered"). K1676 + K1678 do not. T5(D) records
this as a partial F#666 satisfaction — the target-pair survives,
the proxy-only halves do not.

This is structurally distinguishing from F#764 (which was F#666-pure
panel). The new finding F#765 should NOT be tagged as F#666-pure.

## 5-Theorem Stack

### T1 — Artifact shortfall (parent-deliverable-unavailable)

Required artifacts to measure "base upgrade preserves user
adapters or auto-re-crystallizes":

- A working `semver_range_resolver` implementation in
  `pierre/registry/` or equivalent — **absent** (parent T1
  shortfall=2 includes this exactly).
- A `multi_version_base_model_hash_set` registry — **absent**
  (parent T1 shortfall).
- A `multi_version_adapter_registry` — **absent** (parent T1
  shortfall).
- An upgrade-flow entry point (e.g.
  `pierre.registry.upgrade_base(old_hash, new_hash)`) —
  **absent** (no `upgrade_base` symbol in repo).
- A re-crystallization pipeline (auto-fine-tune adapter on new
  base) — **absent** (no `recrystallize_adapter` or equivalent).
- An adapter-quality-floor evaluation harness for K1677's "within
  5%" claim — **absent** (no fixed prompt-set + metric pinned for
  pre/post quality comparison).
- A user-data preservation manifest (adapter-history serialization
  format) — **absent** (no `training_history.json` schema in repo;
  experiment dirs hold per-experiment results, not user-state).

shortfall ≥ 5 missing artefact categories. The upgrade-flow chain
has zero of its required components realisable on the local
platform.

### T2 — Parent-supersession (PROD-deliverable-cascade)

Claim's measurement chain requires the parent
`exp_prod_version_resolution` to be SUPPORTED (or at minimum to
have a working `semver+hash compatibility matrix` artefact). Parent
status: **KILLED**, all three parent KCs FAIL. Until the three
parent blockers (K1662, K1663, K1664) are remediated, no child KC
is reachable.

Cascade-instance index: **4th PROD-deliverable-cascade preempt**
(after F#740, F#741, F#764). **Crosses the promotion threshold.**
The parent-deliverable axis now has three distinct instances
(mlxlm-integration / pip-package-pierre / version-resolution)
producing the same cascade outcome — analyst should promote
"PROD-child-with-KILLED-parent" from compound preempt-axis to
top-level guardrail on next pass.

### T3 — Schema completeness (F#502/F#646 cohort)

DB claim output literal: `success_criteria: [] # MISSING`,
`⚠ INCOMPLETE: missing success_criteria`. Per F#502/F#646, the
schema-completeness gap is a sustained taxonomic signal.

Cohort-hit index:
- F#650 (5th), F#652 (6th), F#763 (7th), F#764 (8th); this is
  **9th**. The 8th hit (F#764) reached the analyst-flagged
  super-family-promotion threshold; the 9th confirms the pattern.
- Recommend analyst promotes `success_criteria=[]` from
  co-indicator to 1st-class preempt-axis on next pass (was
  flagged at 8th, now reinforced at 9th).

### T4 — KC pin count

K1676 ("base upgrade detects incompatible adapters via base-hash
mismatch, blocks silent quality loss"): mechanism named
("base-hash mismatch"), failure mode named ("silent quality loss").
Missing pins: hash algorithm (SHA-256? SHA3?), comparison surface
(weight-tensor hash vs config hash vs combined), "blocks" semantics
(error-out vs warn-and-continue vs disable-adapter), the
quantitative quality-loss floor that "blocks" prevents.

K1677 ("re-crystallize user adapter on new base; quality within
5% of original"): **threshold pinned** (5%), comparison axis named
("of original"). Missing pins: quality metric (PPL? task-accuracy?
behavioural-eval?), benchmark dataset (MMLU-Pro? user-specific
prompts?), training budget for re-crystallization (how many steps?
on what data?), what counts as "user adapter" (per-conversation
vs per-domain), failure path if re-crystallization itself fails.

K1678 ("user data survives upgrade"): scope named ("adapter
weights, training history"). Missing pins: serialization format,
checksum invariant (byte-identical? semantically-equivalent?),
upgrade-rollback path, partial-failure semantics (what if 4 of 5
adapters survive?).

pin_ratio across {epsilon, baseline, host, dataset, scaling-rule}
≈ 0.27 (4/15 average; K1677 contributes 2 epsilon pins, K1676
contributes 1 mechanism pin, K1678 contributes 1 scope pin).
K1676 + K1678 are non-falsifiable as stated (any well-formed
deserialization + any error-on-mismatch satisfies them under
generous read). **K1677 IS falsifiable** ("quality within 5%" is a
falsifiable threshold even with metric ambiguity, because any
metric will produce a number).

### T5 — Source-scope literal breach

Parent `exp_prod_version_resolution` PAPER.md verdict
("KILLED_PREEMPTIVE") explicitly does NOT upgrade to inconclusive:
its T5 records 4/5 source-scope breaches (resolver scope, version
drift, registry, failure path). The child's KCs would attempt to
measure step 2 (upgrade flow) of a chain where step 1 (the
resolver+hash matrix) is verified absent.

Breaches:
- (A) Parent deliverable absent → child measurement-chain step 1
  vacuous. K1676's "base-hash mismatch detection" requires the
  parent's `base_hash_verifying_loader` to exist and be wired into
  upgrade flow. Parent T1 has it as `true` for trivial-load but
  parent T1 has `multi_version_base_model_hash_set: false`,
  meaning the upgrade-flow surface specifically is absent.
- (B) Re-crystallization pipeline absent → K1677's measurement
  vacuous. No `recrystallize_adapter` or training pipeline that
  takes (old_adapter, old_base, new_base) → new_adapter.
- (C) User-data preservation manifest absent → K1678 vacuous. No
  schema for `training_history.json`; per-experiment dirs are not
  user-state.
- (D) F#666 — partial. K1677 has target-metric pair (quality 5%);
  K1676 and K1678 are proxy-only with no target-metric pair.
  Reduced from "panel-wide F#666 violation" (as in F#764) to
  "2-of-3 KC F#666 violation" (K1676, K1678).

4/4 breaches; each is sufficient.

## Decision rule

```
all_block = T1 ∧ T2 ∧ T3 ∧ T4 ∧ T5
defense_in_depth = (any single Ti ⇒ KILL)
verdict = KILLED_PREEMPTIVE
```

## What an unblock would require

The preempt does **not** assert "base-upgrade flow is impossible";
it asserts "the current parent KILL state cannot measure it." To
unblock, all of:

1. Resurrect parent `exp_prod_version_resolution`: build the
   `semver_range_resolver`, `multi_version_base_model_hash_set`,
   `multi_version_adapter_registry`. Re-run parent and reach
   SUPPORTED status with K1662/K1663/K1664 PASS.
2. Implement `pierre.registry.upgrade_base(old_hash, new_hash)`
   and `recrystallize_adapter(old_adapter, old_base, new_base)`.
3. Pin K1677's quality metric + benchmark dataset (MMLU-Pro with
   thinking? F#627-compliant target?) and training budget for
   re-crystallization.
4. Add target-metric KCs paired with K1676 (e.g. "incompatible-
   adapter false-negative rate ≤ 1% on a 100-pair test set") and
   K1678 (e.g. "post-upgrade adapter behavioural-quality regression
   ≤ 0% on a fixed prompt-set"), to satisfy F#666 fully.
5. Populate `success_criteria` field in the DB.

Until then, the experiment is `out-of-scope-pending-parent`.

## Antipattern self-check

- (a) Composition math bug: N/A — no model code.
- (b) `LORA_SCALE`: N/A.
- (c) `shutil.copy` as new adapter: N/A.
- (d) Hardcoded `"pass": True`: explicit theorem booleans, all
  computed.
- (e) Eval-template truncation: N/A.
- (f) Proxy-model substitution: N/A — no model load.
- (g) KC measures wrong object: KCs unchanged; runner reports their
  preempt-status, does not redefine them.
- (h) Smoke run reported as full: `is_smoke=false`; preempt is the
  full content.
- (i) Hallucinated MLX imports: N/A — pure stdlib runner.
- (j) `.backward()` torch-style: N/A.
- (k) Missing `mx.eval` / `mx.clear_cache`: N/A.
- (l) Wrong adapter targets: N/A.

All 12 OK.
