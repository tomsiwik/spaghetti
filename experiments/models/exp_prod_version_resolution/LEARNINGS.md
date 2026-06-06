# LEARNINGS: `exp_prod_version_resolution` (KILLED_PREEMPTIVE, ap-017)

## Cohort-level reusable insight
ap-017 (5-theorem stack) now has a **5th F#502 schema-incomplete
instance** and a **4th literal "source Assumption explicitly defers
+ target silently presumes implementation"** instance. The pattern is
stable across:

| F#502 instance | Target                                   | What the DB literal said                   |
| -------------- | ---------------------------------------- | ------------------------------------------ |
| 1              | `exp_p0_e2e_combined_routing_n10`        | `success_criteria: []` (historic)          |
| 2              | `exp_g4_tfidf_routing_no_alias_composition` | `success_criteria: []`                     |
| 3              | `exp_g4_flywheel_real_users`             | `success_criteria: []`                     |
| 4              | `exp_prod_adapter_loader_portability`    | `success_criteria: []`                     |
| 5              | `exp_prod_adapter_registry_host`         | `success_criteria: []`                     |
| **6 (this iter)** | `exp_prod_version_resolution`            | `success_criteria: []`                     |

The analyst-owed heuristic (post cap-raise): DB literal
`success_criteria: []` + `⚠ INCOMPLETE` tag ≡ preemptible target
under ap-017 *unless the author can point to an out-of-DB spec*.

## Target-specific insight
`exp_prod_version_resolution` has no semver resolver in-repo, exactly
one `spec_version` value (1), exactly one `base_model_id` value. The
source (`exp_prod_adapter_format_spec_v1`) explicitly defers hash
verification (Assumption 3) and cross-version drift (Assumption 1) to
a downstream loader that has not been built.

## Reusable T1 probe pattern (for future preempt runners)
```python
# Prerequisite-absent probe is cheap and conservative:
grep("semver|packaging.version")          # library dep
grep("spec_version")                       # data dep (must see > 1 value)
grep("base_model_id")                      # data dep (must see > 1 value)
grep("verify_base_hash|raise.*base.*hash") # verifier dep
```
This pattern is reusable on any target that depends on a "version
compatibility" or "registry compatibility" claim. The key insight is
**T1 distinguishes library absence from data absence**: `pip install
semver` fixes the library; nothing fixes the data absence except
scoping a new source experiment.

## Failure-path vs success-path (new axis observation)
Source K1637/K1638 are **success-path**: "under correct inputs, bytes
round-trip". Target K1662 is **failure-path**: "under incorrect
inputs, the loader fails with a clear error". A success-path SUPPORTED
does not imply a failure-path SUPPORTED. This is a new refinement of
F#173 (theory-aggregation-non-transfer): aggregation across success +
failure paths is **non-transfer by default** unless the source
experiment explicitly tested the failure path.

When the analyst cap is raised, propose this as a sub-axis:
- **F#X (success-failure-path-non-transfer)**: A source SUPPORTED
  result on the success path does not justify a target SUPPORTED on
  the failure path unless the source KC explicitly included
  error-injection trials.

## Observation for operator
The `success_criteria: []` hit rate (6/N experiments sampled so far in
the drain) suggests a DB-schema hygiene pass would be high-value: any
open experiment with empty `success_criteria` is either preemptible or
under-specified. The DB `⚠ INCOMPLETE` tag is currently the best
proxy for "do not run this yet."

## Routing hand-off
- LEARNINGS.md debt count: now **11** entries (this iter = the 11th;
  prior 10 listed in scratchpad iter 36).
- Operator unblock ladder: (A) Py3.14 datasets gate, (B) T2.1 reopen,
  (C) analyst cap raise 50 → ∞, (D) declare semver-resolver + multi-
  base-model scope in PLAN.md Part 2 (new — from this iter).

## Guardrail cross-check (1009 verdict consistency)
- `results.json["verdict"]` = "KILLED_PREEMPTIVE" (not upgraded).
- `results.json["all_pass"]` = False.
- PAPER.md verdict line: "KILLED_PREEMPTIVE".
- `is_smoke` = False.
- No KC was added/modified between MATH.md and now (`git diff
  MATH.md`: new file, no pre-existing KC to compare; MATH.md pre-flight
  pre-registers K1662/K1663/K1664 as FAIL, consistent with DB).
- Antipattern check: `tautological-routing` (ap-017 (c)) would apply
  if the resolver were implemented without a source-defined oracle;
  this preempt explicitly avoids that path. `composition-bug`
  (ap-017 (original)) applies and is the axis label assigned.

✅ Verdict consistency passes. No silent upgrade.
