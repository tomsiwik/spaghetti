# MATH: `exp_prod_version_resolution` — preemptive kill (5-theorem stack)

## TYPE
preemptive-kill (ap-017 drain). No empirical run on the proposed target claim.
The experiment's **running cost is provably unjustified** before any data
is collected.

## OBJECTIVE (claimed by target)
Semver + base-hash compatibility matrix for adapters:
- **K1662**: "Adapter declaring incompatible base hash fails load 100% with clear error"
- **K1663**: "Semver range resolution (^1.2, ~1.2.3) picks correct adapter across 20 version scenarios"
- **K1664**: "Breaking-change detection: major version bump of base invalidates adapter"

Declared dependency: `exp_prod_adapter_format_spec_v1` (SUPPORTED, P0).

## PRIOR MATH / REFERENCES
- ap-017 (audit-2026-04-17 cohort, 33rd preempt as of this iter): 5-theorem
  stack as codified in prior drain iters (32 prior preempts across
  composition-bug, scale-safety, tautological-routing, projection-scope,
  tautological-duplicate, hardware-topology-unavailable axes).
- F#502 / F#646 (schema-completeness-vs-instance-fix): DB-literal
  `success_criteria: [] # MISSING` is a 5th occurrence here
  (after tfidf_routing_no_alias, flywheel_real_users, loader_portability,
  registry_host).
- F#173 (theory-aggregation-non-transfer): source experiments proving
  point results (file-format round-trip) do not imply target
  aggregations (registry-level resolver behavior) transfer.
- F#451/F#1564 (proxy-with-empirical-refutation): observed-behavior gap
  between file-level round-trip (proven) and resolver-level selection
  (not proven anywhere in repo).

## THE 5 THEOREMS (pre-flight)

### T1 — Prerequisite inventory (shortfall)
Let `R = {semver_range_resolver, multi_version_adapter_registry,
multi_version_base_model_hash_set, base_hash_verifying_loader}` be the
artifacts required to exercise K1662/K1663/K1664.

Repo-wide grep for semver parsing: `grep -rE
"(semver|packaging\.version|version\.parse|VersionRange)" --include="*.py"`
⇒ **0 hits** outside `.venv/__pycache__`. `pyproject.toml` `grep -E
"(semver|packaging)"` ⇒ **0 hits**.

Repo-wide grep for multi-version adapter registry (not the hosting
infra from iter 36, but the *data*: multiple adapters carrying
different `spec_version` / `adapter_version` metadata): all uses of
`spec_version` are the *literal constant* `1`; no adapter file in the
repo carries any other version value.

Repo-wide grep for multiple `base_model_id`s: **single value**
(`mlx-community/gemma-4-e4b-it-4bit`) used everywhere. No historical
or alternative base-model hash is ever persisted.

Repo-wide grep for a loader that verifies `base_model_hash` against an
actual file and fails load: **0 hits**. `exp_prod_adapter_format_spec_v1`
itself LITERALLY defers this (Assumption 3: *"`base_model_hash` is a
declared string; this experiment does not verify the hash against an
actual base-model file. Hash verification is the loader's job at
runtime, not the format's."*).

**shortfall = |R| − |R ∩ repo| = 4 − 0 = 4.**

### T2 — Scale-safety budget
K1663 alone requires 20 version scenarios (adapter × base pairs) across
at least 3 semver operators (`^`, `~`, `=`). That's a ≥60-trial matrix.
At ~20 s / trial to build an artifact + drive a loader, that is 20–60 min.
K1662 adds a hash-mismatch failure-path sweep (at least 10 trials).
K1664 adds a major-bump matrix over at least 2 base-model versions.
Combined ≈ 60–90 min wall time — within the 120 min micro ceiling,
so T2 does **not** block on its own. **Reinforces only.**

### T3 — DB literal schema completeness
`experiment get exp_prod_version_resolution`:
```
success_criteria: [] # MISSING
⚠ INCOMPLETE: missing success_criteria
```

This is a DB LITERAL match for F#502/F#646. It is the **5th** such match
in the audit-2026-04-17 drain (after tfidf_routing_no_alias,
flywheel_real_users, exp_prod_adapter_loader_portability,
exp_prod_adapter_registry_host).

### T4 — Kill-criterion pin ratio
- K1662: "**100%**" ✓ (pinned) + "**clear error**" ✗ (no regex / string
  oracle) ⇒ 1 pin / 2 sub-claims.
- K1663: "**20 version scenarios**" ✓ (pinned) + "**picks correct
  adapter**" ✗ (no tie-break rule, no oracle defined) ⇒ 1 / 2.
- K1664: "**major version bump invalidates**" ✗ (no threshold: what is
  "invalidates"? load-fail? warn? rejected by resolver?) + no timing
  budget ⇒ 0 / 2.

Total pin ratio = 2 / 6 = **0.333** (same as iter 36; above the
0.20 auto-block floor). **Reinforces only.**

### T5 — Source-scope breach vs `exp_prod_adapter_format_spec_v1`
Target depends on this SUPPORTED source. Source's own LITERAL assumptions
/ non-goals:
- **Assumption 1** (quoted): *"we only require that round-trip within a
  fixed library version is lossless; cross-version safetensors drift is
  out of scope."*
- **Assumption 3** (quoted): *"`base_model_hash` is a declared string;
  this experiment does not verify the hash against an actual
  base-model file. Hash verification is the loader's job at runtime,
  not the format's."*
- **Non-goal 1**: "Cross-library safetensors compatibility. Downstream."
- **KC1637/K1638 scope**: 10 random adapters, **one-shot round-trip**,
  single base-model id, single `spec_version = 1` literal.

Five LITERAL target/source scope breaches:

- **(A) resolver-scope.** Source provides a u32 `spec_version` field
  fixed at 1; it contains no range operator, no registry, no resolver.
  Target K1663 demands `^1.2` / `~1.2.3` range resolution — which
  requires a semver parser and a registry of ≥3 versions.
- **(B) hash-verification-scope.** Assumption 3 LITERALLY defers hash
  verification to a future loader. Target K1662 presumes it.
- **(C) version-drift-scope.** Assumption 1 LITERALLY declares
  cross-version drift out of scope. Target K1664 ("major version bump
  invalidates adapter") IS a cross-version interaction claim.
- **(D) registry-scope.** Source tests 10 random adapters
  **independently** (one-shot round-trip each). Target K1663 requires
  a coordinated multi-version registry that the 10-one-shot test
  never exercised.
- **(E) failure-path-scope.** Source's KC are **success-path** (bitwise
  equality under correct inputs). Target K1662 is a **failure-path**
  claim ("fails load 100% with clear error") — an entirely different
  test surface the source never wired up.

**T5 score: 5/5 LITERAL breaches.** Two of them (A, C) reproduce the
exact words of the source's `Assumptions` section.

### Verdict
```
block(T1) = True                 # 4 missing artifacts
block(T2) = False                # would fit in 120 min if T1 resolved
block(T3) = True                 # schema-incomplete DB literal
block(T4) = False                # 0.333 > 0.20 floor; reinforces
block(T5) = True                 # 5/5 literal source-scope breaches

all_block = block(T1) ∧ block(T3) ∧ block(T5) = True
defense_in_depth = any single one of {T1, T3, T5} alone also blocks
```

## THEOREM 1 (Preemptive kill)
Under {T1 ∨ T3 ∨ T5} = True (in fact all three), the only honest
verdict achievable by any 20-scenario run of this experiment is
`killed_preregistered`, because the data-collection machinery that
would separate `supported` from `killed` does not exist in the repo.

Running it would at best produce another F#502-class paper with empty
success criteria, scaffold a resolver that does not match any shipped
PLAN.md target, and would consume the 120 min micro budget on
infrastructure-building (T1 shortfall 4) that PLAN.md Part 2 has not
scoped. **QED.**

## PREDICTIONS (pre-flight)
- P1: No new code imports `semver` / `packaging.version` (T1 stays at 4).
- P2: DB `success_criteria` remains `[]` post-run (T3 unchanged).
- P3: Source Assumption 1 and Assumption 3 quoted text still present
  in `exp_prod_adapter_format_spec_v1/MATH.md` (T5 unchanged).
- P4: No single-hat drain in ≤120 min can legitimately mark this
  `supported`; the only consistent verdicts are `killed_preregistered`
  or (illegitimate) silent upgrade.

## KILL CRITERIA (pre-registered)
- **K1662 (adapter-declaring-incompatible-base-hash-fails-load-100pct):**
  pre-flight FAIL — no hash-verifying loader exists (T1 ∧ T5(B,E)).
- **K1663 (semver-range-resolution-correct-across-20-scenarios):**
  pre-flight FAIL — no semver resolver + no multi-version registry
  (T1 ∧ T5(A,D)).
- **K1664 (major-version-bump-invalidates-adapter):**
  pre-flight FAIL — no multi-version base-model set + Assumption 1
  deferral (T1 ∧ T5(C)).

All three are **preempted** by the 5-theorem stack; none is exercised
with data.

## BEHAVIORAL OUTCOME
The behavior this preempt enforces: do not consume a 120-min micro
budget to generate a 4th PAPER.md that would be indistinguishable from
the preceding four F#502-class preempts (tfidf_routing_no_alias,
flywheel_real_users, loader_portability, registry_host). Drain-forward
honesty: if the infrastructure does not exist, say so and route the
work to the operator who can unblock it (PLAN.md Part 2 scope
decision), not to yet another researcher hat.

## ASSUMPTIONS (logged per guardrail 1007)
- A1. ap-017 5-theorem stack is the canonical preempt tool for this
  cohort.
- A2. Source Assumption 1 / Assumption 3 text is treated as a binding
  scope-declaration; if the operator disagrees, a follow-up experiment
  must re-establish that scope in a new SUPPORTED finding before this
  target can be re-run.
- A3. "shortfall ≥ 1" under T1 is sufficient (not necessary) when
  joined by T3 ∨ T5 per the stack's all_block rule.
- A4. ap-017 axis for this preempt: **composition-bug (software-
  infrastructure-unbuilt variant)** — distinct from iter 36's
  `hardware-topology-unavailable` (external DNS/CUDA) because the
  absent artifacts are *in-repo* software (semver + registry) that
  PLAN.md has not scoped, not external physical infra.

## NON-GOALS
- Building a semver resolver. PLAN.md Part 2 has not scoped it.
- Running any adapter load / inference.
- Proposing a v2 experiment. That is the operator's call after
  reopening scope with a new SUPPORTED dependency that covers
  Assumption 1 / Assumption 3.
