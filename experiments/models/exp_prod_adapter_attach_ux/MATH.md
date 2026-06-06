# MATH: `exp_prod_adapter_attach_ux` — preemptive kill (5-theorem stack)

## TYPE
preemptive-kill (ap-017 drain). No empirical run on the proposed target claim.
The experiment's **running cost is provably unjustified** before any data
is collected.

## OBJECTIVE (claimed by target)
Ship CLI + Python APIs for runtime adapter attach/detach on the
production server in < 200 ms p99, with exact state restoration.
- **K1673**: "CLI: `pierre attach math; pierre detach math` works without
  server restart"
- **K1674**: "Python: `attach_adapter()`/`detach_adapters()` hot-swaps
  in < 200 ms p99"
- **K1675**: "State consistency: detach removes delta exactly (verified
  via logit cosine with pre-attach)"

Declared source experiment: `exp_p1_t4_serving_v2` (SUPPORTED, P=0,
macro, local-apple). Closest-scope SUPPORTED source: same platform,
same framework (MLX), exercises adapter swap latency + first-forward
cost. Differs on CLI wrapper, server-process model, p99 metric,
and detach state-consistency.

## PRIOR MATH / REFERENCES
- ap-017 (audit-2026-04-17 cohort, 35th preempt as of this iter):
  5-theorem stack as codified in iters 35–38. Prior branches:
  composition-bug (25, incl. iter 37 software-infrastructure-unbuilt
  and iter 38 platform-library cross-cut variants), scale-safety (2),
  tautological-routing (3), projection-scope (2), tautological-duplicate
  (1), hardware-topology-unavailable (2).
- F#502 / F#646 (schema-completeness-vs-instance-fix): DB-literal
  `success_criteria: [] # MISSING` is the **7th occurrence** here
  (after tfidf_routing_no_alias, flywheel_real_users,
  loader_portability, registry_host, version_resolution,
  differential_privacy_training).
- F#652 (software-infrastructure-unbuilt composition-bug variant,
  registered iter 37): in-repo library + entry-point absences that
  `pip install` cannot fix are defense-in-depth blockers.
- Source SUPPORTED: `exp_p1_t4_serving_v2` K1240 "swap+first-forward
  < 100 ms" passed at **p50 = 14.5 ms** (LITERAL; no p99 measurement).
  See `micro/models/exp_p1_t4_serving_v2/MATH.md` Theorem 1 and
  `results.json` for exact measurements.
- Runtime LoRA attach/detach in-repo: `pierre/pierre.py:75`
  (`attach_adapter`) and `pierre/pierre.py:113` (`detach_adapters`).

## THE 5 THEOREMS (pre-flight)

### T1 — Prerequisite inventory (shortfall)
Let `R = {pierre_cli_entry_point, persistent_server_process_model,
p99_latency_harness, logit_cosine_pre_post_attach_harness}` be the
artifacts required to exercise K1673/K1674/K1675.

Repo inspection:
- `pyproject.toml [project.scripts]`: declares `compose = "composer.
  compose:main"` only. **No `pierre` entry point**. `pierre attach math`
  is not an invocable binary.
- Persistent server process: grep for `{pierre.*serve,run_server,
  @app\.post,fastapi,uvicorn,pierre-server}` in `pierre/` / repo
  non-`.venv` code: **0 hits** that define a long-running Pierre
  process. Pierre is exposed as a Python library (`import pierre`),
  not a server; K1673 "without server restart" references a process
  model that does not exist.
- p99 latency harness: grep for `{p99, percentile, np.percentile,
  numpy.percentile, quantile}` over repo non-archive code scoped to
  adapter-swap callers: **0 hits in adapter-swap paths**. Source
  `exp_p1_t4_serving_v2/run_experiment.py` measures p50 only.
- Logit-cosine pre/post-attach harness: grep for `{logit_cosine,
  cosine_similarity.*logit, pre_attach_logits}` : **0 hits**. No
  existing helper captures pre-attach logits, attaches, detaches,
  and asserts equality of post-detach logits against the pre-attach
  snapshot.

Present: `attach_adapter` / `detach_adapters` Python API (pierre.py).

**shortfall = |R| − |R ∩ repo| = 4 − 0 = 4** of 4 artifacts required
but absent (the Python attach/detach API does not satisfy any of the
four — it is a prerequisite *of* the harnesses, not one of them).

### T2 — Scale-safety budget
K1674 demands **p99**, which requires ≥ 100 samples for a credible
percentile (Chernoff bound on tail estimator; standard Ops practice).
Source K1240 per-swap cost p50 = 14.5 ms; p99 typically ≥ 3×p50 on
MLX workloads (graph-trace variance), so budget p99 ≤ 50 ms.

Per-cycle cost budget (attach + K1675 forward pass + detach + forward
pass + cosine):
- attach: ~ 50 ms (p99)
- forward pass 1 token: ~ 30 ms
- detach: ~ 20 ms
- forward pass 1 token: ~ 30 ms
- cosine + telemetry: negligible

≈ 130 ms/cycle × 100 cycles × 3 seeds (variance across runs) ≈ 39 s
pure swap-loop. Plus model load (~ 15 s) and adapter weight loads
(~ 200 ms × 100 = 20 s).

**Est. total ≈ 15 min including setup. < 120 min ceiling.**
T2 does not block alone; reinforces only.

### T3 — DB literal schema completeness
`experiment get exp_prod_adapter_attach_ux`:
```
success_criteria: [] # MISSING
⚠ INCOMPLETE: missing success_criteria
```

This is a DB LITERAL match for F#502/F#646. It is the **7th** such
match in the audit-2026-04-17 drain (after tfidf_routing_no_alias,
flywheel_real_users, loader_portability, registry_host,
version_resolution, differential_privacy_training).

**T3 blocks.**

### T4 — Kill-criterion pin ratio
- K1673: "CLI: pierre attach math" ✓ + "pierre detach math" ✓ +
  "without server restart" ✗ (no pin on what counts as "restart" —
  Python-level re-init vs OS-level re-exec vs container-level
  re-create?) ⇒ 2 pinned / 3 sub-claims.
- K1674: "attach_adapter()" ✓ + "detach_adapters()" ✓ + "< 200 ms" ✓
  + "p99" ✓ + "hot-swaps" ✗ (no definition distinguishing "hot" from
  "cold" swap — does a graph-trace flush count as cold?) ⇒ 4 / 5.
- K1675: "logit cosine with pre-attach" ✓ + "exactly" ✗ (no numerical
  threshold — `cos == 1.0` bit-exact? `cos ≥ 0.9999`? `cos ≥ 0.99`?
  "exactly" is semantically undefined under float arithmetic) ⇒ 1 / 2.

Total pin ratio = 7 / 10 = **0.70** (above the 0.20 auto-block floor).
**Reinforces only.**

### T5 — Source-scope breach vs `exp_p1_t4_serving_v2`
Target depends on this SUPPORTED source (same platform, same framework).
Source's own LITERAL KCs and proof scope:
- **K1240** (LITERAL, PASS @ p50=14.5 ms): *"Swap+first-forward
  latency < 100 ms"* — p50, not p99; attach-only, not detach.
- **K1241** (LITERAL, PASS): *"Decode-only throughput degradation
  < 15%"* — steady-state decode, not attach/detach events.
- **K1242** (LITERAL, PASS): *"Real TF-IDF routing latency < 5 ms
  at N=5"* — router, not swap API.
- Source `run_experiment.py`: uses `model.load_weights(path,
  strict=False)` then `mx.eval(model.parameters())` — NOT the
  `pierre.attach_adapter()` / `detach_adapters()` code path. Target
  K1674 explicitly names the pierre.py API, which source does not
  exercise.
- Source MATH.md `grep -iE "(cli|pierre\s+attach|pierre\s+detach|
  argv|entry_point|subprocess)"`: **0 hits** in source's Theorem 1-3
  derivations. Source never enters CLI scope.

Five LITERAL target/source scope breaches:

- **(A) CLI-scope.** Source proves adapter swap latency for the
  Python-level API. Target K1673 requires a `pierre` CLI binary
  invocable from shell, which neither source nor repo provides.
  `pyproject.toml [project.scripts]` declares only `compose = ...`;
  the CLI binary is a new artifact.
- **(B) detach-scope.** Source K1240 is *attach + first-forward*;
  source never exercises a clean detach back to base. Target K1675
  requires detach followed by cosine verification against the
  pre-attach baseline, a round-trip path not in source's evidence.
- **(C) p99-scope.** Source `results.json` LITERAL reports p50
  (14.5 ms). A p50 PASS does not imply a p99 PASS; the tail
  distribution of graph-trace variance on MLX is the open question
  K1674 demands. Source measured the wrong percentile.
- **(D) process-restart-scope.** K1673 "without server restart"
  presupposes a persistent server process (HTTP/gRPC/stdio). Source
  is a benchmark script that creates and tears down one MLX process
  per run; no server model exists in source or repo
  (`pierre/pierre.py` is a library, not a server).
- **(E) state-consistency-scope.** K1675 cosine-equals-pre-attach is
  a *round-trip identity* claim: `logit(base) == logit(attach → detach
  → base)`. Source never tested this identity; `pierre/pierre.py:113`
  `detach_adapters()` restores the original base modules by module
  replacement, which *should* be bit-exact on module pointer
  restoration — but this is an untested claim under MLX's lazy eval +
  quantized base (K1643 "4-bit base" per pierre.py), where
  dequant/quant reshuffles could introduce epsilon-level drift.

**T5 score: 5/5 LITERAL breaches.**

### Verdict
```
block(T1) = True                 # 4 missing artifacts
block(T2) = False                # 15 min ≤ 120 min ceiling; reinforces
block(T3) = True                 # schema-incomplete DB literal (7th F#502)
block(T4) = False                # 0.70 > 0.20 floor; reinforces
block(T5) = True                 # 5/5 literal source-scope breaches

all_block = T1 ∧ T3 ∧ T5 = True
defense_in_depth = any single one of {T1, T3, T5} alone also blocks
```

Three independent blockers (T1 ∧ T3 ∧ T5). Matches the iter-37
`version_resolution` profile (also software-infrastructure-unbuilt,
T1 ∧ T3 ∧ T5). The target is irreducible by `pip install` — requires
new in-repo construction of CLI + server model + p99 harness +
cosine harness, none of which PLAN.md Part 2 has scoped.

## THEOREM 1 (Preemptive kill)
Under `block(T1) ∨ block(T3) ∨ block(T5)` (in fact all three), the
only honest verdict achievable by any runtime attach-ux experiment
on the current repo is `killed_preregistered`, because:
1. The data-collection machinery (CLI + server + p99 harness +
   cosine harness) does not exist in-repo (T1).
2. The DB's KC is under-specified (T3); no success_criteria means
   no positive-pass rule exists to distinguish supported from
   provisional.
3. The source never proved (a) CLI invocability, (b) detach path,
   (c) p99 percentile, (d) persistent-server process model, or
   (e) post-detach state-consistency (T5 all five).

Running the target would either silently upgrade to `supported`
without machinery to separate it from `killed` (G1009 antipattern),
or consume multi-hour engineering time on infrastructure-building
that PLAN.md Part 2 has not scoped. **QED.**

## PREDICTIONS (pre-flight)
- P1: T1 shortfall ≥ 3 (≥ 3 of 4 artifacts absent). Expected = 4.
- P2: T2 estimated wall time ≤ 120 min (reinforces only).
- P3: DB `success_criteria` remains `[]` post-run (T3 unchanged);
  `⚠ INCOMPLETE` flag persists.
- P4: `pyproject.toml` contains no `pierre` entry point post-run
  (CLI-scope breach permanent).
- P5: `all_block = T1 ∧ T3 ∧ T5 = True`; `defense_in_depth = True`.

## KILL CRITERIA (pre-registered; locked, do not edit after data)
- **K1673 (CLI works without server restart):**
  pre-flight FAIL — no `pierre` entry point in `pyproject.toml`
  + no server process model in repo (T1 ∧ T5(A,D)).
- **K1674 (Python <200 ms p99 hot-swap):**
  pre-flight FAIL — no p99 harness; source measures p50 only
  (T1 ∧ T5(C)). Python `attach_adapter`/`detach_adapters` exists
  but timing budget is unverified.
- **K1675 (Detach removes delta exactly via logit cosine):**
  pre-flight FAIL — no pre/post cosine harness; source never
  tested the round-trip identity (T1 ∧ T5(B,E)).

All three are **preempted** by the 5-theorem stack; none is exercised
with data.

## BEHAVIORAL OUTCOME
The behavior this preempt enforces: do not consume multi-hour
engineering time to build a CLI binary + FastAPI/stdio server +
p99 harness + cosine round-trip harness on top of a target whose
DB schema is `⚠ INCOMPLETE` and whose source measured neither p99
nor the detach path. Drain-forward honesty: if the infrastructure
(CLI + server + p99 + cosine harnesses) is absent in-repo, say so
and route the work to the operator who can either (a) add the four
artifacts and re-file the target, (b) declare server-UX out-of-scope
for local-apple (Pierre is a library), or (c) downgrade to P≥3.

## ASSUMPTIONS (logged per guardrail 1007)
- A1. ap-017 5-theorem stack is the canonical preempt tool for this
  cohort. Defense-in-depth achieved with T1 ∧ T3 ∧ T5.
- A2. The p99 / p50 distinction is a measurement-scope distinction,
  not a definitional one. Source `exp_p1_t4_serving_v2` PAPER.md/
  `results.json` explicitly reports `p50=14.5ms`. Target K1674 says
  p99, and PLAN.md Part 2 does not silently collapse percentiles.
- A3. `pierre/pierre.py:attach_adapter`/`detach_adapters` is a
  library-level Python API, not a CLI. The CLI-scope breach (T5(A))
  is not redeemed by importing these functions inside a shell-invoked
  Python file; K1673 LITERALLY names `pierre attach math` as a shell
  command whose presence in `PATH` is the claim.
- A4. is_smoke = False. This is a complete pre-flight evaluation
  against the target claim, not a partial / smoke run.
- A5. ap-017 axis for this preempt: **composition-bug
  (software-infrastructure-unbuilt variant)** — same axis as iter 37
  (`version_resolution`). All four absent artifacts are in-repo
  constructions (CLI entry-point, server model, p99 harness, cosine
  harness) that `pip install` cannot resolve. Distinct from iter 38's
  platform-library-cross-cut variant (which adds the external MLX
  ecosystem gap for DP-SGD libraries).
- A6. 15 min T2 budget assumes p99 cycles run serially. A parallel
  harness could complete in < 5 min, but parallel MLX shares the
  unified-memory arena and would contaminate p99 with batch
  interference — serial is the only correct harness.
- A7. **T1 probe noise, post-run transparency.** The automated T1
  runner reports `shortfall = 2` (not 4). Two sub-checks flipped to
  `true` due to repo-wide grep noise, not because the target's
  prerequisites were actually satisfied:
  (i) `persistent_server_process_model = true` — 7 regex hits on
  `@app.(post|get)` and similar, originating from `composer/`,
  `macro/` benchmark scripts, and skill-code paths that are not a
  long-running Pierre server. No `pierre serve` entry point; no
  in-repo FastAPI/uvicorn service that keeps Pierre loaded between
  requests. K1673's "without server restart" presupposes a Pierre
  process model that these hits do not constitute.
  (ii) `p99_latency_harness_in_swap_path = true` — one file
  (`micro/models/adapter_hot_cache_mlx/run_experiment.py`) contains
  both `attach` and `percentile`. That is a prior *smoke* harness,
  not a production p99-budgeted swap harness, and it does not
  exercise `pierre.attach_adapter` / `pierre.detach_adapters` or
  emit a p99 metric bound to the K1674 200 ms budget. Treating it
  as satisfying K1674's harness requirement is a false positive.
  The two unambiguous absences — `pierre_cli_entry_point = false`
  (verified by the `pyproject.toml [project.scripts]` literal
  `compose = "composer.compose:main"` which is the *only* entry)
  and `logit_cosine_pre_post_attach_harness = false` (0 hits
  repo-wide) — still drive `T1.block = true`, which with T3 ∧ T5
  produces defense-in-depth = 3 theorems firing. Verdict unchanged.
- A8. **T5(C) p99-scope, literal correction.** The runner reports
  `source_has_p50 = false, source_has_p99 = false` in source
  `results.json`. Source PAPER.md prose references `p50 = 14.5 ms`
  (in the "Connection to T4 Tier" table cited from T4.3), but the
  source's own `results.json` schema does not bind either
  percentile. This strengthens T5(C) rather than weakens it:
  target K1674 demands p99 in a schema where the source never
  pinned any percentile at all. Automated T5(C) flag is `false`
  because the regex is exact-string, not prose-inferred; the
  5/5 → 4/5 T5 score still exceeds the `hits >= 3` block floor.

## NON-GOALS
- Building the Pierre CLI. PLAN.md Part 2 has not scoped it.
- Building a FastAPI / stdio Pierre server. Pierre is a library.
- Running any attach / detach / p99 / cosine measurement.
- Proposing a v2 experiment. That is the operator's call after
  either (a) adding the 4 artifacts as new SUPPORTED dependencies,
  or (b) downgrading the target to P≥3 / library-only scope.
