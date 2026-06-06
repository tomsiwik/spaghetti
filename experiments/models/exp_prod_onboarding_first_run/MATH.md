# MATH.md — exp_prod_onboarding_first_run

## Preemptive-Kill Via 5-Theorem Stack
## (PROD-deliverable-cascade, parent-KILLED, 3rd instance after F#740/F#741)

**Verdict: KILLED_PREEMPTIVE** — five independent theorems each
block the claim before any measurement can be produced; any one
sufficient (defense-in-depth).

## Target claim (per DB)

K1670: Time from `pip install` complete to first successful
inference < 30 s on M5 Pro (cold cache OK, warm weights).

K1671: Default bundle ships with base + 5 curated adapters (math,
code, writing, chat, reasoning).

K1672: Zero-config first run: no API key, no manual weight download.

`success_criteria: []` (DB literal `⚠ INCOMPLETE`).

## Parent / source-finding map

**Direct parent (dep): `exp_prod_pip_package_pierre` — KILLED.**
All three of its kill criteria FAIL (unmeasured), per its
`results.json` and PAPER.md:

- B1: `pyproject.toml` `project.name = "lora-compose"`, **not**
  `pierre`. Falsifies clause "`pip install pierre` resolves
  anything."
- B2: `[tool.hatch.build.targets.wheel] packages = ["composer",
  "micro", "macro"]` — `pierre/` is **excluded** from the wheel.
  `import pierre` raises `ModuleNotFoundError` post-install.
- B3: No published artifact under name `pierre` on PyPI / TestPyPI;
  no git tag for `v0.3.0`. No fresh-install path exists at all.

The child's K1670/K1671/K1672 each presuppose a working
`pip install pierre` → `import pierre` → adapter-load chain. The
parent's killed state means none of that chain exists.

**Sister precedents (REUSABLE — PROD-deliverable-cascade family):**

- F#740 (12th F#669 reuse): `exp_pierre_multi_adapter_serving_throughput`
  preempt-KILL because **parent `exp_prod_mlxlm_integration` is
  KILLED** per F#570. Same structural axis: child PROD experiment
  with KILLED parent → child unmeasurable.
- F#741 (13th F#669 reuse): `exp_pierre_adapter_cache_prefill`,
  same parent (`exp_prod_mlxlm_integration` KILLED), same axis.
- This experiment: child PROD experiment with KILLED parent
  (`exp_prod_pip_package_pierre` rather than
  `exp_prod_mlxlm_integration`). 3rd PROD-deliverable-cascade
  instance; promotion candidacy at 4th (within-cluster) or
  cross-cluster reuse.

The parent change (pip-package vs mlxlm-integration) is a different
deliverable axis but the same cascade structure: any PROD child
whose dep chain transits through a KILLED step is unmeasurable
until the dep is repaired.

**F#666-violation (TARGET-GATED KILL guardrail):**

All three child KCs are **proxy-only**:
- K1670: stopwatch timing (operational latency, proxy for UX).
- K1671: bundle config inventory (engineering-checklist proxy).
- K1672: subjective "no API key, no manual download" (presence
  proxy for an engineering invariant).

There is **no target-metric KC** — no task-accuracy, no
behavioural-quality, no oracle-gap measurement that the
fast-onboarding is supposed to enable. Per Finding #666, kill on
proxies-alone is forbidden; equivalently, a "supported" verdict on
proxies-alone is uninterpretable. The pre-reg as written cannot
yield a behavioural conclusion regardless of measurement outcome.

## 5-Theorem Stack

### T1 — Artifact shortfall (parent-deliverable-unavailable)

Required artifacts to time `pip install pierre` → first inference:
- A `pierre` package on PyPI (or TestPyPI, or a local wheel under
  the `pierre` name) — **absent** (B3 of parent KILL).
- A `pierre/__init__.py` shipped in the wheel — **absent** (B2;
  `pierre/` excluded from `[tool.hatch.build.targets.wheel]
  packages`).
- A `pyproject.toml` with `project.name = "pierre"` — **absent**
  (B1; current name is `lora-compose`).
- Five curated bundled adapters (math, code, writing, chat,
  reasoning) installed by the wheel — **absent** (no
  `pierre/bundles/*.safetensors` paths in the repo; the only
  prebuilt adapters live under `micro/models/exp_*/` per-experiment
  dirs, not under a package-bundle directory).
- A first-run entry point (e.g. `pierre.first_run.run()` or a
  console script `pierre`) wired in `[project.scripts]` — **absent**
  (no `pierre` console script in `pyproject.toml`).
- A "no API key, no manual download" runtime: HF cache pre-warmed
  by the package, or weights bundled — **absent** (no bundled
  weight artefacts; HF cache is per-user).

shortfall ≥ 5 missing artefact categories. The `pip install
pierre → first inference` chain has zero of its required steps
realisable on the local platform.

### T2 — Parent-supersession (PROD-deliverable-cascade)

Claim's measurement chain requires the parent
`exp_prod_pip_package_pierre` to be SUPPORTED (or at minimum to have
a working `pip install pierre` artefact). Parent status: **KILLED**,
all three parent KCs FAIL. Until the three parent blockers (B1, B2,
B3) are remediated, no child KC is reachable.

Cascade-instance index: 3rd PROD-deliverable-cascade preempt
(after F#740, F#741). Promotion candidacy at 4th cross-cluster or
2nd within-cluster reuse — track in LEARNINGS for analyst.

### T3 — Schema completeness (F#502/F#646 cohort)

DB claim output literal: `success_criteria: [] # MISSING`,
`⚠ INCOMPLETE: missing success_criteria`. Per F#502/F#646, the
schema-completeness gap is a sustained taxonomic signal (not by
itself a kill reason, but a co-indicator that the experiment was
authored against an underspecified spec).

Cohort-hit index:
- F#650 (5th), F#652 (6th), F#763 (7th); this is **8th**.
- 8th hit reaches the analyst-flagged super-family-promotion
  threshold per scratchpad guidance: track for next-pass analyst
  promotion of `success_criteria=[]` from co-indicator to
  preempt-kill axis.

### T4 — KC pin count (K1670/K1671/K1672 each underpinned)

K1670 ("< 30 s on M5 Pro, cold cache OK, warm weights"): host
pinned, threshold pinned (30 s). Missing pins: dataset/prompt
identity for "first successful inference", warm-weights provenance
(cache hit vs bundled), CPU/GPU thermal state, cold-vs-warm-cache
quantification.

K1671 ("base + 5 curated adapters: math, code, writing, chat,
reasoning"): adapter list pinned by name only. Missing pins:
adapter-version hashes, training-recipe pins, behavioural-quality
floor per adapter (otherwise "ships 5 adapters" is satisfied by
shipping 5 random-init shells).

K1672 ("no API key, no manual weight download"): negative
specification only. Missing pins: enumerable list of "config knobs
that must default to working values", definition of "first run"
(invocation surface).

pin_ratio across {epsilon, baseline, host, dataset, scaling-rule}
≈ 0.20 (1/5 average). K1671 and K1672 are non-falsifiable as
stated — any shipped artefact "satisfies" them under a generous
read.

### T5 — Source-scope literal breach (per parent + F#666)

Parent `exp_prod_pip_package_pierre` PAPER.md §Verdict
("KILLED — infrastructure-blocked") explicitly does **not**
upgrade to inconclusive: it pre-registers theorems 1–3 as
predictions for a future re-attempt **once remediation lands**.

The child's KCs would attempt to measure step 2 (timing) of a chain
where step 1 (the pip-installable artefact) is verified absent.
This is the same source-scope breach pattern named in F#650 T5(B)
and F#763 T5(A,B,C,D), specialised to the "child runs ahead of
KILLED parent" axis.

Breaches:
- (A) Parent deliverable absent → child measurement-chain step 1
  vacuous.
- (B) Default-bundle path (`pierre/bundles/*`) absent → K1671
  vacuous.
- (C) Zero-config invocation surface (`pierre` console script)
  absent → K1672 vacuous.
- (D) F#666 — proxy-only KC panel without target-metric pair →
  no behavioural conclusion possible regardless of timing outcome.

5/5 breaches; each is sufficient.

## Decision rule

```
all_block = T1 ∧ T2 ∧ T3 ∧ T4 ∧ T5
defense_in_depth = (any single Ti ⇒ KILL)
verdict = KILLED_PREEMPTIVE
```

## What an unblock would require

The preempt does **not** assert "fast onboarding is impossible";
it asserts "the current repo + parent KILL state cannot measure
it." To unblock, all of:

1. Resurrect parent `exp_prod_pip_package_pierre`: rename
   `pyproject.name` to `pierre`, add `"pierre"` to wheel packages,
   publish a v0.3.x artefact.
2. Add `pierre/bundles/{math,code,writing,chat,reasoning}.safetensors`
   committed to wheel manifest, with version hashes pinned.
3. Add `[project.scripts] pierre = "pierre.cli:main"` plus a
   `first_run` entry point.
4. Add at minimum one **target-metric KC** paired with K1670 (e.g.
   "first inference produces ≥ X% of MLX-reference behavioural
   quality on a fixed prompt-set"), to satisfy F#666.
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
