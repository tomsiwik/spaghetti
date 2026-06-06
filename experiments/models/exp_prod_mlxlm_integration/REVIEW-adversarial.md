# REVIEW-adversarial: exp_prod_mlxlm_integration

**Verdict:** KILL (confirm).
**Reviewer:** 🔴 Reviewer hat, 2026-04-18.
**One-line reason:** All five preconditions fail by source+filesystem
inspection; KCs K1651/K1652/K1653 are unmeasurable as pre-registered
in MATH.md. Researcher already emitted `experiment complete --status
killed`; this review confirms and adds finding.

## Adversarial checklist

**Consistency (a–d):**
- (a) `results.json["verdict"] = "KILLED"` matches proposed DB status
  `killed`. ✓
- (b) `all_pass = false` and KCs `[1651, 1652, 1653] = false` match
  claim (killed, not supported). ✓
- (c) PAPER.md verdict line: `KILLED — infrastructure_blocked` (line 3);
  TL;DR explicitly says "not PROVISIONAL — proven unmeasurable". ✓
- (d) `is_smoke: false`, `ran: false` — honest preflight-only encoding.
  This is not a smoke-run masquerading as full; it is a precondition
  audit that never reached execution. ✓

**KC integrity (e–g):**
- (e) Experiment dir is untracked in git (fresh 2026-04-18 creation).
  No post-run KC mutation possible — MATH.md §"Pre-registered Kill
  Criteria" matches DB IDs 1651/1652/1653 verbatim. ✓
- (f) No tautology. KCs are falsifiable real-world checks: HTTP 200
  with non-empty response stream; Jaccard < 0.5 on two different
  selector bodies; `(direct − served)/direct < 0.05` on 256-token
  median-of-5. ✓
- (g) `run_experiment.py` functions (`check_*`) measure exactly what
  MATH.md Theorems 1–3 require (entry-point enumeration, `pierre-g4e4b`
  presence, body-schema grep, adapter safetensor size, depends-on
  verdict). ✓

**Code ↔ math (h–m2):**
- (h) No `sum(lora_A` / `add_weighted_adapter(...)` / independent
  safetensor key summing — script is preflight-only, no composition
  path. ✓
- (i) No `LORA_SCALE` in code. ✓
- (j) No routing / no per-sample dispatch in code. ✓
- (k) No `shutil.copy` of sibling adapters. ✓
- (l) No hardcoded `{"pass": True, ...}` — every preflight result is
  computed from observed state, and all-but-one are `False`. ✓
- (m) Target-model disparity: MATH.md names `pierre-g4e4b`; script
  checks for that literal directory/cache entry. No proxy substitution. ✓
- (m2) Skills: not applicable to this hat. The work is filesystem +
  entry-point introspection over a released wheel (`mlx-lm 0.31.2`
  source inspection); no MLX tensor code, no training loop, no model
  load. `/mlx-dev` / `/fast-mlx` invocation is not required for a
  preflight script, and no unidiomatic MLX patterns are present
  (script touches no `mx.*` primitives). ✓

**Eval integrity (n–q):** not applicable (no eval executed).

**Deliverables (r–s):**
- (r) PAPER.md "Predicted vs measured" table present at line 14–20,
  with all three predictions marked FAIL and rationale linked to
  preflight probes. ✓
- (s) Theorems 1–3 are sound software-engineering reasoning with
  cited source locations (`server.py:1155, 1236`, entry-point group
  enumeration, on-disk checks). No mathematical errors; no unsupported
  claims. ✓

## Assumptions logged

- Treated `exp_prod_mlxlm_integration` as an engineering / packaging
  experiment, not a research claim. Applied the preflight-soundness
  standard used for `exp_prod_pip_package_pierre` earlier today.
- Accepted MATH.md's framing that "registered loader" means an
  upstream-provided plugin API, not a fork. The researcher logs this
  assumption under `## Assumptions logged` in MATH.md and PAPER.md;
  alternate framing would re-file as `exp_prod_pierre_server`.

## Cross-experiment signal (forwarded to analyst)

Fourth 2026-04-18 KILL-by-infra-absence after `exp_bench_aime_2026`,
`exp_bench_livecodebench_v6`, and `exp_prod_pip_package_pierre`. Common
blockers:
- **C1 ADAPTER-REBUILD** — `exp_p1_t2_single_domain_training`
  safetensors missing on disk across math/code/medical. Hits 3 of 4
  kills.
- **C2 PIERRE-PACKAGE-RENAME** — cascades from the pyproject rename
  decision.
- **C4 PIERRE-SERVER vs MLX-LM-FORK** — resolving this unblocks the
  serving roadmap.

Draining more infra-blocked experiments without first landing C1/C2/C4
is expected to produce more identical KILLs. Flag for analyst to raise
to the planner.

## Verdict: KILL (confirm)

- `experiment complete exp_prod_mlxlm_integration --status killed` is
  already recorded in DB (verified via `experiment get`). No further
  CLI `complete` needed.
- `finding-add`: emit a finding that codifies today's infra-blocked
  class and references C1/C2/C4.
- Event: `review.killed`.
