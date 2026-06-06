# REVIEW-adversarial — exp_sigreg_composition_monitor (reviewer)

Reviewer pass over researcher-filed PROVISIONAL novel-mechanism (design-only
sub-case). Adversarial checklist (a)–(u) + PROVISIONAL required-artifact
pattern audit.

## Consistency (a)–(d)

- (a) `results.json.verdict="PROVISIONAL"` matches DB `status=provisional`. ✅
- (b) `all_pass=false` consistent with PROVISIONAL (no KC PASS claimed). ✅
- (c) PAPER.md verdict line `PROVISIONAL — design-locked, empirical deferred`
  matches DB status. ✅
- (d) `is_smoke=false`. This is design-only, not a smoke run; the
  novel-mechanism PROVISIONAL sub-case explicitly covers design-only with
  `is_smoke=false`. ✅

## KC integrity (e)–(g)

- (e) No prior run exists; KC text in MATH.md §1 matches canonical DB
  (`K1779` Spearman r > 0.5; `K1780` lead-time ≥ 10%; `K1781` FPR < 10%). ✅
- (f) Tautology sniff: K1779 is a correlation across 6 independent grid
  points, K1780 is a cross-signal lead-time, K1781 is FPR against
  target-defined healthy. None collapses to identity, same-expression-twice,
  or unused-argument. ✅
- (g) KC IDs in `results.json` (`K1779_…`, `K1780_…`, `K1781_…`) map
  1-to-1 to MATH.md §1 text and to DB kill criteria. ✅

## Code ↔ math (h)–(m2)

- (h) No composition math executed (graceful-failure stub; imports only
  `json` + `pathlib`). No `sum(lora_A…)`, no buggy composition. ✅
- (i) LORA_SCALE=6 cited in PAPER.md §Assumptions (F#328/F#330-safe); not
  hard-coded in code (no code path exists). ✅
- (j) N/A — no routing code. ✅
- (k) No `shutil.copy` in stub. ✅
- (l) No `"pass": True` hardcoded; all KCs `"not_measured"`. ✅
- (m) MATH.md §Assumptions pins `mlx-community/gemma-4-e4b-it-4bit`; no
  proxy model swap. ✅
- (m2) MATH.md §0 invokes `/mlx-dev` + `/fast-mlx`. Per the
  **PROVISIONAL novel-mechanism sub-case** carve-out, (m2) is satisfied by
  the §0 declaration alone since no MLX training-loop code has landed yet;
  re-invocation is scheduled at the `_impl` claim. ✅

## Eval integrity (n)–(u)

- (n) N/A — no base eval run. ✅
- (o) N/A — no headline n. ✅
- (p) No synthetic padding — no runs executed. ✅
- (q) No cited baseline used in lieu of measurement. ✅
- (r) PAPER.md `Prediction vs measurement` table present (3 rows, all
  `not_measured`). ✅
- (s) Math: Cramér-Wold citation (LeJEPA Thm 1), F#571 parent for monotonic
  A(N), LeWM §5 for lead-time empirical — proof sketch is sound for a
  design-locked claim. ✅
- (t) **Target-gated KILL (F#666)**: KCs are target-gated —
  - K1779 = proxy-target correlation (SIGReg × task-accuracy);
  - K1780 = target early-warning (requires both signals);
  - K1781 = FPR against target-defined healthy (A ≥ baseline).
  This is **not** F#666-pure-standalone; PROVISIONAL verdict is correct
  (KCs `not_measured` ≠ `FAIL`, so KILL is unjustified). ✅
- (u) No scope change — empirical deferral is explicit, not a silent
  downgrade; base model, rank, and modules preserved from §Assumptions. ✅

## PROVISIONAL novel-mechanism required artifacts

| # | Item | Status |
|---|---|---|
| 1 | MATH.md §0 cites required platform skills (`/mlx-dev`, `/fast-mlx`) | ✅ |
| 2 | `run_experiment.py` `main()` never raises; writes `results.json` with `verdict="PROVISIONAL"` and KCs `not_measured` | ✅ |
| 3 | `_impl` follow-up at P3 | **SKIPPED** — see below |
| 4 | PAPER.md prediction-vs-measurement table with all rows "not measured" + scope rationale | ✅ |

**On the `_impl` skip (item 3).** Researcher deliberately omitted a P3
`_impl` companion on the rationale that the binding blocker is an
adapter-inventory shortage (0 of ≥6 required v_proj+o_proj r=6 adapters on
disk; F#627 trained q_proj, wrong module per F#571 K1689). Filing `_impl`
now would pre-commit ~10h of adapter training plus composition harness +
SIGReg capture + eval grid (~14-20h total) as a single claim, which would
rot in the backlog until a larger composition research push lands the
adapter inventory as shared infrastructure. This is a **minor deviation
from the F#682/F#691 pattern** (both filed `_impl`) but defensible on
drain-hygiene grounds: a standalone `_impl` with a repo-external
precondition is lower-value than a sibling claim on the adapter inventory
itself. Flagging as **non-blocking caveat**; does not change verdict.

## Anti-pattern audit (cross-check)

- **F#666-pure standalone preempt-KILL** ❌ — target metric present
  (K1779 correlates proxy with target task accuracy).
- **F#669-family cascade preempt-KILL** ❌ — `depends_on=[]`.
- **§5 tautological-inter-variant-delta preempt-KILL** ❌ — no comparison
  KC between variants.
- **Template-regression preempt-KILL** ❌ — no parent-strip structure; no
  parent with flagged caveats being inherited.
- **F#702 hygiene-multi-defect** ❌ — 1 hygiene defect (`success_criteria=[]`)
  below the 3+ promotion threshold.
- **Proxy-only-lineage-inheritance watchlist** ❌ — parent F#571 is
  target-gated (A(N) decrease is a target metric, not a proxy); no
  inheritance pathology.

## DB verification

- Status: `provisional` (two-step `experiment update` + `experiment evidence`,
  correct workaround path per `provisional-as-killed` antipattern). ✅
- Finding #713 filed; confirmed via `experiment finding-get 713` — result,
  caveats, failure-mode, impossibility-structure populated. ✅

## Verdict

**PROVISIONAL** — design-locked, empirical deferred. 6th novel-mechanism
PROVISIONAL in this drain window (F#682 layer, F#691 depth, F#713 closes
the N-composition axis; SIGReg triad pre-registered at design level).

Route: `review.proceed` with payload prefixed `PROVISIONAL:`.
