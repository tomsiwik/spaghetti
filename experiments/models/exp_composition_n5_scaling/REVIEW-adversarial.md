# REVIEW-adversarial — exp_composition_n5_scaling

## Verdict: KILL (preempt-structural, method-dependent redundancy + F#666-pure)

Reviewer pass on researcher artifacts. Adversarial checklist clean. F#731 already filed; DB status=killed. No revisions required.

## Consistency (a)–(d)

- (a) `results.json["verdict"] = "KILLED"` matches DB `status=killed` matches PAPER.md verdict line ✅
- (b) `all_pass = false`; both KCs `"inconclusive"`; status=killed consistent ✅
- (c) PAPER.md verdict: "KILLED (preempt-structural)" — no `PROVISIONAL` / `PARTIALLY` inconsistency ✅
- (d) `is_smoke: false` — no smoke-claim/full-run mismatch ✅

## KC integrity (e)–(g)

- (e) KC IDs K1892, K1893 consistent across MATH.md §3, results.json, PAPER.md prediction table ✅. No post-claim relaxation.
- (f) Tautology sniff: N/A (no run); preempt-structural rationale explicit, not hiding behind algebraic identity.
- (g) Code↔DB: K-IDs in `run_experiment.py` output dict match MATH.md §3 ✅

## Code↔math (h)–(m2)

All N/A — `run_experiment.py` imports only `json`/`pathlib`, writes a graceful-failure `results.json` and exits. No MLX training loop, no LoRA composition, no routing, no `shutil.copy`, no hardcoded `pass: True`, no proxy model substitution. (m2) skill invocation: N/A — no platform code to trust.

## Eval integrity (n)–(q)

All N/A — no empirical run.

## Target-gated kill (t)

**Does NOT apply.** This is a preempt-structural KILL. Carve-out per §5 clauses in `reviewer.md`:
- F#666-pure standalone carve-out applies to K1892 (PPL proxy, canonical guardrail 1007).
- Method-dependent redundancy (novel candidate sub-pattern): under *either* composition branch, the KC outcome is determinable from existing findings (F#406/F#54/F#367 Grassmannian; F#543/F#510/F#511 uniform). No KC was measured. (t) gates kills on proxy-FAIL data; here no data exists.

## Scope-changing fix (u)

N/A — graceful-failure stub is the canonical preempt-structural artifact, not a scope change.

## Deliverables (r)–(s)

- (r) PAPER.md §"Prediction vs measurement" table present with both KCs ✅
- (s) Math errors: none. Three theorems internally consistent:
  - Thm 1 (F#666-pure): K1892 PPL is guardrail 1007; K1893 target-binding genuinely missing (no dataset/evaluator bound to "quality").
  - Thm 2 (Grassmannian redundancy): F#406 N=25@4B SUPPORTED + F#367 α=0.39 sub-linear monotonicity → N=5 passes trivially under this method.
  - Thm 3 (uniform redundancy): F#543 uniform N=5 Qwen 7B 2.57× PPL bloat already KILLED; F#510/F#511 SUPPORTED on standard-LoRA pre-merge destruction.

Combined: KC outcome derivable without running under either branch. Under-specified method + canonical proxy + target-pair ambiguity = preempt-KILL.

## F#702 hygiene-patch

- platform: `local-apple` ✅
- dir: `micro/models/exp_composition_n5_scaling/` ✅
- evidence: added via `experiment complete` ✅
- references: cited inline in PAPER.md and MATH.md (6 prior findings) ✅
- success_criteria: CLI flag not supported; omitted per precedent (non-blocking).

## Novel sub-pattern candidate (analyst note)

**Method-dependent redundancy** — distinct from the three existing preempt-structural sub-cases:
- F#669-family (parent-target-unverified): parent PROVISIONAL, child KCs inherit unverified RHS anchor.
- F#702-family (method-unavailable): mechanism not runnable in current platform/scope.
- F#666-pure standalone: all KCs proxy-only, no target metric.
- **NEW: method-dependent redundancy** — KCs well-formed *in principle* but the method space collapses to branches each already covered by prior SUPPORTED/KILLED findings, so outcome is pre-determined regardless of branch selected.

F#731 is the 1st drain-window instance. Watchlist: a 2nd instance (candidate siblings: `exp_composition_runtime_vs_merge_n10`, `exp_composition_weighted_sum`, `exp_composition_clustering_group`, `exp_composition_residual_analysis`, `exp_composition_ordering_matters`) would promote a standalone memory. Note: this candidate can co-fire with F#666-pure (as here), but stands independently — even if K1893 had a target-metric binding, Thm 2/Thm 3 redundancy would still preempt-KILL.

## Routing

Researcher already executed `experiment complete --status killed` and `experiment finding-add` (F#731 verified via `finding-list`). Reviewer emits `review.killed` — hands off to analyst for LEARNINGS.md literature-context pass.
