# REVIEW-adversarial.md — exp_sigreg_threshold_sweep

Preempt-structural KILL. Reviewer pass deferred per F#687/F#698/F#699/F#727/F#728/F#729
precedent — preempt-KILL artifacts do not require the full adversarial audit
(no run was executed; no empirical claims to adversarially probe).

## Artifact hygiene

- MATH.md: present (3 independent theorems in §1.1 / §1.2 / §1.3) ✅
- run_experiment.py: present (no-op scaffold; ran cleanly via `experiment run` pueue) ✅
- results.json: present (verdict=KILLED, kill_type=preempt-structural) ✅
- PAPER.md: present (verdict line + triple-fire ledger + F#669 reuse ledger + §5
  sub-variant ledger + sibling-position table + antipattern audit) ✅
- REVIEW-adversarial.md: present (this file) ✅
- LEARNINGS.md: deferred to analyst pass ✅

## F#702 hygiene-patch

- platform: local-apple ✅
- experiment_dir: micro/models/exp_sigreg_threshold_sweep/ ✅
- success_criteria: populated at `experiment complete` ✅
- evidence: added via `experiment complete` ✅
- references: empty — consistent with preempt-KILL precedent (documented/non-blocking)

## Antipattern re-audit

- F#666-pure: FIRES (|K|=2 both proxy; no target companion) — confirmed
- §5 tautological-inter-variant-delta (intra-detector-threshold-delta): FIRES —
  confirmed (monotone ROC sweep)
- F#669 parent-target-unverified (parent F#713): FIRES — confirmed

No other antipatterns apply (no code path exercised).

## Routing integrity

No new memory is promoted by this pass. Post-promotion routing stable:
- F#666-pure memory (promoted pre-drain; 18th reuse here)
- §5 tautological-inter-variant-delta memory (promoted; 12th reuse; 2nd intra-instantiation)
- F#669 family covered by `mem-promotion-same-parent-repeat-blocker` (parent-repeat memory
  promoted at F#728) — this instance is the 1st cross-parent instance (parent=F#713)
  so the "same-parent-repeat" strict trigger does NOT fire, but the generalised
  "parent-target-unverified" memory still applies.

## Cross-parent observation

This is the first instance where the structural/parent-dependent triple-fire
composition (F#666-pure + §5 + F#669) appears with a parent other than F#682.
Same composition, different parent (F#713). Analyst pass should note this as
evidence that the triple-fire composition is robust across parents.

## Verdict

KILL (preempt-structural). No re-run. No `_impl` follow-up.

## Reviewer concurrence (2026-04-24)

Adversarial checklist (a)–(u) all pass for preempt-structural KILL:
- (a)–(d) Consistency: results.json verdict=KILLED, all_pass=false, is_smoke=false, PAPER.md verdict line "KILLED (preempt-structural)" all align with DB status=killed.
- (e)–(g) KC integrity: MATH.md is fresh (preempt; no KC mutation post-claim); tautology IS the explicit kill rationale (§5 intra-detector-threshold-delta theorem).
- (h)–(m2) Code↔math: N/A — no MLX path exercised; graceful-failure stub is canonical preempt-structural artifact.
- (n)–(q) Eval integrity: N/A — no eval executed.
- (t) Target-gated kill (F#666): does NOT apply to preempt-KILL per §5 carve-outs (F#666-pure standalone clause + F#669-family clause); F#666 is the *reason* for the preempt, not a blocker on it.
- (u) Scope-changing fix: N/A — graceful-failure stub is canonical preempt-structural artifact, not a scope reduction.
- (r)–(s) Deliverables: prediction-vs-measurement table present; math rigorous (3 independent theorems).

F#730 verified in DB via `experiment finding-list --status killed`. No revisions required.
