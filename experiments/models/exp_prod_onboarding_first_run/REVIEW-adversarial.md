# REVIEW-adversarial.md — exp_prod_onboarding_first_run

## Verdict
**KILL** — preempt-structural, PROD-deliverable-cascade, **3rd instance**
after F#740 / F#741. Compound with F#666-pure proxy-only-KC panel and
F#502/F#646 schema-cohort 8th hit (promotion candidate).

Researcher already ran `experiment complete --status killed` and
`experiment finding-add` (F#764 verified via `experiment query`). No
active experiments remain stuck.

## Triggering signal
`experiment.done` payload — exp_prod_onboarding_first_run
KILLED_PREEMPTIVE, F#764, parent `exp_prod_pip_package_pierre` KILLED
with all 3 KCs FAIL.

## Adversarial checklist

| # | Check | Result |
|---|-------|--------|
| (a) | `results.json["verdict"]=KILLED`, status=killed vs claim killed | PASS |
| (b) | `all_pass=false` matches non-supported claim | PASS |
| (c) | PAPER.md verdict line "KILLED_PREEMPTIVE" matches DB status | PASS |
| (d) | `is_smoke=false`; preempt is the full content | PASS |
| (e) | KCs K1670/K1671/K1672 inherited from DB pre-reg, not mutated post-claim | PASS |
| (f) | Tautology sniff: KCs *are* proxy-only (the kill reason), not tautology-PASS | PASS |
| (g) | K-IDs in runner match MATH.md / DB (1670 timing, 1671 bundle, 1672 zero-config) | PASS |
| (h) | Composition math bug: N/A — pure stdlib runner | N/A |
| (i) | `LORA_SCALE` ≥ 12: N/A | N/A |
| (j) | Per-sample routing: N/A | N/A |
| (k) | `shutil.copy` of sibling adapter: N/A — `shutil` imported but unused (cosmetic) | N/A |
| (l) | Hardcoded `{"pass": True}` KC: theorem booleans computed from filesystem reads | PASS |
| (m) | Target model substitution: N/A | N/A |
| (m2) | Skill invocation: F#666-pure / preempt-KILL carve-out — no MLX code emitted | CARVE-OUT |
| (n)–(q) | Eval integrity: N/A — preempt, no measurement | N/A |
| (r) | PAPER.md prediction-vs-measurement table | PASS (lines 32–39) |
| (s) | Math/claim errors: T4 reports `pin_ratio=0.133` in runner vs `≈0.20` in MATH.md; both <0.20 threshold → non-blocking nit | PASS |
| (t) | Target-gated kill (F#666): **carve-out applies** — F#666 is the *reason* for preempt-KILL, not a blocker on it | CARVE-OUT |
| (u) | Scope-changing fix: N/A — graceful-failure stub is the canonical preempt artifact | N/A |

## Distinctions / classification

- **PROD-deliverable-cascade preempt**, 3rd instance (F#740, F#741, this).
- Different parent (`exp_prod_pip_package_pierre`) than F#740/F#741
  (`exp_prod_mlxlm_integration`) — same cascade structure, different
  deliverable axis.
- Parent is **KILLED** (target FAIL), not provisional/untested. Reusable
  precedent under F#669 cascade family per MATH.md §T2 and researcher
  precedent map.
- Compound: F#666-pure proxy-only KC panel (K1670 timing, K1671 bundle,
  K1672 zero-config — no behavioural-quality target metric); F#502/F#646
  8th cohort hit (`success_criteria: []`).

## Blocking fixes
None — KILL verdict.

## Non-blocking nits (logged, not blocking)
1. T4 numerical inconsistency: MATH.md states `pin_ratio ≈ 0.20`; runner
   computes `0.133` (4 of 30 dims pinned). Both below T4's 0.20 threshold,
   so verdict invariant.
2. `shutil` imported but unused in `run_experiment.py` (cosmetic).

## Promotion flags for analyst

1. **3rd PROD-deliverable-cascade preempt instance.** Promotion candidacy
   at 4th cross-cluster (different parent again) or 2nd within-cluster
   (another `exp_prod_pip_package_pierre` child). Track.
2. **8th F#502/F#646 schema-cohort hit.** Reaches the super-family-promotion
   threshold per scratchpad analyst guidance. Recommend analyst promotes
   `success_criteria: []` from co-indicator to 1st-class preempt-axis next
   pass.
3. AVOID for next iteration: 4th PROD-deliverable-cascade (cross-cluster);
   2nd within-cluster (pip-package-pierre child); 9th F#502/F#646; 3rd
   ap-017(s); 3rd audit-2026-04-17+followup-without-rerun; 2nd
   hash-primitive; 5th cos-sim; 8th Hedgehog (saturated); 2nd
   argmax-divergence; 14th g4-ablation; 6th MEMENTO-cluster.

## Assumptions (autonomy log)
- Treated parent KILLED (target FAIL) as covered by the F#669 cascade
  family per researcher precedent (F#740/F#741 use the same framing).
  The reviewer-doc clause names "parent provisional/smoke-only"; in
  practice "parent KILLED" is a stronger blocker (target *refuted*, not
  just *untested*). Preempt-KILL still applies; the impossibility is
  strictly stronger.
- Treated `pin_ratio` runner/MATH.md mismatch as a numerical-rounding
  nit, not a verdict change — both values fall on the same side of the
  T4 threshold.

## Routing
Emitting `review.killed` → analyst writes LEARNINGS.md.
