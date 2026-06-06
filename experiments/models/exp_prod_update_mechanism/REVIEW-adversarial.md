# REVIEW-adversarial.md — exp_prod_update_mechanism

## Verdict: KILL (preempt-structural, parent-supersession sub-case)

4th PROD-deliverable-cascade preempt; F#765 already registered, experiment
already marked `killed` in DB. Reviewer confirms.

## Triggering signal

`experiment.done` payload — KILLED_PREEMPTIVE F#765, parent
`exp_prod_version_resolution` KILLED with K1662/K1663/K1664 all FAIL.
2nd cross-cluster reuse over 3 distinct PROD parents
(mlxlm-integration, pip-package-pierre, version-resolution).

## DB state cross-check

- `experiment query exp_prod_update_mechanism` → finding #765 present.
- `experiment list --status active` → empty (no stuck claim).
- `experiment query exp_prod_version_resolution` → KILLED, evidence link
  to F#765 attached.
- Researcher already executed `experiment complete --status killed` and
  `experiment finding-add` per scratchpad hand-off; reviewer does not
  re-run.

## Adversarial checklist

| Item | Status | Note |
|------|--------|------|
| (a) verdict consistency | PASS | results.json `KILLED` / `status=killed`; DB matches; PAPER.md verdict line `KILLED_PREEMPTIVE`. |
| (b) all_pass vs claim | PASS | `all_pass=false`, status killed — consistent. |
| (c) PAPER.md verdict line | PASS | KILLED_PREEMPTIVE explicit. |
| (d) is_smoke | PASS | `is_smoke=false`; preempt is the full content. |
| (e) KC mutation post-claim | PASS | K1676/K1677/K1678 unchanged from DB pre-reg. |
| (f) tautology sniff | PASS | Each KC preempt-blocked, no `e=0→0` patterns. |
| (g) K-ID semantics | PASS | Code KC strings match DB. |
| (h) composition math bug | N/A | No model code; pure stdlib grep runner. |
| (i) LORA_SCALE | N/A | No LoRA path. |
| (j) per-sample routing | N/A | No routing. |
| (k) shutil.copy as new adapter | N/A | No adapter material. |
| (l) hardcoded `pass: True` | PASS | Theorem booleans computed from disk state. |
| (m) target-model substitution | N/A | No model load. |
| (m2) skill invocation evidence | CARVE-OUT | Preempt-structural — no MLX code emitted. PAPER.md §Antipattern self-check explicitly invokes the m2 carve-out. |
| (n)–(q) eval integrity | N/A | No eval, no n. |
| (r) prediction-vs-measurement table | PASS | PAPER.md §Prediction-vs-measurement present, 5 rows. |
| (s) math errors | PASS | T1 false-negative explicitly disclosed (grep matched parent's *checking* strings, not implementation strings); defense-in-depth carries the kill at 4/5 blocking. |
| (t) target-gated kill F#666 | CARVE-OUT | Preempt-structural KILL: F#666 is a *partial* compound axis (K1677 target-paired survives, K1676+K1678 proxy-only). The verdict does not rest on F#666-proxy-FAIL alone — it rests on T2 parent-supersession. F#666 carve-out per F#669-family clause applies. |
| (u) scope-changing fix | PASS | Graceful-failure stub is the canonical preempt-structural artifact. No silent SFT↔LoRA swap, no max_length reduction, no monitoring disabling. |

All blocking items PASS or carve-out. No revise required.

## Sub-case classification

**Preempt-structural KILL — parent-supersession (F#669-family) sub-case.**
Distinguishing markers vs adjacent clauses:

- NOT F#666-pure standalone — K1677 has target-metric pair (`quality
  within 5% of original`), so the panel is not F#666-pure (only 2-of-3
  KCs proxy-only).
- NOT tautological-inter-adapter-delta — KCs are upgrade-flow primitives,
  not inter-variant deltas.
- NOT F#702 hygiene-patch PROVISIONAL — the defect is parent-state, not
  metadata; not patchable by hygiene-fix + `_impl`.
- IS F#669-family — every child KC transitively requires the parent's
  target (K1662/K1663/K1664) to be SUPPORTED; parent is KILLED with
  all parent KCs FAIL → measurement chain step 1 vacuous.

Per F#669-family preempt-structural clause requirements:
1. MATH.md §1 derives transitivity (T2 parent-supersession). ✓
2. `run_experiment.py` graceful-failure: no MLX, no model load, writes
   `results.json` directly. ✓
3. PAPER.md "Operator unblock" section lists concrete unblock conditions
   (resurrect parent, implement upgrade primitives, populate
   success_criteria, add target-metric KCs to K1676 + K1678). ✓
4. No `_impl` companion — parent's own resurrection is the unblock. ✓

## Promotion signal (for analyst pass)

**4th PROD-deliverable-cascade preempt; 2nd cross-cluster reuse over
3 distinct PROD parents.** The cascade structure is reproducibly
identical across `exp_prod_mlxlm_integration`, `exp_prod_pip_package_pierre`,
and `exp_prod_version_resolution`. Per researcher's promotion recommendation
in PAPER.md and scratchpad: analyst should promote
`PROD-child-with-KILLED-parent` from compound preempt-axis to top-level
guardrail on next pass — future PROD child with KILLED parent should
preempt-KILL on parent-state alone (no 5-theorem stack required).

Also: 9th `success_criteria=[]` cohort hit (F#502/F#646 axis); reinforces
the 8th-hit promotion threshold flagged at F#764.

## Assumptions (autonomy log)

- T1 false-negative is documented and not load-bearing; reviewer does not
  treat it as a structural defect. Defense-in-depth at 4/5 blocking
  theorems satisfies the decision rule.
- F#765 already registered with status `supported` (the finding-record
  verdict, not the experiment verdict; an `experiment finding-add`
  output convention). Reviewer does not re-emit.

## Routing

`review.killed` — researcher's KILL verdict confirmed, experiment closed,
finding registered, drain advances. Analyst writes LEARNINGS.md next.
