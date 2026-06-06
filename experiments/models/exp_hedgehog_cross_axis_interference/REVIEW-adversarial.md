# REVIEW-adversarial.md — exp_hedgehog_cross_axis_interference

**Date:** 2026-04-25 · drain-window iter ~53 · reviewer hat
**Verdict:** **KILL (preempt-structural, F#669-family clause + F#666-schema-defect compound)**
**Routing:** `review.killed` after `experiment complete --status killed --k 1859:fail` + 2 finding-add.

## Summary

3rd consecutive researcher-side preempt-KILL in drain window. NEW sub-mechanism:
**pre-F#770-repair compound F#666+F#669** (distinct from F#776/F#778
post-F#770-repair schema-repair-reveals-F#669 path). Parent
`exp_hedgehog_behavior_adapter_politeness` PROVISIONAL (all 4 KCs untested, no
trained polite-adapter weights). K#1859 also unpaired single KC (F#666
schema-defect, never F#770-repaired per F#771 audit).

## Adversarial checklist (18 items + carve-outs)

| # | Item | Result |
|---|------|--------|
| (a) | results.json verdict ↔ proposed DB status | PASS — both KILLED |
| (b) | all_pass ↔ claim | PASS — all_pass=false matches |
| (c) | PAPER.md verdict line ↔ DB | PASS — "KILLED (preempt-structural...)" |
| (d) | is_smoke ↔ full-run claim | PASS — is_smoke=false, no claim of full-run measurement |
| (e) | KC modification post-claim (git-diff) | PASS — K#1859 byte-for-byte identical to 2026-04-23 DB record (verified via `experiment get`); never F#770-repaired |
| (f) | Tautology sniff | N/A — no measurement |
| (g) | K-ID code measures different quantity than MATH/DB | N/A — no code measures K#1859 |
| (h) | Composition-math bug | N/A |
| (i) | LORA_SCALE ≥ 12 | N/A |
| (j) | Per-sample routing | N/A |
| (k) | shutil.copy as new-domain adapter | N/A |
| (l) | Hardcoded `pass: True` | PASS — `kill_results: {1859: "fail (untested-preempt)"}` |
| (m) | Target-model proxy substitution | N/A |
| (m2) | Skill attestation (`/mlx-dev`, `/fast-mlx`) | **F#669-family carve-out** — no MLX code in `run_experiment.py`; `skill_attestation.{mlx_dev_invoked: false, fast_mlx_invoked: false, carve_out_clause: "F#669-family"}` honest disclosure |
| (n) | Base accuracy 0% / thinking_chars=0 | N/A — no eval |
| (o) | Headline n < 15 | N/A — no n |
| (p) | Synthetic padding | N/A |
| (q) | Cited baseline drift | N/A |
| (r) | Prediction-vs-measurement table | PASS — PAPER.md §2 single-KC table, "UNTESTED-PREEMPT" |
| (s) | Math errors / unsupported claims | PASS — MATH.md §2-§3 derivation transitivity-preempt + schema-defect compound; both grounded in F#669/F#666/F#770/F#771 |
| (t) | Target-gated kill (F#666) | **F#669-family carve-out** — no KC measured (proxy or target); F#666 is *cause* of compound, not blocker |
| (u) | Scope-changing fix antipattern | PASS — MATH.md §6 enumerates 5 shortcuts (surrogate adapter, skip-and-pass, threshold reduction, parent Phase 0 substitution, wait-for-_impl) and rejects each with explicit reasoning |

**18/18 PASS or carve-out N/A. No blocking issues.**

## Verdict-consistency pre-flight (6 items)

1. results.json verdict line: `KILLED` ✓
2. all_pass: `false` ✓
3. PAPER.md verdict line: `KILLED (preempt-structural, F#669-family clause — 16th F#669 reuse; compound F#666+F#669)` ✓
4. is_smoke flag: `false` ✓ (no smoke-vs-full mismatch)
5. KC git-diff: K#1859 unchanged since 2026-04-23 (verified via `experiment get`) ✓
6. Antipattern match: F#669-family preempt-structural + F#666-schema-defect compound; matches NEW sub-form filed as F#NEW2 ✓

## Distinguishing feature (NEW sub-mechanism)

| Path | F#775 (rank_ablation) | F#777 (jepa_scale_sweep) | **This (cross_axis)** |
|------|----------------------|--------------------------|----------------------|
| F#770-repaired? | YES (iter ~36) | YES (iter ~38) | **NO** (excluded per F#771 audit) |
| Diagnosis trajectory | post-repair → schema-repair-reveals-F#669 (F#776) | post-repair → schema-repair-reveals-F#669 (F#778) | **pre-repair → direct cascade with schema-defect alongside** |
| Compound F#666+F#669 | F#666 was repaired before F#669 surfaced | same | **F#666 + F#669 fire simultaneously, never separated** |

This is the 1st pre-F#770-repair compound observation. F#NEW2 captures the
sub-axis (1st of 3 needed for canonicalization).

## Doom-loop self-check (reviewer)

- `python3 .ralph/tools/doom_loop.py` exit=0.
- Prior 2 reviewer iters (~48, ~50): preempt-KILL F#669-family **post-F#770-repair**.
- This iter (~53): preempt-KILL F#669-family **pre-F#770-repair compound F#666+F#669** (NEW sub-mechanism).
- Substantively distinct per `mem-pattern-triple-fire` — different diagnosis trajectory, NEW sub-axis filed.
- NOT a doom-loop.

## Operational note (NOT a finding)

3rd consecutive researcher-side preempt-KILL signals drain-stall reality
(documented in PAPER.md §6 honestly). All remaining P≤2 open entries cascade
off PROVISIONAL parents OR exceed 90-min researcher cap. Operator action to
unlock parent _impl macro budgets is the only structural remediation;
preempt-KILL of cascade children is the only in-cap progress path. If a 4th
consecutive preempt-KILL fires next iter, analyst should append to
HALT_ESCALATION.md.

## Actions executed by this reviewer pass

1. `experiment complete --status killed --dir micro/models/exp_hedgehog_cross_axis_interference --k 1859:fail`
2. `experiment finding-add` ×2 (F#NEW1: F#669 16th reuse, 2nd Hedgehog-cluster, 1st pre-F#770-repair compound; F#NEW2: pre-F#770-repair compound F#666+F#669 sub-axis 1st obs).
3. `experiment finding-list` verification.
4. `ralph emit "review.killed" "..."`.
