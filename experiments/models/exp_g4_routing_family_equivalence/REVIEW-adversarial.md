# REVIEW-adversarial.md — exp_g4_routing_family_equivalence

**Verdict:** KILL (preempt-structural).
**Antipattern:** `tautological-inter-adapter-delta-ignores-base-baseline` —
**3rd drain-window instance** (1st K1552, 2nd K1577/F#704, 3rd K1584/F#709).
Promotion threshold reached (3-instance-promote convention → named §5 clause
in `reviewer.md`, analyst action). Simultaneously 3rd sub-variant of
template-regression watchlist (F#705 / F#708 / F#709 → formal antipattern
promotion, analyst action).

## Independent (a)–(u) pass

- **(a) results.json verdict vs DB status.** `verdict=KILLED` ↔ DB
  `status=killed` (verified via `experiment get exp_g4_routing_family_equivalence --yaml`). PASS.
- **(b) all_pass vs claim.** `all_pass=false` ↔ KILLED. PASS.
- **(c) PAPER.md verdict line.** "KILLED — preemptive, structurally
  uninformative KC." No supported-upgrade language. PASS.
- **(d) is_smoke vs claim.** `is_smoke=false`; preempt-structural semantics
  (not a smoke run). PASS.
- **(e) KC modification post-first-run.** K1584 in results.json byte-matches
  DB K1584: "max pairwise gap < 2pp at N=5" (verified via `experiment get --yaml`).
  No git diff (MATH.md / run_experiment.py are new files on this branch). PASS.
- **(f) Tautology sniff test.** K1584 IS tautological — this is the verdict
  driver. Five-lemma proof in MATH.md §1 (L1 degenerate-equivalence, L2
  prerequisite-gate-unmet, L3 F#477 base-beat unlikely, L4 template-regression
  3rd, L5 antipattern-3rd-instance-promote). KILL is the correct routing for
  a tautological KC. PASS (as KILL).
- **(g) K-ID in code vs MATH.md / DB.** K1584 text consistent across DB,
  MATH.md §1, PAPER.md, results.json. PASS.
- **(h) Composition math bugs.** `run_experiment.py` imports only `json` +
  `pathlib`; no composition path. PASS (vacuously).
- **(i) `LORA_SCALE=20`.** Absent. PASS.
- **(j) `route(val[d][0])`.** Absent. PASS.
- **(k) `shutil.copy`.** Absent. PASS.
- **(l) Hardcoded `{"pass": True}`.** Absent; KC `result="untested"`. PASS.
- **(m) Model mismatch MATH.md ↔ code.** MATH.md §6 explicit: "no MLX /
  Gemma 4 / mlx_lm surface touched." Code confirms (2 stdlib imports only). PASS.
- **(m2) Skill invocation evidence.** MATH.md §0: `/mlx-dev` and `/fast-mlx`
  NOT invoked, documented per preempt-structural exception path (reviewer.md
  §5 F#666-pure / F#669-family clauses — and this new taut-inter-adapter-delta
  clause once §5 promotion lands). PASS.
- **(n) Base acc = 0 artifact.** No eval run. PASS (vacuously).
- **(o) Headline n < 15.** No data. PASS (vacuously).
- **(p) Synthetic padding.** No N-claim measured. PASS (vacuously).
- **(q) Cited baseline drift.** Independently verified parent citations via
  `experiment finding-get`: F#150 caveats contain exact phrase "identical to 4
  decimal places" and "Quality comparison vacuous — zero expert specialization";
  F#166 L2 governing lemma confirmed; F#477 base-beat 2/5 (math +20pp, finance
  +14.7pp; medical −4pp; code +6.7pp, legal +9.3pp) confirmed; F#704 confirmed
  as 2nd-instance precedent of the antipattern with K1577 `delta ≥ 5pp`. PASS.
- **(t) Target-gated kill (F#666).** Does NOT apply: K1584's underlying metric
  is MMLU-Pro accuracy (a target), not a proxy — the tautology is in the
  inter-variant-delta *structure* (no base anchor), not in proxy-substitution.
  Preempt-structural KILL verdict is F#666-orthogonal. PASS (carve-out per §5
  F#669-family precedent, symmetric to this clause).
- **(u) Scope-changing fixes.** No scope change; graceful-failure stub is the
  canonical preempt-structural artifact pattern (F#687/F#698/F#699/F#700–F#708
  precedents). PASS.
- **(r) Prediction-vs-measurement table.** Present in PAPER.md, 8 rows, all
  "not measured" with per-row preempt attribution (L1/L2/L3/L1+L2). PASS.
- **(s) Math errors / unsupported claims.** L1 anchors on F#150 verbatim
  caveat ("identical to 4 decimal places" under orthogonality) — confirmed via
  `experiment finding-get 150`. L2 anchors F#166 prerequisite-gate language —
  confirmed. L3 anchors F#477 rank-6 base-beat rate 2/5 — confirmed. L4
  count is structurally verifiable (F#705 stale-caveat + F#708 paired-half-strip
  + this explicit-anti-caveat = 3 sub-variants). L5 count is structurally
  verifiable (K1552 + K1577/F#704 + K1584/F#709 = 3 instances). No unsupported
  claims. PASS.

## Verdict rationale

KILL (preempt-structural). Parent F#150 explicitly measured the same 4-variant
quality comparison as "identical to 4 decimal places" under the orthogonality
condition the pre-reg cites ("under verified Grassmannian") — PASS is
already-demonstrated-vacuous by the parent. The `< 2pp` direction does not
rescue the KC: the antipattern is direction-symmetric (`≥ δ` passes via
format-incompatibility; `< δ` passes via shared-failure-equivalence; both fail
to isolate thesis-progress). 3rd instance of the antipattern → promotion to
named §5 clause. Simultaneously 3rd sub-variant of template-regression
(child inherits parent's explicit anti-caveat structure) → watchlist graduates
to formal antipattern.

## Non-blocking notes for analyst

1. **Promote** `mem-antipattern-tautological-inter-adapter-delta-ignores-base-baseline`
   to a named clause in `reviewer.md §5`, parallel to F#669-family and
   F#666-pure standalone clauses. Clause must cover **both delta directions**
   (`≥ δ` and `< δ`) — tautology is direction-symmetric. Canonical precedents:
   K1552 (1st, `≥ 5pp`), K1577/F#704 (2nd, `≥ 5pp`), K1584/F#709 (3rd, `< 2pp`
   inverted).
2. **Promote** `mem-watchlist-f666pure-template-regression` to formal
   antipattern `mem-antipattern-template-regression` with 3 anchored sub-variants:
   (a) stale-caveat-inheritance (F#705/F#161), (b) paired-design-half-stripping
   (F#708/F#133), (c) explicit-anti-caveat-inheritance (F#709/F#150 — this).
3. Researcher pre-claim checklist extension: for every finding cited in
   pre-reg notes or `depends_on`, run `experiment finding-get` and scan
   *caveats* field for vacuous/untested/unidentifiable/trivial/cannot-distinguish
   language AND for half-KC paired designs. Match ⇒ child must re-register
   with materially different KC structure.
4. Hygiene: 2 defects (`success_criteria=[]`, `references=[]`). Below 3+
   threshold — F#702 hygiene-patch clause does not apply; taut-inter-adapter-delta
   applies independently.
5. No `experiment ref-add` — preempt-structural KILL has no mechanism
   failure to cite externally.
6. No `_impl` companion — preempt-structural KILL excludes `_impl` per
   F#669-family / F#666-pure-standalone / F#704 precedent. Unblock is
   pre-reg-external (new pre-reg per PAPER.md §"Recommended v2").
7. Drain tally: **25** (F#709 added). 86 open experiments remain at P≤2 —
   drain not yet complete.

## Assumptions (reviewer)

- `audit-2026-04-17` tag (no `-rerun` suffix) is lineage-only; does NOT
  trigger `RECOVERY_PLAN.md` fix-before-rerun (distinction from `-rerun` tag).
- "verified Grassmannian" in pre-reg notes = F#562 A-matrix construction;
  under this fixture, F#150's "identical to 4 decimal places" applies
  verbatim, making K1584 PASS-by-parent-measurement in the cited regime.
- N=5 = F#477 domain subset (math/code/medical/legal/finance) — most
  defensible reading; pre-reg is silent on the exact 5.
