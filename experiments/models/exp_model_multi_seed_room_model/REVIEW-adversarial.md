# REVIEW-adversarial.md — exp_model_multi_seed_room_model (self-review, pre-reviewer)

## Scope
Preempt runner for `exp_model_multi_seed_room_model` (KILLED_PREEMPTIVE).
Verdict over-determined by **3 of 5 automated blocks** (T2 ∧ T3 ∧
T5-K-single). T1 shortfall 2/5 (below threshold 3) — automated does
not inflate; manual re-read per MATH §2 T1 and §4 A6/A7 is 5/5.
T4 pin_ratio 0.00 (reinforce-only).

## Adversarial checklist

- **(a) Verdict consistency**: `results.json` `verdict = "KILLED_PREEMPTIVE"`,
  `all_pass = false`, `is_smoke = false`, `ran = false`. PAPER.md
  verdict line matches. DB will be `Status: killed` post-complete.
- **(b) Metric direction**: blocks listed in `preempt_blocks` match
  the `block: true` entries in `preempt.*`. No silent flip.
- **(c) KC integrity**: K1711 and K1712 pre-registered in DB; both
  recorded as `false` with "target not run" rationale. No
  post-hoc edit.
- **(d) DB ↔ disk parity**: `experiment get` at claim time returns
  `Success Criteria: NONE`, `references: []` absent, INCOMPLETE
  flag. Runner re-probes DB live (T3) and logs the literal line.
- **(e) KC tautology**: none. K1711 and K1712 are untested; no
  measurement was fabricated.
- **(f) KC relaxation**: none. Thresholds recorded exactly.
- **(g) Smoke-vs-full**: `is_smoke = false`. Runner never loaded
  a model; target experiment did not run.
- **(h) Code ↔ math**: T2 arithmetic re-verified:
  3 × (900 + 600 + 5000 + 50) = 3 × 6550 = 19 650 s = 327.5 min.
  Floor: 3 × (900 + 100) = 3000 s = 50.0 min. Matches
  `conservative_min = 327.5` and `floor_min = 50.0` exactly.
- **(h2) A9 honesty**: T1 automated shortfall = 2 (below threshold
  3); manual re-read = 5/5 per MATH §2 T1 (every cooccur hit is a
  false-positive — W_combined hits are in the KILLED parent or
  Finding #571 lineage; seed-merge hits are unrelated macro
  scripts; cv-outlier hits are unrelated experiments). Runner
  does NOT inflate — verdict over-determined by T2 ∨ T3 ∨
  T5-K-single alone without T1.
- **(i) MATH alignment**: T1 probes (1)–(5) correspond to the five
  required artifacts. T2 ceiling = 120 min matches PLAN runtime
  policy. T3 DB probe regex matches the DB pretty-print literal.
  T5-K reads live DB status for parent.
- **(j) Antipattern sweep**: no composition code, no LoRA scale,
  no routing, no `shutil.copy` as new adapter, no hardcoded
  `"pass": True`, no eval-template truncation, no proxy model, no
  KC wrong-object, no N=smoke-reported-as-full. Runner is pure
  stdlib + `experiment get` shell-out.
- **(k) Pin-ratio**: `pin_ratio = 0.00`; `.audit/` directory
  absent. Reinforce-only. Did not fire; did not over-weight.
- **(l) Platform correctness**: pure Python 3 stdlib. No MLX
  imports. No model load. Runs anywhere the CLI `experiment` is
  available.
- **(m) Runner determinism**: regex + dict outputs, no sampling.
  Two runs produce identical results up to wall time.
- **(m2) Runtime honesty**: 2.33 s wall measured; PAPER.md
  "Pure stdlib runner, 2.33 s wall". Matches exactly.
- **(n) Eval hygiene**: zero real eval, no proxy substitution.
- **(o) Data hygiene**: zero data load.
- **(p) RNG hygiene**: zero random.
- **(q) Proxy substitution**: none.
- **(r) Deliverables**: MATH.md, run_experiment.py, results.json,
  PAPER.md, REVIEW-adversarial.md present. LEARNINGS.md is
  analyst-owned (HALT §C cap — still 50/50).
- **(s) Memory/platform**: N/A for runner; target would have been
  macro 48 GB MLX but never ran.

## Independent-verification paths

- **T2**: the 3-seed × 1 000-Q × 5-s arithmetic does not depend
  on any unvalidated assumption about MLX perf; it is a conservative
  upper bound on even a properly-running Gemma-4-E4B + MMLU-Pro
  pipeline. Even if W_combined perf were restored to 165 tok/s,
  the 15 000 s MMLU-Pro eval dominates.
- **T3**: DB probe is a live `experiment get` at runtime.
- **T5-K single**: parent `Status: killed` is a live DB field;
  breach evidence (K1688/K1689 FAIL lines) is read from the
  parent's on-disk `results.json` / `PAPER.md` / `MATH.md`.

## Failure modes (if we re-open)

Operator would need to:
1. Resurrect the KILLED parent `exp_model_room_model_gemma4_speed`
   (K1688 ≥ 150 tok/s, K1689 cos ≥ 0.999) — memory
   `project_room_model.md` records this as SUPERSEDED for N>1;
   operator action is not a reviewer decision.
2. Train 15 seeded Gemma 4 E4B LoRA adapters in a `seed_{0,1,2}/`
   layout (3 seeds × N=5 domains). Currently 0 such files exist.
3. Build an MMLU-Pro harness bound to the Room Model `W_combined`
   output, with identical prompt template and decode policy across
   seeds.
4. Add CV + 2σ outlier analysis on a per-seed-score tuple.
5. Register `success_criteria` and `references` in the DB record
   (current INCOMPLETE flag).

Without (1) the target is unrunnable; even with (2)–(5), K1711 is
ill-posed as a stability claim about a routine whose central claim
(K1688, K1689) is already killed.

## Self-score
- Rigor: honest shortfall (2/5 automated, 5/5 manual) with no
  inflation; verdict over-determined by 3 independent blocks.
- Novelty: 2nd single-parent T5-K; not novel as an axis.
- Risk: low — zero-side-effect runner.

Routes review.killed.

---

## Reviewer ratification (iter 37, 2026-04-19)

**Verdict: KILL (ratify). No revise cycle.**

Adversarial checklist (a)–(s) independently re-verified:

- (a)–(d) Verdict chain consistent: `results.json` `verdict =
  KILLED_PREEMPTIVE`, `all_pass = false`, `is_smoke = false`,
  `ran = false` ↔ PAPER.md verdict line ↔ live DB `Status: killed`
  (K1711 ✗, K1712 ✗).
- (e)–(g) KC integrity: K1711 / K1712 pre-registered; both `false`
  with "target not run" annotation. No tautology, no silent relax.
- (h) T2 arithmetic re-derived independently: 3 × (900 + 600 +
  1000·5 + 50) = 3 × 6 550 = 19 650 s = **327.5 min** exactly,
  floor 3 × (900 + 100) = 3 000 s = **50.0 min** exactly. Matches
  `results.json` literals.
- (h)–(m2) Code ↔ math: runner is pure stdlib + `subprocess`
  shell-out to `experiment get`. Zero MLX, zero composition code,
  zero LoRA scale, zero routing, zero `shutil.copy`, zero eval,
  zero model load. A9 honesty verified: T1 automated shortfall 2/5
  below threshold 3; manual re-read 5/5 valid — every cooccur hit
  is a false-positive (W_combined in KILLED parent or F#571
  lineage; seed-merge in unrelated macro scripts; cv-outlier in
  unrelated experiments). Runner does NOT inflate — verdict
  over-determined by T2 ∨ T3 ∨ T5-K-single alone.
- (n)–(q) Eval hygiene: zero real eval, no proxy substitution, no
  synthetic padding, zero RNG.
- (r)–(s) Deliverables: MATH.md, run_experiment.py, results.json,
  PAPER.md, REVIEW-adversarial.md present. LEARNINGS.md
  analyst-owned (HALT §C cap still 50/50). Runtime 2.33 s.

**T5-K single-parent breach independently verified:**
- Parent `exp_model_room_model_gemma4_speed` live DB `Status:
  killed` (K1688 FAIL 69 tok/s vs 150; K1689 FAIL cos 0.9941 vs
  0.999; K1690 PASS N=1-only). All 5 breach dimensions true:
  (A) speed, (B) quality, (C) F#571 superseded memory breach,
  (D) K1690 N=1 scope breach, (E) KC-target definitional coupling.

**Finding: F#660** — "ap-017 (s4) T5-K single-parent-KILLED,
2nd instance (multi_seed_room_model)". Reviewer read: sibling of
iter 36 `exp_model_loader_portability` instance; both share
single-parent-KILLED structure but distinct parents. Analyst owns
sibling-vs-child placement under F#651 when cap lifts.

**Cohort progress: 41 preemptive-kills** under ap-017. Branches:
- composition-bug: 28 (software-infra-unbuilt lineage unchanged)
- scale-safety: 2, tautological-routing: 3, projection-scope: 2,
  tautological-duplicate: 1, hardware-topology-unavailable: 2,
  (s3) platform-library-absent: 1
- (s4) T5-K single-parent-KILLED: **2** (iter 36 loader_portability
  + iter 45 multi_seed_room_model, F#660 new)
- (s4.2) T5-K double-parent-KILLED: 1 (iter 44)
- (d1) private-data-unobtainable-by-design: 1 (iter 43)

**Non-blocking runner-refinement backlog** (when cap lifts):
- T1 `W_combined` probe: exclude KILLED parent dirs via live DB
  status cross-check (currently false-positives on killed code).
- T1 `seed_merge` probe: require cooccur of `seed=\{0,1,2\}` with
  per-seed adapter-path construction (not dataloader `seed=`).
- T1 CV-outlier probe: require cooccur with MMLU-Pro or Room
  Model specifically.

**Route:** emit `review.killed` → ralph coordinator iter 18 /
researcher iter 46 next. Analyst iter 33 still capped.
