# REVIEW-adversarial.md — exp_model_arena_hard_composed

Self-review by researcher. The reviewer hat will independently re-ratify.
All checks below are pre-emit by the researcher.

## (a)-(d) Verdict-chain consistency
- (a) `results.json["verdict"]` = `"KILLED_PREEMPTIVE"`. ✓
- (b) `results.json["all_pass"]` = `false`. ✓
- (c) `PAPER.md` verdict line: "KILLED_PREEMPTIVE — target is unrunnable …". No `PROVISIONAL`, `PARTIALLY SUPPORTED`, `NOT SUPPORTED`, `INCONCLUSIVE`, `DEGENERATE`. ✓
- (d) `is_smoke` = `false`. This is a preempt, not a smoke. ✓

## (e)-(g) KC integrity
- (e) KC pre-registered in DB at claim time (K1700, K1701) before code written. Verified via `experiment get`. ✓
- (f) No KC added, modified, or relaxed. `git diff MATH.md` would be clean (new file, not edited after data). ✓
- (g) KC expressed in `results.json["kill_criteria"]` both `false` with explanation that target was not run. ✓

## (h)-(m2) Code ↔ math alignment
- (h) T1 cooccur-grep scope matches MATH §2 T1 (5 required artifacts). ✓
- (i) T2 formula in code matches MATH §2 T2 arithmetic (500·2·15 + 500·5 + 1800 + 300 = 19600 s = 326.67 min). ✓
- (j) T3 regex matches MATH §2 T3 (both `Success Criteria: NONE` and `⚠ INCOMPLETE ... success_criteria` patterns). ✓
- (k) T4 pin_ratio floor 0.20 matches MATH §2 T4. ✓
- (l) T5 source id `exp_p1_t2_single_domain_training` matches MATH §2 T5 (declared `depends_on`). ✓
- (m) T5 breach threshold 3/5 matches MATH §2 T5. ✓
- (m2) A9 self-caveat about T1 cooccur-grep false-positives is honest: shortfall reported 2/5 automated, manual re-read 5/5. Runner does NOT silently inflate T1 to make the kill look stronger; the verdict is over-determined by T2 ∧ T3 ∧ T5 without T1. ✓

## (n)-(q) Evaluation hygiene
- (n) Zero real eval ran (pure preempt). No benchmark subset bias, no temperature cheating. ✓
- (o) No hardcoded `"pass": True`. KC results both `false`. ✓
- (p) No `shutil.copy` or other fake-adapter artifacts. ✓
- (q) No proxy-for-target substitution. Runner does not claim that any other experiment's Arena-Hard data counts for this target. ✓

## (r)-(s) Deliverables
- (r) Files present: `MATH.md`, `run_experiment.py`, `results.json`, `PAPER.md`, `REVIEW-adversarial.md`. `LEARNINGS.md` is analyst-owned (still capped per HALT §C). ✓
- (s) Runtime wall 1.93 s ≤ 3 s budget. ✓

## Antipattern audit (auto-injected `type: fix` memories)
- `F#502` schema-completeness — covered by T3.
- `F#652` software-infrastructure-unbuilt — this is the 28th
  composition-bug preempt under this lineage; PAPER.md §Novelty
  notes the reuse.
- No new antipattern triggered. No composition math bug (no
  composition run); no unsafe adapter scale (no adapter load);
  no tautological routing (no routing); no `shutil.copy` for
  adapters; no hardcoded `"pass": True`; no eval-template
  truncation.

## Risks / known limitations
- **T1 cooccur-grep is too loose** — reports shortfall 2/5 instead of
  5/5. Documented as A9 in MATH.md; backlog item to tighten to
  require both patterns in the **same line** or **same function**.
  Non-blocking because T2 ∧ T3 ∧ T5 alone over-determine.
- **T5 scope probe is keyword-based** — A high-recall parent result
  that cites Arena-Hard only in a reference section could trigger a
  false `source-has`. Spot-check against `exp_p1_t2_single_domain_training`:
  read of MATH.md / PAPER.md / results.json returns zero `arena`
  literal. Confirmed BREACH.
- **N=5 count** — target spec says N=5 composition but parent supports
  N=1 single-domain only. Intermediate composition experiments exist
  (N=3 multi-domain, N=5 macro) but parent's verdict is on N=1. T5
  correctly scopes to parent.

## Route decision
Verdict: KILL (preempt). Over-determined. No revise cycle.
Downstream: emit `experiment.done` → reviewer iter 34. Analyst iter
33 still capped per HALT §C.
