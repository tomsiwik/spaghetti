# REVIEW-adversarial.md — P3.C4: Rank-16 Diverse Adapter

## Reviewer sign-off (2026-04-18, V2 audit)

Adversarial checklist (a)–(s) clean; DB status `killed` matches `results.json["summary"]["verdict"]="KILLED"` and PAPER.md headline. KC integrity verified (no KC edits vs 2026-04-11 pre-reg). Cache-bug fix in `run_experiment.py::generate_diverse_training_data` is line-count validated (`n_existing >= N_TRAIN`). Rerun blocked is acknowledged — does not alter strict-KC verdict on documented 2026-04-11 run. Proceeding to Analyst.

## V2 Audit Review (2026-04-18)

**Verdict: KILLED confirmed. Rerun not executable.**

Checked under strict PLAN.md §1:
1. K1205 pre-reg threshold ≥80%; measured 73.3% → FAIL (unambiguous, n=15).
2. `results.json["summary"]["verdict"] = "KILLED"` and `all_pass = false` — written this iteration.
3. PAPER.md verdict line says "KILLED" (no silent upgrade).
4. `is_smoke = false` (full config; cache bug caused 10 effective train examples but
   run metadata recorded `is_smoke=false`).
5. KC not modified (git diff of MATH.md kill criteria = empty since 2026-04-11).
6. Antipattern scan:
   - `training-cache` (flagged tag): confounds data-vs-rank attribution. **Does NOT rescue
     verdict** — 73.3% < 80% regardless of cause. Cache fix applied to `run_experiment.py`.
   - `smoke_as_full`: `n_train_actual=10` vs `n_train_configured=200` annotated in
     results.json; is_smoke flag honest.
   - No composition-math bug, no unsafe adapter scale, no tautological routing, no
     `shutil.copy`-as-adapter, no hardcoded pass, no eval-template truncation.

**Rerun block:** `domain_fused_base/model-000X-of-00004.safetensors` (~15 GB) deleted;
source math adapter also deleted (same pattern as the prior `exp_p2_a1` V2 audit). Rebuilding
the fused base requires retraining the math adapter + B5 fusion (out of 2 h iteration budget).

**Verdict stability:** even a hypothetical rerun with 167 examples cannot change the prior
KILLED finding — that run already measured 73.3% against a pre-registered 80% threshold
and failed. A new run with full data would be a **new experiment** (P3.C5), not a re-do
of P3.C4's pre-registered test.

---

## Original Review (2026-04-11)

**Verdict: PROCEED (KILLED, Finding #471)**

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match evidence (K1205 FAIL: 73.3% < 80%)
- [x] Finding status appropriate (KILLED = kill criteria failed)
- [x] experiment complete + finding-add already run (previous iteration)
- [x] MATH.md present

## Adversarial Concerns

### Non-blocking: Cache Bug Confound
The training data cache check validated file existence but not line count, resulting in
10 examples being used instead of 167. This confounds data vs rank attribution.

**However**: the direction of the confound actually strengthens the rank hypothesis.
Rank-16 + 10 examples (73.3%) beats rank-4 + 167 examples (60%). If rank were not
the bottleneck, we'd expect the opposite. The cache bug makes Theorem 1 a conservative
test — P3.C5 with correct data should only improve.

### Non-blocking: Theorem 1 Coverage Argument
Theorem 1 predicts rank(16) > n_categories(10) → coverage → 80%+. The measured 73.3%
misses threshold. Three candidate explanations given:
1. Data shortage (10 vs 167) — resolved by P3.C5
2. Question-type floor (within-category variation beyond rank)
3. Category count undercounts required style directions

The PAPER.md correctly identifies these are not distinguishable from this experiment
alone. P3.C5 resolves explanation 1; if C5 still fails at ~73-75%, then 2 or 3 is primary.

### Non-blocking: "Hard floor" Characterization
The 4 failures are all in COVERED categories (physics, CS, earth science), not
underrepresented ones. Philosophy and economics questions PASSED. This is correctly
flagged as surprising — if rank/data were the only constraint, covered categories
should be easiest. The token probability floor for certain question formulations
is real and P3.C5 should test whether 167 diverse examples pushes through it.

## Summary

PAPER.md is rigorous, analysis is honest about the confound, prediction table is
complete, impossibility structure is appropriate. KILLED is correct — K1205 FAIL is
unambiguous. P3.C5 (rank-16 + fix cache + 167 examples) is the natural next step.

**Verdict: PROCEED → Analyst writes LEARNINGS.md**
