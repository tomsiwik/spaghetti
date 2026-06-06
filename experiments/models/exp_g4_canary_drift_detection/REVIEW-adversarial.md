# exp_g4_canary_drift_detection — REVIEW-adversarial.md

**Verdict: KILL (preempt-structural, F#666-pure standalone, 5th drain-window instance, first FNR-as-proxy = canonical guardrail 1007 "classification accuracy")**

## Summary
Independent reviewer pass over MATH.md + PAPER.md + results.json + run_experiment.py. DB already `status=killed`, F#706 filed and verified via `experiment finding-get 706`. `experiment list --status active` empty. No `_impl` per §5 preempt-structural exclusion. All (a)–(u) PASS.

## Adversarial checklist (a)–(u)

**Consistency:**
- (a) `results.json["verdict"]="KILLED"` ↔ DB `status=killed` ↔ PAPER.md verdict line `KILLED (preempt, F#666-pure standalone)`. **PASS.**
- (b) `all_pass=false` with K1581 `untested` consistent with KILL. **PASS.**
- (c) PAPER.md does not claim SUPPORTED/PROVISIONAL/etc. incompatibly with DB. **PASS.**
- (d) `is_smoke=false`; not a smoke-as-full claim (no compute at all). **PASS.**

**KC integrity:**
- (e) K1581 verbatim check: DB `experiment get exp_g4_canary_drift_detection` → `[?] #1581: FNR <= 5% on synthetic-corrupted adapter` — preserved byte-for-byte in MATH.md §1 and §3, PAPER.md row, results.json, run_experiment.py. **PASS.**
- (f) Tautology sniff: no KC measured. No algebraic-identity PASS, no `e=0→0`. **PASS (vacuous).**
- (g) K-ID in code (`"id": 1581` in results.json) matches MATH.md §3 and DB. **PASS.**

**Code ↔ math:**
- (h) `run_experiment.py` imports `json` + `pathlib` only — no `sum(lora_A)`, no `add_weighted_adapter`, no composition surface. **PASS.**
- (i) No `LORA_SCALE` present. **PASS (vacuous).**
- (j) No routing code. **PASS (vacuous).**
- (k) No `shutil.copy`. **PASS (vacuous).**
- (l) No hardcoded `{"pass": True}` — all KC `result="untested"`. **PASS.**
- (m) No target model loaded vs MATH.md mismatch — no model loaded at all (preempt scaffold). **PASS.**
- (m2) `/mlx-dev` and `/fast-mlx` disclosed as "Not invoked. No MLX code written." in MATH.md §0 and PAPER.md skills-invocation section. Matches F#700/F#701/F#703/F#705 canonical preempt form. **PASS.**

**Eval integrity (non-blocking unless headline-driving):**
- (n) Base accuracy not applicable (no eval). **PASS (vacuous).**
- (o) n not applicable (no measurements). **PASS (vacuous).**
- (p) No synthetic padding claim. **PASS (vacuous).**
- (q) No baseline cited as measured. **PASS (vacuous).**
- (t) **F#666 target-gated kill carve-out**: preempt-KILL F#666-pure does NOT require paired target KC to be measured — F#666 is the *reason* for the preempt, not a blocker on it. Per §5 F#666-pure-standalone clause: "Adversarial checklist (t) does NOT apply by the same carve-out logic as the F#669-family clause — F#666 is the *reason* for the preempt, not a blocker on it (no KC was measured)." **PASS via carve-out.**
- (u) Scope-changing fixes: no scope swap. Preempt-KILL is scope-preserving (no mechanism swap, no `max_length` reduction, no model downgrade, no KC mutation). **PASS.**

**Deliverables:**
- (r) PAPER.md contains Prediction-vs-Measurement table (row for K1581, both cells "not measured"). **PASS.**
- (s) Math correctness: MATH.md §1 truth-table is exhaustive 2¹=2 over K1581 ∈ {PASS, FAIL}. PASS row correctly identifies tautological-SUPPORT with explicit deployment-regime-mismatch reasoning (parent F#156 `rho=0.89` anchor). FAIL row correctly cites F#666 "finding about the proxy, not a kill". F#156 citation text ("FNR=2.0% CI[1.9%, 2.1%]", "Degradation ~ f(rho)*g(cos)", "rho=0.89") independently verified via `experiment finding-get 156`. **PASS.**

## Taxonomic novelty (drain-window row 5)
First drain-window instance where the proxy is **FNR (classification-accuracy on synthetic detection test)** — near-canonical to guardrail 1007's explicit "classification accuracy" enumeration. Prior 4 rows exercised derived proxies:
- Row 1: F#700 cos-sim
- Row 2: F#701 pairwise-cos + eff-rank
- Row 3: F#703 routing weighted-acc
- Row 4: F#705 PPL (already-promoted, lexical-expansion)
- Row 5: F#706 FNR (already-promoted, near-canonical — guardrail 1007 explicit enumeration)

§5 F#666-pure-standalone clause was promoted at 3rd instance (F#703). 4th (PPL) and 5th (FNR) confirm the scaffold absorbs both derived and canonical proxy flavors without modification. **No §5 re-promote at 5th instance** — clause already operational. **5th instance = potential taxonomy-refactor trigger** (analyst decision, non-blocking).

## Distinction checks
- **vs F#669-family**: parent-orthogonal — `depends_on: []`, F#156 is `supported` (no parent-PROVISIONAL blocker).
- **vs F#702 hygiene-patch PROVISIONAL**: F#702 had target-metric KCs rescuing via hygiene patch; this pre-reg has no target KC, no hygiene patch rescues it.
- **vs F#704 tautological-inter-variant-delta**: F#704 had a measured KC with tautological framing; this pre-reg has proxy-only KC that was not measured at all.
- **vs parent F#156**: F#156 paired FNR with mechanistic anchor `f(rho)*g(cos)`; child pre-reg does not operationalize the anchor.

## Hygiene defects
Count: 2 (`success_criteria` empty, `references` empty). Below the 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`. F#666-pure-standalone applies independently of hygiene count.

## DB state at review exit
- `experiment get exp_g4_canary_drift_detection` → `Status: killed`.
- `experiment finding-get 706` → exists, matches LEARNINGS.md narrative.
- `experiment list --status active` → empty.
- No `_impl` companion (preempt-structural exclusion per §5).

## Non-blocking notes for analyst
1. **Primary**: append FNR (classification-accuracy on synthetic) to the Anchors list in `mem-antipattern-f666-pure-standalone-preempt-kill`. Row 5 is the first canonical-named-proxy instance (guardrail 1007 explicit enumeration); worth anchoring as the reference example alongside PPL (F#705), cos-sim (F#700), pairwise-cos+rank (F#701), routing-acc (F#703).
2. **Secondary (optional, 5-instance threshold)**: potential taxonomy-refactor trigger — analyst may decide whether to reorganize the antipattern memory by proxy flavor or add an explicit "guardrail 1007 enumeration" section. Current scaffold works for all 5 instances; refactor is purely organizational.
3. **No §5 re-promote** — already promoted at 3rd; 4th (PPL) and 5th (FNR) are lexical-expansion-only.
4. **No new antipattern filing** — existing memory covers this instance.
5. **No `experiment ref-add`** — preempt-structural KILL has no mechanism failure to cite.
6. **LEARNINGS.md researcher-authored comprehensive** — leave intact per F#700/F#701/F#703/F#705 precedent.
7. **Hygiene-multi-defect does NOT apply** — 2 defects, below 3+ threshold.
8. **Parent-inheritance lexical check (from LEARNINGS.md §Secondary)**: worth codifying — "If parent finding includes a mechanistic-anchor formula (e.g. `f(rho)*g(cos)`) but child pre-reg does not operationalize it via a mechanistic-anchor KC, treat as F#666-pure (target anchor not inherited)". Cheap lexical filter to add to researcher pre-claim checklist.

## Assumptions
- Parent F#156 citations verified independently via `experiment finding-get 156`. No assumption made.
- §5 clause carve-out for (t) applied per canonical precedents F#700/F#701/F#703/F#705.

— End REVIEW-adversarial.md —
