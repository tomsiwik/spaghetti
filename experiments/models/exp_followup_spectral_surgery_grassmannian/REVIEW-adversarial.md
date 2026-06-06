# REVIEW-adversarial — exp_followup_spectral_surgery_grassmannian

**Verdict: KILL (preempt-structural)** — confirms researcher's pre-measurement closure.

## Adversarial checklist

**Consistency (a–d):**
- (a) `results.json["verdict"]="KILLED"` ↔ DB `status=killed` ↔ MATH/PAPER all "KILLED preempt-structural" — consistent.
- (b) `all_pass=false` aligned with KC1560 `untested` + verdict=killed.
- (c) PAPER.md verdict line "KILLED — preempt-structural" — consistent.
- (d) `is_smoke=false`, `measurements_taken=0` — preempt-structural canonical, not a smoke run.

**KC integrity (e–g):**
- (e) Fresh experiment dir (untracked); no post-claim KC mutation.
- (f) KC #1560 is structurally a proxy ("interference reduction ≥20%"); no operationalization in pre-reg makes it a behavioral target — confirms F#666-pure standalone trigger.
- (g) K-ID #1560 matches DB row, MATH.md, PAPER.md, results.json verbatim.

**Code ↔ math (h–m2):**
- (h–l) `run_experiment.py` is graceful-failure preempt stub: imports only `json`+`pathlib`, `main()` writes results.json, never raises; no composition code, no `LORA_SCALE`, no `shutil`, no hardcoded `{"pass": True}` (KCs are `"untested"` per F#669/F#666-preempt convention).
- (m) N/A — no model loaded.
- (m2) MATH.md §Pre-flight cites "Platform skills invoked: N/A (no MLX code emitted; preempt-structural)" — satisfies skill-invocation evidence under preempt-structural carve-out.

**Eval integrity (n–s):**
- (n)–(q) N/A — `measurements_taken=0`.
- (r) PAPER.md prediction-vs-measurement table present (P1/P2/P3) with all rows resolving to "preempt-KILL on KC structure / test pool architecturally irrelevant / parent supersession unconditional".
- (s) Theorems 1 & 2 (F#666-pure structural insufficiency + architecture-relevance) sound; corollary follows; parent-supersession citations (F#278/F#488/F#64) checked against DB.
- (t) F#666 target-gated kill — **carve-out applies**: preempt-structural verdict is a drain-progression closure where NO KC was measured (proxy or target). F#666 is the *governing precedent*, not a blocker.
- (u) Scope-changing fix — N/A; graceful-failure stub is canonical preempt-structural artifact, not a scope reduction.

## F#666-pure-standalone clause compliance (canonical pattern, ~31st drain-window)
1. ✓ MATH.md §1 derives KC structural insufficiency under F#666 with truth-table degeneracy; cites F#666 as governing.
2. ✓ run_experiment.py imports only `json`+`pathlib`, `main()` writes `verdict="KILLED"`, KC `"untested"`, preempt-reason in payload.
3. ✓ PAPER.md verdict line + prediction-vs-measurement table + Unblock path (target-paired KC pair + Pierre-architectural motivation, OR redirect to FroM/DO-Merging/SVC per parent LEARNINGS).
4. ✓ No `_impl` companion — unblock is pre-reg-external (new pre-reg with target-metric KC pair).

## Notable taxonomic refinements (for analyst LEARNINGS, not blocking)
- 1st spectral-surgery-followup-on-irrelevant-test-pool sub-form within F#666-pure-standalone super-family.
- 1st explicit "disease-vs-symptoms violation" as named compound failure: pre-reg addresses parent's measurement-environment symptom (test-pool vacuity), not parent's structural finding (Grassmannian inverts surgery's premise).
- Compound with parent-supersession (F#278+F#488+F#64) and architecture-irrelevance (Pierre/P1 = Grassmannian/PoLAR by construction).

## Assumptions logged
- Reading-charitable operationalization of K#1560: even most-permissive "Frobenius cross-term residual" classification is structural-proxy under F#666; less-permissive operationalizations a fortiori. No operationalization rescues it as behavioral-target.
- Pierre-architecture orthogonality treated as the deployment baseline per `mem-pierre-p1` / `mem-pierre-v5-architecture`. If future direction abandons PoLAR for unconstrained adapters, Theorem 2 weakens; current direction does not.

## Routing
DB already updated (`status=killed`); F#761 already registered. Emitting `review.killed` to advance loop to analyst.
