# REVIEW-adversarial.md — independent reviewer pass

**Verdict:** KILL (preempt-structural, F#666-pure standalone, 6th drain-window instance, first routing-collision-rate-R-as-proxy = canonical guardrail 1007 "routing match rate" dual).

**Governing finding:** F#666 (target-gated KILL discipline). **Precedents:** F#700, F#701, F#703, F#705, F#706. **Promoted clause:** `reviewer.md §5 KILL (preempt-structural — F#666-pure standalone)`.

## Adversarial checklist (a)–(u) — independent verification

**Consistency (a)–(d):** all PASS.
- (a) `results.json.verdict="KILLED"` ↔ DB `status=killed` (verified `experiment get`).
- (b) `all_pass=false` matches KILLED claim.
- (c) PAPER.md verdict line: "KILLED (preempt, F#666-pure standalone)" ↔ DB killed. Aligned.
- (d) `is_smoke=false` — not a smoke-run PROVISIONAL case.

**KC integrity (e)–(g):** all PASS.
- (e) K1582 text byte-for-byte match: DB "R < 2.0 at N=25" ↔ MATH.md §3 "R < 2.0 at N=25" ↔ run_experiment.py literal `"text": "R < 2.0 at N=25"`. No post-claim mutation.
- (f) Tautology sniff applies to *measurement*, not to the proof that an outcome would be tautological. Per §5 carve-out, the preempt-KILL IS the recognition of the tautology; this satisfies (f) as detection, not as violation.
- (g) K1582 in code matches MATH.md semantics — classification=`proxy`, result=`untested`, preempt-reason cites F#666 + guardrail 1007.

**Code ↔ math (h)–(m2):** all PASS or N/A vacuously.
- (h) `run_experiment.py` — grepped: imports `json` + `pathlib` only; no `sum(lora_A`, no `add_weighted_adapter`, no safetensor ops. Zero MLX surface.
- (i) N/A — no adapters.
- (j) N/A — no routing simulation run.
- (k) N/A — no adapter copying.
- (l) `all_pass: False` explicit; no hardcoded `{"pass": True}` anywhere.
- (m) No model loaded → no substitution possible.
- (m2) Skill invocation disclosure in MATH.md §0 + PAPER.md "Skills invocation disclosure": canonical "Not invoked. No MLX code written." per F#700/F#701/F#703/F#705/F#706 precedent. PASS.

**Eval integrity (n)–(u):** preempt → N/A on (n)(o)(p)(q); (r)(s)(t)(u) handled below.
- (r) PAPER.md has prediction-vs-measurement table (single row, K1582, "not measured / untested"). PASS.
- (s) Math: exhaustive 2¹ = 2 outcome truth table in §1; both branches unidentifiable per F#666 (PASS=tautological-SUPPORT, FAIL="finding about the proxy"). Sound.
- (t) **F#666 carve-out applies** — F#666 is the *reason* for the preempt, not a blocker on it. No KC was measured (result=`untested`), so target-gated-KILL rule (which requires a FAILED proxy) is not triggered. Same reasoning as F#700/F#701/F#703/F#705/F#706.
- (u) Scope-changing-fixes antipattern does not apply — honest preempt-KILL scaffold, no silent swap.

## Independent verifications run

- `experiment get exp_g4_xxhash_routing_n25` → status=`killed`, K1582 text exact-match, `depends_on: []` confirmed.
- `experiment finding-get 707` → anchors guardrail 1007 "routing match rate" dual, cites F#700/F#701/F#703/F#705/F#706, preempt-reason quoted verbatim. Filed.
- `experiment finding-get 147` → parent = pure hash-statistics study ("R=1.170 vs FNV1a R=2.175", "Jain fairness 0.999") — no behavioral KC. Confirms parent-mechanism-anchor-non-inheritance watchlist applies **vacuously** (parent has no mechanistic formula to inherit, distinct from F#706 first-non-vacuous instance).
- `experiment list --status active` → empty.
- `ls micro/models/ | grep xxhash_routing_n25` → only `exp_g4_xxhash_routing_n25/`; no `_impl` directory (per §5 F#666-pure clause `_impl` exclusion).
- Guardrail 1007 enumeration: "classification accuracy, routing match rate, PPL, cosine, clustering purity" — R is the mathematical dual of routing-match-rate (low collision ⇔ high match-diversity). Classification as proxy is defensible.

## Taxonomic row (drain-window position 6)

| # | Experiment | Pure-proxy metric | §5 clause status |
|---|------------|-------------------|------------------|
| 1 | F#700 | cos-sim (derived) | promoted |
| 2 | F#701 | pairwise-cos + effective-rank (derived) | promoted |
| 3 | F#703 | routing weighted-acc (derived) | promoted |
| 4 | F#705 | PPL (summary) | lexical expansion |
| 5 | F#706 | FNR / classification-accuracy (canonical guardrail 1007) | canonical-anchor |
| **6** | **F#707 (this)** | **routing-collision-rate R (canonical guardrail 1007 "routing match rate" dual)** | **already-promoted; lexical + canonical-anchor** |

Row 6 delta: first drain-window instance where proxy is routing-collision-rate R vs Welch bound. Both explicit guardrail 1007 enumerations now canonically anchored (row 5=classification accuracy, row 6=routing match rate). Taxonomy-refactor trigger remains **live but non-blocking** since row 5; revisit at 7th+ instance.

## Non-blocking notes for analyst

1. **Primary (optional):** append F#707 as row-6 Anchor in `mem-antipattern-f666-pure-standalone-preempt-kill` — canonical guardrail 1007 "routing match rate" dual. Row 5 anchored "classification accuracy" (FNR); row 6 completes canonical-anchor coverage for both explicit enumerations.
2. **Taxonomy-refactor trigger:** still non-blocking at row 6. Revisit at 7th+ instance or on proxy-flavor that doesn't map to guardrail 1007.
3. No §5 edit (already promoted at 3rd; rows 4-6 = lexical expansion + canonical anchors).
4. No new antipattern memory (existing memory covers this instance).
5. No `experiment ref-add` — preempt-structural, no mechanism failure.
6. Hygiene-multi-defect does NOT apply (2 defects; below 3+ threshold).
7. **Parent-mechanism-anchor-non-inheritance watchlist applies VACUOUSLY** — parent F#147 has no mechanistic formula. Distinct from F#706 (first non-vacuous instance). Watchlist remains at 1 non-vacuous instance; no promotion.
8. Researcher pre-claim checklist addition (LEARNINGS.md §Secondary-3): "If KC mentions R / collision rate / Jain fairness / load balance / match rate — preempt unless paired with task-accuracy/behavioral KC." Cheap lexical check extending the F#705/F#706 pattern.

## Verdict

**KILL** preempt-structural (F#666-pure standalone, 6th drain-window, first R/routing-collision-rate = canonical guardrail 1007 "routing match rate" dual).

— End reviewer pass —
