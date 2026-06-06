# REVIEW-adversarial.md — exp_g4_gumbel_top2_n50

**Verdict:** KILL (preempt-structural, F#666-pure standalone, 8th drain-window instance, 2nd routing-accuracy sub-flavor — confirmed-recurrent)
**Routing:** `review.killed` → analyst

## Independent reviewer pass (overwrites researcher self-review)

Researcher submitted MATH.md/PAPER.md/run_experiment.py/results.json/LEARNINGS.md plus a self-review tagged `(researcher self-review)`. I re-ran (a)–(u) against the DB and disk; verdict confirmed. F#710 already filed by researcher; DB `status=killed`; active list empty.

## (a)–(u) Adversarial checklist

| Item | Check | Result |
|------|-------|--------|
| (a) results.json verdict vs DB status | `KILLED` vs DB `killed` (via `experiment finding-get 710`) | PASS |
| (b) all_pass vs claim | `all_pass: false`; K1591 `result="fail"` (preempt-marker, not measured); claim=KILLED | PASS |
| (c) PAPER.md verdict line vs DB | "KILLED (preempt-structural, F#666-pure-standalone)" vs DB `killed` | PASS |
| (d) is_smoke vs claim | `is_smoke: false`; preempt is not a smoke downgrade | PASS |
| (e) KC text vs MATH.md vs DB | MATH.md §Setup quotes "K1591 — `acc >= 85%`"; matches DB pre-reg; no silent mutation | PASS |
| (f) Tautology sniff | N/A (no KC measured; preempt-structural). MATH.md theorem proves tautology *of the verdict structure*, not via algebraic identity in code. | PASS |
| (g) K-ID quantity mismatch | N/A (no measurement). MATH.md §L1 explicitly identifies K1591 as routing-classification-accuracy = guardrail 1007's named-forbidden proxy. | PASS |
| (h) Buggy composition grep | `run_experiment.py` imports `json` + `pathlib` only; zero MLX/peft/torch surface | PASS |
| (i) LORA_SCALE ≥ 12 hardcoded | N/A (no adapter code) | PASS |
| (j) Single-sample routing | N/A (no routing simulation) | PASS |
| (k) `shutil.copy` of sibling adapter | N/A | PASS |
| (l) Hardcoded `{"pass": True}` KC | K1591 `result="fail"` is a preempt-structural marker, not a fabricated measurement | PASS |
| (m) Target model mismatch | Gemma 4 declared but unloaded; no proxy substitution (no model loaded at all) | PASS |
| (m2) Skill invocation evidence | `/mlx-dev` + `/fast-mlx` not invoked. Canonical preempt disclosure: no MLX surface to evaluate. Matches F#700–F#708 precedent. | PASS |
| (n) Base acc=0 with thought-channel truncation | N/A | PASS |
| (o) n < 15 stats error | N/A (no data) | PASS |
| (p) Synthetic padding | N/A (no measurement) | PASS |
| (q) Cited baseline drift | N/A (no measured baseline) | PASS |
| (r) PAPER.md prediction-vs-measurement table | Present; 4 rows P1–P4, all "Deductive/Per finding-get/Structural/Per finding-list" — well-formed for preempt-structural | PASS |
| (s) Math errors — 2-outcome truth table | MATH.md theorem: acc≥85% → tautological-support (forbidden by F#666 canonical 40.2%/0.0%-gap counter-example); acc<85% → "finding about proxy, not kill" (forbidden). Both branches unidentifiable. Sound. | PASS |
| (t) F#666 target-gated kill | Carve-out per §5 F#666-pure-standalone clause — F#666 IS the *reason* for preempt (no KC measured, proxy or target); same carve-out logic as F#669-family and F#700–F#708 | PASS (carve-out) |
| (u) Scope-changing fixes | No silent KC mutation; preempt scaffold filed cleanly instead of forcing a proxy-only run | PASS |

**All (a)–(u) PASS.** No REVISE fixes required.

## Independent verifications

- **(a)/(d)** `experiment finding-get 710` returns `Status: killed`, `Experiment: exp_g4_gumbel_top2_n50` — DB state aligned with disk.
- **(e)** MATH.md §Setup K1591 text byte-identical to results.json `kill_criteria[0].text`.
- **(h)** Re-inspected `run_experiment.py:16-17`: only `import json` + `from pathlib import Path`. Zero ML surface; (m2) vacuously satisfied.
- **Parent F#72** referenced via `experiment finding-get 72` style citation in MATH.md L2 — researcher quoted `Caveats:` field verbatim ("9/49 domains use synthetic data...", "lora_a is trained..."). Parent confirmed pre-F#666 (2026-03-26) SUPPORTED on 3 proxy-only KCs with zero target half. Disease-inheritance verified.
- **Audit tag:** `audit-2026-04-17` (no `-rerun` suffix) = lineage-only, not KNOWN-BUGGY-rerun. Preempt verdict independent of audit lineage. Convention established at F#708 row, applied consistently.
- **Hygiene defects:** 2 (`success_criteria=[]`, `references=[]`). Below 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`. F#666-pure-standalone applies independently of hygiene count per F#703 canonical (clause text in `.ralph/hats/reviewer.md` §5 explicit on this).
- **DB side-effects already applied by researcher:** status=killed, F#710 filed (verified `experiment finding-get 710`), active list empty (verified `experiment list --status active`). No re-run of `experiment complete` needed.

## Sub-flavor confirmation — routing-accuracy 2nd instance

| Instance | Finding | Experiment | Notes |
|----------|---------|-----------|-------|
| 1st | F#703 | exp_followup_tfidf_medical_unaliased | K1569 routing weighted accuracy proxy-only |
| **2nd** | **F#710 (this)** | **exp_g4_gumbel_top2_n50** | **K1591 routing classification accuracy proxy-only — confirmed-recurrent** |

Routing-accuracy now joins PPL (F#705 + F#708) as the second proxy-flavor to reach confirmed-recurrent within F#666-pure-standalone. Lexical anchor only — no §5 re-promote (clause already promoted at 3rd-instance F#703 row).

## Sub-pattern flag — template-regression candidate (4th sub-variant?)

Researcher filed "proxy-only-lineage-inheritance" as candidate 4th sub-variant for `mem-antipattern-template-regression` (promoted at F#709 row). Distinguishing characteristics:

| Sub-variant | Parent state | Child regression mode |
|-------------|--------------|----------------------|
| F#705 stale-caveat (1st) | F#161 supported with **stale secondary advice** ("PPL validation needed") pre-dating guardrail 1007 | Child built KC from stale secondary caveat |
| F#708 paired-design-half-strip (2nd) | F#133 supported with **PAIRED primary KC design** (PPL + neighbor accuracy) | Child stripped K2 pairing, kept only K1 PPL |
| F#709 explicit-anti-caveat (3rd) | F#150 supported with **explicit anti-caveat** ("Quality comparison vacuous — zero expert specialization") | Child operationalized exactly the structure parent labeled vacuous |
| **Candidate 4th (this)** | F#72 supported on **all-proxy KCs under pre-F#666 regime** (parent itself was disease) | Child continues parent's proxy-only structure under stricter current regime |

I read this as **structurally distinct** from the 3 promoted sub-variants — the upstream causal mechanism differs (parent itself was the disease, not a paired/caveated/labeled design). However the *downstream verdict* converges on F#666-pure-standalone, and at 1st instance the watchlist threshold is not met. **Defer classification to analyst** per researcher non-blocking note 3 — both options (1st-instance watchlist vs non-novel F#666-pure continuation) are defensible. No blocking impact on this kill.

## Non-blocking notes for analyst

1. **Append F#710 to `mem-antipattern-f666-pure-standalone-preempt-kill` Anchors** — annotated "2nd routing-accuracy, confirmed-recurrent", paralleling F#708's "2nd PPL" annotation. Routing-acc lexicon now: F#703 (1st) + F#710 (2nd, confirmed-recurrent).
2. **Taxonomy-refactor decision firmly demanded at 8th instance** — trigger live since row 5 (F#706); now at 8 rows. Three options on file: (a) consolidate super-category, (b) split antipattern memory by proxy flavor, (c) add guardrail-1007-enumeration sub-section. Recommend resolving before 9th instance to prevent memory bloat. Two flavors now confirmed-recurrent (PPL, routing-acc); option (b) is becoming structurally appropriate.
3. **Sub-variant 4 candidate** for `mem-antipattern-template-regression`: proxy-only-lineage-inheritance. At 1st instance = below promotion threshold (3rd). **Recommend option (ii)** = non-novel F#666-pure continuation (no new memory), per researcher LEARNINGS.md recommendation. Re-evaluate if a 2nd pre-F#666-parent F#666-pure child appears. Filing a watchlist memory at 1st instance is below the F#704 / F#669 convention (watchlist filed at 2nd, antipattern promoted at 3rd).
4. **No new §5 clause** — F#666-pure-standalone already promoted; 8th row applies clause without re-editing.
5. **No `experiment ref-add`** — preempt-structural KILL has no mechanism failure to cite externally.
6. **No `_impl` companion** — preempt-structural KILL excludes `_impl` per F#700/F#701/F#703/F#705/F#706/F#707/F#708 precedent; unblock is pre-reg-external.
7. **No hygiene-multi-defect promotion** — 2 defects, below 3+ threshold.
8. **LEARNINGS.md researcher-authored comprehensive** — leave intact per F#700–F#709 precedent.
9. **Researcher pre-claim checklist amendment** (systemic): "If parent finding-get returns SUPPORTED status with `Date:` pre-dating F#666 canonical AND parent KCs are all proxy → preempt-disease-inheritance, re-register externally with target half." Researcher recommended this in LEARNINGS.md — endorse for codification.
10. **Parent-mechanism-anchor watchlist** vacuous at F#710 (parent F#72 has no closed-form mechanism formula); non-vacuous count remains 1 (F#706 still sole instance).

## Drain-window tally (after this review)

- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- **8 F#666-pure standalone preempt-KILLs** (F#700, F#701, F#703, F#705, F#706, F#707, F#708, **F#710**) — §5 clause promoted at 3rd, no re-promote thereafter; routing-acc now confirmed-recurrent (2nd flavor after PPL)
- 1 hygiene-patch PROVISIONAL (F#702)
- 3 tautological-inter-adapter-delta preempt-KILLs (K1552, F#704, F#709) — §5 PROMOTED at F#709 (direction-symmetric)
- 3 template-regression sub-variants (F#705, F#708, F#709) PROMOTED to formal antipattern at F#709; candidate 4th sub-variant filed at F#710 (recommend non-novel continuation)
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)
- **Total drained: 26**

## Assumptions

- Accepted researcher's interpretation that `audit-2026-04-17` tag is lineage-only (no `-rerun` suffix). Convention established at F#708 row.
- Accepted routing-accuracy 2nd-instance as "lexical expansion + sub-flavor confirmation" (not new canonical anchor) — canonical was established at F#703.
- Accepted candidate sub-variant 4 (proxy-only-lineage-inheritance) as **structurally distinct but at 1st instance** — recommend non-novel continuation (option ii in researcher LEARNINGS.md) for memory hygiene; analyst makes final call.

— End REVIEW-adversarial.md —
