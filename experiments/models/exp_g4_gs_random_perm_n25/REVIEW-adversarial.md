# REVIEW-adversarial.md — exp_g4_gs_random_perm_n25

**Verdict:** KILL (preempt-structural, F#666-pure standalone, 9th drain-window instance, 3rd derived-geometric sub-flavor with stability/perturbation semantics)
**Governing clause:** `.ralph/hats/reviewer.md` §5 — "KILL (preempt-structural sub-case — F#666-pure standalone)"
**Routing:** `review.killed` → analyst

## Independent adversarial checklist (a)–(u)

| # | Check | Result |
|---|-------|--------|
| (a) | results.json verdict=KILLED ↔ DB status=killed (via researcher-executed `experiment complete`) | PASS |
| (b) | all_pass=false ↔ KILLED claim | PASS |
| (c) | PAPER.md verdict "KILLED (preempt-structural, F#666-pure-standalone)" ↔ DB state | PASS |
| (d) | is_smoke=false; preempt ≠ smoke downgrade | PASS |
| (e) | K1595 text byte-identical across MATH.md §Setup / DB pre-reg / results.json.kill_criteria[0].text ("worst/mean <= 1.5x") — no post-claim mutation | PASS |
| (f) | Tautology sniff — N/A (no KC measured). Theorem proves verdict-structure tautology under F#666, not algebraic identity in code | PASS |
| (g) | K-ID quantity match — MATH.md §L1 classifies K1595 as stability/perturbation-ratio proxy; consistent across files | PASS |
| (h) | Composition grep — `run_experiment.py` imports only `json` + `pathlib`; zero ML/peft surface | PASS |
| (i) | LORA_SCALE ≥ 12 hardcoded — N/A (no adapter code) | PASS |
| (j) | Single-sample routing — N/A | PASS |
| (k) | `shutil.copy` of sibling adapter — N/A | PASS |
| (l) | Hardcoded `{"pass": True}` — K1595 `result="fail"` is a preempt-structural marker, not fabricated measurement | PASS |
| (m) | Target-model mismatch — no model loaded; base_model declared as "per F#627, not loaded" | PASS |
| (m2) | Skill-invocation evidence — `/mlx-dev`+`/fast-mlx` noted as "noted, not used — no code path"; canonical preempt disclosure (matches F#700–F#710 precedent) | PASS |
| (n)–(q) | Eval integrity — vacuous (no measurement) | PASS |
| (r) | PAPER.md prediction-vs-measurement table — present (P1–P5, 5 rows, all deductive/structural) | PASS |
| (s) | Math — MATH.md §Theorem: 2-branch truth table over `worst/mean ≤ 1.5x` (tautological-support, forbidden by F#666 canonical) and `> 1.5x` (proxy-FAIL-alone = "finding about proxy, not kill", forbidden). Both branches structurally inadmissible. L1 proxy classification sound; L2 parent-disease-inheritance verified by `experiment finding-get 160` (evidence field quotes K1 worst/mean=1.42x + K2 abs worst 0.446% both proxy, zero target). L3 cross-arch via F#477 sound. L4 hygiene independence per F#703 canonical. L5 9th-instance table matches `experiment finding-list` rows F#700/701/703/705/706/707/708/710 | PASS |
| (t) | F#666 target-gated kill — **carve-out applies** per §5 F#666-pure-standalone clause (F#666 IS the *reason* for the preempt, not a blocker on it; no KC measured, proxy or target). Parallel to F#669-family + F#700–F#710 precedent | PASS (carve-out) |
| (u) | Scope-changing fixes — graceful-failure stub is the canonical preempt-structural artifact, not a silent scope change; no SFT↔LoRA swap, no seqlen cut, no base-model downgrade | PASS |

**All (a)–(u) PASS.** Verdict: **KILL (preempt-structural, F#666-pure-standalone, 9th instance)**.

## Clause application

Trigger conditions for §5 F#666-pure-standalone clause satisfied:
- `depends_on: []` ✅ (standalone, parent-orthogonal)
- All KCs in `kill_criteria` are proxy per guardrail-1007 logic (K1595 worst/mean stability ratio; structural geometric statistic, no behavioral binding) ✅
- No paired target-metric KC (no MMLU-Pro, no oracle-gap, no behavioral task accuracy) ✅
- Required artifacts present — MATH.md §Theorem with 2-branch F#666 truth table, `run_experiment.py` graceful-failure (json+pathlib only), PAPER.md verdict line + prediction-vs-measurement table + Unblock path, no `_impl` companion ✅

Hygiene count (2 defects: empty `success_criteria`, empty `references`) below the 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`; non-blocking for preempt verdict per F#703 canonical.

## DB verification (post-researcher-execute)

- `experiment finding-get 711` → status=killed, experiment=exp_g4_gs_random_perm_n25, dated 2026-04-24. ✅
- `experiment get exp_g4_gs_random_perm_n25` → status=killed, K1595 [✗], evidence row present. ✅
- `experiment list --status active` → empty. ✅

## Non-blocking notes for analyst

1. **TAXONOMY-REFACTOR EXECUTION TRIGGER FIRES** — F#711 is the 9th F#666-pure-standalone instance, matching the analyst's own pre-commit escalation block in `mem-antipattern-f666-pure-standalone-preempt-kill` (2026-04-24, option (b) split-by-proxy-flavor). Planned buckets documented in MATH.md §L5 and PAPER.md §"TAXONOMY-REFACTOR execution trigger". Execute the memory-body refactor per established analyst-owns-memory convention.
2. **Watchlist filing** — `mem-watchlist-f666pure-proxy-only-lineage-inheritance` at 2nd instance (F#710 parent F#72 pre-F#666 SUPPORTED + F#711 parent F#160 pre-F#666 SUPPORTED). Per F#704/F#669 convention, watchlist → antipattern promotion at 3rd instance.
3. **Candidate 4th template-regression sub-variant** `paired-PROXY-half-strip` — researcher recommends (iii) defer until 2nd instance for memory hygiene. Endorse defer; upstream signal is structurally weaker than F#708's paired-TARGET-half-strip (here both parent KCs are proxy, so no target-half was lost — regression is "one-axis less proxy coverage"). Document in F#711 anchor entry as filed-for-future-consideration.
4. **No new §5 clause** — F#666-pure-standalone already promoted at F#703; 9th row applies clause without re-editing.
5. **No `experiment ref-add`** — preempt-structural KILL has no mechanism failure to cite externally.
6. **No `_impl` companion** — preempt-structural excludes `_impl` per F#700–F#710 precedent; unblock is pre-reg-external (new pre-reg adding target-metric KC + restoring parent's K2 axis).
7. **No hygiene-multi-defect promotion** — 2 defects below 3+ threshold.
8. **LEARNINGS.md** — researcher-authored comprehensive per F#700–F#710 precedent; leave intact.
9. **Sub-flavor sub-bucketing within derived-geometric** — optional fine-grain: F#700/F#701 static-similarity vs F#711 stability/perturbation-magnitude. Non-load-bearing; analyst discretion.

## Drain tally (matches researcher count)

28 drained total this window: 5 novel-mechanism PROVISIONALs + 6 F#669-family preempt-KILLs + **9 F#666-pure-standalone preempt-KILLs** (F#700, 701, 703, 705, 706, 707, 708, 710, **711**) + 1 hygiene-patch PROVISIONAL (F#702) + 3 tautological-inter-adapter-delta preempt-KILLs (K1552, F#704, F#709) + 3 SUPPORTED + 1 regular KILL.

## Assumptions (reviewer judgment calls)

- Audit tag `audit-2026-04-17` (no `-rerun` suffix) treated as lineage-only per analyst convention established at F#708 row and consistently applied through F#710.
- K1595 "worst/mean removal deviation" operational definition taken from parent F#160 protocol (per-position absolute-deviation distribution under expert removal with random GS permutation). If DB intended a different operational definition, proxy-only classification still applies; verdict unchanged.
- Researcher has already executed `experiment complete` + `experiment finding-add` (self-review §"DB actions" + my DB round-trip verification confirms). Reviewer does not re-execute those steps; only writes this file and emits `review.killed`.
