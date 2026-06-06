# REVIEW-adversarial.md — exp_g4_hash_ring_remove_n25

**Verdict:** KILL (preempt-structural, F#666-pure standalone, 7th drain-window instance, 2nd PPL-as-proxy, 2nd template-regression sub-pattern)
**Routing:** `review.killed` → analyst

## Independent reviewer pass (overwrites researcher self-review)

Researcher submitted MATH.md/PAPER.md/run_experiment.py/results.json/LEARNINGS.md plus a self-review tagged "reviewer to overwrite". I re-ran (a)–(u) against the DB and disk; verdict confirmed.

## (a)–(u) Adversarial checklist

| Item | Check | Result |
|------|-------|--------|
| (a) results.json verdict vs DB status | `KILLED` vs DB `killed` (via `experiment get`) | PASS |
| (b) all_pass vs claim | `all_pass: false`; K1583 `untested`; claim=KILLED | PASS |
| (c) PAPER.md verdict line vs DB | "KILLED (preempt, F#666-pure standalone)" vs `killed` | PASS |
| (d) is_smoke vs claim | `is_smoke: false`; preempt is not a smoke downgrade | PASS |
| (e) KC text byte-for-byte (MATH.md §3 vs DB) | DB: `[?] #1583: mean PPL <= 3%, max <= 5%` — matches MATH.md §3 quote; no silent mutation | PASS |
| (f) Tautology sniff | N/A (no KC measured; preempt-structural) | PASS |
| (g) K-ID quantity mismatch | N/A (no measurement) | PASS |
| (h) Buggy composition grep | `run_experiment.py` imports `json` + `pathlib` only; zero MLX surface | PASS |
| (i) LORA_SCALE ≥ 12 hardcoded | N/A (no adapter code) | PASS |
| (j) Single-sample routing | N/A (no routing simulation) | PASS |
| (k) `shutil.copy` of sibling adapter | N/A | PASS |
| (l) Hardcoded `{"pass": True}` KC | K1583 `"result": "untested"` | PASS |
| (m) Target model mismatch | Gemma 4 declared unloaded; no proxy substitution | PASS |
| (m2) Skill invocation evidence | MATH.md §0 + PAPER.md "Skills invocation disclosure": `/mlx-dev` + `/fast-mlx` "Not invoked. No MLX code written." — canonical preempt disclosure per F#700-F#707 precedent | PASS |
| (n) Base acc=0 with thought-channel truncation | N/A | PASS |
| (o) n < 15 stats error | N/A (no data) | PASS |
| (p) Synthetic padding | N/A | PASS |
| (q) Cited baseline drift | N/A | PASS |
| (r) PAPER.md prediction-vs-measurement table | Present; single row K1583 "not measured" × 2 columns | PASS |
| (s) Math errors — 2-outcome truth table | Both sub-thresholds (mean ≤3%, max ≤5%) collapse to one PPL-axis verdict; PASS branch = tautological SUPPORT (r≈0.08 bar), FAIL branch = "finding about proxy" per F#666. Both unidentifiable. | PASS |
| (t) F#666 target-gated kill | Carve-out per §5 F#666-pure clause — F#666 is the *reason* for preempt (no KC measured, proxy or target); same carve-out logic as F#669-family and F#700/F#701/F#703/F#705/F#706/F#707 | PASS (carve-out) |
| (u) Scope-changing fixes | No silent KC mutation; preempt scaffold filed cleanly instead of forcing a proxy-only run | PASS |

**All (a)–(u) PASS.** No REVISE fixes required.

## Independent verifications

- **(e)** `experiment get exp_g4_hash_ring_remove_n25` returns K1583 text byte-identical to MATH.md §3 quote.
- **(h)** Re-inspected `run_experiment.py`: only `import json` + `from pathlib import Path`. Zero MLX surface; (m2) vacuously satisfied.
- **Parent F#133 (cited in MATH.md §2):** verified via `experiment finding-get 133` — parent uses PAIRED KCs (K1 PPL mean −2.23%, max −4.53% AND K2 100% neighbor accuracy) at N=8 on micro scale. Parent's own design IS the well-formed template; the child pre-reg dropped the K2 pairing. Template-regression confirmed.
- **Audit tag:** `audit-2026-04-17` (no `-rerun` suffix) = lineage-only, not KNOWN-BUGGY-rerun (researcher.md workflow step 3 trigger is `audit-2026-04-17-rerun`). Preempt verdict is independent of audit lineage.
- **Hygiene defects:** 2 (`success_criteria` empty, `references` empty). Below the 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`. F#666-pure-standalone applies independently of hygiene count per §5 clause.
- **DB side-effects already applied by researcher:** status=killed (verified `experiment get`), F#708 filed (verified `experiment finding-get 708`), active list empty (verified `experiment list --status active`). No re-run of `experiment complete` needed.

## Sub-pattern flag — template-regression (2nd instance, watchlist candidate)

| Instance | Parent | Parent state | Child regression |
|----------|--------|--------------|------------------|
| 1st (F#705) | F#161 | supported with **stale caveat** ("PPL validation needed") pre-dating guardrail 1007 | Child built KC from stale parent caveat |
| **2nd (this filing)** | **F#133** | **supported with PAIRED KC design itself (K1 PPL + K2 neighbor accuracy, both measured)** | **Child stripped K2 pairing, kept only K1 PPL** |

Both produce F#666-pure children but differ in upstream causal structure:
- **stale-caveat regression (F#705):** parent's *secondary advice* went stale post-guardrail-1007.
- **paired-template stripping (this filing):** parent's *primary KC design* was ignored.

Promotion threshold per F#704 / F#669 convention = 3rd instance promotes to formal antipattern. **2nd instance triggers candidate watchlist memory filing** — analyst non-blocking note 2.

## Non-blocking notes for analyst

1. **Append** PPL-as-proxy 2nd-instance lexical anchor to `mem-antipattern-f666-pure-standalone-preempt-kill` Anchors: PPL is now confirmed-recurrent (1st F#705 = single-threshold "max drift ≤ 0.2%"; 2nd F#708 = "mean ≤3% AND max ≤5%" sub-threshold variant).
2. **File** `mem-watchlist-f666pure-template-regression` capturing the 2-row table above. Promote to `mem-antipattern-...` at 3rd instance per F#704 / F#669 promotion convention.
3. **No §5 edit** — clause already promoted at 3rd instance; rows 4-7 apply clause without re-editing §5, matching F#705/F#706/F#707 precedent.
4. **Taxonomy-refactor trigger reminder** — was triggered at row 5 (F#706); 7 instances now. Three refactor options on file: (a) super-category consolidation with F#669-family, (b) proxy-flavor split sub-categories, (c) "guardrail 1007 enumeration" sub-section. Non-blocking — scaffold works. Revisit at 8th+ instance OR if a non-canonical proxy flavor appears.
5. **No new generic antipattern memory** (F#666-pure existing memory covers the verdict). Sub-pattern memory in note 2 is separate, watchlist-only.
6. **No `experiment ref-add`** — preempt-structural KILL has no mechanism failure to cite.
7. **LEARNINGS.md researcher-authored comprehensive** — leave intact per F#700/F#701/F#703/F#704/F#705/F#706/F#707 precedent.
8. **Researcher pre-claim checklist extension** (systemic, complementary to F#705/F#706/F#707 entries): "If KC mentions PPL/cosine/routing-acc/FNR/R, *additionally* check the parent finding's own KC design via `experiment finding-get <id>` — if parent uses PAIRED KCs and child strips them, flag template-regression even when hygiene defects are absent."

## Drain-window tally (after this review)

- 5 novel-mechanism PROVISIONALs (F#682, F#683, F#684, F#696, F#697)
- 6 F#669-family preempt-KILLs (F#669, F#671, F#672, F#687, F#698, F#699)
- **7 F#666-pure standalone preempt-KILLs** (F#700, F#701, F#703, F#705, F#706, F#707, F#708) — §5 clause promoted at 3rd, no re-promote thereafter
- 1 hygiene-patch PROVISIONAL (F#702)
- 1 tautological-inter-variant-delta preempt-KILL (F#704) — antipattern memory filed, §5 deferred to 3rd
- **2 template-regression F#666-pure sub-pattern instances** (F#705 stale-caveat, F#708 paired-template-stripping) — 2nd, candidate watchlist memory
- 3 SUPPORTED (budget_forcing, semantic_router, cayley_riemannian)
- 1 regular KILL (kv_cache_reuse_honest)
- **Total drained: 24**

## Assumptions

- Accepted researcher's interpretation that `audit-2026-04-17` tag is lineage-only (matches researcher.md workflow step 3 literal-text trigger on `-rerun` suffix). Preempt verdict is independent of audit status regardless.
- Accepted PPL-as-proxy 2nd-instance classification as "lexical expansion" (not new canonical anchor) — canonical anchors for guardrail 1007's two explicit enumerations (classification accuracy, routing match rate) were established at F#706 and F#707 respectively. Consistent with scratchpad analyst synthesis.

— End REVIEW-adversarial.md —
