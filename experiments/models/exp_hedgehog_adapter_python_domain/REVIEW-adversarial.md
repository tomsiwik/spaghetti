# REVIEW-adversarial — exp_hedgehog_adapter_python_domain

**Reviewer pass:** 2026-04-24 (independent). Overwrites researcher self-review per sibling precedent (F#683/F#684/F#696). 4th Hedgehog-axis design-only filing, 2nd axis-domain sibling (JS done at F#696, Rust + SQL still open at P=2).

## Verdict: **PROVISIONAL** (novel-mechanism design-only sub-case)

## Adversarial checklist (a)–(u)

**Consistency (a–d).** `results.json["verdict"]="PROVISIONAL"` (L41) ↔ `PAPER.md` L3 "Verdict: PROVISIONAL" ↔ DB `status=provisional` (verified via `experiment get`). `all_pass=false`; both KCs `"untested"` (L37-40). `is_smoke=false` (L2) — correct: design-only filing that never touched the model, not a smoke run. ✓

**KC integrity (e–g).** Fresh directory — `git log -- <path>` returns empty; all 6 artifacts are untracked at review time. No post-run KC mutation is possible. K1844/K1845 text in `run_experiment.py` L6-10 matches MATH.md §4 table matches DB-registered text (`experiment get` verbatim). No tautology — both KCs `"untested"`, nothing passes. ✓

**Code ↔ math (h–m2).** No composition math. `LORA_SCALE = 6.0` (L59) ≤ 8 per F#328/F#330. No routing. No `shutil.copy` adapter-as-new-domain. No `{"pass": True, ...}`. `STUDENT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"` (L48) and `TEACHER_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"` (L49) match MATH.md §0 verbatim — no proxy substitution. (m2): MATH.md §0 L9 explicitly cites `/mlx-dev` (array/nn/training patterns, `mx.eval`, `mx.clear_cache`, `nn.value_and_grad`) + `/fast-mlx` as hard-gated skills before `_impl`'s training loop lands. Satisfies design-only (m2). ✓

**Eval integrity (n–u).** (n)–(q) N/A — no eval executed. (r) PAPER.md L9-12 has the prediction-vs-measurement table with kill-condition column, 2 rows, all "not measured". (s) MATH.md §3 theorem is explicitly labeled "(informal)" with "Proof sketch" — honest. (t) F#666 **does not apply** — KILL requires both proxy AND target FAIL; both here are `untested`, not FAIL. (u) Scope-preservation: PAPER.md §"Why not silently downscale" explicitly rejects (a) proxy teacher→E4B, (b) cos-sim→CE swap, (c) skip K1844 baseline, (d) reduce `N_STEPS`. `run_experiment.py` raises `NotImplementedError` in each phase rather than stubbing. No silent scope swap. ✓

## Novel-mechanism design-only 4-part check (reviewer.md §PROVISIONAL)

1. **MATH.md §0 skills cited** — L9: `/mlx-dev` + `/fast-mlx`. ✓
2. **`main()` never raises; writes PROVISIONAL + untested KCs** — verified L276-335: every `NotImplementedError` is caught in `try/except`, `blockers` list accumulates, `RESULTS_FILE.write_text(...)` at L343 always runs. Scaffold executed in 1.6 s producing `results.json` with 5 structured blockers. ✓
3. **`_impl` follow-up at P=3 with KCs inherited verbatim** — verified via `experiment get exp_hedgehog_adapter_python_domain_impl`: P=3, depends-on parent, K1951/K1952 text-inherit K1844/K1845 verbatim. Researcher filed this inline per `mem-antipattern-impl-follow-up-delegation` — reviewer did NOT need to file it. ✓
4. **PAPER.md prediction-vs-measurement table** — L9-12, all "not measured", kill-condition column, status column. ✓

All 4 satisfied. PROVISIONAL is the canonical verdict.

## Assumptions (judgment calls)

- **Sub-case.** Novel-mechanism (cos-sim distillation not in `mlx_lm.lora` CLI) not macro-scope (compute-only). Same as F#683/F#684/F#696.
- **KC-count asymmetry.** 2 KCs pre-registered in DB (K1844+K1845) vs JS sibling's 4. Researcher correctly refused to retro-attach non-interference KCs post-hoc (would violate KC-lock). Non-interference deferred to triple-composition child `exp_hedgehog_triple_composition_3domain`. Acceptable — the 2-KC set still satisfies F#666 (proxy+target pair).

## Non-blocking notes for analyst

1. **KC-count asymmetry across axis-domain siblings** (JS=4, Python=2) — MATH.md §A7 documents as pre-registration choice, not defect. Forward implication: triple-composition child's cross-axis interference analysis partially substitutes for the missing non-interference KCs here. Not an antipattern candidate — DB pre-reg choice, not silent relaxation.
2. **`_impl` priority drift across siblings** — politeness/procedural siblings at P=1; JS + this at P=3 (matches reviewer.md spec). Non-blocking doc drift, optional hygiene.
3. **26 B teacher not cached** — shared blocker across all 4 Hedgehog `_impl`s + `exp_model_knowledge_gap_26b_base` (currently P=3). A single ~14 GB pre-cache step unblocks all.
4. **No antipattern candidates** — clean application of `mem-antipattern-novel-mechanism-single-iteration-scope` + `mem-antipattern-impl-follow-up-delegation`.

## Actions

- Finding filed: **F#697** (via `experiment finding-add --status provisional`).
- DB status already `provisional` (researcher two-step workaround); no re-issue.
- Emit `review.proceed` with `PROVISIONAL:` prefix + `_impl` ID.
