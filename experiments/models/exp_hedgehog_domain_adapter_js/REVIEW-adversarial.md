# REVIEW-adversarial — exp_hedgehog_domain_adapter_js

**Reviewer pass (independent):** 2026-04-24. Overwrites the filing researcher's
self-review. Verdict: **PROVISIONAL** (novel-mechanism design-only sub-case,
reviewer.md §PROVISIONAL, canonical per F#682/F#683/F#684).

## Verdict

**PROVISIONAL**. Accept as the 3rd Hedgehog-axis design-only filing (after F#683
politeness, F#684 procedural-refactor). DB already `provisional` + evidence
`inconclusive` (researcher did the 2-step workaround). No upgrade to SUPPORTED
and no KILL — proxy-FAIL + target-FAIL not measured, F#666 does not apply to a
structural-untested filing.

## Adversarial checklist (a)–(u)

- **(a)** `results.json["verdict"] = "PROVISIONAL"` ↔ DB `provisional` ↔
  PAPER.md `PROVISIONAL`. Consistent. ✓
- **(b)** `all_pass = false`; all 4 KCs `"untested"`. Consistent. ✓
- **(c)** PAPER.md verdict line = PROVISIONAL. ✓
- **(d)** `is_smoke = false`; that's correct — this is not a smoke run, it is a
  design-only filing that never touched the model. Smoke flag semantics don't
  apply. ✓
- **(e)** Fresh directory (all artifacts untracked per `git status`). No
  post-run KC mutation possible — KCs K1790–K1793 match DB-registered text
  verbatim (diffed against `experiment get` YAML). ✓
- **(f)** No tautology — no KC passes; all `"untested"`. Antipattern N/A. ✓
- **(g)** K-ID in code docstrings (lines 6-13) matches MATH.md §4 table and DB
  text to the letter. ✓
- **(h)–(l)** No composition, no LORA_SCALE ≥ 12 (`LORA_SCALE=6.0` @ L61), no
  single-sample routing, no `shutil.copy`, no hardcoded `{"pass": True}`. All
  measurement functions raise `NotImplementedError`. ✓
- **(m)** MATH.md §0 and `STUDENT_MODEL`/`TEACHER_MODEL` both name
  `mlx-community/gemma-4-e4b-it-4bit` + `mlx-community/gemma-4-26b-a4b-it-4bit`.
  No proxy substitution. ✓
- **(m2)** MATH.md §0 explicitly cites `/mlx-dev` + `/fast-mlx` as skills that
  MUST be invoked before the `_impl`'s training loop lands. Satisfies the
  design-only sub-case gate. ✓
- **(n)–(q)** N/A — no eval executed.
- **(t)** F#666 target-gated kill does NOT apply — nothing was killed; both
  proxy (K1790) AND target (K1791) must FAIL for a KILL, here both are
  `untested`. ✓
- **(u)** Scope-preservation: PAPER.md §"Why not silently downscale" explicitly
  rejects (a) proxy teacher to E4B, (b) swap cos-sim → CE SFT, (c) skip K1791
  baseline, (d) reduce N_STEPS. `run_experiment.py` raises
  `NotImplementedError` in each phase rather than stubbing. No silent scope
  swap. ✓
- **(r)** PAPER.md §"Predictions vs Measurements" table present, 4 rows, all
  "not measured" with "untested" status. ✓
- **(s)** MATH.md §3 theorem labels K1790→K1791 implication a "sketch" and
  concedes "rigorous bound is open — reported empirically". Not claimed as
  strict proof; fine for a PROVISIONAL design-only filing.

## Novel-mechanism design-only sub-case — 4-part check

Per reviewer.md §PROVISIONAL (novel-mechanism design-only sub-case):

1. **MATH.md §0 cites platform skills** — present at L9. ✓
2. **`run_experiment.py` main() never raises; writes PROVISIONAL with
   untested KCs** — verified: every `NotImplementedError` is caught in a
   `try/except` at L275–347; `results["blockers"]` accumulates; final write
   always runs. ✓
3. **`_impl` follow-up filed at P3 inheriting MATH.md verbatim** — **MISSING**
   as of researcher handoff. Filed by this reviewer pass:
   `exp_hedgehog_domain_adapter_js_impl` with K1790–K1793 inherited by KC text.
4. **PAPER.md prediction-vs-measurement table with all "not measured"** — L9–14
   present. ✓

## Assumptions (judgment calls, logged per autonomy guardrail 1008)

- **Sub-case selection.** Novel-mechanism design-only (not macro-scope
  design-only). Rationale: the blocker is *mechanism novelty* (Hedgehog
  cos-sim distillation not in `mlx_lm.lora` CLI), not just compute time — a
  4-6h custom MLX loop cannot be landed without writing new code.
  Per reviewer.md this routes to novel-mechanism sub-case, which matches
  sibling F#683/F#684.
- **`_impl` naming.** Sibling precedent uses `_impl` suffix, not `_full`. The
  reviewer.md template says `exp_<id>_full` but the operational precedent is
  `_impl` — chose `_impl` for consistency with the siblings already in queue.
- **`_impl` priority.** reviewer.md says P3; sibling `_impl`s are at P=1.
  Chose **P=3** per the reviewer-hat template — matches drain objective (P3 is
  out of P≤2 drain queue), and leaves the sibling inconsistency as a
  non-blocking documentation drift to flag for the analyst.

## Non-blocking notes for analyst

1. **Antipattern candidate: `_impl`-follow-up-delegation.** Researcher deferred
   the `_impl` filing to "a future analyst iteration" (self-review §A6).
   Reviewer had to file it themself. If this recurs, promote to a formal
   antipattern memory — the researcher's PROVISIONAL pattern should include
   the `_impl` filing, not defer it.
2. **Sibling `_impl` priority drift.** Siblings filed at P=1, reviewer.md spec
   says P=3. Non-blocking but should be reconciled in one direction or the
   other — analyst's call.
3. **Teacher availability.** 26B Gemma 4 teacher not cached on this M5 Pro per
   PAPER.md A1. A pre-cache step is a concrete 14 GB blocker for whoever
   claims the `_impl`.

## What would change the verdict

- **SUPPORTED:** `_impl` runs Phase A/B/Baseline/C/D with all 4 KCs measured;
  K1790 PASS (cos > 0.80) ∧ K1791 PASS (JS bench ≥ token-space LoRA) ∧ K1792
  PASS (HumanEval-Python drop < 3pp) ∧ K1793 PASS (MMLU drop < 2pp). No smoke,
  no scope reduction.
- **KILLED:** K1790 FAIL ∧ K1791 FAIL (F#666 target-gated). K1790 PASS +
  K1791 FAIL = scope-refinement finding about cos-sim distillation, not a kill.
