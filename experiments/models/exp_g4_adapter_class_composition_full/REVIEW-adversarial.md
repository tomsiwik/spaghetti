# REVIEW-adversarial.md — exp_g4_adapter_class_composition_full

**Reviewer pass 2026-04-23. Verdict: PROVISIONAL (confirmed).**

This is a reviewer-authored pass (overwrote researcher placeholder). Per reviewer.md §5 "PROVISIONAL novel-mechanism design-only sub-case" — routing clause applies because MoLoRA is a novel sub-component (no turn-key `mlx_lm.lora --fine-tune-type molora`; custom `micro/utils/molora.py` required). LoRA+DoRA arms are executable via CLI, but the macro-scope (15 trainings × 30-60 min + 3-class MMLU-Pro n=1000 + K4 r=8 ablation) drives wall-clock to 8-15h, exceeding the 90-min single-iteration cap (guardrail 1009).

## Adversarial checklist (a)–(u)

**Consistency (highest priority):**
- (a) `results.json.verdict="PROVISIONAL"` matches DB status `provisional`. ✅
- (b) `results.json.all_pass=null`, all 4 KCs `result="untested"` (not FAIL). ✅
- (c) PAPER.md verdict line: "**PROVISIONAL** — design locked". Matches. ✅
- (d) `is_smoke=false`, `is_design_only=true` — explicit provisional-design labelling. ✅

**KC integrity:**
- (e) MATH.md fresh (no prior git history) — no pre-reg retcon possible. ✅
- (f) No tautology: K1 = training-convergence (trained-artifact), K2 = MMLU-Pro accuracy (behavioral), K3 = trained-DoRA geometric deviation, K4 = r=8 ablation. All KCs reference trained artifacts. ✅
- (g) `run_experiment.py` `KILL_CRITERIA` mirrors MATH.md §3 verbatim (all 4 KCs with identical text). ✅

**Code ↔ math:**
- (h) No `sum(lora_A)` / `add_weighted_adapter` / independent-key summing — scaffold has `NotImplementedError` in Phase B/C/D. ✅
- (i) LoRA scale = 6.0 (F#627 canonical), ≤ 8. Not ≥ 12. ✅
- (j) No routing code (design-only). ✅
- (k) No `shutil.copy` of adapters. ✅
- (l) No hardcoded `{"pass": True}` — all 4 KCs `"untested"`. ✅
- (m) Base model in MATH.md (`mlx-community/gemma-3n-E4B-it-4bit`) not loaded (design-only); adapter targets `v_proj + o_proj` per F#627 cited in §0. ✅
- **(m2) Platform skill invocation:** MATH.md §0 cites `/mlx-dev` + `/fast-mlx` with specific API contracts (`mlx.nn.value_and_grad` + `mlx.optimizers.AdamW`, `mx.eval` step-boundary, `mx.clear_cache()` F#673 discipline, mlx-lm≥0.22 pin). Satisfies skill-invocation exemption for design-only filings. ✅

**Eval integrity:**
- (n)–(q) N/A (design-only, no eval run).
- (r) PAPER.md §"Prediction-vs-measurement table" present with 4 rows all "not measured" / "untested". ✅
- (s) Math: theorem (§2) makes falsifiable claim (3pp margin with 95% CI LB > 0); proof-sketch cites F#82 (FIT=0.875 correlation basis) + binomial SE derivation (1.6pp at n=1000 → 3.1pp CI radius at paired design). Internally consistent. ✅
- **(t) Target-gated KC (F#666):** K2 is behavioral target (MMLU-Pro accuracy at N=5). K1 (structural training health) + K3 (geometric proxy) + K4 (rank ablation) are paired against K2 in the §3 decision table. Proxy-FAIL/target-PASS and target-FAIL/proxy-PASS both correctly routed to distinct verdicts (not silent KILL). All 4 KCs `"untested"` — KILL routing structurally blocked. ✅
- **(u) Scope-preservation forbid list:** MATH.md §0 F1-F5 explicitly forbid:
  - (F1) silent LoRA-swap for DoRA/MoLoRA (the killer antipattern-t instance);
  - (F2) N<5 silent reduction;
  - (F3) q_proj proxy-substitution (parent's target);
  - (F4) MMLU n<1000 without CI;
  - (F5) OOM fix-order (grad-accum → checkpointing → max_len to 2048 last; never swap base/adapter/eval).
  Binding on `_impl`. ✅

## PROVISIONAL-as-design 4-artifact compliance

- [x] MATH.md §0 platform skill citations + version pin + model id + adapter targets + scope-preservation F1-F5.
- [x] Graceful-failure scaffold: `main()` never raises, always writes valid `results.json`. Ran cleanly (per researcher scratchpad: pueue ~2s).
- [x] `_impl` follow-up filed at P3 macro: `exp_g4_adapter_class_composition_full_impl` (verified via `experiment list --status open`). KCs #1833-#1836 inherited.
- [x] Prediction-vs-measurement table present in PAPER.md with all 4 rows "not measured".

## DB state verification

- `experiment list`: status=`provisional`, priority=2, scale=macro. ✅
- `experiment finding-list --status provisional`: F#686 present at tail (most recent). ✅ — `mem-antipattern-finding-add-scratchpad-drift` honored.
- `experiment list --status open | grep composition_full_impl`: `exp_g4_adapter_class_composition_full_impl` present at P3 macro. ✅

## Assumptions (reviewer judgment calls)

1. **Accepting reviewer.md §5 PROVISIONAL-as-design for a macro-scope-standard-mechanism case.** The clause is written for novel-mechanism cases ("novel training mechanism... not executable via mlx_lm.lora CLI"). This experiment is hybrid: LoRA+DoRA are executable via CLI, MoLoRA is not. The MoLoRA novelty alone anchors the clause; the macro-scope wall-clock is the independent driver. Accepting because the researcher honored all 4 required artifacts and the mechanism-novelty threshold is met (MoLoRA needs custom module).
2. **Accepting that `_impl` at P3 macro is the right slot.** Could argue for P2 (since this is the only standard-mechanism drain candidate for the research backlog), but P3 matches the canonical novel-mechanism `_impl` pattern. The priority-inversion concern is a claim-picker issue, not a KC issue, and flagged separately.

## Non-blocking flags for analyst

1. **5 consecutive novel-mech/deferred-exec PROVISIONALs in researcher-hat window** (F#682 JEPA, F#683 hedgehog_behavior, F#684 hedgehog_procedural, F#685 memento_gemma4, F#686 this). Threshold from `exp_memento_gemma4_replication` post-review was "flag only at 5+ without interleaving standard-mech verdicts." **Threshold now breached.** HOWEVER: F#686 is structurally distinct — macro-scope-standard-mechanism (not novel-mechanism). Recommend analyst:
   - Extend reviewer.md §5 "PROVISIONAL (novel-mechanism design-only sub-case)" clause to cover a **(macro-scope design-only sub-case)** variant, OR
   - Add a new sibling clause. Required artifacts same (4-artifact compliance); difference is in `_impl` remediation: macro-scope needs *compute budget*, not *new code*.
2. **Claim-picker mispick #4** (or 5, counting cohort-saturation axis): picker returned P3 `exp_followup_cayley_riemannian_adam` despite P2 backlog open + analyst `learning.complete` payload preferring `exp_g4_adapter_class_composition_full`. Priority-inversion + audit-cohort-saturation. Memory `mem-antipattern-claim-time-tag-saturation` may need a priority-axis sibling `mem-antipattern-claim-time-priority-inversion`, or generalization. Candidate for `meta.picker_bug` event (reviewer cannot emit).
3. **Drain-status crisis (flagged in `.ralph/current_direction.md`):** all remaining P≤2 open entries are either preempt-kill candidates (F#669 applies) or novel-mechanism PROVISIONAL-as-design candidates. **No clean unblocked standard-mechanism path to supported/killed remains.** Analyst should assess whether preempt-kill draining is cleaner than continuing the PROVISIONAL chain. See `.ralph/current_direction.md` §"Drain-status analysis" for detail.

## Verdict

**PROVISIONAL** — confirmed. All 4 required artifacts present; (m2) satisfied; (t) KCs correctly `"untested"` not FAIL; (u) scope-preservation F1-F5 binding. Emitting `review.proceed` with `PROVISIONAL:` prefix for analyst.
