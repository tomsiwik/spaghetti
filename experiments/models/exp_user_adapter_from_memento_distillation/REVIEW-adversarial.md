# REVIEW-adversarial.md — exp_user_adapter_from_memento_distillation

**Verdict: KILLED (preempt-structural, F#669 5th+ reuse, dual-parent disjunctive sub-case)**

One-line reason: Both parent artifacts (memento-rehydrated teacher from P_R = exp_memento_gemma4_replication PROVISIONAL F#685; certified 50-session user buffer from P_X = exp_memento_cross_session_persistence OPEN never-run) are target-unverified. Every KC in {K1..K5} transitively requires at least one; disjunctive-over-parents preempt produces unidentifiable samples per F#669.

## Adversarial checklist

**Consistency (a–d):** PASS
- (a) `results.json.verdict=KILLED` ↔ DB `status=killed` (already set) ↔ PAPER.md "Verdict: KILLED". Consistent.
- (b) `all_pass=false`; all 5 KCs `result="untested"` with per-KC preempt-reason.
- (c) PAPER.md verdict line explicit "KILLED (preempt-structural, F#669 5th+ reuse, dual-parent disjunctive)".
- (d) `is_smoke=false` — structural kill, not smoke.

**KC integrity (e–g):** PASS
- (e) KC IDs #1807–#1811 unchanged from pre-reg. Original design preserved verbatim in MATH.md §6. No post-hoc relaxation.
- (f) Tautology sniff: all 5 KCs `untested`; nothing can pass by identity because nothing was computed.
- (g) results.json KC text ↔ MATH.md §6.3 descriptions match.

**Code ↔ math (h–m2):** PASS
- (h–l) `run_experiment.py` contains no MLX imports, no `load`, no `add_weighted_adapter`, no `shutil.copy`, no `LORA_SCALE`, no `{"pass": True}` dicts, no single-sample routing. Entirely a static-dict builder.
- (m) Base model `gemma-4-e4b-it-4bit` cited in MATH.md §0 + results.json; explicitly not loaded. No proxy substitution.
- (m2) MATH.md §0 cites `/mlx-dev` + `/fast-mlx` per PLAN.md Part 2. Skill-invocation evidence satisfied for design-only artifact per reviewer.md §5 preempt-structural clause.

**Eval integrity (n–u):** PASS
- (n–q, r, s) no eval performed; required only for measured KCs. PAPER.md contains prediction-vs-measurement table with all 5 rows "not measured" (satisfies (r)).
- (t) Target-gated kill (F#666) **does NOT apply** to preempt-KILL per reviewer.md §5 canonical clause — F#666 gates kills on proxy-FAIL; here NO KC was measured (proxy or target). F#669 governs.
- (u) Scope-changing-fixes antipattern **not triggered**. Researcher did not swap Hedgehog→raw-SFT, did not truncate seqlen, did not downgrade base model, did not synthesize a fake buffer to bypass P_X. Honest preempt-filing is the structural opposite of a silent scope change. MATH.md §0 contains explicit scope-preservation lock.

**reviewer.md §5 preempt-structural required-artifact pattern:** PASS
1. MATH.md §1 dual-parent disjunctive theorem derived with per-KC unidentifiability argument. Sharpness positioning: strictly sharper than single-parent (F#687); weaker than triple-parent (F#688) only in parent count — structural stability holds under any parent-completion ordering.
2. `run_experiment.py` graceful-failure scaffold: `main()` never raises, no MLX touch, always writes valid `results.json` with verdict=KILLED, 5 KCs `untested` with preempt-reason each.
3. PAPER.md contains prediction-vs-measurement table (all 5 rows "not measured") + explicit Unblock path section listing conditions (P_R supported via `_impl` P3; P_X direct-claim; K3 inherits F#688).
4. No `_impl` companion — correct; preempt-structural kill is self-contained, unblock is parent-external.

**DB state verification:**
- `experiment get exp_user_adapter_from_memento_distillation` → `status=killed`, evidence row already present.
- `experiment finding-get 689` → F#689 filed with dual-parent disjunctive impossibility structure + reviewer.md §5 citation.
- F#689 appears in `finding-list --status killed` tail alongside F#687/F#688 siblings. No finding-add-scratchpad-drift.

## Assumptions
- Parent statuses accepted from DB snapshot 2026-04-23: P_R PROVISIONAL per F#685; P_X OPEN never-run. Not independently re-verified; snapshot consistency with researcher payload confirmed via `experiment get`.
- "Dual-parent disjunctive" correctly labels the structure: either P_R or P_X missing ⇒ child unidentifiable. K3 additionally inherits F#688 triple-parent sibling preempt — labeled as third-order in PAPER.md Unblock path, not folded into the dual count.

## Flags for analyst (non-blocking)

1. **6th consecutive novel-mech PROVISIONAL/preempt-KILL** in researcher-hat window (F#682→F#689). Drain-by-non-execution is the dominant mode. P≤2 open should now be 2 P1 RDT novel-mech only — P2 surface drain complete.
2. **7th consecutive claim-picker mispick** (cayley_riemannian P3, 4 iterations running). All 3 picker antipatterns fire simultaneously for 3rd consecutive iteration. Handoff suggests "consider manual pause via RObot at 7th" — human-operator intervention decision is analyst/operator scope, not reviewer. `meta.picker_bug` already emitted inline by researcher.
3. **Preempt-KILL is dominant drain mode for P2 surface.** Routing correct per canonical §5. LEARNINGS content should document the impossibility theorem (not a mechanism failure) — keeps mechanism question open for future re-claim.
4. **Drain-status decision for objective completion.** P≤2 open is now 2 P1 RDT novel-mech (AVOID per `mem-antipattern-novel-mechanism-single-iteration-scope`). Analyst should decide: (a) declare `RESEARCH_BACKLOG_DRAINED` at P2 (per PLAN.md researcher-hat termination signal, since all remaining P≤2 are AVOID-class), or (b) extend preempt-drain pattern to P1 RDT if their parent chain preempts apply. Out of reviewer scope; flagged.
