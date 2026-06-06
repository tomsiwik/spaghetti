# REVIEW-adversarial — `exp_model_long_context_adapter_stability`

**Verdict.** PROVISIONAL (macro-scope design-only sub-case, compute-budget variant per F#768 pattern).

**Reviewer pass.** 2026-04-25, post `experiment.done` PROVISIONAL-escalation request from researcher (doom-loop break — prior 2026-04-24 iter RELEASED-to-OPEN).

---

## Disk state

- `MATH.md` (13 KB) — §3 derives proof-first prior with three regimes (linear-perturbation §3.1, V/O structural protection §3.2, range-extrapolation §3.3 LongLoRA). KCs K1706/K1707 inherited from DB byte-for-byte.
- `run_experiment.py` (6.6 KB) — refusal scaffold; `main()` returns 0, never raises uncaught exception; writes `results.json` directly with `verdict="PROVISIONAL"` and KCs `"untested"`.
- `results.json` — `verdict="PROVISIONAL"`, `is_smoke=false`, `all_pass=false`, blocker `COMPUTE_BUDGET_EXCEEDS_DRAIN_ITERATION`, ratio 8.0× single-iter budget.
- `PAPER.md` (8.5 KB) — verdict line "PROVISIONAL", prediction-vs-measurement table (5 rows untested), explicit "Why PROVISIONAL not RELEASED/KILLED" rationale, 6-check verdict-consistency pre-flight.
- `LEARNINGS.md` — absent (analyst's deliverable, deferred per workflow).

---

## Adversarial checklist

**Consistency:**
- (a) `results.json["verdict"]` = `"PROVISIONAL"` ↔ DB target `provisional` — **PASS**.
- (b) `all_pass=false` ↔ KCs untested — **PASS**.
- (c) PAPER.md verdict line is "PROVISIONAL" — **PASS**.
- (d) `is_smoke=false`, no smoke-as-full issue — **PASS**.

**KC integrity:**
- (e) MATH.md K1706/K1707 strings match DB-registered KCs byte-for-byte (NIAH within 5pp at 8k/32k/128k; RULER within 3pp on all subtasks). Dir is untracked-fresh; no prior version to diff against. **PASS**.
- (f) No tautology — KCs measure NIAH retrieval rate vs RULER 13-subtask scores, not algebraic identities. No KC was measured. **PASS**.
- (g) K-IDs in scaffold match MATH.md §5 and DB. **PASS**.

**Code ↔ math:**
- (h) No composition code executed (refusal scaffold). MATH.md §6 documents `Σ_i B_i @ A_i` as the IMPL-time requirement (correct). **PASS**.
- (i) LORA_SCALE — documented as ≤8 in MATH.md §6 per F#328/#330; not invoked here. **PASS**.
- (j) No routing in scaffold. **PASS**.
- (k) No `shutil.copy` of sibling adapters. **PASS**.
- (l) No hardcoded `{"pass": True, ...}` — all KCs marked `"untested"`. **PASS**.
- (m) Target model in MATH.md (§6) is `mlx-community/gemma-4-e4b-it-4bit`; matches `BASE_MODEL_ID` constant in `run_experiment.py:43`. No proxy substitution. MATH.md §6 explicitly distinguishes from `26b-a4b-it-4bit`. **PASS**.
- (m2) Skill invocation evidence — MATH.md §6 explicitly cites `/mlx-dev` and `/fast-mlx` as required-at-IMPL-time, "explicitly NOT performed in this design-only iteration because no platform code is being written here (refusal scaffold only)." Satisfies the design-only sub-case carve-out. **PASS**.

**Eval integrity:**
- (n)–(s) — N/A in design-only iteration. No eval performed.
- (t) Target-gated kill (F#666) — paired KC pair preserved (K1706 proxy + K1707 target). KILL is not the verdict; F#666 does not block PROVISIONAL. **PASS**.
- (u) Scope-changing fix antipattern — graceful-failure scaffold that refuses to silently proxy on the context-length axis IS the canonical preempt/PROVISIONAL artifact, not a scope reduction. PAPER.md §"Why this experiment is filed PROVISIONAL rather than KILLED or RELEASED" defends explicitly against silent 8k→128k extrapolation. **PASS**.

**Deliverables:**
- (r) PAPER.md prediction-vs-measurement table present with all 5 rows "untested" and clear rationale. **PASS**.
- (s) MATH derivation sound — §3.2 V/O structural-protection argument uses correct attention algebra (`softmax(QK^T)` unchanged by V/O perturbations); §3.3 cites LongLoRA as load-bearing for the 128k uncertainty.

All (a)–(u) clear for PROVISIONAL. KILL is unsafe (would discard the load-bearing 128k question; §3.3 grants genuine uncertainty per LongLoRA). RELEASED-to-OPEN was the prior 2026-04-24 verdict; repeating would be doom-loop.

---

## Verdict: PROVISIONAL

Macro-scope design-only sub-case (compute-budget variant). Pattern matches F#768 (model-cache variant of the same shape).

## Routing

- DB workaround per reviewer.md §PROVISIONAL applied:
  1. `experiment update --status provisional --dir micro/models/exp_model_long_context_adapter_stability/`
  2. `experiment evidence ... --verdict inconclusive`
  3. `experiment finding-add --status provisional ...` → **F#769 registered**.
  4. Verified via `experiment finding-get 769` (record present with linked experiment, failure-mode, impossibility-structure).
- No `_impl` companion filed in this iteration (scaffold's `reclaim_path` already documents the IMPL workflow; PAPER.md "Suggested follow-ups" lists three content-level decompositions, not workflow-required).
- Emit `review.proceed` with `PROVISIONAL:` prefix to analyst hat for `LEARNINGS.md`.

## Assumptions

- A1. Doom-loop break via PROVISIONAL escalation (not RELEASE-to-OPEN, not KILL) is the structurally-different action per researcher hat §0; pattern is F#768 precedent.
- A2. Hygiene defects (success_criteria + references empty per `experiment get`) are non-blocking for PROVISIONAL — the verdict is about the compute-budget blocker, not pre-reg hygiene.
- A3. The compute-budget reason (`COMPUTE_BUDGET_EXCEEDS_DRAIN_ITERATION`) is a sub-form distinct from F#768's `BASE_MODEL_NOT_CACHED` — both fit the macro-scope-resource-blocked PROVISIONAL super-family. Whether to register a new finding number or extend F#768 is the analyst's framing decision; this review files the new finding to make the sub-form discoverable.
