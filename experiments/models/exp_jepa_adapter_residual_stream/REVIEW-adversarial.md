# REVIEW-adversarial — `exp_jepa_adapter_residual_stream`

**Reviewer pass (overwrites researcher self-review placeholder).**

## Verdict

**PROVISIONAL — confirmed.** Design-only experiment; 4 KCs pre-registered and target-gated per F#666; no empirical run. Scope-preservation honored (antipattern-t), F#666 honored (no proxy-alone kill), false-kill avoided. Route via two-step workaround (already executed by researcher: `experiment update --status provisional` + `experiment evidence … --verdict inconclusive`).

## One-line reason

Nothing measured → nothing falsified → KILL unjustified; nothing supported → PROCEED unjustified; design is grounded in arxiv:2603.19312 and arxiv:2511.08544 with canonical target-gated KCs → PROVISIONAL is the honest verdict.

## Adversarial checklist (reviewer independent pass)

**Consistency:**
- (a) `results.json["verdict"]="PROVISIONAL"` = DB status `provisional` = PAPER.md verdict line `PROVISIONAL`. ✓
- (b) `all_pass: false` consistent with "untested" KCs. ✓
- (c) No SUPPORTED claim anywhere; PROVISIONAL consistent across all three. ✓
- (d) `is_smoke: false` — intentional. The scaffold neither ran nor smoked; PROVISIONAL reflects "design locked, implementation deferred," not "smoke run." Edge case but defensible — the reviewer PROVISIONAL criteria (smoke OR structural-PASS+target-not_measured) don't cover this scenario, but it is *more* conservative than either (nothing ran at all), so KILL/PROCEED are both strictly worse.

**KC integrity:**
- (e) KCs in `MATH.md §5` exactly match `results.json["kc"]` keys; no post-hoc relaxation (no data collected). ✓
- (f) Tautology sniff: K#1766 EP rejection on adapter activations (not self-referential); K#1767 loss ratio (genuine learning-dynamics measurement); K#1768 vs independent baseline; K#1769 ablation gap. No tautology. ✓
- (g) K-IDs match MATH.md text 1:1 (K#1766/1767/1768/1769). ✓

**Code ↔ math:**
- (h) No `sum(lora_A)`, no `add_weighted_adapter(combination_type="linear")`, no composition at all. ✓ N/A
- (i) `LORA_SCALE = 6.0` ≤ 8 per F#328/F#330. ✓
- (j) No routing. ✓ N/A
- (k) No `shutil.copy` of sibling adapter. ✓
- (l) All KCs `"untested"`, not faked. ✓
- (m) `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` = MATH.md target model. No proxy substitution. ✓
- (m2) **Non-blocking flag**: `run_experiment.py:11` references "Skills invoked: /mlx-dev + /fast-mlx (documented in MATH.md §0)" but MATH.md has no §0. The skill-invocation declaration is missing from MATH.md. For the PROVISIONAL design pass this is non-blocking (no MLX training code runs). **The follow-up `_impl` iteration MUST invoke `/mlx-dev` + `/fast-mlx` before writing Phase B, and add a MATH.md §0 to that effect, or the reviewer will REVISE on (m2).**

**Eval integrity:**
- (n)–(q) N/A — nothing measured.
- (t) **Target-gated kill honored**: 2 proxy/target pairs registered. K#1766 (structural) + K#1768 (target accuracy); K#1767 (dynamics proxy) + K#1769 (ablation target). No proxy-alone kill path exists. PROVISIONAL with all four untested is fine; no F#666 violation possible. ✓
- (u) **Scope-preservation**: researcher explicitly refused silent swap from JEPA→standard-LoRA when Phase B proved un-implementable in one iteration. `train_jepa_adapter` raises `NotImplementedError` with a structured blocker list (lines 177–188); Phase C/D correctly mark dependent blockers. No max_length reduction, no model downgrade, no KC dropping. Textbook-correct handling of antipattern-(u). ✓

**Deliverables:**
- (r) PAPER.md §"Prediction vs. measurement" table present with P1–P4 → K#1766–K#1769 mapping and explicit "untested" status. ✓
- (s) Math: LeJEPA Thm 1 citation correct (Cramér-Wold + Epps-Pulley consistency); SIGReg formulation matches Eq. 7; stopgrad pattern correct; rank-matching argument honest (head discarded at inference, param-match caveat explicit in §3.2). No errors spotted.

## Non-blocking flags for analyst

1. **Design-as-experiment edge case.** PROVISIONAL for a never-run design is not in the reviewer's canonical PROVISIONAL criteria (smoke-OR-target-not-measured). Filing this verdict preserves the math but risks a "scaffolding-is-science" antipattern if repeated. Analyst may consider formalizing as a `type: fix` memory: **"novel-mechanism single-iteration scope"** — mechanisms requiring custom training loops (JEPA, recurrent-depth, distillation-with-aux-loss) should be scoped to dedicated implementation iterations, not researcher-hat single-iteration attempts. Researcher flagged this in LEARNINGS.md §4; endorsing.

2. **Skill-invocation gap** `run_experiment.py:11` claims `MATH.md §0` documents `/mlx-dev + /fast-mlx` invocation, but that section doesn't exist. Non-blocking for this PROVISIONAL (no MLX code runs); **blocking for `_impl` follow-up.**

3. **Follow-up naming**: PAPER.md uses `_impl` suffix; reviewer.md workflow specifies `_full` convention. Both are accepted; `_impl` is clearer for an implementation-only follow-up.

## Assumptions (reviewer judgment calls)

- **R1.** A "never-run design" PROVISIONAL is a legitimate reviewer verdict when (a) KCs are pre-registered and target-gated, (b) no silent scope reduction was applied, and (c) no falsifier was collected. The alternative (KILL) would fabricate a failure; PROCEED would fabricate a success. Logged.
- **R2.** Filing a `_impl` follow-up at P3 is correct per objective discipline (P≤2 backlog drain). Not extending the drained bucket.
