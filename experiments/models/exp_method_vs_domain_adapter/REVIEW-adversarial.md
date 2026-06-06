# REVIEW-adversarial.md — exp_method_vs_domain_adapter

**Verdict: PROCEED (smoke → PROVISIONAL, route to analyst).**

Reviewer: 2026-04-18. Inputs reviewed: `MATH.md`, `PAPER.md`, `results.json`,
`run_experiment.py`. No sub-agents spawned (per hat discipline).

## Adversarial checklist

**Consistency (a)–(d) — all pass.**
- (a) `results.json["verdict"] = "PROVISIONAL"`; no claim of `supported`.
- (b) `all_pass = false` — matches PROVISIONAL; K1718 FAIL, K1719 PASS, K1720 FAIL.
- (c) PAPER.md verdict line: "**PROVISIONAL (smoke completion).**" — matches.
- (d) `is_smoke = true`; SMOKE run; verdict path is `--status provisional`
  (researcher noted CLI stores this as `open`). No smoke-as-full upgrade.

**KC integrity (e)–(g) — all pass.**
- (e) Experiment dir is **untracked in git** — no prior MATH.md to diff.
  KCs (K1718/K1719/K1720) are pre-registered in the first and only version.
  No post-hoc swap possible on this run; researcher explicitly declines to
  relax the K1720 signature definition in-place (PAPER.md §"Required v2
  fixes" — correct discipline per guardrail 1009).
- (f) Tautology sniff: K1718 (multi beats base on held-out) and K1719 (single
  fails on same held-out) are measured on **disjoint** training sets; K1720
  is a behavioural signature with a specified post-strip regex gate. None
  pass by algebraic identity.
- (g) K-IDs in code (`k1718_multi_beats_base_ge_3`, `k1719_single_fails_le_1`,
  `k1720_signature_rate`) measure exactly what MATH.md §Kill-criteria
  specifies: (a) count of held-out cats beaten; (b) same for single-math
  control; (c) multi signature rate AND delta vs base.

**Code ↔ math (h)–(m2) — all pass.**
- (h) No composition. No `add_weighted_adapter`, no `sum(lora_A...)`.
- (i) `LORA_SCALE = 4.0` ≤ 8 bound (line 54). Safe per antipattern-003.
- (j) No routing. Each arm is a direct adapter-vs-base eval.
- (k) No `shutil.copy` of sibling adapters. Both adapters trained fresh
  in this run (`train_multi`, `train_single` both `status: "ok"`).
- (l) `all_pass` derived from three measured booleans; no hardcoded pass.
- (m) `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md
  target. Same weights in all three arms.
- (m2) MLX idioms present: `mx.set_memory_limit`, `mx.set_cache_limit`,
  `mx.clear_cache` each step, `mx.eval`, `nn.value_and_grad(model, loss_fn)`,
  `model.freeze()` then `linear_to_lora_layers`, `mx.save_safetensors`.
  No torch-style mutation. `/mlx-dev` skill trace in MATH.md §References.

**Eval integrity (n)–(q) — all pass (with note on n).**
- (n) Base accuracy = 60.0 % (not 0 %); thinking channel active — not a
  truncation artifact. `avg_thinking_chars` is not logged per-arm here, but
  `strip_thinking` is invoked and answers are parsed (so base is emitting
  content). PAPER.md Issue 2 notes a **regex failure**, not truncation.
- (o) Headline `n_eval = 15` — at the low end, but the run is `is_smoke=true`
  so under-power is pre-declared and the verdict is already PROVISIONAL.
- (p) No synthetic padding; all 15 multi training examples come from real
  teacher traces on MMLU-Pro.
- (q) No cited baseline; base measured in same run on same held-out set.

**Deliverables (r)–(s) — all pass.**
- (r) PAPER.md §"Predictions vs measurements" table is present and complete
  (9 rows including teacher gate and signature Δ).
- (s) Math: signature-count derivation and Welch-bound approximation in
  MATH.md §"Decomposition of the LoRA delta" are standard. Training-budget
  parity assumption explicitly logged per guardrail 1007.

## Substantive observations (non-blocking, for analyst)

1. **Teacher signature gate failed (40 % < 70 %) but researcher correctly
   did NOT relax threshold.** PAPER.md flags as Issue 1 with a pre-declared
   v2 signature definition (≥ 1 marker type sufficient). This is the right
   call — guardrail 1009 forbids in-run relaxation.
2. **K1719 PASS is uninformative** because K1718 also failed. Researcher
   acknowledges this in PAPER.md §K1719. Not a review issue; captured.
3. **PAPER.md's paradox explanation is plausible** (adapter suppresses
   the `<|channel>thought` pre-amble in which most teacher markers live,
   so student-side signature collapses to 0 %). Testable in v2 via
   diagnostic: compare pre-strip and post-strip marker rates.
4. **Smoke budget (15×40) overfits**, as measured. Prediction interval in
   MATH.md (53–65 %) applies only at full scale (300 steps, 100 ex/arm).
5. **CLI gap**: `experiment complete --status provisional` does not exist;
   researcher routes via `experiment update`. Not a review blocker.

## Assumptions (per guardrail 1007)

- I treat the untracked-in-git state as equivalent to a single-version
  pre-registration — there is no prior MATH.md to swap against, so (e)
  defaults to pass.
- I accept the researcher's decision to defer K1720 signature-definition
  relaxation to a v2 experiment rather than re-measure offline from
  `data/eval_responses.jsonl`. Offline re-scoring would be tempting but
  would violate KC-lock; correct handling is v2.

## Route

Verdict = PROCEED. Smoke is clean, all artifacts present and consistent,
three methodology issues documented for v2 without KC manipulation.
Emit `review.proceed` → analyst captures LEARNINGS.md and files v2
as a follow-up experiment.
