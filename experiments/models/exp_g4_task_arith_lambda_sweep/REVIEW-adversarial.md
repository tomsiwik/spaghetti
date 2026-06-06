# REVIEW-adversarial — exp_g4_task_arith_lambda_sweep

**Verdict: KILL (preemptive-kill confirmed, 18th cohort, 17th composition-bug branch)**

## Adversarial checklist (17 items)

Consistency (a–d):
- (a) results.json `verdict="KILLED_PREEMPTIVE"` ↔ DB `status=killed` ↔ PAPER.md `Verdict: KILLED` → PASS
- (b) `defense_in_depth=true` with T1/T3/T4 = fail; K1608 = fail → consistent
- (c) PAPER.md verdict line = "KILLED (preemptive)", no PROVISIONAL/SUPPORTED language
- (d) `is_smoke` absent (runner is structural check, not training) → N/A

KC integrity (e–g):
- (e) K1608 text unchanged pre-run vs post-run (`experiment get` confirms "lambda=0.5 best within 2pp on MMLU-Pro") → no KC edit
- (f) No tautology — kill is from missing structure (inventory/SC), not algebraic identity
- (g) K-ID matches DB description

Code ↔ math (h–m2):
- (h) Runner is pure stdlib (pathlib + subprocess + json), no `sum(lora_A)` / `add_weighted_adapter` — N/A
- (i) No LORA_SCALE — N/A (no training)
- (j)–(m) No routing / shutil.copy / hardcoded-True / proxy substitution — N/A
- (m2) Skill invocation N/A (no MLX / model loading; pure stdlib structural check)

Eval integrity (n–q):
- (n)–(q) No eval run — N/A (preemptive-kill)

Deliverables (r–s):
- (r) PAPER.md has prediction-vs-measurement table (5 rows, T1–T5) → PASS
- (s) T5 runner false-negative (BitNet/MLP substring check against finding-get summary which uses "orthogonal adapters" + "d=17.2M" without literal "BitNet"/"MLP") acknowledged in PAPER §T5 runner false-negative with manual verification path; MATH-level T5 independently verifiable. Defense-in-depth T1 ∨ T3 ∨ T4 each alone blocks SUPPORTED.

## Direct verification

- T1 inventory: `ls micro/models/exp_p1_t2_single_domain_training/adapters/` = {code, math, medical} → shortfall=2 ✓
- T3 success_criteria: `experiment get` returns "Success Criteria: NONE — add with: experiment success-add" ✓
- T4 KC keywords: K1608 text has 0 of {epsilon, baseline, pooled, discipline, domain} ✓
- T5 F#164 existence: `experiment finding-get 164` returns supported result on BitNet-2B MLP orthogonal adapters, λ∈[0,0.5] (monotonic "higher values untested"), self-caveat "lambda > 0.5 not tested" — confirms scope non-transfer to K1608 λ∈{0.67, 1.0} Gemma 4 v_proj MMLU-Pro.

Defense-in-depth: T1 ∨ T3 ∨ T4 each individually block SUPPORTED. T2 (iter-budget) + T5 (F#164 non-transfer) reinforce.

## Routing signal for analyst

No new antipattern. Reinforces:
- ap-017 partial-cascade-insufficiency (17 → 18 instances; composition-bug 17 + scale-safety 1)
- ap-framework-incomplete (success_criteria=[])
- ap-scale-misclassified (F#164 BitNet-2B MLP PPL → Gemma 4 v_proj MMLU-Pro)

Register F#164 non-transfer as reusable one-line preempt (e) under ap-017, alongside (a) F#306, (b) F#13/F#14, (c) F#44, (d) F#45.

## Reviewer assumptions

- T5 "runner false-negative" is cosmetic: F#164 summary from `finding-get` does not include literal "BitNet"/"MLP" substrings, but the referenced experiment (`exp_lora_soups_cat` with d=17.2M orthogonal MLP adapters) establishes scope. Accepted as MATH-level theorem; defense-in-depth (T1/T3/T4) protects the verdict independently.
- Runner T5 regex gap is non-blocking but worth patching cohort-wide: future runners should scan the referenced experiment's MATH.md for base-model scope, not the condensed finding-get summary.
