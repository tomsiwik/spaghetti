# REVIEW-adversarial — exp_g4_5domain_real_hf

**Verdict:** KILL (ratify researcher's KILLED_PREEMPTIVE)
**Reviewer iter:** 17 (16th audit-2026-04-17 cohort preempt; 1st N=5)
**Date:** 2026-04-19

## 17-item checklist

| # | Item | Result |
|---|---|---|
| a | results.verdict (KILLED_PREEMPTIVE) vs DB status (killed) | PASS — consistent |
| b | all_pass=true ∧ status=killed | PASS — predictions confirm preempt |
| c | PAPER.md verdict line "KILLED_PREEMPTIVE" | PASS — matches DB |
| d | is_smoke=false; claim is preempt (not full-run upgrade) | PASS |
| e | KC edited post-claim? `git status`: dir untracked, no history; KCs from DB unchanged | PASS — no edits |
| f | Tautology sniff: P1–P5 measure orthogonal facts (fs, DB text, count, time, regex) | PASS |
| g | K-ID code measures DB-stated quantity? K1604/K1605 are unmeasurable as stated; runner verifies *that* unmeasurability via P1–P5 (preempt proxies, explicitly documented) | PASS |
| h | LoRA composition bugs (sum lora_A, etc.) | N/A — pure-stdlib, no model |
| i | LORA_SCALE ≥ 12 hard-coded | N/A — no training |
| j | route(val[d][0]) routing | N/A |
| k | shutil.copy mislabel | N/A |
| l | hardcoded `{"pass": True}` | PASS — passed values are computed (`len(hits)==0`, `T_total_min ≥ iter_budget_min`, etc.) |
| m | target model in MATH ≠ loaded model | N/A — no model loaded |
| m2 | skill-invocation evidence (mlx-dev/fast-mlx) | N/A — no MLX code |
| n | base acc 0% w/ thinking_chars 0 | N/A — no eval |
| o | headline n<15 | N/A — preempt, not eval |
| p | synthetic padding | N/A |
| q | cited baseline drift | N/A |
| r | prediction-vs-measurement table in PAPER.md | PASS — present (P1–P5) |
| s | math errors / unsupported claims | PASS — see spot-check below |

## 5-theorem spot-check (verified)

- **T1 (adapter-count):** `find` confirms 2 F#44-matched on disk (code, medical). math + universal-thinking are not in F#44 domain set {legal, python, creative, code, medical}. Shortfall=3. ✓
- **T2 (iter-budget breach):** T2.1 V3 mean train time 20.92 min/adapter × 3 missing = 62.76 min = 2.09× over 30-min iter budget. Under 2h micro ceiling. ✓
- **T3 (framework-incomplete):** `experiment get` returns `Success Criteria: NONE`. ✓
- **T4 (KC under-spec):** K1604/K1605 text contains 0/4 of {MMLU-Pro, GSM8K, HumanEval, PPL}. ✓
- **T5 (F#44 non-transfer):** F#44 finding-get verified: BitNet-2B-4T base, PPL-only eval, self-caveats include "train/val contamination severe", "Two domains (python, creative) show training DIVERGENCE", "PPL-only eval (no task eval)", "macro Qwen converged: cos=0.142, 142× worse". F#44 explicitly disclaims its own transfer status. ✓

Defense-in-depth: T1 ∨ T3 ∨ T5 individually block `supported`. T2 + T4 reinforce. No single theorem load-bearing.

## MATH→runner reconciliation accepted

PAPER.md §reconciliation explicitly documents shortfall MATH=2 → runner=3 (strict F#44 match). Tighter bound strengthens T1+T2; no KC change. Not a guardrail-1009 violation — runner's stricter check is internal tightening, not DB KC edit.

## Routing implications

**No new antipattern.** Reinforces:
- ap-017 partial-cascade-insufficiency (instance 15 → 16 with this confirm)
- ap-framework-incomplete
- ap-scale-misclassified (iter-budget-breach variant for N=5)
- ap-domain-count-mismatch (KC eval-task unpinned)

**Reusable preempt registered (per analyst iter 16 update):** F#44 non-transfer is the new one-line preempt for any future "5-domain real HF compose on Gemma 4" claim, compounding with T1 (F#44-matched-count) and T3 (framework-incomplete).

## Cohort-drain status (post-confirm)

**16 consecutive preemptive-kills** in audit-2026-04-17 cohort this session. First N=5 instance; prior 15 were N=25. Theorem-1 threshold scales with KC denominator; T2 shifts macro-wall-clock-breach → iter-budget-breach for smaller N. Pattern holds.

Remaining cohort (per researcher iter 20 forecast): N=25 `_vproj_compose_n25_clean`, `_tfidf_ridge_n25_clean` (T1+T2 macro+T4); N=5/14 `_compose_bakeoff_top3`, `_dare_sparsified_n5` (T1 by F#44-match-count, T3+T4+T5 unchanged). Operator unblock (success_criteria add + macro multi-adapter training OR KC re-scope to ≤T2.1-available domains) remains the only accelerator.

## Reviewer assumptions (disclosed)

- F#44 domain set inferred as {legal, python, creative, code, medical} from F#44 caveats text. PAPER.md acknowledges robustness to this assumption (T3+T4+T5 hold under any F#44 domain set).
- Iter-budget guardrail is `ralph.yml` 30-min single-iter discipline; T2 weaker than N=25 macro-ceiling-breach but still structurally blocks single-iter execution.
- No git history for MATH.md/KCs because experiment dir is untracked (.gitignore-style new dir); KC text comes from DB (immutable post-claim per guardrail 1009), so the no-edit check is satisfied by DB inspection.

## Verdict

**KILL — ratified.** DB status=killed already; finding registration + event emit only.
