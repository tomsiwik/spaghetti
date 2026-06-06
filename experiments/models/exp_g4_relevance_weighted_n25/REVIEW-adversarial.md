# REVIEW-adversarial — exp_g4_relevance_weighted_n25

**Verdict:** KILL (preemptive)
**Date:** 2026-04-19
**Reviewer iter:** 14 (post-cascade drain, 14th consecutive audit-2026-04-17 preemptive-kill)

## 17-item adversarial checklist

| # | Item | Status | Evidence |
|---|---|---|---|
| a | `results.json["verdict"]` vs DB status | PASS | verdict=KILLED_PREEMPTIVE ↔ DB status=killed |
| b | `all_pass` vs claim | PASS | all_pass=true refers to preemptive-predictions P1–P5 all passing (i.e. kill is confirmed). Not claiming supported. |
| c | PAPER.md verdict line | PASS | "Verdict: KILLED_PREEMPTIVE" at top |
| d | `is_smoke` | PASS | false |
| e | KC git-diff | PASS | DB K1602 text "diff >= 5pp" unchanged (fresh exp dir; no MATH.md git history modifying threshold; preemptive-kill is not a KC edit) |
| f | Tautology sniff | PASS | Five independent theorems, none algebraic identities |
| g | KC-in-code vs MATH/DB | PASS | K1602 "diff >= 5pp relevance-weighted vs equal-weight compose at N=25, MMLU-Pro" — runner doesn't measure it, correctly declares untestable |
| h | Buggy composition code | N/A | Runner is pure-fs predicate verifier (no `add_weighted_adapter`, no `sum(lora_A)`) |
| i | `LORA_SCALE>=12` | N/A | no LoRA code |
| j | Per-sample routing | N/A | no routing |
| k | `shutil.copy` stub adapters | N/A | no adapter creation |
| l | Hardcoded `{"pass": True}` | N/A | predicates computed from disk/subprocess |
| m | Model mismatch | N/A | no model loaded |
| m2 | Skill invocation | N/A | pure-stdlib + `subprocess("experiment get")`; no MLX calls → /mlx-dev not required |
| n–q | Eval integrity | N/A | no model eval; preemptive-kill |
| r | Prediction-vs-measurement table | PASS | PAPER.md §"Prediction vs Measurement" 5/5 PASS |
| s | Math errors / unsupported claims | PASS | see theorem spot-check below |

## Five-theorem spot-check

- **T1 (adapter-count):** ✓ Inventory verified on disk — `micro/models/exp_p1_t2_single_domain_training/adapters/{code,math,medical}/adapters.safetensors` (4,999,229 B each, mtime Apr 19 04:46, T2.1 V3 window). 3 specialist + 1 universal = 4 < 25. Shortfall 21.
- **T2 (wall-clock):** ✓ Arithmetic: (1352.7+840.0+1572.8)/3 = 1255.2s = 20.92 min; 21×20.92 = 439.3 min = 7.32 h. 7.32 h / 2 h = 3.66× over micro ceiling; / 30 min = 14.64× over iter budget. Matches l2_norm_compose_n25 (iter 16) T2 number — same empirical basis.
- **T3 (success_criteria=[]):** ✓ `experiment get exp_g4_relevance_weighted_n25` confirms "Success Criteria: NONE". ap-framework-incomplete applies.
- **T4 (MMLU-Pro pigeonhole):** ✓ 14 disciplines (Wang 2024, arxiv:2406.01574 Table 2). 25 > 14 → min 11 collisions. "Relevance-weighted over disjoint domains" is ill-defined under collision.
- **T5 (F#137 non-transfer):** ✓ `finding-get 137` confirms Status=supported, +9.34pp, r=0.990, 2026-03-28, PPL-probe mechanism. BitNet-2B architecture (ternary, different norm + attention) ≠ Gemma 4 E4B (RMSNorm + QK-pre-projection norm + MQA per MLX_GEMMA4_GUIDE.md). PPL-probe calibration is quantization-sensitive; r=0.990 has never been measured on 4-bit Gemma 4. Transfer basis unestablished.

## Defense-in-depth

Three load-bearing arguments (any one sufficient): T1 counting, T2 wall-clock, T5 architectural non-transfer.
Two supporting (not load-bearing alone but reinforce): T3 framework-incomplete, T4 MMLU-Pro pigeonhole (both preclude a clean "supported" even after 21-adapter training).

## Assumptions / judgment calls

- Accepted ap-017 (partial-cascade-insufficiency scope addendum, 11th→14th instance) per analyst iter 13 registry update. No new antipattern required.
- Accepted `all_pass=true` semantic: true = "all preemptive-kill predictions fired" = kill is complete. Not a supported-claim; PAPER.md and DB status=killed agree.

## Routing implications

14th consecutive cohort preemptive-kill. Remaining N=25 cohort members (`_1overN_correct_delta`, `_vproj_compose_n25_clean`, `_tfidf_ridge_n25_clean`) will reproduce the same T1+T2+T4 kill trio on claim. Operator unblock (macro-batch 21-adapter training + KC success_criteria additions) remains the only accelerator. Analyst should cite ap-017 partial-cascade-insufficiency + ap-framework-incomplete, not create new antipatterns.

## Verdict: KILL (preemptive). Register finding, emit review.killed.
