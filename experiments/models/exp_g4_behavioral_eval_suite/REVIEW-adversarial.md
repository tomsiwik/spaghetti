# REVIEW-adversarial — exp_g4_behavioral_eval_suite

**Verdict: KILL** (pre-registered precondition probe; DB already `status=killed`).

13th consecutive `audit-2026-04-17` cohort probe-KILL, same single upstream
blocker (`exp_p1_t2_single_domain_training` rerun) as Findings #605, #606,
#608, #610, #611, #612, #613, #615, #616, #617, #618, #619.

## 17-check adversarial audit

| # | Check | Result | Evidence |
|---|---|---|---|
| a | results.json verdict vs DB status | PASS | `verdict=KILLED` matches `status=killed` |
| b | `all_pass` vs claim | PASS | `all_pass=false` matches KILLED |
| c | PAPER verdict line | PASS | "**Verdict: KILLED**" — no provisional/partial |
| d | `is_smoke` flag | PASS | `is_smoke=false` (full probe) |
| e | MATH.md KC git-diff | PASS | Dir untracked; no post-run KC mutation |
| f | Tautology sniff | PASS | Probe is pure file-existence; no algebraic identity |
| g | K-ID quantity match | PASS | K1593 maps AUC ≥ 0.85 across 4 benchmarks; code returns FAIL because UNMEASURABLE |
| h | Composition bug pattern | N/A | No composition, no `add_weighted_adapter`, no `sum(lora_A` |
| i | LORA_SCALE ≥ 12 | N/A | No MLX, no LoRA scale |
| j | Single-sample routing | N/A | No routing |
| k | shutil.copy adapter mislabeling | N/A | No adapter creation |
| l | Hardcoded `{"pass": True}` | PASS | All `pass` fields computed from probe output |
| m | Target model vs loaded | N/A | No model loaded |
| m2 | MLX skill invocation | N/A | No MLX code in probe |
| n | base=0 + thinking=0 eval artifact | N/A | No eval ran |
| o | n < 15 | N/A | Probe not eval |
| p | Synthetic padding | N/A | No eval |
| q | Cited baseline drift | N/A | No baseline cited |
| r | Prediction-vs-measurement table | PASS | Present in PAPER.md with 4 rows + K1593 |
| s | Math errors / unsupported claims | PASS | UNMEASURABLE is honest; no fabricated AUC |

17/17 PASS or N/A.

## Independent verification

- `ls micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/`
  → each contains **only** `adapter_config.json`, 0 safetensors. P1 genuinely
  FAILs.
- `find ... -name "*.safetensors"` → zero results. Consistent with `n_safetensors=0`.
- results.json `benchmarks_wired`: mmlu_pro=true, gsm8k/humaneval/medmcqa=false.
  Matches grep of candidate runners.
- P3 binds correctly to P1 (no adapters → no per-sample labels).
- DB `experiment get` shows `status: killed`, evidence `K1593 FAIL: AUC UNMEASURABLE`.

## Assumptions

- ap-017 (fabricated metric from missing upstream) is the dominant risk class
  for this cohort; honoring the pre-registered tripwire and refusing to run
  MLX on absent adapters avoids it.
- No double-complete: DB already `status=killed`. Skipping `experiment complete`
  to avoid overwriting the existing kill evidence.

## Non-blocking note (orchestrator)

Queue still returning cohort members after 6 analyst escalations. This review
does not attempt to fix that — orchestrator claim-queue filter on
`tag=audit-2026-04-17` or first-class promotion of the upstream rerun remains
the real fix. Flagging for analyst's next `learning.complete`.

## Route

- `experiment finding-add` → register #620 (13th cohort KILL, same upstream).
- Emit `review.killed`.
