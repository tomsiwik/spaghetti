# REVIEW-adversarial — exp_g4_compose_multiseed_cv

**Verdict:** KILL (confirm)
**Reviewer:** reviewer hat, 2026-04-19
**Wall:** ≤ 5 min, 5 tool calls.

## One-line reason
12th consecutive cohort `audit-2026-04-17` precondition-probe KILL. K1590
UNMEASURABLE (P1/P2/P3 all fail). DB already `status=killed`; artifacts
consistent end-to-end.

## Adversarial checklist (17 items)

| # | Check | Result |
|---|---|---|
| (a) | results.json verdict=KILLED ↔ DB status=killed | PASS |
| (b) | all_pass=false ↔ status=killed | PASS |
| (c) | PAPER.md verdict line = KILLED ↔ DB status=killed | PASS |
| (d) | is_smoke=false, status=killed (no upgrade attempt) | PASS |
| (e) | MATH.md git-diff: file untracked single snapshot, K1590 unchanged | PASS |
| (f) | Tautology sniff: K1590 = CV across seeds; not algebraic identity | PASS |
| (g) | K-ID 1590 in code measures the same precondition gate as MATH/DB | PASS |
| (h) | Composition bug grep (sum lora_A / linear merge): no composition code | N/A |
| (i) | LORA_SCALE ≥ 12 hardcoded: probe loads no model | N/A |
| (j) | per-sample routing on single sample: no routing | N/A |
| (k) | shutil.copy adapter masquerade: none | N/A |
| (l) | hardcoded `{"pass": True}`: probe `pass` derives from real file checks | PASS |
| (m) | target model ≠ loaded model: no model loaded (probe-only) | N/A |
| (m2) | skill invocation evidence: no MLX needed (file-existence + JSON read) | N/A |
| (n) | base acc=0% w/ thinking=0: no eval | N/A |
| (o) | n<15: probe, not headline eval | N/A |
| (p) | synthetic padding: no synthetic adapters | N/A |
| (q) | drifted cited baseline: not used | N/A |
| (r) | PAPER.md prediction-vs-measurement table | PASS |
| (s) | math errors / unsupported claims | PASS |

## Independent verification

- `experiment get exp_g4_compose_multiseed_cv` → `Status: killed`,
  evidence string already records K1590 fail.
- `find micro/models -path '*/seed*/*.safetensors'` → **0 files**. P1 genuinely fails.
- `cat micro/models/exp_p1_t2_single_domain_training/results.json` → `verdict=KILLED`,
  `all_pass=false`, no `lora_scale` field. P2 genuinely fails.
- P3: no landed cohort MMLU-Pro baseline (whole cohort is probe-killed).

## Theorem audit (MATH.md)

Lipschitz chain `|M(f_{s₁}) − M(f_{s₂})| ≤ L · ||W_{s₁} − W_{s₂}||` is correct as
stated. Conclusion is honest: KC reduces to an *empirical* variance question, not
a theorem-level guarantee — appropriate for a reproducibility CV.

## Cohort context

12th consecutive `audit-2026-04-17` cohort precondition-probe KILL.
Prior chain: Findings #605/#606/#608/#610/#611/#612/#613/#615/#616/#617/#618.
All gate on the same upstream:
`exp_p1_t2_single_domain_training` rerun at LORA_SCALE=5, max_tokens≥512, rank
sweep `{2,4,6,12,24}`, grad-SNR per layer logged, 5+ disjoint domains.

## Routing

No double-complete (DB already `status=killed`). Emit `review.killed` →
analyst hat. Register Finding via `experiment finding-add`.

## Assumptions

- Took the 0 seed-safetensor file count at the repo level as authoritative
  for P1 (no seeds train under any cohort dir today).
- Treated `_audit_note` + `_reconstruction_note` blob in the upstream
  results.json as the canonical reason for the upstream KILL — same as the
  prior 11 cohort REVIEWs.
- Did not attempt to relitigate the upstream rebuild scope here; that
  belongs to the analyst hat's escalation channel.

## Non-blocking orchestrator note

Cohort claim queue still returning members after 12 KILLs. The fix is *not*
analyst-hat escalation (it has not changed queue behavior over 4 iterations);
it is an `experiment claim` filter on `tag=audit-2026-04-17` until the upstream
rebuild lands. Out of scope for the reviewer hat — flagged for the analyst.
