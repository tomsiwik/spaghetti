# REVIEW — exp_g4_energy_vs_ridge_routing

**Verdict**: KILL (already `status=killed` in DB; this review ratifies.)

**One-liner**: 10th consecutive audit-2026-04-17 cohort precondition-probe KILL.
P1/P2/P3 all FAIL on independently verified disk/DB state. K1588 correctly
UNMEASURABLE. No fabricated delta, no heavy MLX burned.

## Independent verification

| Probe | Claim | Verified by reviewer |
|---|---|---|
| P1 | 0 safetensors, 3 `adapter_config.json` stubs in `{code, math, medical}` under `exp_p1_t2_single_domain_training/adapters/` | `ls` confirms 3 stub-only dirs, 0 `.safetensors` |
| P2 | `exp_g4_ridge_routing_n25_mcq` upstream verdict=KILLED | results.json shows `verdict: KILLED`, K1616 measured=0.839 < threshold 0.9 |
| P3 | `energy_gap_topk_routing` base_model is not Gemma 4 | results.json shows `"model": "microsoft/BitNet-b1.58-2B-4T"` — wrong base per antipattern |

## Adversarial checklist (17 items)

Consistency (a)–(d): PASS — results.json KILLED, all_pass=false, PAPER.md
KILLED, `is_smoke=false`, `precondition_probe=true`.

KC integrity (e)–(g): PASS — MATH.md pre-registered K1588 and the
P1/P2/P3 FAIL→KILLED rule before any run. No relaxation. UNMEASURABLE is
the honest outcome (no tautology).

Code↔math (h)–(m2): N/A — `run_experiment.py` is a pure file-existence/
JSON-read probe. No composition code, no `sum(lora_A)`, no
`LORA_SCALE`, no `shutil.copy`, no routing, no model load. m2 skill
invocation N/A because no MLX code is executed; MATH.md explicitly cites
the cohort standing rule for skipping heavy work.

Eval integrity (n)–(q): N/A — no accuracy metric is computed. Probe
wall-time 0.003 s.

Deliverables (r)–(s): PASS — PAPER.md has the prediction-vs-measurement
table, documents assumptions (rule 1007), and cites the cohort's 9 prior
blocking findings plus Finding #182 motivation.

## Assumptions logged

1. Claim-queue mismatch: analyst's prior `learning.complete` asked for
   out-of-cohort picks, but `experiment claim` auto-returned a cohort
   member and no unclaim flag exists. Probing + KILLing is the cheapest
   honest response (0.003 s vs 2–4 h wasted heavy run). Logged in both
   MATH.md §Assumptions and PAPER.md §Assumptions.
2. P3 "not Gemma 4" call is conservative: `energy_gap_topk_routing` was
   run on BitNet. Any older-base AUC would still fail the
   `base_model contains gemma-4` rule. No fabrication.

## Reviewer action

- Finding registered for Ralph audit trail.
- No DB state change: `status=killed`, K1588 `fail` already set. Calling
  `experiment complete` again would duplicate evidence — declined.
- Routing: `review.killed` → analyst for LEARNINGS.md.

## Non-blocking observation (orchestrator)

This is the 10th KILL on the same upstream. Cohort is 100 % saturated.
Analyst should repeat the prior learning.complete escalation: promote
`exp_p1_t2_single_domain_training` retrain (LORA_SCALE=5,
max_tokens ≥ 512, 5 disjoint domains, rank sweep {2,4,6,12,24},
grad-SNR logging) to first-class blocking task, and filter the
`audit-2026-04-17` tag out of the claim queue until it lands.
