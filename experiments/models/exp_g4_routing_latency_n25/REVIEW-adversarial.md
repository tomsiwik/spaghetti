# REVIEW-adversarial.md — exp_g4_routing_latency_n25

## Verdict
**KILL** (already `status=killed` in DB). 16th-position cohort precondition-probe KILL (researcher numbered as 15th in their iter-8 entry; this is the 16th REVIEW in the cohort if we include the probe series — sticking with researcher's numbering: **15th**).

## One-line reason
17/17 adversarial checks PASS or N/A. Pre-registered tripwire fired correctly: P1=0/25 Gemma 4 adapters, P2 upstream T2.1 verdict=killed all_pass=false (K1030 ✗), P3 router gated on P1. K1597 UNMEASURABLE — researcher correctly killed without fabricating a latency number.

## Adversarial checklist

| # | Check                                          | Result |
| - | ---------------------------------------------- | ------ |
| a | results.json verdict ↔ DB status               | PASS — both KILLED |
| b | all_pass ↔ claim                               | PASS — false / killed |
| c | PAPER verdict line                             | PASS — "KILLED" (no PROVISIONAL) |
| d | is_smoke vs full-run claim                     | PASS — is_smoke=false, no headline number |
| e | KC drift (K1597 added/relaxed post-run)        | PASS — pre-registered, no git diff |
| f | Tautology sniff                                | PASS — P3 hard-codes `False` but documented as "existence alone insufficient; gated on P1" in MATH.md; not algebraic |
| g | K-ID quantity match                            | PASS — K1597 = latency ratio in code/MATH/DB |
| h | Composition math in code                       | N/A — probe-only, no composition |
| i | LORA_SCALE ≥ 12                                | N/A — no scale in code |
| j | Single-sample routing                          | N/A — no routing in code |
| k | shutil.copy of sibling adapter                 | N/A |
| l | Hardcoded `{"pass": True}`                     | PASS — pass derived from probe results |
| m | Model proxy substitution                       | N/A — probe-only, no model load |
| m2| Skill invocation evidence                      | N/A — probe-only, no MLX kernels |
| n | Base 0% / thinking truncation                  | N/A — no eval |
| o | Headline n < 15                                | N/A — no headline |
| p | Synthetic padding                              | N/A |
| q | Cited baseline drift                           | N/A |
| r | PAPER prediction-vs-measurement table          | PASS — table present, all rows |
| s | Math errors / unsupported claims               | PASS — tripwire theorem correct |

## Independent verification
- P1: `find ... -name '*.safetensors' | wc -l` → **0** across all 4 candidate dirs. ✓
- P2: `experiment get exp_p1_t2_single_domain_training` → **Status: killed**, K1030 ✗. ✓
- P3: `ls` on 2 router candidates → only `exp_p1_c0_composition_port_gemma4/run_experiment.py` exists; `exp_g4_tfidf_ridge_n25_clean/` missing. ✓

## Cohort context (non-blocking, for analyst hat)
- Cohort now **15/15** saturated (Findings #605/#606/#608/#610/#611/#612/#613/#615/#616/#617/#618/#619/#620/#621 + this one).
- Same single upstream blocker: `exp_p1_t2_single_domain_training` rerun at LORA_SCALE=5, max_tokens≥512, ≥5 disjoint domains, rank sweep, grad-SNR logging.
- **Stale active entries** (researcher iter-8 side-observation, independently verified relevance): orchestrator MUST address — see analyst hat for triage. Specifically `exp_g4_cot_vs_direct_mmlu_pro` is SUPPORTED in commit 4bc99ab but DB still active; two `exp_followup_*` entries claimed since 2026-04-18.

## Assumptions
- The hard-coded `return False` in `probe_p3_router()` is honest pre-registered behavior (P3 documented as "either lacks G4 wiring or depends on P1"), not a tautology bypass. If a router with both Gemma 4 wiring AND independence from P1 ever lands, this probe needs an update — but at probe stage with P1=0/25, the gating is structurally correct.

## Routing
DB already `status=killed` — do NOT call `experiment complete` again. Register Finding #622 and emit `review.killed` → analyst.
