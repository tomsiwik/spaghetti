# PAPER.md — exp_g4_e2e_mmlu_pro_thinking

## Verdict
**KILLED_PREEMPTIVE.** K1618 (beats 62.1% MMLU-Pro thinking baseline)
is structurally unreachable via six independent theorems (see MATH.md).
16th P11/g4-adjacent preemptive kill in the audit-2026-04-17 sweep;
8th confirmed instance of antipattern-017 (stub adapters consumed).

## 1. Claim
The full E2E Pierre pipeline on Gemma 4 E4B 4-bit — ridge router over
domain adapters, delta-sum composition of top-k selected adapters,
thinking mode enabled at decode — beats the 62.1% MMLU-Pro base+thinking
baseline (Finding #536).

## 2. Why this is closed pre-flight
Six mutually independent kill drivers, any one of which forces K1618
FAIL:

1. **Stub adapters (ap-017 #8).** 5 of 5 adapter paths in
   `adapters/{math,bash,python,sql,medical}` contain only
   `adapter_config.json` + `tokenizer_config.json`, no safetensors.
   Registry-pointed paths
   (`micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/`)
   are also stub-only. Under Theorem 1 (MATH.md), the pipeline
   forward pass y = x + Σ α_i (B_i A_i x) reduces to y = x. Δ = 0.
2. **Cascade upstream open (ap-020).** Domain-adapter training
   (`exp_p1_t2_single_domain_training`) has **status=open**; it has
   never produced weights.
3. **Cascade upstream killed (ap-020).** Ridge routing at N=25
   (`exp_g4_ridge_routing_n25_mcq`) is **status=killed** as of
   2026-04-18: K1616 FAIL at test_acc=0.8387 vs target 0.90 (F#502
   hidden-state ridge ties TF-IDF at 83.9% vs 84.2%).
4. **F#536 suppression.** Any non-thinking-trained LoRA adapter + thinking
   mode = 50.4% = -11.7pp vs baseline. Linearity preserves suppression
   under delta-sum (Theorem 2, MATH.md).
5. **F#560 unsolved sub-problem.** The only thinking-compatible
   training attempt (math+code 2000 ex) produced -14.5pp on MMLU-Pro.
   Theorem 3: sum of negative components is negative.
6. **F#478 knowledge-gap closure.** Gemma 4 4B has no exploitable
   knowledge gap for basic rank-6 LoRA on advanced 10-option MCQ.
   MMLU-Pro falls in the closure region.

Additionally: K1618 text specifies no threshold, no MDE, no n
(framework-incomplete per PLAN.md #1009).

## 3. Predictions (pre-flight verification)
All six predictions from MATH.md verified; see `results.json`.

| # | Prediction | Measurement | Pass |
|---|------------|-------------|------|
| P1 | 0/5 local domain adapters have safetensors | stub_count=5 | ✓ |
| P2 | 0/3 registry-pointed adapters have safetensors | stub_count=3 | ✓ |
| P3 | `exp_p1_t2_single_domain_training` status ≠ supported | status=open | ✓ |
| P4 | `exp_g4_ridge_routing_n25_mcq` status = killed | status=killed | ✓ |
| P5 | F#536 anchors 62.1% baseline + -11.7pp adapter drag | confirmed | ✓ |
| P6 | DB `success_criteria` empty | len=0 | ✓ |

## 4. Method
Pre-flight filesystem + DB checks; no model load, no evaluation. The
theorems in MATH.md close K1618 via derivation; the pre-flight verifies
the premises of those theorems.

## 5. Results
`results.json` — verdict=KILLED_PREEMPTIVE, all_pass=true, is_smoke=false.
K1618 result=fail.

## 6. Dependency state table

| Stage | Experiment | Status | Blocks E2E? |
|-------|------------|--------|-------------|
| 1 | Base+thinking measurement | F#536 supported | no (anchor exists) |
| 2a | Domain adapter training | `exp_p1_t2_single_domain_training` open | **yes** |
| 2b | Thinking-compatible domain training | `exp_p11_thinking_adapter_universal` killed (F#560) | **yes** (open research) |
| 3 | Ridge router at N | `exp_g4_ridge_routing_n25_mcq` killed (K1616) | **yes** |
| 4 | Delta-sum composition | depends on 2+3 | **yes** (cannot run) |

Three of four stages are unbuilt or killed. A 4-stage series pipeline
with three broken links has failed end-to-end.

## 7. Salvageable sub-findings
- **Delta-sum preserves mode-suppression (Theorem 2).** Linear
  composition propagates thinking-suppression; if a single summand
  suppresses thinking, Σ suppresses thinking. Design-time closure
  rule. Distinct from F#536 (empirical, single adapter).
- **Pipeline cascade-closure rule.** Stage-wise unreached stages in a
  series composition close the outcome. Candidate pattern-level
  antipattern (generalizes beyond this experiment).

## 8. Unblock path — P11.HARNESS rebuild
1. Produce 5 domain adapters with non-trivial safetensors trained
   with `enable_thinking=True` (solving F#560's open question).
2. Verify per-adapter Δ ≥ 0 vs base+thinking on MMLU-Pro (condition
   for delta-sum to be a positive operator).
3. Rebuild ridge router at the same N, test_acc ≥ 0.90 (K1616
   re-attempt).
4. Fix K1618 spec: threshold, MDE, n.

Until stage (1)-(2) is empirically resolved, K1618 remains
unreachable by derivation, not just by measurement.

## 9. Related work / DB references
- F#536 — MMLU-Pro thinking baseline 62.1% (exp_bench_mmlu_pro_thinking).
- F#478 — Gemma 4 4B knowledge-gap closure (exp_p4_b1_hard_question_eval).
- F#560 — thinking-universal math+code -14.5pp (exp_p11_thinking_adapter_universal).
- F#502 — TF-IDF ≈ hidden-state ridge at N=25 (exp_g4_ridge_routing_n25_mcq).
- Prior cascade kills in this sweep (14 P11/g4-adjacent, 7 prior ap-017
  instances): `exp_followup_composition_correct_delta`,
  `exp_followup_routing_multi_sample_ppl`,
  `exp_followup_competitive_gsm8k_200n`, `exp_g4_routed_beats_base_think`,
  `exp_g4_25domain_real_hf`, et al.

## 10. Open threads for Analyst
- Bump antipattern-017 from 7 → 8 confirmed instances (this is the 8th
  in 2 days).
- Promote Theorem 2 (delta-sum preserves mode-suppression) to a
  standalone closure-rule finding. Cites F#536 as empirical anchor;
  distinct contribution is the **linearity extension** covering
  sum-composed operators.
- Promote Theorem 5 (pipeline cascade-closure) to a
  pattern-level antipattern candidate. First explicit statement of the
  rule; M0/C0 prior kills used the same logic implicitly.
- `current_direction.md` still says "remaining P=1 open:
  exp_p1_t5_user_local_training, exp_g4_e2e_mmlu_pro_thinking" —
  after this iteration only `exp_p1_t5_user_local_training` remains
  P=1.
