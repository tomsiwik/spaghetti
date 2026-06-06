# PAPER — exp_g4_relevance_weighted_n25

**Verdict:** KILLED_PREEMPTIVE
**Date:** 2026-04-19
**K1602:** fail

## Summary

K1602 ("diff >= 5pp relevance-weighted vs equal-weight compose at N=25, MMLU-Pro,
Gemma 4 E4B") is mathematically unreachable given current state. Five independent
structural theorems (MATH.md) close the criterion without training. Cascade unblock
from T2.1 V3 SUPPORTED (2026-04-19, 3 Gemma 4 adapters: math/code/medical) is
NECESSARY but NOT SUFFICIENT — 4/25 available ≪ 25 required. Pattern matches
exp_g4_l2_norm_compose_n25 (iter 16), this is the 14th consecutive audit-2026-04-17
cohort preemptive-kill in the current drain session.

## Prediction vs Measurement

| Pred | Claim | Prediction | Measurement | Pass |
|---|---|---|---|---|
| P1 | No safetensors in exp dir | 0 files | 0 files | ✓ |
| P2 | `success_criteria = []` in DB | "Success Criteria: NONE" | found | ✓ |
| P3 | Available adapters < 25 | 3-4 available | 4 total (3 T2.1 + 1 universal) | ✓ |
| P4 | Training 21 missing adapters ≥ 2h | ≥ 120 min | 439.3 min (7.32h) | ✓ |
| P5 | MMLU-Pro categories < N=25 | 14 < 25 | 14 cats; 25 > 14 | ✓ |

All 5 PASS → verdict = KILLED_PREEMPTIVE.

## Kill criteria

| KC | Result | Evidence |
|---|---|---|
| K1602 diff ≥ 5pp relevance-weighted vs equal-weight | **fail** | T1 (3/25 < 25) ∧ T2 (7.32h >> 2h) ∧ T3 (success_criteria=[]) ∧ T4 (MMLU-Pro-14 < 25 pigeonhole) ∧ T5 (F#137 BitNet → Gemma 4 non-transfer). Five independent structural blocks. |

## Findings / Caveats

1. **Partial cascade is insufficient for N-denominator KCs.** T2.1 V3 landed 3 Gemma 4
   adapters, unblocking all KCs with N ≤ 3 (e.g. composition-bakeoff-n=3). N=25 KCs
   still preempt-kill on Theorem 1 (adapter-count). 21 additional adapters needed
   before this cohort becomes attempt-able; per Theorem 2 that is macro-scale work
   requiring operator approval.

2. **Finding #137 does not transfer.** F#137 established PPL-probe relevance-weighting
   at +9.34pp on BitNet-2B (different architecture, different quantization). Gemma 4
   E4B uses RMSNorm + QK-pre-projection norm + MQA; the PPL-probe oracle signal's
   r=0.990 on BitNet has never been measured on 4-bit Gemma 4. Future cites of F#137
   as a transfer basis should read the Gemma-4 ladder as unproven.

3. **ap-framework-incomplete applies.** `success_criteria: []` makes the `supported`
   verdict vacuous even if K1602 were reachable. Operator must add positive criteria
   before any re-claim. Per DB KC integrity rule (guardrail 1009), this researcher
   does not edit KC post-claim.

4. **Cohort drain pattern confirmed.** Identical preemptive-kill shape as
   exp_g4_l2_norm_compose_n25 (iter 16) and exp_g4_batched_lora_k1 (iter 17). 14th
   cohort preemptive-kill in session.

## Assumptions

- T2.1 V3 adapters (math/code/medical, 4.9 MB each) remain on disk in
  `micro/models/exp_p1_t2_single_domain_training/adapters/` at re-run time.
- MMLU-Pro discipline count = 14 (Wang et al. 2024, arxiv:2406.01574 Table 2). No
  MMLU-Pro-Plus or extension with N ≥ 25 disciplines is assumed.
- "Relevance-weighted" per F#137 uses PPL-probe per-adapter; no alternative signal
  (activation, logit_diff) was re-tried — F#137 reported those with r=0.023 and
  r=-0.245 respectively (essentially noise).

## Next steps (operator)

- Option A: Approve macro-scale training of 21 Gemma 4 adapters (7.32h wall-clock).
  After landing, re-claim this experiment; K1602 becomes attempt-able but still
  structurally blocked by T4 (MMLU-Pro-14 pigeonhole) and T5 (F#137 non-transfer) —
  operator should reconsider whether N=25 on MMLU-Pro is the right measurement.
- Option B: Redesign KC as N=4 (or ≤ current adapter count) and re-scope against
  MMLU-Pro's 14 disciplines. This is a new experiment (new KC, not a KC-edit).
- Option C: Keep experiment killed and move to non-cohort P≤2 work (recommended
  per analyst iter 14 routing).
