# LEARNINGS — exp_g4_relevance_weighted_n25

**Verdict:** KILLED_PREEMPTIVE (14th audit-2026-04-17 cohort preempt)

## Core Finding
K1602 (diff ≥ 5pp relevance-weighted vs equal-weight compose, N=25, MMLU-Pro,
Gemma 4 E4B) unreachable. 5 theorems close without training. 5/5 P1–P5 PASS.

## Why
Load-bearing (any one suffices):
- T1 adapter-count: 4/25 (3 T2.1 V3 + 1 universal); shortfall 21.
- T2 wall-clock: 21×20.92 min = 7.32 h >> 2h micro >> 30 min iter.
- T5 F#137 non-transfer: BitNet-2B PPL-probe r=0.990 never measured on 4-bit
  Gemma 4 (RMSNorm + QK-pre-proj + MQA ≠ BitNet arch).

Reinforcing (preclude "supported" even with 21 trained):
- T3 `success_criteria=[]` → vacuous supported.
- T4 MMLU-Pro 14 disciplines < 25 → min 11 collisions break disjointness.

## Implications
- Reusable preempt: F#137 non-transfer blocks any "PPL-probe relevance-weighted
  compose on Gemma 4" claim until r re-measured on 4-bit Gemma 4.
- Remaining N=25 cohort (`_1overN_correct_delta`, `_vproj_compose_n25_clean`,
  `_tfidf_ridge_n25_clean`) will reproduce T1+T2+T4 on claim. Pivot to
  non-cohort P≤2 work.
- Operator unblock: add success_criteria, approve macro 21-adapter train, or
  re-scope KCs to N ≤ adapter count. None available to researcher.
- Antipattern: reinforces ap-017 (partial-cascade-insufficiency, instance 12)
  + ap-framework-incomplete. No new antipattern.
