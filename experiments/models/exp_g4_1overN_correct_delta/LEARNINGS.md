# LEARNINGS — exp_g4_1overN_correct_delta

**Date:** 2026-04-19
**Verdict:** KILLED_PREEMPTIVE
**Cohort drain:** 15th consecutive audit-2026-04-17 preemptive-kill; 14th
partial-cascade-insufficiency instance.

## Core Finding

K1603 ("1/N beats equal(scale=1) and additive ≥3pp at N=25 MMLU-Pro compose,
Gemma 4 E4B") is structurally unreachable under current state via 5
independent theorems: T1 adapter-count (4/25), T2 wall-clock (7.32h ≫ 2h),
T3 success_criteria=[] (vacuous "supported"), T4 MMLU-Pro pigeonhole (14<25),
T5 F#13/#14 BitNet-2B-N=5 → Gemma-4-N=25 non-transfer (F#13 UPDATE 2026-03-26
literally: "0/20 pairwise transfers >2%, the benefit is 1/N regularization
not knowledge sharing"). Any of {T1, T2, T5} alone blocks SUPPORTED.

## Why

- T2.1 V3 cascade unblocked 3 Gemma 4 adapters (math/code/medical) +
  1 universal thinking adapter = 4 on disk; K1603 fixes denominator at 25.
  Cascade is NECESSARY but NOT SUFFICIENT (ap-017 partial-cascade scope).
- F#13 UPDATE explicitly demoted the "1/N preserves gains" interpretation
  to "1/N is regularization, not knowledge sharing" — the directional +3pp
  claim has no measured basis on Gemma 4 at N=25.
- Operator approval required for macro-batch 21-adapter training (7.32h),
  per guardrail 1010 (target hardware budget) and 1008 (anti-stuck).

## Implications for Next Experiment

- **F#13/F#14 → Gemma-4-N=25 non-transfer is a reusable one-line preempt**
  for any future "1/N scaling" / "compose-catastrophe-relief" claim on
  Gemma 4 until regularization dynamics re-measured on 4-bit Gemma 4 E4B.
- Remaining N=25 cohort (`exp_g4_vproj_compose_n25_clean`,
  `exp_g4_tfidf_ridge_n25_clean`) will reproduce T1+T2+T4 — same drain
  template, no new antipattern.
- Researcher iter 20: pivot OFF audit-2026-04-17 cohort entirely. Claim
  open P≤2 non-cohort experiments for actual runs (or HALT_ESCALATE if
  none remain unblocked). Operator unblock = success_criteria addition
  + macro 21-adapter training OR KC re-scope to N ≤ 4.
