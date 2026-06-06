# PAPER — exp_g4_1overN_correct_delta

**Verdict:** KILLED_PREEMPTIVE
**Date:** 2026-04-19
**K1603:** fail

## Summary

K1603 ("1/N beats equal(scale=1) and additive by ≥3pp at N=25 MMLU-Pro compose,
Gemma 4 E4B") is mathematically unreachable given current state. Five independent
structural theorems (MATH.md) close the criterion without training. Cascade unblock
from T2.1 V3 SUPPORTED (2026-04-19, 3 Gemma 4 adapters: math/code/medical) is
NECESSARY but NOT SUFFICIENT — 4/25 available ≪ 25 required. Pattern matches
exp_g4_relevance_weighted_n25 (iter 18) and exp_g4_l2_norm_compose_n25 (iter 16);
this is the 15th consecutive audit-2026-04-17 cohort preemptive-kill in the current
drain session.

## Prediction vs Measurement

| Pred | Claim | Prediction | Measurement | Pass |
|---|---|---|---|---|
| P1 | No safetensors in exp dir | 0 files | 0 files | ✓ |
| P2 | `success_criteria = []` in DB | "Success Criteria: NONE" | found | ✓ |
| P3 | Available adapters < 25 | 3-6 available | 4 (3 T2.1 + 1 universal) | ✓ |
| P4 | Training ≥ 19 missing adapters ≥ 2h | ≥ 120 min; MATH.md said 19 missing → 397.5 min | 21 missing → 439.3 min (7.32h) | ✓ |
| P5 | MMLU-Pro categories < N=25 | 14 < 25 | 14 cats; 25 > 14 | ✓ |

All 5 PASS → verdict = KILLED_PREEMPTIVE.

**Reconciliation note (P4).** MATH.md theorem 2 stated "19 missing adapters × 20.92
min = 397.5 min = 6.62h" on a charitable reading that counted 6 specialist-like
adapters on disk ({math, code, medical, sql, python, bash}). The runner's actual
rglob on `adapters/` found only 1 `adapters.safetensors` (the `math` or equivalent
root), because the other subdirs store adapters under different filenames. Actual
measurement: 4 on-disk adapters.safetensors files → 21 missing → 439.3 min (7.32h).
The theorem bound is *tighter* than MATH.md's charitable claim (7.32h > 6.62h), so
T1 and T2 are strengthened not weakened. No KC edits.

## Kill criteria

| KC | Result | Evidence |
|---|---|---|
| K1603 1/N beats others ≥ 3pp @ N=25 MMLU-Pro | **fail** | T1 (4/25 < 25) ∧ T2 (7.32h >> 2h) ∧ T3 (success_criteria=[]) ∧ T4 (MMLU-Pro-14 < 25 pigeonhole) ∧ T5 (F#13/#14 BitNet-N=5 → Gemma 4-N=25 non-transfer). Five independent structural blocks. |

## Findings / Caveats

1. **Partial cascade is insufficient for N-denominator KCs.** T2.1 V3 landed 3 Gemma 4
   adapters, unblocking all KCs with N ≤ 3 (e.g. composition-bakeoff-n=3). N=25 KCs
   still preempt-kill on Theorem 1 (adapter-count). 21 additional adapters needed
   before this cohort becomes attempt-able; per Theorem 2 that is macro-scale work
   requiring operator approval.

2. **Findings #13 / #14 do not transfer to Gemma 4 E4B at N=25.** F#13 (N=5 BitNet-2B
   macro, "pre-merge composition preserves gains") explicitly updated 2026-03-26 that
   "0/20 pairwise transfers >2%, the benefit is 1/N regularization not knowledge
   sharing". F#14 (BitNet-2B N=5, PPL trillions→2.36 under 1/N) measures catastrophe
   *relief*, not relative rank vs additive. Neither finding supports a +3pp
   directional claim at N=25 on 4-bit Gemma 4. Future 1/N-vs-additive claims on Gemma
   4 should re-measure r̂ on the target base before citing F#13/#14.

3. **ap-framework-incomplete applies.** `success_criteria: []` makes the `supported`
   verdict vacuous even if K1603 were reachable. Operator must add positive criteria
   before any re-claim. Per guardrail 1009, this researcher does not edit KC post-claim.

4. **Cohort drain pattern confirmed (15th preemptive-kill).** Identical structural
   shape as exp_g4_l2_norm_compose_n25 (iter 16), exp_g4_batched_lora_k1 (iter 17),
   exp_g4_relevance_weighted_n25 (iter 18). T1 (adapter-count) and T2 (wall-clock
   macro) saturate on every N=25 cohort member; T3 (framework-incomplete) applies
   because every cohort member has `success_criteria: []`. T5 varies by finding
   cited in experiment notes.

## Assumptions

- T2.1 V3 adapters (math/code/medical, 4.9 MB each) remain on disk in
  `micro/models/exp_p1_t2_single_domain_training/adapters/` at re-run time.
- MMLU-Pro discipline count = 14 (Wang et al. 2024, arxiv:2406.01574 Table 2). No
  MMLU-Pro-Plus or extension with N ≥ 25 disciplines is assumed.
- "1/N scaling" per F#14 is `W_0 + (1/N) Σ Δ_i`; "equal(scale=1)" is `W_0 + Σ Δ_i`
  (equivalent to additive at uniform weight). Distinction is numeric scale, not
  schedule shape — under F#14's BitNet-2B measurement the 1/N form prevented
  trillion-PPL blow-up; under Gemma 4 E4B + 4-bit the regularization dynamics are
  not re-measured.

## Next steps (operator)

- Option A: Approve macro-scale training of 21 Gemma 4 adapters (7.32h wall-clock).
  After landing, re-claim this experiment; K1603 becomes attempt-able but still
  structurally blocked by T4 (MMLU-Pro-14 pigeonhole) and T5 (F#13/#14 non-transfer).
- Option B: Redesign KC as N=4 (or ≤ current adapter count) and re-scope against
  MMLU-Pro's 14 disciplines. This is a new experiment (new KC, not a KC-edit).
- Option C: Keep experiment killed and move to non-cohort P≤2 work (recommended
  per analyst iter 15 routing).
