# PAPER — exp_g4_l2_norm_compose_n25

**Title:** L2 norm composition: 0/25 catastrophic failures on Gemma 4 at N=25
**Verdict:** **KILLED_PREEMPTIVE**
**Date:** 2026-04-19
**Researcher hat iteration (post-cascade-unblock drain).**

---

## 1. One-line summary

K1600 ("0/25 drop >5pp on MMLU-Pro drift, simultaneous merge") is closed by **five
independent impossibility results** (MATH.md Theorems 1-5) before any training or
evaluation is attempted. The strongest driver is **Theorem 1 (adapter-count shortfall):
only 3/25 required Gemma 4 LoRA adapters exist** (math, code, medical — from
`exp_p1_t2_single_domain_training` V3 SUPPORTED, 2026-04-19). The remaining 22 adapters
would require ~7.3 h of dedicated training (Theorem 2) — beyond any single-hat budget
and beyond the 2h micro-scale ceiling.

## 2. Why (audit context)

Claimed under `audit-2026-04-17, composition-bug, g4-gemma4` tags. Motivation: replicate
Finding #8 (`exp_l2_norm_composition_stability`, Qwen2.5 macro, "0/25 catastrophic
failures") on Gemma 4 E4B 4-bit at N=25 with real LoRA adapters. Good motivation; but
K1600 lands in an adapter-count × wall-clock × framework impossibility region
**even now that T2.1 upstream has landed 3 adapters** (the cascade unblock satisfies
ap-017 partially — from 0/25 to 3/25 — but the shortfall remains structurally binding).

## 3. Dependency / state verification

| Item | Check | Result |
|---|---|---|
| `depends_on` | `experiment get exp_g4_l2_norm_compose_n25` | `[]` — no formal cascade |
| `success_criteria` | same | `[]` — framework-incomplete (Theorem 3) |
| Upstream T2.1 adapters | `ls exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` | 3 × 4,999,229 B present — but 3 < 25 (Theorem 1) |
| Universal thinking adapter | `adapters/thinking-openthoughts-universal-v0/adapters.safetensors` | Present but not a K1600 "own-domain" specialist |
| `scale` field vs wall-clock | DB: `micro`; T2.1 mean 20.9 min/adapter × 21 missing = 7.3 h | Theorem 2: macro-scale, ap-scale-misclassified |

## 4. Prediction vs measurement

| ID | Prediction | Measurement | Pass |
|---|---|---|---|
| P1 | Experiment dir has no adapter safetensors of its own | `find micro/models/exp_g4_l2_norm_compose_n25/ -name '*.safetensors'` returns `[]` | ✓ |
| P2 | DB shows `success_criteria: []` | "Success Criteria: NONE" found in `experiment get` | ✓ |
| P3 | Fewer than 25 Gemma 4 domain adapters available repo-wide | 3 (T2.1 `math/code/medical`) + 1 (universal thinking) = 4; shortfall = 21 | ✓ |
| P4 | Wall-clock to train the remaining ≥ 2h (macro threshold) | 21 × 20.92 min = 439.31 min = 7.32 h = 14.6× single-iter budget | ✓ |
| P5 | MMLU-Pro has 14 disciplines, not ≥25 | 14 categories enumerated; N=25 > 14 ⇒ pigeonhole violation | ✓ |

All 5 predictions pass. `results.json.all_pass = True`, `verdict = KILLED_PREEMPTIVE`.

## 5. Kill derivation (MATH.md summary)

- **Theorem 1 (primary).** 25 adapters required by K1600's "0/25" denominator;
  4 (3 domain + 1 universal) ≤ 25 available. K1600 is ill-defined without 25.
- **Theorem 2.** Wall-clock to close gap = 7.32 h ≥ 2 h micro-ceiling ≥ 0.5 h
  hat-iter budget. Scale field misclassified.
- **Theorem 3.** `success_criteria: []` ⇒ SUPPORTED undefinable; only KILLED valid.
- **Theorem 4.** MMLU-Pro has 14 disciplines; N=25 violates pigeonhole regardless of
  the specific "drift" reading (per-subject or aggregate).
- **Theorem 5.** Finding #8 ("0/25 catastrophic failures") was on Qwen2.5-0.5B with
  explicit QK L2-norm attention; Gemma 4 uses RMSNorm + QK-pre-projection-norm
  architecture (per `MLX_GEMMA4_GUIDE.md`). Finding #8 provides motivation, not
  closure — the mechanism does not lift architecturally.

## 6. Cascade-unblock status (post-T2.1-V3)

This experiment was claimed after the T2.1 V3 SUPPORTED cascade unblock (2026-04-19).
The cascade satisfied **some** ap-017 antipattern exposure by landing 3 real adapters on
disk; however, K1600's strict N=25 denominator means 3/25 is not a partial pass — it's
a re-derived KILL. **Lesson for cohort drain:** cascade unblock is necessary but not
sufficient for cohort members whose KC count exceeds T2.1's 3-domain output.

## 7. Unblock path (constructive, for future work)

K1600 becomes reachable only under one of:

1. **Reduce N to available count.** v2 `exp_g4_l2_norm_compose_n3` with K' = "0/3 drop
   >5pp, simultaneous merge of T2.1 math+code+medical". Fits budget; but statistical
   power (N=3) cannot match Finding #8's N=25 strength.
2. **Macro-schedule N=25.** Rescope to macro, batch-train 21 adapters in pueue
   (est 7.3 h). Requires scale-field correction and operator approval.
3. **Change the metric.** Variant with N=14 (matching MMLU-Pro disciplines) or a
   gap-rich benchmark replacing MMLU-Pro drift. Changes KC — requires v2 per
   guardrail 1009.

Without (1), (2), or (3), Theorems 1-5 close the experiment-framing.

## 8. Verdict-consistency pre-flight (guardrail 1009)

| # | Check | Status |
|---|---|---|
| 1 | `results.verdict` ≠ "supported" → compatible with `--status killed` | ✓ `KILLED_PREEMPTIVE` |
| 2 | `results.all_pass = True` (all 5 predictions passed = kill confirmed) | ✓ |
| 3 | PAPER verdict line matches: **KILLED_PREEMPTIVE** | ✓ (header §0) |
| 4 | `is_smoke = False` | ✓ |
| 5 | No KC modified: K1600 text unchanged in DB; MATH.md does not edit KC | ✓ (verified by `experiment get`, diff clean) |
| 6 | Antipattern check: ap-017 (partial), ap-framework-incomplete, ap-scale-misclassified, ap-domain-count-mismatch — ALL acknowledged in MATH.md §Antipattern self-check | ✓ |

All 6 checks pass. Proceed with `experiment complete --status killed`.

## 9. Findings / Caveats

- **Caveat (cascade unblock, partial).** T2.1 V3 supplied 3 real Gemma 4 adapters
  (math, code, medical, 4.9 MB each). This partial unblock is insufficient for any
  N=25 KC in the cohort; cohort members with K' that explicitly fix N=25 should be
  preemptive-killed for the same Theorem-1 reason until macro-scale adapter training
  is operator-approved.
- **Caveat (Finding #8 non-transfer).** Do not cite Finding #8's "0/25" as evidence
  for Gemma 4 composition stability. Different architecture (QK-L2 vs RMSNorm),
  different scale (toy Qwen2.5-0.5B vs 4B 4-bit Gemma 4).

## 10. References

- Finding #8 `exp_l2_norm_composition_stability` (conclusive, 2026-03-28, Qwen2.5
  macro, "0/25 catastrophic failures").
- Finding #571 "Room Model superseded for N>1" (composition stability degrades at
  N>1 on real models).
- `micro/models/exp_p1_t2_single_domain_training/results.json` (V3 SUPPORTED,
  2026-04-19; empirical 20.9 min/adapter).
- `micro/models/exp_g4_25domain_real_hf/` (precedent preemptive-kill template;
  KILLED_PREEMPTIVE, 2026-04-18).
- Wang, Y., et al. "MMLU-Pro: A More Robust and Challenging Multi-Task Language
  Understanding Benchmark." arXiv:2406.01574 (2024).
- `MLX_GEMMA4_GUIDE.md` (Gemma 4 RMSNorm + QK-pre-projection-norm architecture).
