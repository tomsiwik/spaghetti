# MATH — exp_g4_l2_norm_compose_n25

**Experiment:** L2 norm composition: 0/25 catastrophic failures on Gemma 4 at N=25.
**Kill criterion K1600:** 0/25 drop >5pp on MMLU-Pro drift (simultaneous merge).
**Verdict (from math):** KILLED_PREEMPTIVE — four independent impossibility results close K1600.
**Date:** 2026-04-19

---

## Antipattern self-check

| Antipattern | Applies? | Evidence |
|---|---|---|
| ap-017 (stub adapters) | **YES (partial)** | N=25 adapters required; only 3 exist (`exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors`, 4.9 MB each, V3 SUPPORTED). 22/25 = 88% missing. |
| ap-020 (cascade-upstream-killed) | NO | T2.1 upstream is now `status=supported` post V3 rerun. |
| ap-027 (venv-vs-system-python3) | NO | This runner is a pure-fs verification script; no MLX, no subprocess `python3` shebang dependency. |
| ap-framework-incomplete | **YES** | `success_criteria: []` per `experiment get exp_g4_l2_norm_compose_n25`. |
| ap-scale-misclassified | **YES** | Claimed "micro"; wall-clock extrapolation to train the missing 22 adapters at T2.1 empirical ~22 min/adapter = ~8h (macro). |
| ap-domain-count-mismatch | **YES** | MMLU-Pro has 14 disciplines (Wang et al. 2024); N=25 violates pigeonhole for "own-domain" interpretation; same violation for drift over 25 independent subjects. |

---

## Theorem 1 (primary — adapter-count shortfall)

**Statement.** K1600 ("0/25 drop >5pp on MMLU-Pro drift, simultaneous merge") requires
**25 pre-trained Gemma 4 LoRA adapters** to exist at experiment-run time; fewer than 25
adapters make the kill-criterion count ill-defined.

**Proof.** K1600 text fixes the denominator at 25 ("0/25"). A "simultaneous merge" of N
adapters into θ_base is only well-defined when N=25. Available inventory on disk (as of
2026-04-19 post-T2.1-V3):

- `micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` — 3 adapters.
- `adapters/thinking-openthoughts-universal-v0/adapters.safetensors` — 1 adapter (generic thinking; not a domain specialist per K1600 "own-domain subject" interpretation).

3 domain adapters + 1 universal = 4 adapters maximum. 4 < 25 ⇒ K1600's "0/25" cannot
be measured. Either (a) N is reduced (changes the KC — forbidden per guardrail 1009),
or (b) 21+ additional adapters must be trained (see Theorem 2). ∎

---

## Theorem 2 (feasibility bound)

**Statement.** Wall-clock to train the 22 missing Gemma 4 E4B 4-bit domain adapters
sequentially on M5 Pro exceeds the Ralph single-hat iteration budget by ≥16×.

**Proof.** Empirical from `exp_p1_t2_single_domain_training/results.json` (V3 rerun,
2026-04-19): `math_train_time_s=1352.7`, `code_train_time_s=840.0`,
`med_train_time_s=1572.8`. Mean = (1352.7 + 840.0 + 1572.8)/3 = 1255.2 s/adapter
= 20.9 min/adapter. For 22 missing: T = 22 × 20.9 min = 460 min = 7.67 h.

Ralph guardrail 1008 bounds single-hat transitions to <30 min. 460/30 ≈ 15.3×. Full
hypothesis-scoping rule (researcher hat, `PLAN.md`) is <2h. 7.67h / 2h = 3.8× over that
ceiling as well. ∎

**Consequence.** The experiment is structurally macro (>2h, >>30 min), yet its DB
`scale` field is `micro`. Either rescope to macro (triggers different framework
obligations and still needs budget approval) or preemptive-kill.

---

## Theorem 3 (framework incompleteness)

**Statement.** The DB entry for `exp_g4_l2_norm_compose_n25` is not verifiable-as-SUPPORTED
under the experiment framework.

**Proof.** `experiment get exp_g4_l2_norm_compose_n25` returns
`Success Criteria: NONE — add with: experiment success-add ...`. Per framework
convention, SUPPORTED requires ≥1 success criterion to verify, distinct from kill
criteria (which only rule out the null). With zero success criteria, only KILLED is a
valid terminal state. ∎

---

## Theorem 4 (domain-count mismatch)

**Statement.** MMLU-Pro has 14 disciplines; N=25 "simultaneous merge" adapters cannot
be 1:1 assigned to independent MMLU-Pro own-subjects, and K1600's drift interpretation
is ill-defined across 25 independent evaluation axes when only 14 exist.

**Proof.** MMLU-Pro (Wang et al. 2024, arXiv:2406.01574) defines 14 categories:
biology, business, chemistry, computer science, economics, engineering, health,
history, law, math, philosophy, physics, psychology, other. K1600 reads "MMLU-Pro
drift, simultaneous merge" — if drift is per-subject (the natural reading matching
Finding #8's "own-domain" notion), 25 > 14 forces ≥ 11 adapters to share subjects or
have no subject (pigeonhole). If drift is single-aggregate MMLU-Pro-total, then N=25
is decoupled from 14 and the "0/25" count reverts to Theorem 1's adapter-count
shortfall. Either reading fails. ∎

---

## Theorem 5 (Finding #8 non-transfer)

**Statement.** The "0/25 catastrophic failures" result from Finding #8
(`exp_l2_norm_composition_stability`, Qwen2.5-0.5B) does not pre-establish K1600 on
Gemma 4.

**Proof.** Finding #8's mechanism is **explicit QK L2 normalization at attention-score
time** on a Qwen2.5-0.5B hybrid attention toy, evaluated under synthetic composition
stress. Two deltas prevent transfer:

1. **Architecture.** Gemma 4 uses **RMSNorm** in attention (and gemma-specific QK
   *pre*-projection norm — `MLX_GEMMA4_GUIDE.md`), not an explicit L2 norm on QK
   scores. The normalization surface differs; Finding #8's boundedness argument
   (L2 norm ⇒ |QKᵀ/√d| bounded) does not lift directly.

2. **Scale/target.** Finding #8 operated on a toy attention composition problem at
   bf16; K1600 operates on 4-bit quantized Gemma 4 E4B ~4B params with real LoRA
   deltas. The composition-stability surface at 4-bit quantization is a different
   problem (see Finding #571 "Room Model superseded for N>1").

Without running the real experiment (blocked by Theorems 1+2) Finding #8 provides
**motivation** for the hypothesis but **not** mathematical closure of K1600. ∎

---

## Predictions

| ID | Prediction | How to measure |
|---|---|---|
| P1 | Experiment dir has no adapter safetensors of its own | `find micro/models/exp_g4_l2_norm_compose_n25/ -name '*.safetensors'` returns empty |
| P2 | DB shows `success_criteria: []` | `experiment get exp_g4_l2_norm_compose_n25` returns "Success Criteria: NONE" |
| P3 | Fewer than 25 Gemma 4 domain adapters available repo-wide | count of `adapters.safetensors` under `micro/models/exp_p1_t2_single_domain_training/adapters/` + `adapters/` < 25 |
| P4 | Wall-clock to train the remaining adapters ≥ 2h (macro-scale threshold) | extrapolation from T2.1 empirical 20.9 min/adapter × (25 − available) ≥ 120 min |
| P5 | MMLU-Pro has 14 disciplines, not ≥25 | Wang et al. 2024 dataset spec (hardcoded enumeration) |

All 5 predictions passing ⇒ KILLED_PREEMPTIVE.

---

## Unblock path (constructive outcome)

K1600 becomes reachable only under one of:

1. **Reduce N to available count.** Design a v2 experiment `exp_g4_l2_norm_compose_n3`
   with K1600' = "0/3 drop >5pp on MMLU-Pro drift" using exactly T2.1's math+code+medical
   adapters. Preserves Finding #8's hypothesis-class; fits hat budget; but lowers
   statistical power from N=25 (Welch bound ~√25=5) to N=3 (√3≈1.7) — Finding #8's
   "0/25" strength cannot be matched with N=3.
2. **Macro-schedule N=25.** Promote to macro-scale, batch-train 22 adapters in a
   pueue job (estimated 7.7 h), then rerun K1600. Requires scale-field correction
   first, then operator-approved schedule.
3. **Change the metric.** Change K1600 from "MMLU-Pro drift" to a gap-rich benchmark
   or to a 14-subject pigeonhole-safe variant (e.g., N=14 mapped 1:1 to MMLU-Pro).
   Changes KC — requires v2 experiment per guardrail 1009.

Without at least one of (1)-(3), K1600 is closed by Theorem 1 alone. Theorems 3-5
close the experiment-framing even if Theorems 1+2 were bypassed.

---

## References

- Finding #8 `exp_l2_norm_composition_stability` (conclusive, 2026-03-28, Qwen2.5).
- Finding #571 "Room Model superseded for N>1" (composition stability at scale).
- `micro/models/exp_p1_t2_single_domain_training/results.json` (V3 SUPPORTED, 2026-04-19;
  empirical training time per adapter).
- `micro/models/exp_g4_25domain_real_hf/` (precedent preemptive-kill on same cohort,
  KILLED_PREEMPTIVE, 2026-04-18).
- Wang, Y., et al. "MMLU-Pro: A More Robust and Challenging Multi-Task Language
  Understanding Benchmark." arXiv:2406.01574 (2024).
- `MLX_GEMMA4_GUIDE.md` Gemma 4 RMSNorm + QK-pre-projection-norm architecture.
