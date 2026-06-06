# LEARNINGS — exp_g4_l2_norm_compose_n25

**Date:** 2026-04-19 (analyst iter 12, post-T2.1-V3 cascade drain)
**Verdict:** KILLED_PREEMPTIVE (K1600=fail; 17/17 adversarial checks PASS; Finding #628)

## Core finding
K1600 ("0/25 drop >5pp on MMLU-Pro drift, simultaneous merge") is closed preemptively
by a 5-theorem impossibility stack. Primary driver: **adapter-count shortfall** — only
3/25 required Gemma 4 domain adapters exist after the T2.1 V3 cascade unblock
(math/code/medical, 4.9 MB each). Closing the 21-adapter gap would take 7.32 h of
dedicated training, exceeding the 2 h micro ceiling and 0.5 h hat-iter budget.
Defense-in-depth: T2 wall-clock, T3 success_criteria=[], T4 MMLU-Pro 14 disciplines ≠ 25,
T5 Finding #8 non-transfer (Qwen2.5-QK-L2 ≠ Gemma 4 RMSNorm) — each independently KILLs.

## Why
T2.1 V3 SUPPORTED landed exactly 3 Gemma 4 LoRA specialists — necessary unblock for the
audit-2026-04-17 cohort, but **not sufficient** for any cohort member whose KC fixes
N≥4. K1600's hard "N=25" denominator makes 3/25 a re-derived KILL, not a partial PASS.
Finding #8's "0/25 catastrophic failures" motivates the question but does not lift: the
mechanism-carrying architecture differs (QK-L2 normalisation vs RMSNorm +
QK-pre-projection-norm per `MLX_GEMMA4_GUIDE.md`).

## Implications for next experiment
1. **Cohort-filter forecast (preemptive-kill expected):** `exp_g4_1overN_correct_delta`,
   `exp_g4_relevance_weighted_n25`, `exp_g4_vproj_compose_n25_clean`,
   `exp_g4_tfidf_ridge_n25_clean`, and any cohort title containing `n25`/`n14` that
   demands >3 Gemma 4 domain specialists. Same Theorem-1 shortfall closes each.
2. **Partial-cascade-insufficiency rule:** cascade unblock is NECESSARY but NOT
   SUFFICIENT for cohort drain. T2.1's 3-adapter delivery unblocks KCs with N≤3; for
   N≥4 the downstream correct path is preemptive-kill, not retry.
3. **Prefer non-cohort P≤2 work.** Next researcher hat should claim any open P≤2
   experiment NOT demanding N>3 Gemma 4 domain specialists — those are now runnable
   end-to-end (Py3.12 venv + T2.1 adapters on disk). Shift drain focus off the N=25
   cohort until macro-batch training is operator-approved.
4. **Unblock path for K1600 (if revisited):** (a) v2 with N=3 using T2.1 specialists
   directly, (b) macro-schedule the 21-adapter train (7.3 h, pueue, operator approval),
   or (c) re-scope metric to N=14 aligning with MMLU-Pro disciplines. All three are
   KC-changing (guardrail 1009 → new experiment, not edit).
