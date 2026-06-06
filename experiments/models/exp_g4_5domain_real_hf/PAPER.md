# PAPER — exp_g4_5domain_real_hf

**Verdict: KILLED_PREEMPTIVE**
**Date:** 2026-04-19
**Scale:** micro (pure-fs verification; no model load; ~1 s wall-clock)
**Tags:** audit-2026-04-17, composition-bug, g4-gemma4

## Claim (as posed)

> On Gemma 4 E4B, 5-domain real HF LoRA adapters compose such that **≥4/5
> domains improve own-domain** (K1604) and **0/5 degrade base >3%** (K1605).
> Motivation: transfer Finding #44 (BitNet-2B-4T 5-domain real HF composition,
> supported 2026-03-20) to Gemma 4 E4B with MMLU-Pro / GSM8K / HumanEval eval.

## Verdict rationale (5 independent structural blocks)

Per `MATH.md`, five theorems close K1604 ∧ K1605 preemptively. No single
theorem is load-bearing alone; any two of {T1, T3, T5} suffice to block
`supported` without running Gemma 4 training.

| Theorem | Block |
|---|---|
| T1 | Adapter-count shortfall: 2 F#44-matched adapters on disk (code, medical) < 5 required. Shortfall 3. |
| T2 | Wall-clock iter-budget breach: 3 × 20.92 min = 62.76 min > 30 min iter budget (2.09× over). |
| T3 | Framework-incomplete: `success_criteria: []` — `supported` verdict vacuous. |
| T4 | KC under-specification: K1604 and K1605 text lacks eval task, base, and delta-threshold. |
| T5 | F#44 non-transfer: architectural (BitNet-2B ternary ≠ Gemma 4 RMSNorm+QK-pre-norm+MQA), metric (PPL≠task-eval, r≈0.08), and self-caveat contamination. |

## Prediction vs measurement

| P | Predicted | Measured | Match |
|---|---|---|---|
| P1 | No safetensors in exp dir | `hits=[]` | ✓ PASS |
| P2 | `success_criteria: []` | `"Success Criteria: NONE found"` | ✓ PASS |
| P3 | F#44-matched adapters < 5 | 2 matched (code, medical); shortfall 3 | ✓ PASS |
| P4 | Training missing exceeds 30-min iter budget | 62.76 min (3 missing × 20.92 min) = 2.09× over | ✓ PASS |
| P5 | K1604/K1605 text lacks MMLU-Pro/GSM8K/HumanEval/PPL | 0 eval keywords found in KC text | ✓ PASS |

**all_pass = true → verdict = KILLED_PREEMPTIVE**

## MATH → runner reconciliation

MATH.md Theorem 1 charitable estimate counted **code + medical + (math or
thinking-universal) as 2–3 of 5 F#44-matched**. Runner applied strict domain-
name matching against `F44_DOMAINS = {code, medical, python, legal, creative}`
and found exactly 2 (code, medical). **math** and **thinking-universal** were
correctly excluded as not in the F#44 domain set. Shortfall 3, tighter than
MATH.md's shortfall-2. This strengthens T1 (less of a cohort slot filled by
on-disk inventory) and strengthens T2 (3 × 20.92 = 62.76 min > MATH.md's
2 × 20.92 = 41.84 min estimate). No KC changed. All T1/T2 conclusions
re-derive at the tighter bound. No re-registration needed.

## Assumptions (disclosed, non-load-bearing)

- F#44 domains identified as {legal, python, creative, code, medical} based on
  F#44 Result line ("5 domains, rank-16 LoRA") and F#44 caveats explicitly
  naming "python" and "creative" as divergent. If the actual F#44 domain set
  differs, **T1 weakens proportional to F#44-match-count**, but T3 (framework-
  incomplete), T4 (KC under-specification), and T5 (architectural non-transfer)
  each independently block `supported`. Robust to the assumption.
- T2.1 V3 empirical adapter throughput (20.92 min/adapter mean) assumed to
  hold for the 3 missing F#44 domains. If throughput is slower for legal /
  python / creative HF datasets (plausible: larger context, different doc
  structure), T2 strengthens; if faster, T2 weakens but T1/T3/T4/T5 are
  unaffected.
- "Eval task" for K1604/K1605 left deliberately unpinned in experiment notes;
  any specific choice (MMLU-Pro vs GSM8K+HumanEval) doesn't satisfy all 5
  domains (creative has no canonical match). T4 holds under every choice.

## Relation to cohort

15th consecutive preemptive-kill in the **audit-2026-04-17** cohort this
session. Preceding kills (selected): exp_g4_l2_norm_compose_n25,
exp_g4_batched_lora_k1, exp_g4_relevance_weighted_n25, exp_g4_1overN_correct_delta.
Pattern: `audit-2026-04-17` members fail on ≥1 of {framework-incomplete,
Theorem-1-adapter-count, KC-under-specification, prior-finding-non-transfer}.

First N=5 (not N=25) cohort instance this session. Theorem-1 threshold scales
with KC denominator (5 instead of 25) but adapter inventory (2 F#44-matched on
disk) is still insufficient. Theorem-2 shifts from "macro-scale wall-clock
breach" (6–7h) to "iter-budget breach" (1 h) — weaker but still structurally
blocking. Theorem-5 pivots from F#13/F#14 (1/N) to F#44 (real HF), maintaining
non-transfer argument against a different ancestor finding.

## Cohort forecast for analyst

Remaining audit-2026-04-17 N=5 / N=25 cohort members per prior analyst iter 15
forecast will reproduce the pattern:

- N=25: `exp_g4_vproj_compose_n25_clean`, `exp_g4_tfidf_ridge_n25_clean` —
  T1 adapter-count (2–4/25) + T2 macro-wall-clock breach + T4 KC under-spec.
- N=5 / N=14: any `exp_g4_compose_bakeoff_top3`, `exp_g4_dare_sparsified_n5`,
  etc. — T1 proportional to F#44-match-count, T3+T4+T5 unchanged.

Operator unblock (add `success_criteria` + approve macro 2–21-adapter training
OR re-scope KCs to ≤ T2.1-available domains) remains the only cohort
accelerator.

## Routing signal

No new antipattern. Reinforces: ap-framework-incomplete, ap-017 partial-
cascade-insufficiency (instance 15), ap-scale-misclassified (iter-budget-breach
variant for N=5), ap-domain-count-mismatch (KC eval-task unpinned).

F#44 non-transfer argument is the reusable one-line preempt for any future
"5-domain real HF compose on Gemma 4" claim, compounding with T1 adapter-count
and T3 framework-incomplete.
