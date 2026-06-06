# LEARNINGS — exp_g4_5domain_real_hf

**Verdict:** KILLED_PREEMPTIVE (16th audit-2026-04-17 cohort preempt; 1st N=5)
**Date:** 2026-04-19

## Core Finding

K1604 ∧ K1605 ("5-domain real HF LoRA adapters compose on Gemma 4 E4B, ≥4/5 own-domain improve, 0/5 degrade >3%") blocked by a 5-theorem stack before any training. Any one of {T1 adapter-count, T3 framework-incomplete, T5 F#44 non-transfer} blocks `supported` alone; T2 iter-budget breach (62.76 min = 2.09× over 30-min hat-iter) and T4 KC under-specification (0/4 eval keywords in K1604/K1605 text) reinforce.

## Why

Cohort structure, not hparams:
- **T1** inventory: `find` confirms 2 F#44-matched adapters on disk (code, medical) against F#44 domain set {legal, python, creative, code, medical}; math and universal-thinking correctly excluded. Shortfall=3.
- **T3** `experiment get` returns `Success Criteria: NONE` — `supported` vacuous by guardrail 1009.
- **T5** F#44 self-caveats (finding-get) literally say "train/val contamination severe", "Two domains (python, creative) show training DIVERGENCE", "PPL-only eval (no task eval)", "macro Qwen converged: cos=0.142, 142× worse". F#44 disclaims its own transfer status; architectural gap (BitNet-2B ternary ≠ Gemma 4 E4B RMSNorm+QK-pre-norm+MQA) + metric gap (PPL vs task-eval, r≈0.08) compound.
- First N=5 cohort instance: Theorem-1 threshold scales with KC denominator; T2 shifts macro-wall-clock-breach → iter-budget-breach (weaker but still structurally blocking).

Defense-in-depth: T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED. No single theorem load-bearing.

## Implications for Next Experiment

- **Pivot off audit-2026-04-17 cohort.** Remaining members (`_vproj_compose_n25_clean`, `_tfidf_ridge_n25_clean` N=25; `_compose_bakeoff_top3`, `_dare_sparsified_n5` N=5/14) will reproduce T1+T3+T5 on claim. Claim open P≤2 non-cohort experiments for actual runs.
- **F#44 non-transfer one-line preempt** is reusable for any future "5-domain real HF compose on Gemma 4" claim, compounding with T1 (F#44-match-count) and T3 (framework-incomplete). Already captured in ap-017 source list.
- **Operator unblock** (success_criteria add + macro 2–21-adapter training OR KC re-scope to ≤T2.1-available domains) remains the only cohort accelerator. Do NOT attempt adapter-training rescues at micro scale — they breach iter budget.
- **No new antipattern** — reinforces ap-017 (partial-cascade-insufficiency instance 15→16), ap-framework-incomplete, ap-scale-misclassified (iter-budget variant for N=5), ap-domain-count-mismatch (KC eval-task unpinned).
