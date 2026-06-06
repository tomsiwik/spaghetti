# LEARNINGS.md — P11.Z0: Plan-and-Solve Prompting

**Status**: KILLED (K1529 FAIL)
**Date**: 2026-04-14

---

## Core Finding

Plan-and-Solve (PS) prompting provides zero improvement over direct-answer on thinking-enabled Gemma 4 E4B: P1_ps = -2.9pp, P2_ps_plus = -1.1pp vs P0_direct. The best prompt was no prompt — direct answer wins.

## Why

Wang et al. (arXiv:2305.04091) designed PS for models WITHOUT extended thinking (GPT-3/4 completions). For thinking-enabled models, planning is already executed internally via the thinking mechanism before the first response token. External PS instructions compete with, not complement, native planning. The planning capacity is bounded by T (thinking), not I (instruction).

## Implications for Next Experiment

1. Use **direct-answer format** in all P11 benchmark evaluations (s1K, LIMO, GRPO). Never add PS prompt overhead.
2. P0 drift (-6.7pp vs Finding #530) is unresolved — exp_p11_baseline_eval must re-establish MMLU-Pro+thinking baseline with careful thinking engagement tracking.
3. Wang et al. PS gains do not transfer to reasoning models. Any future prompt-engineering hypothesis targeting thinking models must account for this.
