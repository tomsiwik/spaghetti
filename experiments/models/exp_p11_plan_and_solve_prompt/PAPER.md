# PAPER.md — P11.Z0: Plan-and-Solve Prompting on MMLU-Pro

**Experiment**: exp_p11_plan_and_solve_prompt  
**Date run**: 2026-04-14  
**Status**: KILLED (K1529 fails)

---

## Prediction vs Measurement

| Metric | Prediction | Measured | Pass? |
|--------|-----------|----------|-------|
| P0_direct MMLU-Pro + thinking | ~62.1% (Finding #530) | **55.4%** | — (drift: -6.7pp) |
| Best prompt MMLU-Pro ≥ 64.1% | ≥64.1% | **55.4%** (P0_direct) | **K1529 FAIL** |
| PS+ ≥ PS accuracy | True | **54.3% > 52.5%** | **K1530 PASS** |
| Token ratio (best/P0) ≤ 2.0× | ≤2.0× | **1.00×** (best=P0) | **K1531 PASS** |

---

## Results by Prompt

| Prompt | Accuracy | Avg Tokens | Delta vs P0 |
|--------|----------|------------|-------------|
| P0_direct | 55.4% (155/280) | 567 | — |
| P1_ps (Plan-and-Solve) | 52.5% (147/280) | 909 | -2.9pp |
| P2_ps_plus (PS+) | 54.3% (152/280) | 896 | -1.1pp |

**Best prompt**: P0_direct (direct answer, no plan-and-solve template)

---

## Key Finding: Thinking Subsumes Plan-and-Solve

Plan-and-Solve instructions provided **zero improvement** over direct answer. P1_ps actually degraded by 2.9pp. P2_ps_plus degraded by 1.1pp.

This confirms the pre-registered failure mode from the adversarial review: **Gemma 4 E4B with extended thinking already performs internal step decomposition**. The PS prompt instruction is redundant — the model's thinking mechanism handles planning automatically.

---

## P0 Drift (-6.7pp): Critical Observation

P0_direct + thinking = 55.4% in this run vs 62.1% in Finding #530.

**Possible causes** (in order of likelihood):
1. **Different question sample**: MMLU-Pro has 12K+ questions; different random seed → different difficulty distribution
2. **Thinking engagement**: No `avg_thinking_chars` tracked in this experiment; unclear if thinking engaged consistently across all 280 questions
3. **Prompt format interaction**: PS experiment used slightly different system prompt structure vs baseline eval

**Implication**: The P0_direct baseline in this run is unreliable as a comparison point. K1530/K1531 results stand, but the -6.7pp P0 drift must be disclosed in any downstream reference.

---

## Delta Analysis (delta_vs_P0, not delta_vs_Finding530)

Since P0_direct = 55.4% in this run:
- P1_ps delta vs P0: -2.9pp (PS hurts accuracy)
- P2_ps_plus delta vs P0: -1.1pp (PS+ approximately neutral but slightly hurt)

Using **delta_vs_P0** (as specified in REVIEW-adversarial.md): PS never matches P0, let alone improves it by 2pp.

---

## Kill Verdict

- K1529: **FAIL** — best = 55.4% (need ≥64.1%). Prediction refuted.
- K1530: **PASS** — PS+ (54.3%) > PS (52.5%). Consistent with Wang et al. finding that PS+ > PS.
- K1531: **PASS** — best prompt is P0_direct; ratio = 1.0×.

**Status: KILLED** — Plan-and-Solve prompting provides no improvement on thinking-enabled Gemma 4 E4B.

---

## Impossibility Structure

Plan-and-Solve was designed for models WITHOUT extended thinking (Wang et al. tested on GPT-3/4 completions). For models with explicit thinking chains (Gemma 4, DeepSeek-R1), the planning step is already executed internally before the response token is generated. External PS instructions compete with, rather than complement, the model's native planning.

**Impossibility theorem (informal)**: For a model M with thinking mechanism T that generates planning tokens before answer tokens, PS instructions I provide no information gain over T. The planning capacity is bounded by T, not I.

---

## Implications

1. **Do NOT use PS prompts** in downstream P11 benchmark evaluations (s1K, LIMO, GRPO comparisons). Use direct-answer format.
2. **P0 drift needs investigation**: Re-establish MMLU-Pro baseline in exp_p11_baseline_eval with careful thinking tracking.
3. **Wang et al. result bounded**: PS gains from arXiv:2305.04091 do not transfer to thinking-enabled models. Hypothesis killed.
