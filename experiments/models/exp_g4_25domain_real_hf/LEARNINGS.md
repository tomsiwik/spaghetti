# LEARNINGS — exp_g4_25domain_real_hf

**Status:** KILLED_PREEMPTIVE (reviewer-endorsed, 2026-04-18)
**K1606:** FAIL (≥10pp MMLU-Pro on 25 Gemma 4 LoRA adapters, structurally unreachable)

## Core Finding

**Gemma 4 4B × basic LoRA × MMLU-Pro is a closed-region for ≥10pp gains.**
Any proposal whose success requires a ≥10pp own-domain MMLU-Pro delta via r≤8
q_proj-only LoRA on Gemma 4 4B lands inside F#478's impossibility region and
must be killed at design time, not ran.

## Why

Five independent theorems close K1606 pre-flight (MATH.md Thm 1–5), any one of
which is individually decisive:
1. **F#478 closure (primary):** no exploitable knowledge gap for basic LoRA on
   Gemma 4 4B at advanced eval.
2. **F#442 δ_format ≈ 0:** MMLU-Pro baseline 56–88%; no format-rescue lane like
   F#424's 22–82pp (which ran on 4% baselines).
3. **`success_criteria: []`:** framework-incomplete, SUPPORTED undefinable.
4. **Wall-clock 8.7h:** 17.4× single-iteration budget.
5. **Pigeonhole 25 > 14:** can't 1:1 map to MMLU-Pro's 14 disciplines.

## Implications for Next Experiment

- **Reusable closure rule:** preemptively kill any experiment matching
  `{base=Gemma4-4B} × {method=basic LoRA r≤8 q_proj} × {eval=MMLU-Pro} × {ask=Δ≥10pp}`.
- **Unblock path requires structural change** (not hparams): swap base (Qwen3
  gap-rich), or swap eval (proprietary/domain-specific, not MMLU-Pro), or swap
  data (advanced-subdomain, not HF basics), or swap N (14 aligned to disciplines
  with explicit success criteria).
- **Feasibility constant** (reusable): 20.88 min/adapter r=6 q_proj on Gemma 4
  E4B 4-bit; multiply by N for macro-sizing.
- **Backlog drain:** review `audit-2026-04-17` remaining P=1 entries for same
  closure-region membership before claim — likely yields more preemptive kills.
