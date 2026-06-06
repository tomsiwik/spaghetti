# LEARNINGS.md — exp_hedgehog_behavior_adapter_conciseness_impl

**Verdict:** PROVISIONAL (smoke iter, F#666 carve-out — K#1966 deferred to `_full`)
**Date:** 2026-04-25 (drain-window iter ~94, analyst pass)
**Finding registered:** F#789 (provisional)

## Core Finding

K#1965 deterministic length reduction PASSes at 26.17% (base mean=256.0 tokens, student mean=189.0, n=8 held-out). Real measurement on tokenizer counts — **not** heuristic regex collapse. Proxy cos=0.957; Phase B loss 0.164→0.039 (4.2× over 30 steps). Adapter `adapters/hedgehog_concise_r8/` (84 modules, r=8, scale=6.0, v_proj+o_proj). 26.17% is a **lower bound** because base hit max_tokens=256 ceiling on 8/8 prompts.

## Why

Conciseness is the 1st **deterministic-proxy** entry in the Hedgehog _impl cluster. Token count from the same tokenizer is a numerical measurement, structurally distinct from the K2-collapse antipattern (`mem-antipattern-proxy-target-stage-mismatch`, 3-instance promoted at formality_impl iter ~67) where K2 was a heuristic regex judge that collapsed to `heuristic_only`. K#1965 escapes that failure mode by construction: the kill-criterion is arithmetic, not interpretive. Strong cos sim on attention output also confirms structural acquisition before behavioral measurement — an ordering F#783/F#784/F#786 each respected.

## Implications for Next Experiment

- **`_full` priority lifted:** raise `max_tokens` to ≥1024 for both base and student so the K#1965 ceiling artifact (8/8 base capped) is removed; expect a larger Δ. Add K#1966 (MMLU 100-question subset) for F#666 target half. Budget 3-5h M5 Pro 48GB; no API key required.
- **Pair K#1965 with a quality target:** "concise but wrong" failure mode is invisible to token count alone. `_full` should add a correctness-preservation target — either MMLU non-interference (already planned as K#1966) or per-prompt LLM-as-judge correctness on the 8 student outputs.
- **Deterministic-proxy preference encoded:** when designing future _impl KCs, prefer arithmetic measurements (token count, edit distance, perplexity-on-fixed-text) over heuristic regex judges. `mem-antipattern-proxy-target-stage-mismatch` is the negative pattern; conciseness K#1965 is the positive escape.
- **`linear_to_lora_layers` shim is now 4-deep recurrence** (politeness/refactor/formality/conciseness). Manual fallback works (84 modules attached, training converges every time) so it is a code-quality fix-pattern, not a kill-trigger. Promoting as antipattern to surface in future _impl scaffolding.
