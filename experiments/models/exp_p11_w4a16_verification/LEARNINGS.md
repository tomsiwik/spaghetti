---
experiment: exp_p11_w4a16_verification (P11.K1)
status: pending-results (pueue task 4, queued)
updated: 2026-04-14
---

# LEARNINGS: W4A16 Quantization Verification

## Core Finding (pre-results)
Design verified: W4A16 lossless reasoning theorem (arXiv:2504.04823) predicts <5pp gap between 4-bit and 8-bit MMLU-Pro+thinking. This directly answers whether the 7.3pp gap to Google's 69.4% is quantization-induced or not.

## Why This Matters
From Finding #530 (exp_bench_mmlu_pro_thinking): 4-bit Gemma 4 hits 62.1% MMLU-Pro with thinking; Google reports 69.4% for the same model. If W4A16 gap <5pp, quantization is NOT the bottleneck — the gap must come from adapter quality, prompting, or distribution shift. If gap ≥5pp, switch to 8-bit base.

## Key Design Decisions
- **Correct thinking token regex**: `<|channel>thought.*?<channel|>` — fixes s1K Phase 4a bug (wrong `<think>` regex gave 12.5% with 0 thinking chars)
- **Clean memory teardown**: `del model, tokenizer` + `mx.metal.clear_cache()` + `mx.eval()` between loads — safe on 48GB M5 Pro
- **280 questions, seed=42** — matches exp_bench_mmlu_pro_thinking for direct comparison

## What to Watch for in Results
- **K1540 FAIL (expected)**: gap < 5pp → quantization not the bottleneck → CLoQ + prompting experiments remain the right levers
- **K1540 PASS (unexpected)**: gap ≥ 5pp → switch to 8-bit base, accept +2.7GB memory for ~5pp reasoning gain
- **Cross-check**: W4A16 Phase 2 (4-bit, correct regex) should match s1K Phase 4a re-eval. If they diverge, investigate seed/data drift.

## Implications for Next Experiment
If K1540 FAIL: confirms adapter training quality is the gap — LIMO (higher-quality reasoning traces) and GRPO (RL refinement) are the correct next steps.
If K1540 PASS: architectural pivot to 8-bit base before any further adapter training.
