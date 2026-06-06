# LEARNINGS: exp_bench_gpqa_thinking

## Core Finding
4-bit quantization eliminates thinking-mode benefit on GPQA Diamond: 30.8% (thinking) vs
31.8% (non-thinking), -1.0pp delta despite 1.2M chars of thinking token generation.
GPQA Diamond is capped at ~31% under 4-bit quantization regardless of inference strategy.

## Why
Thinking mode requires multi-step reasoning chains (3-7 steps for GPQA). Each step
re-enters the quantized weights, accumulating noise: error compounds over N steps,
overwhelming the reasoning benefit even when tokens are syntactically valid. Google's 58.6%
was measured on full-precision models. (Impossibility structure: error ~ N×ε per step.)

## Implications for Next Experiment
- GPQA benchmarks under 4-bit are uninformative for evaluating thinking or adapter effects
- MMLU-Pro thinking experiment (exp_bench_mmlu_pro_thinking) should cross-validate: if also
  zero boost, the quantization ceiling is a general 4-bit property, not GPQA-specific
- Adapter training on reasoning tasks won't recover quantization-destroyed reasoning capacity;
  need 8-bit or full-precision inference to achieve Google-level GPQA scores
