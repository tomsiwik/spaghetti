---
experiment: exp_p1_t4_serving_v2
status: supported
finding: 503
---

# LEARNINGS — T4.3v2: MLX Adapter Serving

## Core Finding
MLX hot-swap adapter serving is production-ready: true swap cost ~1ms, zero graph recompilation penalty, and 3.2% decode degradation — all well within thresholds.

## Why
MLX lazy evaluation and the `load_weights()` + `mx.eval()` pattern replace adapter tensors in-place without triggering graph retrace. The forward pass takes the same ~13.5ms before and after swap (Hu et al. 2021, arxiv:2106.09685 for LoRA overhead model).

## Critical Insight: stream_generate vs raw forward
`stream_generate()` adds ~106ms constant overhead per call (tokenizer, sampler, Python generator) — this is API cost, not swap cost. The initial T4.3 loophole feared 122ms swap latency; 106ms of that is unavoidable generation API overhead, completely independent of adapter swapping. True swap cost: 0.25ms load + 0.73ms eval = ~1ms.

## Prediction Gap
MATH.md predicted 0.6% decode overhead from FLOP analysis; actual is 3.2%. Memory access patterns dominate over raw FLOPs at this model size. Still well under 15% threshold — lesson: FLOP counting underestimates memory bandwidth effects for LoRA.

## Implications for Next Experiment
T4 line is fully closed (routing + serving + format compat). The e2e serving pipeline is: route 0.15ms → swap (if needed) 1.0ms → first token ~120ms (API constant) → decode 81.5 tok/s. Next focus: P1 T5 or higher-tier experiments scaling to N=25 domains with real behavioral benchmarks.
