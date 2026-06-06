# Peer Review: Metal Kernel Profiling

## NotebookLM Findings

Skipped -- this experiment is primarily an engineering profiling exercise, not a mechanism experiment with novel mathematical claims. NotebookLM deep review would add limited value.

## Mathematical Soundness

### MATH.md has no proof

MATH.md contains a roofline-style calculation (memory-bound throughput = model_size / bandwidth), not a theorem-proof-QED block. This is appropriate for a profiling experiment, but it means the experiment operates at the "provisional" tier of evidence by project standards.

### The roofline calculation itself

The theoretical estimate is straightforward and correct in structure:

- T_memory = 1.7 GB / 273 GB/s = 6.2 ms --> 161 tok/s

However, PAPER.md then reveals the actual packed model size is 1.18 GB, not 1.7 GB. Using the correct size:

- T_memory = 1.18 GB / 268.6 GB/s = 4.39 ms --> 228 tok/s

This means measured 165.6 tok/s is actually 73% of the corrected theoretical max, not 105%. The claim "there is no gap" and "slightly EXCEEDS the theoretical estimate" is based on comparing against the wrong model size (1.7 GB instead of 1.18 GB).

**This is the central error of the paper.** The 1.7 GB figure appears to be the unpacked bf16 weight size, not the packed ternary size. The experiment's own prior work (inference_speed_10x) correctly computed: 273 GB/s / 1.179 GB = 231.6 tok/s theoretical max, and measured 171.8 tok/s = 74.2% utilization. That is consistent with this experiment's 165.6 tok/s.

So the actual gap is ~1.4x (228/165.6), not 1.05x as claimed.

### Prediction vs. measurement

MATH.md predicted the gap breakdown:

| Bottleneck | Predicted | Measured |
|-----------|-----------|---------|
| Memory bandwidth | ~50% of 3.5x gap | No 3.5x gap; bandwidth is 100% at large tensors |
| Ternary unpacking | ~20% of gap | 125ms total but not on decode path |
| Kernel dispatch | ~15% of gap | 5% eval overhead |
| Python/graph | ~10% of gap | Not isolated |
| mx.compile closing 20-40% of gap | 21% at seq=1, ~0% at longer seq | Partially confirmed |

The predictions were made against a 3.5x gap that turned out not to exist (it was a KV cache issue). The predictions are therefore untestable against their original framing. This is not a failure of the experiment -- discovering the measurement artifact is valuable -- but it means the quantitative predictions were never verified.

## Novelty Assessment

### This is a re-discovery of a known result

The inference_speed_10x experiment (already in the codebase) measured:
- 171.8 tok/s base model (internal timing)
- 74.2% bandwidth utilization
- 1.179 GB actual model size
- Identified Python overhead as 10.6%

This experiment measured:
- 165.6 tok/s with KV cache
- ~73% bandwidth utilization (when correctly computed)
- 1.18 GB model size (same)
- Identified mx.compile as 21% speedup at seq=1

The only genuinely new measurements are:
1. mx.compile speedup quantification across sequence lengths (useful)
2. Eval boundary overhead at 5% (useful)
3. Memory bandwidth curve from 1MB to 1GB (useful reference data)

The "surprise finding" (no 3.5x gap) was already known from inference_speed_10x. The prior experiment had already identified that early measurements were artifacts of including Python overhead.

## Experimental Design

### Strengths

1. **Methodical profiling**: Six phases covering bandwidth, components, compilation, eval boundaries, ternary overhead. Well-structured code.
2. **Proper warmup and repetition**: 10 warmup + 100 measurement iterations for most tests.
3. **The bandwidth curve** (Phase 6) is genuinely useful -- showing dispatch-bound regime below 10MB is good data.
4. **mx.compile across sequence lengths** shows the optimization only matters for seq=1, which is the generation case. Good insight.

### Weaknesses

1. **The "no cache" comparison is misleading.** The 27.2 tok/s figure measures single-forward-pass time for a 13-token prompt, treating it as "one token per forward." This is not autoregressive generation without KV cache -- it is full-prompt processing. Actual autoregressive generation without KV cache would be O(n^2) in context length, getting progressively worse. The 6.1x "speedup" of KV cache is really just the ratio of prompt-processing to single-token decode, which are different operations.

2. **Component timing with per-layer eval (Phase 2)** introduces measurement artifacts. The -74% "overhead" at seq=1 (negative overhead, meaning components sum to 135% of total) is correctly noted as eval-per-layer serialization. But this means the component breakdown does not reflect actual execution at all -- it reflects an artificial eager-mode execution. The paper acknowledges this but still presents the per-component numbers as if they are meaningful.

3. **The summary hardcodes model_size_gb = 1.7** (line 548 of run_experiment.py) even though the ternary overhead phase identifies the actual packed size. This propagates the incorrect theoretical estimate into the final gap calculation.

4. **No generation quality check.** The experiment uses `generate_step` for throughput but does not verify that the generated tokens are sensible. For a profiling experiment this is acceptable, but worth noting.

## Macro-Scale Risks (advisory)

1. **The 1.4x remaining gap** (73% bandwidth utilization) is the real optimization target. At macro scale with adapter composition, this gap compounds with adapter overhead (33-48% from prior experiments).

2. **mx.compile's 21% speedup at seq=1** is actionable and should be integrated into the serving pipeline. At longer sequences the benefit vanishes, confirming the system is compute-bound for prefill.

3. **The bandwidth curve data** (dispatch-bound below 10MB) is relevant to adapter composition design: individual adapter weight reads are small matrices that will be dispatch-bound, not bandwidth-bound. This supports the pre-merge strategy over runtime LoRA.

## Verdict

**REVISE**

The experiment contains genuinely useful profiling data, but the central claim is wrong and the main finding was already known.

### Required fixes

1. **Correct the theoretical throughput calculation.** Use the actual packed model size (1.18 GB), not the unpacked estimate (1.7 GB). The corrected theoretical max is ~228 tok/s, and the measured 165.6 tok/s represents 73% utilization, consistent with inference_speed_10x. Remove the claim that throughput "exceeds" the theoretical limit.

2. **Acknowledge inference_speed_10x.** That experiment already measured 171.8 tok/s and 74.2% bandwidth utilization. This experiment's "surprise finding" (no 3.5x gap) is a re-discovery. The paper should cite it and frame the contribution as additional profiling detail (compile impact, eval overhead, bandwidth curve), not as a new discovery.

3. **Fix the no-cache comparison framing.** The 27.2 tok/s figure is not "generation without KV cache." It is prompt processing latency divided by prompt length. Clarify what is actually being measured.

4. **Write a formal proof or downgrade to provisional.** Per project standards, without a theorem/proof/QED block, the finding status should be `provisional`, not `supported`. A roofline calculation is not a proof. If the experiment is purely empirical profiling (which is fine), label it as such.

5. **Fix hardcoded model size in run_experiment.py** (line 548-549). Either read actual model size from loaded weights or use the correct 1.18 GB figure.
