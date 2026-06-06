# Learnings: exp_training_speed_optimization

## Core Finding

Batch size is the only material training optimization for BitNet-2B-4T LoRA
fine-tuning on M5 Pro. Batch=8 + mx.compile achieves 7.52x samples/sec throughput
(69.9 vs 9.3). Python-level tricks (GC disable, pre-tokenization, step wrapping)
are all <2% — noise at 2.4B parameter scale where each step is >100ms of GPU compute.

**Critical nuance:** Throughput ≠ training speed. For fixed 200-step training,
batch=1 completes in 21s vs 143s for batch=4 (7x slower wall-clock). Batching
only helps when processing MORE data per domain (>500 samples). The "speedup"
is in GPU utilization, not wall-clock adapter training time.

## Critical Discoveries

### 1. Batch=1 is optimal for current 200-step regime

Each gradient update takes 107ms at batch=1 vs 716ms at batch=4. For 200 gradient
updates (the current training budget), batch=1 finishes in 21.4s while batch=4
takes 143.2s. The throughput metric (samples/sec) is misleading for fixed-step
training — it measures GPU utilization, not time-to-adapter.

**Implication:** Keep batch=1 for the current pipeline. Only switch to batch≥4
when training on >1000 samples per domain.

### 2. mx.compile is context-dependent

- batch=1: -8% (recompilation overhead dominates, high variance std=41ms)
- batch≥4: +17% (graph large enough to amortize, fixed-shape padding eliminates recompilation)

This is consistent with exp_mx_compile_full_pipeline finding: mx.compile is
redundant for generation (async_eval hides dispatch) but helps training
(synchronous mx.eval exposes dispatch overhead) — but ONLY at batch≥4.

**Requirement:** All sequences must be padded to fixed MAX_SEQ_LENGTH before
batching. Variable-length sequences cause recompilation that destroys the benefit.

### 3. Python overhead is invisible at 2.4B scale

GC disable: +0.0%. Pre-tokenization: -1.1% (noise). Step wrapping: +1.2% (noise).
At 107ms per step dominated by 2.4B parameter forward+backward, saving 0.1ms is
invisible. This contradicts small-model MLX benchmarks where GC/dispatch overhead
is significant — at production scale, the GPU dominates.

### 4. Combined optimizations stack multiplicatively

batch=8 (6.2x) × mx.compile (1.2x) = 7.4x theoretical, 7.5x observed. The two
mechanisms are independent: batching converts vec-mat to mat-mat (compute efficiency),
compilation eliminates dispatch overhead (scheduling efficiency).

## Contradicting Evidence

### FLOP estimate inconsistency (from adversarial review)

MATH.md theoretical minimum is 133ms (3.6T FLOPs / 27 TFLOPS), but observed
minimum is 92ms. Either the FLOP count overestimates (likely: LoRA forward skips
most base weight computation) or the M5 Pro delivers >39 TFLOPS in practice.
The theoretical analysis is wrong but the empirical results are correct.

### Convergence comparison across batch sizes

Batch=1 final loss 2.80, batch=4 1.76, batch=8 0.84. These solve different
optimization problems (different data volume per step, no LR adjustment).
K2 "convergence preserved" means "loss decreases" not "same solution reached."

## Cross-References

1. **exp_mx_compile_full_pipeline:** mx.compile redundant for generation, confirmed
   context-dependent for training (helps at batch≥4 only).
2. **exp_inference_speed_10x:** 172 tok/s base established. Training bottleneck is
   orthogonal to serving bottleneck (compute-bound vs bandwidth-bound).
3. **exp_memory_budget_analysis:** N_max=853 adapters at 48GB. Memory budget for
   batch>1 training not yet validated against this budget.
4. **fast-mlx guide:** "Evaluating the graph incurs some overhead, so don't do it
   too frequently" — at batch=1, this overhead is invisible at 2.4B scale.

## What This Changes

**Pipeline recommendation:**
- batch=1 for current 200-step/500-sample training (21s/adapter)
- batch=4 + mx.compile for future large-dataset training (67.5 samples/sec)
- Skip GC disable, pre-tokenization, step wrapping everywhere

**No new experiment needed.** The training pipeline is optimized for the current
regime. If training budget increases (more data per domain), re-evaluate with
batch=4 + mx.compile. Memory profiling at batch=8 is needed before production use.
