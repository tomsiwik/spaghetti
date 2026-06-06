# Mathematical Foundations: Competitive Benchmark

## 1. Mechanism Definition

### Pre-merge Composition

Given base weight matrix W in R^{out x in} and N domain adapters with
Grassmannian A_i in R^{in x r} and trained B_i in R^{r x out}:

```
W_composed = W + (1/N) * alpha * sum_{i=1}^{N} B_i^T @ A_i^T
```

where alpha = 20.0 (LoRA scaling factor), N = 5, r = 16, so effective
per-adapter scale = alpha/N = 4.0.

The composed model has identical architecture to the base model (nn.Linear
layers with modified weights). No per-token overhead during inference.

### Parameter Efficiency Argument

Our system:
- Base: BitNet-2B-4T = 2.4B params, ternary packed ~700MB on disk, ~1.7GB in memory (bf16 unpacked)
- Per adapter: rank-16, 7 target layers x 26 blocks = 182 LoRA pairs
  Each A: (in, 16), each B: (16, out). Total per adapter ~1.9KB (ternary B)
- 5 adapters: ~9.5KB total adapter storage
- Pre-merged: no additional inference memory (absorbed into W)

Qwen2.5-3B:
- 3.09B params, FP16 ~6.2GB, 4-bit quantized ~1.8GB
- No modularity, no domain specialization without fine-tuning

Gemma-2-2B:
- 2.61B params, FP16 ~5.2GB, 4-bit quantized ~1.5GB

### Memory Analysis

| System | Params | Memory (inference) |
|--------|--------|--------------------|
| BitNet-2B + SOLE (pre-merge) | 2.4B + ~10KB | ~1.7GB (bf16 unpacked) |
| Qwen2.5-3B (4-bit) | 3.09B | ~1.8GB |
| Gemma-2-2B (4-bit) | 2.61B | ~1.5GB |

K3 threshold: our system must use less memory than Qwen2.5-3B quantized.
At ~1.7GB vs ~1.8GB, this should PASS but the margin is thin. The bf16
unpacking of ternary weights is the bottleneck (same issue identified in
exp_e2e_demo_pipeline_mlx).

## 2. Why Pre-merge Composition Should Work

The pre-merge approach was proven in exp_e2e_demo_pipeline_mlx:
- +44.1% mean PPL improvement across 5 domains
- All domains positive (34-61% improvement)
- 0% per-token overhead (merged into base weights)

The uniform 1/N weighting was the best task-accuracy strategy in
exp_task_accuracy_real_benchmarks:
- GSM8K: base=38%, uniform=54% (+16pp)
- MMLU: mixed results, but uniform >= base on 3/5 domains

The mathematical guarantee: Grassmannian A matrices ensure
||Delta_W_i^T Delta_W_j|| <= (alpha/r)^2 ||B_i|| ||A_i^T A_j|| ||B_j||
With A_i perp A_j, interference -> 0. Mean |cos| = 0.00125 empirically.

## 3. What Breaks It

### Scale Disadvantage
Qwen2.5-3B has 1.3x more parameters than our base. Even with adapters,
our effective model capacity is lower. On general knowledge tasks (MMLU),
parameter count matters: more params = more stored knowledge.

Expected failure mode: MMLU general subjects where adapters don't help.

### Format Mismatch (Known Issue)
Our adapters were trained on instruction-response pairs, not multiple-choice.
From exp_task_accuracy_real_benchmarks: routing HURTS MMLU on 4/5 domains.
Uniform composition partially mitigates this but cannot fix fundamental
format incompatibility.

### Quantization Comparison Confound
We compare bf16-unpacked ternary (effectively ~2-bit weights upcast) against
4-bit quantized dense models. The ternary base has inherently lower precision
per parameter. This is not a fair comparison in parameter efficiency terms --
it is a fair comparison in MEMORY BUDGET terms (which is what matters for
deployment).

## 4. Kill Criteria Analysis

**K1 (> 60% benchmarks worse than Qwen2.5-3B):** We have 6 benchmarks
(GSM8K + 5 MMLU domains). Worse on > 60% means worse on >= 4/6.

Prior data suggests:
- GSM8K: uniform=54%. Qwen2.5-3B published ~65-70% on GSM8K. Likely LOSE.
- MMLU medical: uniform=40%. Qwen2.5-3B published ~55-60%. Likely LOSE.
- MMLU code: uniform=40%. Qwen2.5-3B ~50-55%. Likely LOSE.
- MMLU math: uniform=35%. Qwen2.5-3B ~45-50%. Likely LOSE.
- MMLU legal: uniform=45%. Qwen2.5-3B ~45-50%. Toss-up.
- MMLU finance: uniform=45%. Qwen2.5-3B ~45-50%. Toss-up.

**Honest prediction: K1 will likely FAIL (4-5/6 benchmarks worse).**
Qwen2.5-3B is a much stronger model with 30% more parameters and
instruction tuning. Our adapters were not designed for MMLU-style evaluation.

**K2 (worse than base on any benchmark):** Prior data shows uniform
composition improves GSM8K by 16pp and is neutral-to-positive on most
MMLU domains. Should PASS unless temp=0.0 changes the dynamics.

**K3 (memory exceeds Qwen2.5-3B):** ~1.7GB vs ~1.8GB. Should PASS
with thin margin.

## 5. Worked Example

For a GSM8K problem with 5 domain adapters (medical, code, math, legal, finance):

1. Load base model: W_base for each of 182 linear layers
2. For each layer, compute: delta = (1/5) * 20.0 * sum(B_i^T @ A_i^T)
   = 4.0 * sum(B_i^T @ A_i^T)
3. Set W_new = W_base + delta
4. Run inference with W_new (standard nn.Linear forward pass)

At d=2560, r=16:
- A_i: (2560, 16) = 40,960 elements
- B_i: (16, 2560) = 40,960 elements (ternary: {-1, 0, 1})
- delta per layer: (2560, 2560) = 6.5M elements
- Total deltas across 182 layers: ~1.2B delta elements (but not stored -- merged in place)

## 6. Connection to Architecture

This experiment is the GATE experiment -- it determines whether the
BitNet-SOLE architecture is competitive with monolithic alternatives.

If K1 PASSES: Architecture is viable, proceed to production serving.
If K1 FAILS: Architecture needs a stronger base model (3B+ ternary)
or smarter composition (routing-weighted, not uniform).

The result directly informs Track A (Own Our Ternary Base) priority:
if the architecture loses at 2B, we need to either scale up the base
or improve adapter effectiveness.
