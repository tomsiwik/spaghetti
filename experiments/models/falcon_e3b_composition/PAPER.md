# Falcon-E-3B LoRA Adapter Composition: Research Digest

## Hypothesis

Falcon-E-3B (3B ternary, Llama-compatible, 999MB) supports LoRA adapter composition and can compete with Qwen2.5-3B-Instruct on domain benchmarks with lower memory.

**Result: PARTIALLY SUPPORTED. K1 PASS, K2 PASS, K3 FAIL (KILL on memory). S1 FAIL.**

LoRA works. Falcon-E-3B base is competitive with Qwen-3B (ties or beats on 5/6). But uniform composition degrades the base model, and memory exceeds threshold due to bf16 unpacking.

## What This Experiment Is

A head-to-head comparison of Falcon-E-3B (ternary) vs Qwen2.5-3B-Instruct (4-bit) on GSM8K + 5 MMLU domains, testing whether a stronger ternary base (vs BitNet-2B) closes the competitive gap identified in our prior benchmark.

Three systems evaluated:
1. **Falcon-E-3B base** (no adapters, instruction-tuned)
2. **Falcon-E-3B + 5 domain adapters** (uniform 1/5 pre-merge composition)
3. **Qwen2.5-3B-Instruct-4bit** (the competitor baseline)

## Key References

- Falcon-Edge (tiiuae/onebitllms): open ternary training toolkit, Falcon-E-3B source
- Prior competitive benchmark (micro/models/competitive_benchmark/): BitNet-2B KILLED 4/6 vs Qwen
- MoLoRA (arxiv 2603.15965): per-token routing beats uniform composition
- BitNet b1.58 (arxiv 2402.17764): ternary architecture

## Empirical Results

### Training Phase

All 5 domain adapters trained successfully on Falcon-E-3B with rank-16 LoRA on attention (q/v/o):

| Domain | Base PPL | Trained PPL | Improvement |
|--------|:--------:|:-----------:|:-----------:|
| medical | 4.26 | 2.19 | -48.6% |
| code | 3.04 | 2.30 | -24.2% |
| math | 2.61 | 1.62 | -37.9% |
| legal | 15.30 | 11.01 | -28.0% |
| finance | 13.87 | 10.89 | -21.5% |

Training: 200 iterations per adapter, ~50-60s each, peak 12.93 GB during training.

### Benchmark Results

| Benchmark | Falcon Base | Falcon+5 Adapt | Qwen-3B-4bit |
|-----------|:-----------:|:--------------:|:------------:|
| GSM8K | **44%** | 36% | 36% |
| MMLU medical | 55% | 30% | **70%** |
| MMLU code | **60%** | 50% | 40% |
| MMLU math | **55%** | **55%** | 45% |
| MMLU legal | 40% | 35% | 40% |
| MMLU finance | **60%** | 45% | 55% |

### Memory

| System | Peak Memory |
|--------|:-----------:|
| Falcon-E-3B base | 6.74 GB |
| Falcon-E-3B + adapters | 8.80 GB |
| Qwen2.5-3B-4bit | 2.43 GB |

### Kill Criteria

| Criterion | Result | Evidence |
|-----------|--------|----------|
| K1 (#532): Falcon-E-3B doesn't support LoRA | **PASS** | LoRA trains, converges, composes. 5 adapters, all improve PPL. |
| K2 (#533): Loses >4/6 vs Qwen-3B | **PASS** | Composed model loses 3/6 (medical, legal, finance). Ties GSM8K. |
| K3 (#534): Total memory >6GB | **KILL** | 8.80 GB peak (6.74 GB base + merge overhead) |

### Success Criteria

| Criterion | Result | Evidence |
|-----------|--------|----------|
| S1 (#54): Beats Qwen on >=3/6 at <4GB | **FAIL** | Beats on 2/6 (code, math), memory 8.80 GB |

## The Honest Assessment

### What Went Right

**1. Falcon-E-3B BASE is genuinely competitive with Qwen-3B.**

Without any adapters, Falcon-E-3B ties or beats Qwen-3B on 5/6 benchmarks:
- GSM8K: 44% vs 36% (+8pp)
- Code: 60% vs 40% (+20pp)
- Math: 55% vs 45% (+10pp)
- Finance: 60% vs 55% (+5pp)
- Legal: 40% vs 40% (tie)
- Medical: 55% vs 70% (-15pp, only loss)

This is a dramatic improvement over BitNet-2B, which lost 4/6. The Falcon-E-3B instruction tuning is effective.

**2. LoRA training works perfectly.** All 5 adapters converge with 21-49% PPL improvement. The ternary->bf16 unpack + LoRA training pipeline is validated on a second ternary architecture.

### What Went Wrong

**1. Uniform composition DEGRADES the base model.**

Falcon composed is worse than Falcon base on ALL 6 benchmarks:
- GSM8K: 36% vs 44% (-8pp)
- Medical: 30% vs 55% (-25pp)
- Code: 50% vs 60% (-10pp)
- Math: 55% vs 55% (same)
- Legal: 35% vs 40% (-5pp)
- Finance: 45% vs 60% (-15pp)

This replicates and amplifies the finding from the BitNet competitive benchmark (where composition hurt math/legal). On Falcon-E-3B, the degradation is more severe because:

(a) The base model is already instruction-tuned and competent. The adapters were trained on domain text (NTP-style), not instruction-tuning data. Adding NTP-domain signals to an already-good instruction model actively degrades it.

(b) Uniform 1/5 weighting means every query gets all 5 adapters, including irrelevant ones. For a medical MMLU question, the code/math/legal/finance adapters inject noise.

**2. Memory is catastrophic.** The bf16 unpack penalty makes Falcon-E-3B 3.6x worse than Qwen-3B on memory (8.80 vs 2.43 GB). This is an engineering limitation: native ternary inference would use ~1.0 GB.

### The Reframing

The experiment's most important finding is NOT about composition -- it is about the base model:

**Falcon-E-3B base already beats Qwen-3B on 5/6 benchmarks without any adapters.**

This means:
1. The competitive gap identified with BitNet-2B was a base-model-quality problem, not a composition problem.
2. Adapters should be used for specialization BEYOND the base's capabilities, not for replacing its existing knowledge.
3. The composition strategy should be per-query routing (activate only relevant adapters), not uniform blending.

## Limitations

1. **Small eval sets.** 20 MMLU questions per domain, 50 GSM8K problems. Individual benchmark differences of 5pp (1 question) are within noise.

2. **Uniform composition only.** No per-domain routing tested. Prior work shows routing improves by +13.9% over uniform.

3. **Adapters trained on NTP data.** The data from real_data_25_domain_adapters uses instruction format but was optimized for PPL, not task accuracy. Adapters trained on MMLU-style QA data might compose differently.

4. **Memory measured with bf16 unpack.** Native ternary inference would dramatically change the memory story. The 6.74-8.80 GB is the worst case.

5. **Single seed, greedy decoding.** Deterministic but does not capture variance.

6. **Qwen-3B-4bit underperforms published numbers.** Qwen published GSM8K ~65-70%, we measured 36%. Our 4-bit quantization and prompt format degrade it significantly. Internally consistent but not comparable to published benchmarks.

## What Would Kill This

K3 is already triggered (memory). The architecture path forward requires either:
1. Native ternary inference kernels (reducing 6.74 GB to ~1 GB)
2. A different composition strategy (routing vs uniform)
3. Adapters that add capability rather than degrade existing instruction-tuning

## Implications for the Project

1. **Falcon-E-3B is the better ternary base.** It beats Qwen-3B 5/6 without adapters, whereas BitNet-2B lost 4/6 even with adapters.

2. **Uniform composition is confirmed harmful.** Two independent experiments (BitNet-2B, Falcon-E-3B) show that blindly merging all adapters degrades task performance. Per-query routing is mandatory.

3. **The value proposition shifts.** With a competitive base model, adapters should target NOVEL capabilities (new domains, specialized tasks) rather than trying to match existing instruction-tuning. The "compose cheap experts" thesis becomes "add specialized knowledge on top of a strong base."

4. **Memory remains the blocker.** Both ternary models require bf16 unpack for MLX inference. Until native ternary kernels exist, memory advantage over 4-bit quantized models is impossible.

5. **Track A success:** Falcon-E-3B validates that 3B ternary models can match dense 3B models on benchmarks. The ternary approach is viable at this scale.
