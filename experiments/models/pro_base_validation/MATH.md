# Pierre Pro Base Validation: Memory and Performance Analysis

## 1. Objective

Validate that Qwen3-4B-4bit loads and runs on M5 Pro 48GB with sufficient
quality to serve as a composition base. This is a baseline measurement
(Type 1: verification), not a novel mechanism experiment.

## 2. Model Architecture

Qwen3-4B (Qwen3ForCausalLM):

| Parameter | Value |
|-----------|-------|
| hidden_size (d) | 2560 |
| num_hidden_layers (L) | 36 |
| num_attention_heads | 32 |
| num_key_value_heads (GQA) | 8 |
| head_dim | 128 |
| intermediate_size | 9728 |
| vocab_size | 151936 |
| max_position_embeddings | 40960 |
| activation | SiLU |
| quantization | 4-bit, group_size=64 |
| tie_word_embeddings | True |

## 3. Memory Analysis

### 3A. Parameter Count

**Embedding:**
- embed_tokens: V * d = 151936 * 2560 = 388,956,160

**Per transformer layer (L=36):**
- Q projection: d * d = 2560 * 2560 = 6,553,600
- K projection: d * (d/4) = 2560 * 640 = 1,638,400 (GQA: 8 KV heads vs 32 Q heads)
- V projection: d * (d/4) = 2560 * 640 = 1,638,400
- O projection: d * d = 2560 * 2560 = 6,553,600
- gate_proj: d * i = 2560 * 9728 = 24,903,680
- up_proj: d * i = 2560 * 9728 = 24,903,680
- down_proj: i * d = 9728 * 2560 = 24,903,680
- input_layernorm: d = 2560 (RMSNorm, no bias)
- post_attention_layernorm: d = 2560
Total per layer: 91,102,160

**Final:**
- norm: d = 2560 (RMSNorm)
- lm_head: tied with embed_tokens = 0 extra

**Total params:** 388,956,160 + 36 * 91,102,160 + 2,560
= 388,956,160 + 3,279,677,760 + 2,560
= 3,668,636,480 (~3.67B params)

Note: Qwen reports "4B" which rounds up. The actual count is ~3.67B.

### 3B. 4-bit Quantized Memory

At 4-bit quantization with group_size=64:
- Each weight: 4 bits = 0.5 bytes
- Group scales: fp16 overhead per 64 elements = 2 bytes / 64 = 0.03125 bytes/element
- Effective bits per weight: 4 + 16/64 = 4.25 bits = 0.53125 bytes

**Quantized model memory:**
~3.67B * 0.53125 bytes = ~1.95 GB

Some tensors (embeddings, layernorms) stay in bf16:
- embed_tokens (bf16): 388,956,160 * 2 bytes = 777.9 MB
- layernorms (bf16): 36 * 2 * 2560 * 2 bytes = 368.6 KB (negligible)

**Predicted total weight memory: ~2.7 GB** (quantized layers + bf16 embeddings)

### 3C. KV Cache

At sequence length S with batch size 1:
- Per layer: 2 * n_kv_heads * head_dim * S * 2 bytes (bf16)
  = 2 * 8 * 128 * S * 2 = 4096 * S bytes
- All layers: 36 * 4096 * S = 147,456 * S bytes

At S=512: 147,456 * 512 = 75.5 MB
At S=2048: 147,456 * 2048 = 302.0 MB

**Predicted peak memory (weights + KV at S=512): ~2.8 GB**

This is well within the 48GB M5 Pro budget (5.8% utilization).

### 3D. Composition Headroom

With base at ~2.8 GB, remaining budget: 48 - 8 (system) - 2.8 = 37.2 GB

LoRA adapter at rank-16 on attention (Q, K, V, O):
- Per adapter per layer: 4 * 2 * d * r * 2 bytes (bf16)
  = 4 * 2 * 2560 * 16 * 2 = 655,360 bytes = 640 KB
- All 36 layers: 36 * 640 KB = 22.5 MB per adapter

**Max adapters on M5 Pro: 37.2 GB / 22.5 MB = ~1,653 adapters**
(Far exceeds any practical N.)

## 4. Throughput Analysis

### 4A. Memory Bandwidth Bound

M5 Pro memory bandwidth: ~273 GB/s

For autoregressive generation (batch=1), each token requires reading all weights:
- Weight read per token: ~2.7 GB (quantized weights)
- KV update per token: ~147 KB (negligible vs weight read)

**Theoretical max tok/s = 273 GB/s / 2.7 GB = ~101 tok/s**

With typical bandwidth utilization of 60-75% on MLX:
**Predicted tok/s: 60-76 tok/s**

### 4B. Comparison with BitNet-2B

Previous base (BitNet-2B-4T):
- 172 tok/s base, 97 tok/s with adapter
- ~1.22 GB memory

Qwen3-4B-4bit is ~2.2x larger in memory, so we expect:
- ~2.2x slower generation (proportional to weight read)
- Predicted: ~78 tok/s base (172 / 2.2)

## 5. Benchmark Predictions

Based on published Qwen3 technical reports and community benchmarks:

### 5A. MMLU / MMLU-Pro

Qwen3-4B is positioned between Qwen2.5-3B and Qwen2.5-7B.
Published benchmarks (full precision):
- Qwen2.5-3B: MMLU ~68%
- Qwen3-4B: MMLU ~72% (estimated from Qwen3 blog post scaling)
- Qwen2.5-7B: MMLU ~75%

4-bit quantization typically degrades MMLU by 1-3 percentage points.
**Prediction: MMLU-Pro subset 65-72%** (5-shot, 200 questions)

Note: MMLU-Pro is harder than MMLU. Published MMLU-Pro scores for 4B class:
~45-55%. We will measure both standard MMLU-style and MMLU-Pro.

### 5B. GSM8K

Qwen3-4B with chain-of-thought:
- Qwen2.5-3B: ~65% GSM8K
- Qwen3-4B: ~75% (estimated)
- 4-bit quant degradation: 2-5%

**Prediction: GSM8K ~70-75%** (8-shot CoT, 100 problems)

### 5C. Instruction Following

IFEval-style tests (format compliance):
- Models at this scale typically score 60-80% on format tasks
- Qwen3 was trained with thinking mode support

**Prediction: IFEval ~65-75%**

## 6. Kill Criteria (Derived)

| ID | Criterion | Source | Threshold |
|----|-----------|--------|-----------|
| K808 | Model loads on M5 Pro 48GB | Memory analysis (Sec 3) | Peak < 40GB |
| K809 | MMLU >= 60% | Sec 5A predictions | 60% is minimum useful for composition |

K808: The memory analysis predicts ~2.8 GB peak. This is 7% of the 48GB budget.
The model will load unless there is a software bug.

K809: Published benchmarks for Qwen3-4B at full precision show ~72% MMLU.
Even with 4-bit degradation (worst case -5%), we expect >= 67%.
The 60% threshold is conservative — a model below this would be too weak
to serve as a meaningful composition base.

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   The model is 2.7 GB quantized; 48 GB hardware has 17x headroom. Loading cannot fail from memory.

2. Which existing theorem(s) does the proof build on?
   Shannon rate: bits_per_weight * n_params = total_bits = memory. Bandwidth-bound latency = memory / bandwidth.

3. What specific numbers does the proof predict?
   Memory: ~2.7 GB weights, ~2.8 GB peak. Throughput: 60-76 tok/s. MMLU: 65-72%.

4. What would FALSIFY the proof?
   Memory > 10 GB (would indicate bf16 fallback or cache explosion). tok/s < 30 (would indicate non-bandwidth-bound regime).

5. How many hyperparameters does this approach add?
   Count: 0 (this is a measurement, not a method)

6. Hack check: N/A — baseline measurement, no method being tested.
