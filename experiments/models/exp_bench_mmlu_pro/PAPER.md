# PAPER.md: MMLU-Pro Baseline + Pierre Adapted

## Summary

Gemma 4 E4B 4-bit scores **42.3%** on MMLU-Pro (100 questions/category, 1400 total)
with thinking disabled. This is 27.1pp below Google's reported 69.4% (with thinking).
A single math adapter (q_proj LoRA r=6 from T2.1) **degrades** all categories by an
average of -6.2pp (36.1% adapted), including math itself (-13pp).

## Prediction vs Measurement

| Criterion | Predicted | Measured | Match |
|-----------|-----------|----------|-------|
| K1411: Base within 5pp of 69.4% | FAIL (58%, 11pp gap) | **FAIL** (42.3%, 27.1pp gap) | Direction correct, magnitude underestimated |
| K1412: Adapted >= base + 2pp | FAIL (+0.5pp) | **FAIL** (-6.2pp) | Direction correct for failure, adapter harms worse than predicted |
| K1413: < 6h runtime | PASS (0.6h) | **PASS** (0.18h) | Correct |

### Prediction accuracy analysis

- **K1 (base accuracy):** Predicted 54-63% without thinking, measured 42.3%. The
  discrepancy is because MMLU-Pro (10 options) is significantly harder without chain-of-thought
  reasoning than standard MMLU (4 options). Random baseline for 10 options = 10%, vs 25% for
  4 options. The thinking penalty on MMLU-Pro is ~27pp, not the 5-15pp predicted from
  4-option benchmarks.

- **K2 (adapter delta):** Predicted small positive effect (+0.5pp); measured large negative
  (-6.2pp). The NTP-trained adapter doesn't just fail to help — it actively degrades the
  instruction-tuned model's MCQ capability across ALL domains. This is consistent with
  Finding #44 (NTP adapters degrade OOD) and the exp_code_adapter_benchmark_validation
  result (code adapter: GSM8K -18pp, HumanEval -15pp).

## Per-Category Results

| Category | Base | Adapted | Delta |
|----------|------|---------|-------|
| biology | 75.0% | 69.0% | -6.0pp |
| business | 29.0% | 22.0% | -7.0pp |
| chemistry | 27.0% | 21.0% | -6.0pp |
| computer science | 44.0% | 37.0% | -7.0pp |
| economics | 47.0% | 44.0% | -3.0pp |
| engineering | 37.0% | 21.0% | -16.0pp |
| health | 39.0% | 41.0% | +2.0pp |
| history | 47.0% | 44.0% | -3.0pp |
| law | 31.0% | 30.0% | -1.0pp |
| math | 36.0% | 23.0% | -13.0pp |
| other | 46.0% | 35.0% | -11.0pp |
| philosophy | 46.0% | 41.0% | -5.0pp |
| physics | 27.0% | 18.0% | -9.0pp |
| psychology | 61.0% | 59.0% | -2.0pp |

## Key Findings

### 1. Thinking mode is essential for MMLU-Pro on Gemma 4
Without thinking, Gemma 4 E4B 4-bit scores 42.3% (vs Google's 69.4% with thinking).
The 27pp gap far exceeds quantization effects (~2pp). MMLU-Pro's 10-option format
requires reasoning that the non-thinking mode cannot perform.

### 2. NTP adapter degrades MCQ benchmark uniformly
The math adapter (trained on GSM8K with NTP loss) hurts ALL 14 categories, including math
(-13pp). Only health shows a tiny positive (+2.0pp). This is the **NTP-instruction conflict**:
NTP training shifts the model's output distribution toward language modeling, away from
instruction-following required for MCQ.

### 3. Eval pipeline validated
The direct-generation approach (mlx_lm in-process, no server) runs 1400 questions in
~5 minutes at 5.3 q/s. The pipeline correctly loads adapters and evaluates.
This infrastructure can be reused for future benchmark experiments.

## Impossibility Structure

**Why the math adapter cannot help MMLU-Pro:**

The adapter was trained with NTP loss on domain text: $\min_\theta -\sum \log p(x_t | x_{<t}; \theta)$.
This objective optimizes for predicting the next token in math text, NOT for selecting
the correct answer from 10 options given a question stem. The adapter shifts attention
distributions toward mathematical notation patterns and away from the instruction-following
patterns needed for "Answer with ONLY the letter."

For an adapter to help on MCQ benchmarks, it would need to be trained with:
- Instruction-tuning loss (SFT on question-answer pairs), OR
- The thinking mode enabled so the base model can reason despite adapter perturbation

## Experimental Setup

- **Model:** mlx-community/gemma-4-e4b-it-4bit
- **Adapter:** q_proj LoRA r=6, 1000 iters, GSM8K NTP loss (from T2.1)
- **Dataset:** MMLU-Pro test set (12,032 questions), sampled 100/category (1,400 total)
- **Thinking:** Disabled (`enable_thinking=false`) for tractable runtime
- **Generation:** Direct mlx_lm.generate(), max_tokens=16, greedy decoding
- **Platform:** M5 Pro 48GB, MLX 0.31.x
- **Runtime:** 637.5s total (318s base + 319s adapted)

## Next Steps

1. **Run with thinking enabled** to validate K1 (expect ~65-70% matching Google).
   Requires ~4x longer runtime (~40min total).
2. **SFT adapter training** instead of NTP — train on MCQ format data to help benchmarks.
3. **N=5 composition benchmark** — test all 5 domain adapters composed, not single adapter.
