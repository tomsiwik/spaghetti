# PAPER.md — P0: v_proj+o_proj Adapter Quality Floor — KILLED

## Summary

Training v_proj+o_proj adapters with real HuggingFace datasets (450 examples, 1000
iterations) produces **WORSE** behavioral quality than P8's 10 handcoded examples
(200 iterations) in 4/5 domains. The root cause is data distribution mismatch:
task-completion data shifts the output distribution away from explanatory vocabulary.

**Finding #506:** Data distribution matters more than quantity for behavioral quality.

## Prediction vs Measurement

| Prediction | Predicted | Measured | Match |
|-----------|-----------|----------|-------|
| K1320 All 5 domains >= 60% | 65-80% | math=50%, code=10%, med=15%, legal=25%, fin=15% | **FAIL** |
| K1321 Mean >= 50% | ~67% | **23%** | **FAIL** (44pp short) |
| K1322 Legal >= 40% | 50-65% | **25%** | **FAIL** |
| K1323 Train <= 30 min | 10-15 min | **12.6 min max** | PASS |

## Why Predictions Failed

The MATH.md predicted that more diverse data = better vocabulary coverage. This was
wrong because it conflated data **quantity** with data **distribution alignment**.

### The actual mechanism

v_proj+o_proj adapters directly modify the output token distribution (Finding #504).
Training data determines WHICH tokens the distribution shifts toward:

| Dataset | Token Distribution | Effect on Explanatory Vocab |
|---------|-------------------|---------------------------|
| P8 handcoded | Dense domain terminology in explanatory prose | **Increases** vocab density |
| GSM8K | Arithmetic steps, numbers, "####" | Mixed (+30% math, only survivor) |
| CodeAlpaca | Code blocks, syntax, imports | **Decreases** prose vocab (-38%) |
| MedMCQA | Short MCQ answers "(B) [term]" | **Decreases** rich explanations (-41%) |
| Finance-Alpaca | Mixed financial instructions | **Decreases** prose vocab (-32%) |
| Alpaca (legal) | Generic short answers | **Decreases** domain terms (-31%) |

GSM8K is the only dataset where step-by-step solutions naturally contain mathematical
terminology similar to the eval queries. This explains why math is the sole survivor.

## Detailed Results

### Training

| Domain | Dataset | N train | Iters | Time (min) |
|--------|---------|---------|-------|-----------|
| Math | GSM8K | 450 | 1000 | 12.2 |
| Code | CodeAlpaca-20k | 450 | 1000 | 8.2 |
| Medical | MedMCQA | 450 | 1000 | 12.6 |
| Legal | Alpaca-filtered | 450 | 1000 | 9.2 |
| Finance | Finance-Alpaca | 450 | 1000 | 10.0 |

### Behavioral Quality (vocab density)

| Domain | Base Mean Vocab | Adapted Mean Vocab | Delta | Improvement Rate | P8 Rate |
|--------|----------------|-------------------|-------|-----------------|---------|
| Math | 1.50 | 1.95 | +30% | **50%** | 55% |
| Code | 2.50 | 1.55 | -38% | **10%** | 50% |
| Medical | 2.45 | 1.45 | -41% | **15%** | 70% |
| Legal | 1.45 | 1.00 | -31% | **25%** | 35% |
| Finance | 2.00 | 1.35 | -32% | **15%** | 50% |

## Root Cause Analysis

### The Disease (not the symptom)

The symptom was "low adapter quality." The assumed disease was "insufficient training data."
The actual disease is **training-evaluation distribution mismatch**.

P8's handcoded data was specifically designed with:
1. Dense domain glossary terms
2. Explanatory prose format (matching eval queries)
3. Rich conceptual answers

HuggingFace datasets optimize for task completion:
1. Code datasets produce code, not explanations
2. MCQ datasets produce letter answers, not rich medical text
3. Math datasets produce arithmetic, not conceptual explanations

### Impossibility Structure

The failure is geometrically inevitable: v_proj+o_proj modifies the output token
distribution. Training on code tokens literally teaches the model to output code
tokens instead of explanatory prose tokens. You cannot increase domain vocabulary
density by training on a distribution that is sparse in domain vocabulary.

### What This Means for P0

Two paths forward:
1. **For behavioral quality (generation):** Train on vocabulary-dense explanatory data
   (like P8 but with 30-50 diverse examples per domain instead of 10)
2. **For benchmark accuracy (tasks):** Train on HuggingFace datasets, evaluate on
   matching benchmarks (GSM8K accuracy, HumanEval pass@1, not vocab density)

These are **different adapters for different purposes**. The P0 goal of "generation
quality AND benchmarks" may require domain-aware training data curation.
