# MATH.md — P0: v_proj+o_proj Adapter Quality Floor

## Problem Statement

Finding #504 established v_proj+o_proj as the correct projection target for behavioral
text quality (all 5 domains improved vs q_proj). Finding #505 proved composition is NOT
the bottleneck. The remaining lever is **adapter solo quality**, which was limited by
minimal training: 10 hardcoded examples cycled to 80, 200 iterations.

This experiment asks: **what is the minimum training configuration that produces
behavioral quality above the 60% vocabulary improvement threshold?**

## Theorem 1: Training Signal Sufficiency (from LoRA convergence)

**Theorem.** A LoRA adapter with rank r on projection matrices W_v, W_o learns
a rank-r perturbation ΔW = BA^T. The quality of this perturbation depends on:
1. **Data diversity** — the adapter must observe sufficient domain-specific token
   distributions to learn the vocabulary shift
2. **Gradient steps** — the adapter must converge to a useful local minimum

**From LoRA (Hu et al., 2106.09685):** At rank r, the adapter has 2×r×d parameters
per layer. For Gemma 4 E4B with d=2816, r=16, targeting v_proj+o_proj on 16 layers:
2 × 16 × 2816 × 2 × 16 = 2,883,584 parameters = 2.88M.

**Data requirement from Finding #149:** Data saturation occurs at N=200-500 unique
examples. Below N=200, overfitting dominates. P8 used N=10 unique examples — far below
the saturation floor.

**Training duration from Finding #472:** rank-16 + 150 diverse examples → 93.3%
style compliance. The key insight: rank determines capacity, but data diversity
determines what the capacity is used for. With N=10 examples, the adapter memorizes
specific phrasings rather than learning domain vocabulary distribution.

**Prediction:** With N=500 unique examples and 1000 iterations, the adapter will
sample each example ~2× per epoch (batch_size=2, 1000 steps × 2 / 500 = 4 passes).
This provides sufficient gradient signal for convergence without overfitting.

## Theorem 2: Vocabulary Density as Behavioral Proxy

**Claim.** The vocabulary improvement rate (fraction of responses where adapted model
uses more domain terms than base) measures behavioral quality because:

```
vocab_improvement_rate = P(|D_adapted ∩ glossary| > |D_base ∩ glossary|)
```

where D is the set of tokens in a generated response and glossary is the domain
vocabulary set. This is a strict behavioral test: the model must *generate* domain
terms, not merely recognize them.

**From P8 (Finding #504):** v_proj+o_proj directly modifies the output token distribution
(Theorem 1 in P8 MATH.md). Training on domain-rich text teaches the adapter which
vocabulary to inject. More diverse training examples → more vocabulary coverage →
higher improvement rate.

**Prediction:** 500 diverse examples with real domain text (GSM8K step-by-step proofs,
CodeAlpaca implementations, medical explanations, legal reasoning, financial analysis)
will cover substantially more domain vocabulary than 10 hardcoded examples.

## Quantitative Predictions

| Domain | P8 Result (10 ex, 200 iter) | Predicted (500 ex, 1000 iter) | Basis |
|--------|---------------------------|------------------------------|-------|
| Math   | 55%                       | 70-80%                       | GSM8K has rich mathematical vocabulary in step-by-step solutions |
| Code   | 50%                       | 65-75%                       | CodeAlpaca has full implementations with programming terms |
| Medical| 70%                       | 75-85%                       | Already strong; more data should push further |
| Legal  | 35%                       | 50-65%                       | Largest gap; real legal text is more vocabulary-dense than hardcoded examples |
| Finance| 50%                       | 65-75%                       | Financial instruction data has diverse terminology |

**Key prediction:** Legal domain will show the largest improvement (35% → 50-65%)
because it had the sparsest hardcoded training data and will benefit most from
diverse real legal text.

**Kill criteria mapping:**
- K1320: All 5 domains >= 60% → predicted PASS (math 70%, code 65%, medical 75%, legal 50%, finance 65%)
  - **Risk:** Legal may fall short at 50%. If K1320 fails, it will be due to legal alone.
- K1321: Mean >= 50% → predicted PASS (mean ~67%)
- K1322: Legal >= 40% → predicted PASS (50-65%)
- K1323: Training <= 30 min/domain → predicted PASS (T2 showed 10-22 min/domain)

## Training Configuration

| Parameter | P8 (prior) | This experiment | Rationale |
|-----------|-----------|-----------------|-----------|
| Data source | 10 hardcoded Q&A | HuggingFace datasets | Real domain distribution |
| N unique examples | 10 | 500 | Finding #149 saturation at 200-500 |
| Iterations | 200 | 1000 | Finding #421 used 1000 steps |
| Rank | 16 | 16 | Same (capacity proven sufficient) |
| Targets | v_proj+o_proj | v_proj+o_proj | Finding #504 |
| Layers | 16 (last) | 16 (last) | Consistent with P8 |
| Batch size | 2 | 2 | Memory constraint |
| LR | 1e-4 | 1e-4 | Standard for LoRA |

## Datasets

| Domain | Dataset | Format |
|--------|---------|--------|
| Math | openai/gsm8k | Step-by-step solutions (generative) |
| Code | sahil2801/CodeAlpaca-20k | Instruction → code output |
| Medical | medalpaca/medical_meadow_medical_flashcards | Medical Q&A (generative) |
| Legal | pile-of-law subset or instruction data | Legal reasoning (generative) |
| Finance | gbharti/finance-alpaca | Financial instruction following |
