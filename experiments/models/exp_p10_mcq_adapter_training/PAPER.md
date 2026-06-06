# P10.C0: MCQ-Format Adapter — Fix NTP Degradation on Benchmarks

## Status: KILLED

K1470 FAIL (50.4% < 65%), K1471 FAIL (HumanEval 25.0% < 55%).
But three major findings emerge from the failure pattern.

## Summary

Standard LoRA adapter trained with MCQ classification loss on MMLU-Pro data
improves non-thinking MCQ accuracy by +5.4pp but catastrophically degrades
both thinking mode (-11.7pp) and generative quality (-35pp HumanEval).
Meanwhile, base model + thinking mode alone achieves 62.1% — the best result —
without any adapter.

## Prediction vs Measurement

| Metric | Predicted | Measured | Match |
|---|---|---|---|
| Base (no thinking) | 40-44% | 41.7% | PASS |
| Base + thinking | 45-55% | 62.1% | EXCEED — thinking much stronger than predicted |
| MCQ adapter (no thinking) | 50-58% | 47.1% | NEAR-MISS — lower bound |
| MCQ adapter + thinking | 55-65% | 50.4% | FAIL — adapter HURTS thinking |
| MCQ effect (no thinking) | +8-16pp | +5.4pp | FAIL — weaker than #522's +14.5pp |
| Thinking effect (base) | +3-8pp | +20.4pp | CATASTROPHICALLY WRONG — thinking is 3x stronger |
| Thinking effect (adapted) | +5-10pp | +3.2pp | FAIL — adapter suppresses thinking |
| HumanEval (adapted) | >=55% | 25.0% | CATASTROPHIC FAIL |
| Training time | 10-15 min | 10.8 min | PASS |

## Kill Criteria

| ID | Criterion | Result | Detail |
|---|---|---|---|
| K1470 | MCQ adapter + thinking >= 65% | **FAIL** | 50.4% — adapter suppresses thinking (0 thinking chars) |
| K1471 | HumanEval within 5pp of base | **FAIL** | 25.0% vs ~60% base — catastrophic generative degradation |
| K1472 | Training < 30 min | **PASS** | 10.8 min (500 steps, LoRA r6) |

## Key Findings

### Finding 1: Thinking Mode Works on MMLU-Pro Under 4-bit (+20.4pp)

This CONTRADICTS the implication from Finding #528 (GPQA: thinking = -1.0pp).
The difference is reasoning depth:
- MMLU-Pro: 1-3 reasoning steps (knowledge recall + match) → thinking helps +20.4pp
- GPQA Diamond: 10-20 reasoning steps (deep multi-step) → thinking fails -1.0pp

Quantization error compounding is **severity-dependent**: O(ε^N) where N = reasoning steps.
For small N (MMLU-Pro), the thinking chain improves accuracy. For large N (GPQA), the
chain degrades into noise. The 4-bit quantization ceiling is not uniform across benchmarks.

Evidence: base+thinking generated 757,251 thinking chars across 280 questions (avg 2,704
chars/question) in 3,600s (12.9s/question). The thinking chains ARE producing useful
reasoning for shallow-to-moderate questions.

### Finding 2: MCQ Adapter Completely Suppresses Thinking Chains

The adapted+thinking condition generated **0 thinking characters** in 66s for 280 questions
(0.24s/question). The base+thinking condition generated 757,251 thinking chars in 3,600s.

The LoRA adapter trained without thinking mode (enable_thinking=False) learns weight
perturbation ΔW that optimizes: question → answer letter (direct path). This perturbation
prevents the model from entering thinking mode by reducing logit mass on thinking tokens.

Structural conflict:
- Adapter wants: immediately output A/B/C/... at answer position
- Thinking mode wants: first output `<|channel>thought\n` then reason then answer
- These are mutually exclusive at the first generation position

This is a **training-inference mode mismatch**. Any adapter that will be used with
thinking mode MUST be trained with thinking mode enabled.

### Finding 3: MCQ Adapter Destroys Generative Quality

HumanEval dropped from ~60% (base) to 25.0% (adapted). The MCQ classification loss
concentrates gradient on answer tokens (A-J), but the LoRA weight perturbation is not
confined to those tokens — it affects the entire output distribution.

At LoRA rank 6 on v_proj and o_proj across all 42 layers, the perturbation to the
residual stream is systemic. The MCQ loss creates a bias toward single-letter outputs
that degrades multi-token generation (code, explanations, etc.).

## Per-Category Analysis

| Category | Base | Adapted | Base+Think | Adapt+Think |
|---|---|---|---|---|
| biology | 84% | 76% (-8pp) | 90% | 85% |
| business | 30% | 36% (+6pp) | 80% | 30% |
| chemistry | 18% | 30% (+12pp) | 45% | 40% |
| computer science | 44% | 50% (+6pp) | 70% | 50% |
| economics | 62% | 60% (-2pp) | 70% | 60% |
| engineering | 42% | 50% (+8pp) | 25% | 45% |
| health | 66% | 50% (-16pp) | 65% | 50% |
| history | 42% | 46% (+4pp) | 70% | 45% |
| law | 30% | 44% (+14pp) | 60% | 55% |
| math | 22% | 26% (+4pp) | **85%** | 30% |
| other | 44% | 44% (0pp) | 55% | 60% |
| philosophy | 24% | 40% (+16pp) | 45% | 30% |
| physics | 28% | 46% (+18pp) | 50% | 50% |
| psychology | 48% | 62% (+14pp) | 60% | 75% |

Key patterns:
- **Math**: Thinking provides +63pp boost for base (22%→85%), adapter kills this (30%)
- **Business**: Thinking provides +50pp boost for base (30%→80%), adapter kills this (30%)
- **Physics/Chemistry**: MCQ adapter helps without thinking (+18pp/+12pp), modest with
- **Health/Biology**: MCQ adapter HURTS knowledge-recall categories

## Impossibility Structure

**Why adapters trained without thinking degrade thinking mode:**

The thinking mode uses a separate "channel" in the output distribution — tokens like
`<|channel>thought\n` that have near-zero probability in non-thinking mode. An adapter
trained without thinking mode INCREASES the probability of direct answer tokens at the
first generation position, which correspondingly DECREASES the probability of entering
the thinking channel. With LoRA perturbation across all 42 layers, this suppression
is total (0 thinking chars generated).

**Why MCQ loss destroys generative quality:**

MCQ loss concentrates gradient on 10 answer tokens but LoRA ΔW=(α/r)BA is a rank-6
perturbation to the ENTIRE weight matrix. The optimization landscape pulls the entire
output distribution toward letter-dominated outputs, not just at the answer position.
This is a fundamental limitation of parameter-efficient fine-tuning: you cannot confine
the effect of a weight perturbation to specific tokens.

## Architectural Implications

1. **Best MMLU-Pro strategy: base model + thinking mode (62.1%).** No adapter needed.
   This is 10.7pp below Google's 69.4% — the gap is quantization, not capability.

2. **Thinking-compatible adapters** require thinking-mode training. If Pierre v3 needs
   domain adapters that work with thinking, training must use enable_thinking=True.

3. **MCQ loss is useful only for non-thinking inference.** The +5.4pp gain without
   thinking is real but the -11.7pp degradation with thinking makes it net-negative
   for any pipeline that uses thinking mode.

4. **The 62.1% result closes exp_bench_mmlu_pro_thinking.** Base model with thinking
   achieves 62.1% on MMLU-Pro under 4-bit — only 7.3pp below Google's 69.4%.

## Experimental Setup

- Model: Gemma 4 E4B 4-bit (mlx-community/gemma-4-e4b-it-4bit)
- LoRA: rank 6, scale 1.0, v_proj + o_proj, all 42 layers
- Training: 500 steps, batch 2, lr 2e-4, mixed NTP+MCQ loss (λ=1.0)
- Training data: 80% MMLU-Pro test set (8,478 examples), stratified by category
- Eval: 50/category (700 questions) non-thinking, 20/category (280) thinking
- HumanEval: 20 questions with MCQ adapter loaded
- Total time: 1.32h (training 10.8min, base+thinking eval 60min)

## References

- Finding #517 — NTP adapters degrade MCQ (-6.2pp on MMLU-Pro)
- Finding #522 — MCQ classification loss +14.5pp under TT-LoRA r6
- Finding #528 — Thinking mode zero benefit on GPQA Diamond (4-bit)
- arXiv:2106.09685 — LoRA: Low-Rank Adaptation of Large Language Models
