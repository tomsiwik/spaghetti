# E6 PAPER: First Strategy Adapter — Systematic Decomposition via Hedgehog Distillation

## Status: PROVISIONAL (smoke N=20, strong kill signal)

## Prediction vs Measurement

| Metric | Predicted | Measured | Pass/Fail |
|--------|-----------|----------|-----------|
| Training convergence | loss ratio ≥ 2.0 | 2.47 (0.108→0.044) | PASS |
| K_struct cos-sim | > 0.85 | 0.9587 | PASS |
| K2028 any domain +3pp | ≥1 domain improves | 0 domains improve | FAIL |
| K2029 2+ domains +3pp | ≥2 domains improve | 0 domains improve | FAIL |
| MMLU non-interference | drop < 5pp | drop = 10pp | FAIL |

### Per-Domain Results

| Domain | Base Acc | Adapter Acc | Delta (pp) | N |
|--------|----------|-------------|------------|---|
| GSM8K (math) | 15.0% | 0.0% | -15.0 | 20 |
| ARC-C (science) | 85.0% | 70.0% | -15.0 | 20 |
| MMLU-STEM | 80.0% | 75.0% | -5.0 | 20 |
| MMLU-general | 75.0% | 65.0% | -10.0 | 20 |

## Verdict: PROVISIONAL with strong KILL signal

K_struct PASS + K2028 FAIL → F#666 tautological-proxy pattern. The adapter matches teacher attention (cos=0.96) but degrades accuracy across all domains. This is the exact pattern documented in F#666.

## Mechanism Analysis

### 1. GSM8K Catastrophic Failure
The adapter makes GSM8K predictions null — the model generates text that contains NO extractable numbers despite explicit instructions ("Give ONLY the final numerical answer"). Base model extracts numbers from 3/20 outputs; adapter from 0/20. The decomposition attention patterns override the format instruction, causing the model to begin generating decomposed reasoning text instead of direct numerical answers.

### 2. MCQ Degradation (-5pp to -15pp)
For MCQ tasks (ARC-C, MMLU), the adapter produces valid letters but makes worse choices. The decomposition attention pattern shifts the model's internal processing in ways that interfere with direct answer selection. The adapter learns "how the teacher processes input" but this processing is optimized for decomposed generation, not concise answers.

### 3. Root Cause: Attention-matching ≠ behavioral transfer for strategies

The Hedgehog politeness adapter (F#783) succeeded because politeness is a SURFACE behavior — the attention pattern that produces polite language directly maps to polite output tokens. Systematic decomposition is a STRUCTURAL behavior — it changes the reasoning process, not the surface form.

Hedgehog cos-sim distillation matches attention outputs on the INPUT pass. But strategy effects manifest during GENERATION. The mismatch:

- **Teacher**: processes input with decomposition system prompt → generates step-by-step reasoning (teacher attention during generation is NOT captured by training)
- **Student**: matches teacher's INPUT attention patterns → but generation behavior is NOT constrained by the training objective

The adapter learns to make the model's input processing look like "decomposition-prompted processing," but this doesn't teach the model to actually decompose during generation. Instead, it shifts the model's output distribution in unpredictable ways — here, away from concise answers.

### 4. Comparison with E1 (F#801)

E1 failed because mean-difference extraction captures format signal, not strategy content. E6 avoids that failure (we train directly), but hits a NEW failure: attention matching on input processing doesn't transfer strategy behavior to generation. The strategy signal is in the generation process, not the input processing.

## Key Finding

**Hedgehog per-layer cos-sim distillation is effective for surface behaviors (politeness) but antagonistic for reasoning strategies (decomposition).** The method transfers attention patterns that shift output distribution away from direct answers, degrading accuracy rather than improving it.

This is a method-level failure, not a statistical artifact — all 4 domains degrade, and GSM8K adapter outputs contain zero extractable numbers.

## Implications

1. **E7 (Strategy Transfer Validation)** will hit the same wall if it uses Hedgehog distillation
2. **E9 (Composable CoT Primitives)** needs a different composition mechanism than attention matching
3. **Strategy adapters require generation-aware training** — SFT on strategy-eliciting data, or RL from strategy compliance rewards, not input-side attention matching
4. **E8 (Behavioral Eval v2)** should still be pursued — we need evals that detect strategy application even when accuracy is unchanged

## Smoke Gate

| Gate | Result |
|------|--------|
| A1: Phase B converges 2x | PASS (2.47x) |
| A2: cos-sim ≥ 0.85 | PASS (0.9587) |
| A3: GSM8K base ≥ 20% | FAIL (15%) |
| A4: adapter persists | PASS |

A3 failure is informative: Gemma 4 E4B without CoT only achieves 15% on GSM8K with single-number-output format. This confirms that short-form eval without thinking mode is inappropriate for reasoning benchmarks, but this doesn't change the adapter's negative direction.

## Assumptions

- enable_thinking=False is appropriate per F#790 (avoids truncation artifacts)
- max_tokens=8 is sufficient for MCQ and single-number GSM8K answers
- Smoke N=20 is sufficient to detect method-level failures (4/4 domains negative)
