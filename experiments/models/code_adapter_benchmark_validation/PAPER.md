# Code Adapter Benchmark Validation: Proof Verification Report

## Theorem (Restated from MATH.md)

The Superficial Alignment Hypothesis (LIMA, 2305.11206) predicts that an SFT code
adapter cannot degrade pre-trained knowledge (MMLU), while Code Reasoning Transfer
(Aryabumi et al., 2405.20535) predicts it should improve structured tasks (GSM8K,
HumanEval) by activating latent reasoning circuits.

## Predictions vs Measurements

| Prediction (from framework)             | Measured       | Match? |
|----------------------------------------|----------------|--------|
| MMLU: |delta| <= 2pp (LIMA)             | +8.0pp         | PARTIAL -- improved, not just preserved |
| GSM8K: adapter >= base + 5pp            | -18.0pp        | NO -- adapter DEGRADES math |
| HumanEval: adapter >= base + 10pp       | -15.0pp        | NO -- adapter DEGRADES code |
| Base MMLU >= 25% (above random)         | 38.0%          | YES |
| Base GSM8K >= 5% (above floor)          | 58.0%          | YES |
| Base HumanEval > 0% (above floor)       | 60.0%          | YES |

## Hypothesis

**Original:** Code adapter + BitNet-2B-4T improves real benchmark scores by >=10%
on structured tasks (math, code) while not degrading prose tasks.

**Verdict: KILLED.** The code adapter degrades both structured tasks substantially
while unexpectedly improving MMLU general knowledge.

## What This Experiment Found

### Key Results

| Benchmark   | Base BitNet-2B-4T | + Code Adapter | Delta   |
|-------------|-------------------|----------------|---------|
| MMLU (50)   | 38.0% (19/50)     | 46.0% (23/50)  | +8.0pp  |
| GSM8K (50)  | 58.0% (29/50)     | 40.0% (20/50)  | -18.0pp |
| HumanEval (20) | 60.0% (12/20)  | 45.0% (9/20)   | -15.0pp |

### Interpretation

1. **BitNet-2B-4T is stronger than expected.** 58% GSM8K and 60% HumanEval with
   no adapter at all demonstrates the base model has substantial pre-trained
   capability. This is consistent with Microsoft's published benchmarks.

2. **The code adapter HURTS the tasks it was expected to help.** This is the
   opposite of the prediction. The adapter trained on codeparrot/github-code-clean
   with rank-16 LoRA at scale=20.0 degrades both code generation (HumanEval
   -15pp) and math reasoning (GSM8K -18pp) on standardized benchmarks.

3. **The code adapter HELPS MMLU.** +8pp on MMLU suggests the adapter improves
   the model's ability to follow multiple-choice instructions, consistent with
   LIMA (SFT teaches format). The base model sometimes outputs verbose explanations
   instead of single letters; the adapter's instruction-following training helps
   extract the correct answer format.

4. **Finding #208 (code adapter universal best) was measuring format compliance,
   not capability.** The prior experiment used custom execution-based metrics on
   a simple instruction-following task format ("### Instruction / ### Response").
   The code adapter won because it best teaches THIS specific format, not because
   it improves reasoning. On standardized benchmarks with diverse prompt formats,
   the adapter actually degrades performance.

### Root Cause Analysis

The most likely explanation is **format-capability decoupling:**

- **Format improvement:** The SFT adapter teaches the model to follow instructions
  in the "### Instruction / ### Response" format. This helps on MMLU (where the
  prompt uses this format) and on our custom evals (which also use this format).

- **Capability degradation:** The LoRA perturbation at scale=20.0 disrupts the
  base model's pre-trained representations for code generation and math reasoning.
  The adapter was trained on github-code-clean data (raw code, not problems-and-solutions),
  which teaches code syntax but may interfere with the model's ability to solve
  novel problems step-by-step.

- **The TWO-WORLD problem resurfaces:** Finding #208 used the same instruction
  format for evaluation as for training. Standardized benchmarks use different
  formats and test different skills. The adapter creates a "two-world" split
  where it excels in its training format but degrades elsewhere.

### Inference Speed Impact

- Base model evaluation: 513.2s (120 problems)
- Adapter model evaluation: 2658.8s (120 problems)
- The adapter model is **5.2x slower** due to the LoRA computation overhead
  in the TernaryLoRALinear forward pass (STE quantization + extra matmuls).

## Kill Criteria Assessment

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| K614: MMLU non-degradation | delta >= -2pp | +8.0pp | **PASS** |
| K615: Structured improvement | GSM8K OR HumanEval delta >= 5pp | -18pp, -15pp | **FAIL** |
| K616: Above random baseline | Any benchmark above random | All above | **PASS** |

**Overall: KILLED by K615.**

## Limitations

1. **n=50/50/20 is small.** With n=50, a single question flip changes accuracy
   by 2pp. The directional signals are clear (-18pp, -15pp) but exact values
   have wide confidence intervals.

2. **Single prompt format.** All prompts used "### Instruction / ### Response"
   format. The base model may perform differently with its native prompt format.
   However, we kept the format constant between base and adapter for fair comparison.

3. **LoRA scale=20.0 was not ablated.** A lower scale might preserve base
   capabilities while still gaining format benefits. This was not tested.

4. **SFT training data was raw code, not benchmark-style problems.** A code
   adapter trained on problem-solution pairs (not raw code) might perform
   differently.

## What Would Kill This Further / Next Steps

1. **Ablate LoRA scale:** Test scale=1.0, 5.0, 10.0, 20.0 to find the sweet
   spot where format improves but capability does not degrade.

2. **Test without instruction format:** Run the base model with a simpler prompt
   (no "### Instruction" wrapper) to see if the base model performs even better
   natively.

3. **Train on problem-solution data:** Instead of raw code from github-code-clean,
   train the adapter on GSM8K training data or code-contest solutions.

## Implications for the Project

This experiment fundamentally challenges Finding #208 (code adapter universal best):

- **The adapter teaches format, not capability.** Finding #208's execution-based
  metrics were measuring instruction-following quality, not domain expertise.

- **Standardized benchmarks disagree with custom evals.** This is a critical
  validation: our custom eval framework (Finding #210) was validated for detecting
  RELATIVE differences between configs, but the absolute meaning of "better" was
  anchored to instruction-following format, not benchmark capability.

- **Base model is already strong.** BitNet-2B-4T achieves 58% GSM8K and 60%
  HumanEval without any adapter. The composition project should consider whether
  adapters are needed at all for this model, or whether the use case is specifically
  format adaptation for downstream applications.

## Key References

- LIMA (Zhou et al., 2305.11206): Superficial Alignment Hypothesis
- Code Reasoning Transfer (Aryabumi et al., 2405.20535): Code training improves reasoning
- Finding #208: Code SFT adapter is universal best (now challenged)
- Finding #210: Behavioral eval framework validated (now contextualized)
