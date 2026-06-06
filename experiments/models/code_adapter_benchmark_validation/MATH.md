# Code Adapter Benchmark Validation: Mathematical Framework

## Experiment Type: Guided Exploration

This is a **Type 2 (guided exploration)** experiment. The proven framework is the
Superficial Alignment Hypothesis (LIMA, 2305.11206) combined with our Finding #208
(code adapter is universal best across 5 domains). The unknown being explored is
whether this advantage transfers to standardized benchmarks (MMLU, GSM8K, HumanEval).

## A. Failure Mode Identification

**Potential degenerate behavior:** The code adapter's apparent superiority in Finding #208
could be an artifact of:
1. **Evaluation metric overfitting:** Custom execution-based metrics may not correlate
   with standardized benchmark performance
2. **Format compliance, not knowledge:** SFT teaches the model to output in expected
   formats (code blocks, "the answer is X") without improving actual reasoning
3. **Domain leakage:** Code training data (codeparrot/github-code-clean) may contain
   benchmark-like problems, creating false improvement signal

**Is this the root cause or a symptom?** The disease is that SFT adapter evaluation
has been done exclusively with custom metrics. Standardized benchmarks provide an
independent, externally validated measurement of the same capability claims.

## B. The Right Question (Reframe)

**Wrong question:** "Does the code adapter improve benchmark scores?"

**Right question:** "Does the Superficial Alignment Hypothesis predict that a code SFT
adapter, which teaches structured output format and activates latent reasoning circuits,
will measurably improve performance on tasks requiring structured reasoning (GSM8K,
HumanEval) while preserving or improving general knowledge retrieval (MMLU)?"

The answer comes from two established results:
1. **LIMA (Zhou et al., 2305.11206), Section 4:** A model's knowledge and capabilities
   are learned during pre-training; SFT teaches format and style. Therefore, a code
   adapter should NOT degrade MMLU (it does not erase pre-trained knowledge).
2. **Code Reasoning Transfer (Aryabumi et al., 2405.20535), Section 3:** Training on
   code data improves reasoning capabilities even on non-code tasks, because code
   enforces logical structure. Therefore, a code adapter SHOULD improve GSM8K
   (math reasoning benefits from code-style logical structure).

## C. Predictions From the Framework

### Prediction 1: MMLU Non-Degradation (from LIMA)
If SFT only teaches format without erasing knowledge, then:
- Base MMLU accuracy: A_base
- Adapter MMLU accuracy: A_adapter
- Prediction: |A_adapter - A_base| <= 0.02 (within noise)
- The code adapter teaches instruction-following format; MMLU tests knowledge recall
  in multiple-choice format. SFT should not degrade recall of pre-trained knowledge.

### Prediction 2: GSM8K Improvement (from Code Reasoning Transfer)
If code training activates reasoning circuits (Aryabumi et al., 2405.20535):
- Base GSM8K accuracy: G_base
- Adapter GSM8K accuracy: G_adapter
- Prediction: G_adapter >= G_base + 0.05 (at least 5pp improvement)
- GSM8K requires step-by-step mathematical reasoning. Code training teaches
  sequential logical decomposition. Finding #208 showed 7x improvement on custom
  math eval (7/10 vs 1/10 correct).

### Prediction 3: HumanEval Improvement (direct domain match)
Code adapter directly trains on code data:
- Base HumanEval pass@1: H_base
- Adapter HumanEval pass@1: H_adapter
- Prediction: H_adapter >= H_base + 0.10 (at least 10pp improvement)
- This is the most direct test: code adapter on code benchmark.

### Prediction 4: Above-Random Baseline
- MMLU random baseline: 25% (4-choice multiple choice)
- If BOTH base and adapter score < 25% on MMLU, the model lacks sufficient
  pre-trained knowledge for meaningful evaluation (K616 trigger).

## D. Kill Criteria Derivation

**K614 (MMLU degradation):** Derived from LIMA. If SFT teaches format without
erasing knowledge, MMLU degradation > 2pp means the adapter IS erasing knowledge,
violating the Superficial Alignment Hypothesis for this model.
- Threshold: A_adapter < A_base - 0.02
- If triggered: The code SFT data contains anti-correlated information, or
  LoRA scale 20.0 is too aggressive for knowledge preservation.

**K615 (No real gain on structured tasks):** Derived from Code Reasoning Transfer.
If code training does NOT improve GSM8K AND HumanEval by at least 5pp each,
then the reasoning transfer effect does not manifest at this model scale/quality.
- Threshold: (G_adapter - G_base < 0.05) AND (H_adapter - H_base < 0.05)
- If triggered: BitNet-2B-4T's ternary weights may limit the expressiveness of
  LoRA perturbations, or the adapter was undertrained.

**K616 (Below random):** Sanity check.
- Threshold: A_base < 0.25 AND G_base < 0.05 AND H_base = 0.0
- If triggered: The base model is too weak for benchmark evaluation.
  Prior work suggests BitNet-2B-4T is competitive with similar-scale models,
  so this would indicate an implementation error.

## E. Assumptions & Breaking Conditions

1. **BitNet-2B-4T has sufficient pre-trained knowledge for MMLU.** If violated,
   K616 triggers. Microsoft reports competitive performance on standard benchmarks
   for this model.

2. **SFT adapter at LoRA scale 20.0 does not catastrophically interfere with
   base model weights.** If violated, K614 triggers. Prior experiments show
   the adapter is well-behaved at this scale.

3. **Multiple-choice format is compatible with the instruction template.**
   The model must be able to output "A", "B", "C", or "D" reliably.
   If it cannot follow this format, MMLU scores will be artificially low.

4. **GSM8K answer extraction works with this model's output format.**
   Finding #210 validated answer extraction for custom math problems.
   GSM8K uses "#### [number]" format which we already support.

5. **HumanEval execution is safe and feasible.** We will sandbox code execution
   to prevent harmful operations. The model must generate syntactically valid
   Python that can be executed.

## F. Worked Example

Not applicable for benchmark evaluation (no matrix computation). Instead, the
"worked example" is the benchmark protocol itself:

**MMLU Example:**
```
Question: What is the capital of France?
A) London  B) Paris  C) Berlin  D) Madrid
Model output: "B"
Correct: B -> score = 1.0
```

**GSM8K Example:**
```
Question: Janet has 12 apples. She gives 3 to her friend. How many does she have?
Model output: "Janet starts with 12 apples. She gives away 3. 12 - 3 = 9. #### 9"
Extract answer: 9.0
Correct answer: 9.0
Match: True -> score = 1.0
```

**HumanEval Example:**
```python
def has_close_elements(numbers, threshold):
    # Model generates function body
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
# Execute test cases -> pass@1 = 1.0
```

## G. Complexity & Architecture Connection

- **Inference cost:** Same as base model + LoRA overhead (~1% additional FLOPs
  for rank-16 LoRA on d=2560 model)
- **Memory:** Base model (~1.7GB) + adapter (~50MB) fits easily in 48GB
- **Benchmark runtime:** 50 MMLU + 50 GSM8K + 20 HumanEval = 120 generations
  at ~200 tokens each. Expected runtime: ~15-25 minutes on M5 Pro.

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   The Superficial Alignment Hypothesis (LIMA) predicts SFT cannot erase pre-trained
   knowledge, making MMLU degradation impossible under correct SFT.

2. **Which existing theorem(s) does the proof build on?**
   LIMA (Zhou et al., 2305.11206, Section 4): SFT teaches format, not knowledge.
   Code Reasoning Transfer (Aryabumi et al., 2405.20535, Section 3): Code data
   activates cross-domain reasoning.

3. **What specific numbers does the proof predict?**
   - MMLU: |delta| <= 2pp
   - GSM8K: adapter >= base + 5pp
   - HumanEval: adapter >= base + 10pp
   - Base MMLU >= 25% (above random)

4. **What would FALSIFY the proof (not just the experiment)?**
   If code SFT degrades MMLU by >5pp, LIMA is wrong for ternary models (SFT
   CAN erase knowledge in low-precision settings).

5. **How many hyperparameters does this approach add?**
   Count: 0 (we use existing adapter with existing LoRA scale)

6. **Hack check:** No fixes being stacked. This is pure evaluation.
