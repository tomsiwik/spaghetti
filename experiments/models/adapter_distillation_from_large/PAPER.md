# Adapter Distillation from Large Teacher: Research Digest

## Hypothesis

Distilling knowledge from Qwen2.5-7B into ternary LoRA adapters on
BitNet-2B-4T via sequence-level knowledge distillation produces adapters
with at least 5% better PPL than self-supervised training on the same
domain data.

**Verdict: KILLED.** Distilled adapters are 34.4% WORSE than self-supervised
adapters across all 5 domains (0/5 improved).

## What This Experiment Did

1. Loaded Qwen2.5-7B-Instruct-4bit (teacher) on Apple Silicon via MLX
2. Generated 100 domain-specific text samples per domain using chat-format
   prompts from the original training datasets
3. Trained rank-16 LoRA adapters on BitNet-2B-4T (student) using the
   teacher-generated text (distilled condition)
4. Trained identical adapters on the original training data (self-supervised
   control condition)
5. Evaluated both on the SAME original validation data

## Key References

- Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
  -- soft targets transfer dark knowledge
- TinyBERT (Jiao et al., 2020, arxiv 1909.10351) -- layer-wise distillation
- MiniLLM (Gu et al., 2024, arxiv 2306.08543) -- KD for LLMs
- Our exp_generation_quality_test LEARNINGS: quality bottleneck motivates this

## Empirical Results

### Per-Domain PPL on Original Validation Data

| Domain   | Baseline (500-sample) | Self-Supervised (100-sample) | Distilled | Imp vs SS |
|----------|----------------------:|-----------------------------:|----------:|----------:|
| python   |                  2.22 |                         2.35 |      2.95 |    -25.6% |
| math     |                  3.60 |                         4.06 |      6.42 |    -58.1% |
| medical  |                  4.74 |                         6.08 |      6.62 |     -8.9% |
| legal    |                 16.53 |                        19.52 |     26.55 |    -36.0% |
| creative |                  4.92 |                         5.46 |      7.83 |    -43.5% |
| **Avg**  |              **6.40** |                     **7.50** |  **10.08**| **-34.4%**|

### Kill Criteria

- **K1 FAIL**: Distilled PPL is 34.4% WORSE than self-supervised (threshold: 5% better).
  Not a single domain improved. All 5 are worse.
- **K2 PASS**: Peak memory 6.16 GB, well within 40GB budget.

### Training Convergence

All distilled adapters converged (loss decreased 40-55% over 200 steps).
The adapters DID learn -- they learned the WRONG distribution.

| Domain   | First-50 Loss | Last-50 Loss | Converged |
|----------|:-------------:|:------------:|:---------:|
| python   |         0.688 |        0.355 | Yes       |
| math     |         0.835 |        0.548 | Yes       |
| medical  |         1.247 |        0.676 | Yes       |
| legal    |         1.512 |        0.917 | Yes       |
| creative |         1.318 |        0.893 | Yes       |

Note: distilled training losses are LOWER than self-supervised (0.355 vs 0.699
for python), because the teacher-generated text has lower entropy (more predictable
Qwen-style output). Lower training loss + higher eval PPL = classic distribution
mismatch.

## Why This Failed: Distribution Mismatch

The core failure is a **distribution mismatch between teacher output and
evaluation data**. This is not a capacity problem or a training problem --
it is a data problem.

### The Mechanism

1. **Original training data** (from HuggingFace datasets):
   - Medical: "Very low Mg2+ levels correspond to low PTH levels which in
     turn results in low Ca2+ levels." (1 sentence, 17 words)
   - Python: raw code snippets
   - Math: step-by-step GSM8K solutions

2. **Teacher-generated data** (from Qwen2.5-7B-Instruct):
   - Medical: "The relationship between magnesium (Mg2+), parathyroid
     hormone (PTH), and calcium (Ca2+) levels is complex and interconnected,
     particularly in the context of kidney disease and bone health. Here's
     a detailed overview: ### 1. **Mg2+ Levels and PTH** ..." (200+ words)
   - Python: "Certainly! Below is a clean and correct Python function..."
   - Math: verbose explanations with markdown formatting

3. **Evaluation**: on original-style data (terse flashcards, raw code, GSM8K).

The distilled adapter learns to predict Qwen's verbose, markdown-heavy,
"Certainly!"-prefixed style. When evaluated on terse domain text, this
learned distribution is a terrible fit. The adapter's logits are optimized
for predicting explanation tokens, not flashcard-style facts.

### Evidence from Training Losses

The distilled adapter achieves LOWER training loss (0.355 vs 0.699 for
python) because Qwen's text is lower-entropy (more predictable boilerplate
like "Certainly!" and markdown headers). But this predictability is
IRRELEVANT to the evaluation distribution.

This is the same PPL-quality disconnect documented in generation_quality_test:
optimizing for one distribution's PPL does not transfer to another.

### Literature Confirmation

This failure mode is well-documented:
- **DistilBERT** (Sanh et al., 2019) succeeds because teacher and student
  share the same tokenizer AND the distillation data is the SAME distribution
  as the evaluation data (GLUE tasks).
- **MiniLLM** (Gu et al., 2024) uses reverse KL specifically to handle
  teacher-student distribution gaps -- but still assumes same-distribution
  evaluation data.
- **TinyBERT** (Jiao et al., 2020) adds data augmentation to bridge the
  distribution gap -- but within the same task distribution.

None of these papers test cross-distribution sequence-level distillation
(teacher generates different-style text, student evaluated on original style).

## What This Means for the Project

### Sequence-Level Distillation is Dead for This Use Case

When teacher and student use different tokenizers AND the evaluation
distribution differs from the teacher's generation distribution, sequence-level
KD cannot work. The adapter learns the wrong distribution.

### Potential Fixes (Not Tested)

1. **Same-tokenizer teacher**: Use a model with the same tokenizer as
   BitNet-2B-4T (unlikely -- BitNet uses its own 32K vocab). Or use
   logit-level distillation with vocabulary projection.

2. **Distill on evaluation-distribution data**: Instead of having the
   teacher generate new text, have it produce soft labels (logit
   distributions) on the ORIGINAL training data. Requires vocabulary
   alignment between teacher and student tokenizers.

3. **Style-matched generation**: Constrain the teacher to generate text
   in the same style/format as the original data. For medical flashcards:
   "Generate a single-sentence medical fact about [topic]." This would
   reduce the distribution gap but limits the teacher's knowledge transfer.

4. **DPO/RLHF instead of distillation**: Train adapters with preference
   optimization rather than distillation. This was already identified in
   generation_quality_test LEARNINGS as the most promising fix.

## Limitations

1. Only 100 samples per domain (vs 500 in baseline). The self-supervised
   control with 100 samples showed higher PPL than the 500-sample baseline
   (7.50 vs 6.40), confirming sample count matters. But this does not
   explain the distilled vs self-supervised gap (distilled uses the same
   100-sample count).

2. Teacher was 4-bit quantized. Full-precision teacher might generate
   higher-quality text. But the distribution mismatch problem is fundamental
   and would persist.

3. Only temperature=0.7 tested. Different temperatures might produce
   different text styles. But the core issue is distribution mismatch,
   not generation quality.

## What Would Kill This

Already killed. K1 FAIL: -34.4% vs the required +5.0%.

The result is unambiguous: 0/5 domains improved, average degradation is
34.4%, worst case (math) is 58.1% worse. No hyperparameter tuning can
fix a fundamental distribution mismatch.
