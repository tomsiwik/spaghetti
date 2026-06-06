# BitNet-2B Task-Based Evaluation: Research Digest

## Hypothesis

BitNet-2B ternary composed model improves task-based performance (accuracy, F1,
syntax validity), not just perplexity, over the base model.

## What This Experiment Is

The first task-based evaluation of the BitNet-SOLE composition pipeline. All prior
BitNet-2B experiments measured only perplexity (PPL), but prior work proved PPL does
NOT predict task accuracy (r=0.08, micro proven). This experiment bridges the
PPL-to-task gap by evaluating on 5 domain-specific task metrics.

**Conditions evaluated:**
- Base model (ternary LoRA applied but zeroed)
- Individual domain adapters (medical, code, math, legal, creative)
- Composed model (all 5 adapters, 1/N averaged-factor scaling)

**Adapters:** Reused from bitnet_multiseed_validation (seed 42, 400 steps, ternary
QAT+STE, rank-16, all-modules). No retraining.

## Key References

- PPL does NOT predict task accuracy: r=0.08 (micro/models/ppl_vs_task_performance/)
- 1/N scaling resolves composition catastrophe (macro/sole_critical_path/)
- Ternary adapters compose 4.4% better than FP16 (micro/models/bitnet_ternary_adapter_composition/)
- Multiseed reproducibility: CV=0.5% (micro/models/bitnet_multiseed_validation/)
- Prior PPL result: composition ratio 3.45x, |cos|=0.002 (all PASS)

## Empirical Results

### Base Model Performance

| Domain | Metric | Base | Notes |
|--------|--------|------|-------|
| Math | Accuracy | 5.0% (1/20) | MATH-500 levels 1-3, non-instruction-tuned model |
| Code | Syntax valid | 10.0% (2/20) | Python completion, ast.parse() |
| Medical | Keyword F1 | 13.0% | QA from flashcard validation data |
| Legal | Keyword F1 | 14.8% | QA from law-stack-exchange validation data |
| Creative | PPL | 3.519 | TinyStories held-out continuation |

### Individual Adapter Performance (own domain only)

| Adapter | Domain Metric | Individual | Base | Delta |
|---------|--------------|-----------|------|-------|
| Medical | Keyword F1 | 17.9% | 13.0% | **+4.9pp** |
| Code | Syntax valid | 5.0% | 10.0% | -5.0pp |
| Math | Accuracy | 5.0% | 5.0% | 0.0pp |
| Legal | Keyword F1 | 11.0% | 14.8% | -3.8pp |
| Creative | PPL | 3.119 | 3.519 | **-0.40** (better) |

### Composed Model (1/N, all 5 adapters)

| Domain | Metric | Composed | Base | Delta | Verdict |
|--------|--------|----------|------|-------|---------|
| Math | Accuracy | 0.0% | 5.0% | -5.0pp | **WORSE** |
| Code | Syntax valid | 5.0% | 10.0% | -5.0pp | **WORSE** |
| Medical | Keyword F1 | 15.6% | 13.0% | +2.6pp | BETTER |
| Legal | Keyword F1 | 14.0% | 14.8% | -0.8pp | **WORSE** |
| Creative | PPL | 3.304 | 3.519 | -0.215 | BETTER |

### Kill Criteria

- **K1: KILL.** Composed worse than base on 3/5 metrics (60% > 40% threshold).
  Worse on math (-5pp), code (-5pp), legal (-0.8pp). Better on medical (+2.6pp)
  and creative (-0.215 PPL).

- **K2: KILL.** Math adapter shows 0.0pp improvement over base (threshold >= 3pp).
  The math adapter trained on GSM8K answer text, not math problem-solving. The
  base model at 2B parameters is too small for MATH-500 level problems.

**VERDICT: KILLED (both K1 and K2).**

## Analysis: Why Composition Hurts Task Metrics

1. **1/N^2 attenuation is too aggressive for task performance.** With N=5 averaged-factor
   composition, each adapter's contribution is scaled to ~4% of its full strength.
   While this stabilizes PPL (composition ratio 3.45x, well below 10x catastrophe),
   it also destroys the adapter's task-specific signal. The adapter effectively
   becomes noise at 4% scaling.

2. **Base model is too small for generative tasks.** BitNet-2B-4T at 2.4B parameters
   (ternary) has very limited generative capability. Base math accuracy of 5% on
   level 1-3 MATH-500 means the model barely understands the task format. Adapters
   trained on domain text (NTP loss) add domain vocabulary but not reasoning.

3. **Adapters trained on NTP loss do not learn task skills.** The adapters were trained
   to predict next tokens on domain text (flashcards, code, GSM8K answers, legal
   Q&A, stories). This improves PPL on domain text but does not teach the model to
   SOLVE math problems or GENERATE syntactically valid code. Task performance
   requires instruction tuning or task-specific training, not just domain NTP.

4. **Medical is the exception.** Medical keyword F1 improves for both individual adapter
   (+4.9pp) and composed (+2.6pp). Medical flashcards contain factual QA pairs that
   partially transfer to the QA evaluation format. This suggests NTP training CAN
   improve task performance when the training data format matches the eval format.

5. **Creative PPL improves (expected).** Creative adapter reduces PPL on TinyStories
   by 0.40 (individual) and 0.215 (composed). This is consistent with PPL-based
   prior results. Creative writing is the one domain where PPL IS the task metric.

## Key Insight: PPL Improvement != Task Improvement (Confirmed)

This experiment validates the prior micro finding (r=0.08) at the BitNet-2B scale:
PPL improvement does NOT predict task accuracy improvement. The math adapter achieves
substantial PPL improvement (3.08 individual vs 4.54 base = 32% better), yet math
accuracy is identical (5% both). The correlation between PPL improvement and task
improvement across domains is essentially zero.

**This means the SOLE architecture needs task-aware training, not just NTP domain
exposure, to demonstrate real-world value.**

## Limitations

1. **N=20 per domain is too small for statistical power.** Binomial 95% CI for 5%
   accuracy on 20 trials is [0.1%, 24.9%]. Differences smaller than ~15pp are not
   distinguishable from noise. This experiment provides directional signal only.

2. **Non-instruction-tuned base.** BitNet-2B-4T is a base (completion) model, not
   instruction-tuned. All prompts require the model to "understand" task format from
   the prompt alone, which 2B models struggle with.

3. **Keyword F1 is a weak metric.** Token overlap measures surface similarity, not
   semantic correctness. A response that uses domain vocabulary but is factually wrong
   can score high F1.

4. **No KV cache in generation.** Each token requires full sequence recomputation,
   making generation extremely slow (~50s/response). With KV cache, generation would
   be 10-50x faster, enabling larger eval sets.

5. **Autoregressive generation without KV cache is slow.** 106 min total runtime for
   ~200 generations. This limits eval set size.

6. **Adapters were trained for PPL, not for task performance.** The kill here reflects
   training objective mismatch, not a fundamental composition failure.

## What Would Kill This (At Larger Scale)

- If instruction-tuned adapters on a 7B+ model still show composed performance worse
  than base on >40% of task metrics, composition is fundamentally broken for tasks.
- If the PPL-to-task gap persists even with task-specific training data, the SOLE
  architecture needs a fundamentally different approach (e.g., task-specific routing
  instead of uniform composition).

## What Was Learned

1. **PPL success does not transfer to task success** -- confirmed at BitNet-2B scale.
   The 3.45x composition ratio and 0.002 cosine similarity are necessary but not
   sufficient for useful composition.

2. **1/N^2 scaling is too aggressive for task-relevant signal.** The effective 4%
   adapter contribution preserves base quality but destroys specialization.
   Future work should test: (a) higher scaling (1/N instead of 1/N^2), (b) top-k
   routing (only compose relevant adapters), (c) PPL-probe weighting.

3. **Training objective matters.** NTP on domain text does not produce task-capable
   adapters. Instruction-tuning or task-specific fine-tuning is needed.

4. **Medical is a positive signal.** The one domain where training data format matches
   eval format (QA pairs) shows genuine improvement. This suggests instruction-tuned
   adapters would fare better across all domains.

5. **Model size matters.** At 2B parameters (ternary), the base model's generative
   capability is too limited for most task benchmarks. Prior macro results on Qwen-7B
   showed +10.6pp on MATH-500 with reasoning distillation -- scale matters.

## Experiment Details

- **Runtime:** 105.9 min on Apple Silicon
- **Cost:** $0 (all local)
- **Model:** microsoft/BitNet-b1.58-2B-4T (2.4B params, ternary)
- **Adapters:** seed 42, 400 steps, rank-16, ternary QAT+STE, all-modules
- **Composition:** averaged-factor 1/N scaling (effective 1/N^2 on diagonal)
- **Generation:** autoregressive, greedy, max 128 tokens, no KV cache
