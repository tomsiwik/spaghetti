# PPL vs Task Performance: Research Digest

## Hypothesis

Perplexity improvement from domain-specific LoRA experts correlates with
downstream task accuracy (Pearson r >= 0.5).

**Falsifiable:** If Pearson r(PPL improvement, accuracy improvement) < 0.5
across 5 domains, or if the expert with the best PPL improvement does not
have the best accuracy improvement, the hypothesis is killed.

## What This Experiment Is

A direct test of whether perplexity -- the metric SOLE uses for shadow scoring
and clone-and-compete evolution -- is a valid proxy for actual task performance.

We train a base character-level transformer on 5 synthetic structured domains,
then train domain-specific LoRA experts. For each expert, we measure both
held-out perplexity (what SOLE evolution optimizes) and task-specific accuracy
(what users actually care about). If these don't correlate, the evolution
mechanism optimizes the wrong signal.

Five synthetic domains with deterministic correct answers:

| Domain | Format | Task |
|--------|--------|------|
| arithmetic | "2+3=5" | Solve addition |
| reverse | "abc>cba" | Reverse strings |
| repeat | "ab*3=ababab" | Repeat patterns |
| sort | "bca>abc" | Sort characters |
| parity | "1011>even" | Determine bit parity |

## Lineage in the Arena

```
macro/lora_moe_benchmark (4-domain MoE, PPL only)
  |
  +-- THIS: micro/models/ppl_vs_task_performance
  |   (5 domains, PPL vs task accuracy correlation)
  |
  +-- [BLOCKED] exp_code_execution_self_learning
      (needs task accuracy signal, not just PPL)
```

## Key References

- Shadow scoring in VISION.md: uses per-token perplexity as the quality signal
  for clone-and-compete evolution.
- Hoffmann et al. 2022 (Chinchilla): scaling laws use loss (= log PPL) as the
  primary training metric, implicitly assuming PPL predicts downstream quality.
- Gadre et al. 2024 (DataComp-LM): find perplexity on curated data predicts
  downstream benchmarks at scale, but with significant variance per-task.

## Empirical Results

### Per-Domain Results (Seed 42)

| Domain | Base PPL | Expert PPL | PPL Improv | Base Acc | Expert Acc | Acc Improv |
|--------|----------|------------|------------|----------|------------|------------|
| arithmetic | 2.73 | 2.59 | +5.1% | 0.675 | 0.865 | +19.0pp |
| reverse | 5.26 | 6.69 | **-27.0%** | 0.810 | 0.905 | +9.5pp |
| repeat | 1.56 | 1.48 | +4.8% | 1.000 | 1.000 | +0.0pp |
| sort | 4.32 | 4.64 | **-7.4%** | 0.720 | 0.860 | +14.0pp |
| parity | 1.70 | 1.65 | +3.0% | 0.865 | 1.000 | +13.5pp |

Key observation: **reverse** and **sort** experts get WORSE perplexity but
BETTER task accuracy. The expert learns the structured answer pattern but
loses some ability to model the prompt distribution, inflating full-sequence
PPL while improving the task-relevant output.

### Correlation Across Seeds

| Seed | Pearson r | K1 (r >= 0.5) | K2 (best PPL = best Acc) |
|------|-----------|---------------|--------------------------|
| 42 | 0.046 | KILL | PASS |
| 123 | 0.218 | KILL | PASS |
| 7 | -0.011 | KILL | FAIL |
| **Mean** | **0.084 +/- 0.097** | **0/3 PASS** | **2/3 PASS** |

### Kill Criteria Assessment

**K1: Pearson r >= 0.5?**
Mean r = 0.084. All three seeds below threshold (max = 0.218).
**KILLED.**

**K2: Best PPL improvement = Best accuracy improvement?**
Passes 2/3 seeds (coincidental alignment on arithmetic and parity).
**PARTIAL FAIL.**

**Overall verdict: KILLED.** PPL improvement does not correlate with task
accuracy improvement. The correlation is essentially zero (r = 0.08).

## The Mechanism: Why PPL and Accuracy Diverge

The divergence has a clean mathematical explanation. Full-sequence perplexity
averages over ALL token positions:

    PPL = exp(-(L_prompt + L_answer) / (T_prompt + T_answer))

But task accuracy depends ONLY on the answer portion (greedy decode after
the delimiter). A domain expert that specializes in producing correct answers
may lose prompt modeling quality because:

1. It trains only on one domain (not the mixed distribution the base saw)
2. The answer patterns are highly structured (deterministic), making them
   easy to learn even as the prompt distribution shifts
3. PPL penalizes ANY prediction error equally, but accuracy only cares
   about the top-1 prediction on answer tokens

The **reverse** domain is the clearest example: the expert learns to reverse
strings perfectly (+9.5pp accuracy), but the character bigram distribution in
reversed strings is very different from forward strings, causing PPL to spike
on the output side of the sequence too (the "reverse-ness" of the text is
lower probability under the general model, even when correct).

## Implications for SOLE Architecture

This result has direct consequences for the SOLE evolution mechanism:

1. **Shadow scoring with raw PPL is misleading.** A clone that improves task
   accuracy by 14pp could be pruned because its PPL increased 7%.

2. **Answer-conditioned PPL** (computing loss only after the task delimiter)
   would be a better proxy. This requires knowing where the "answer" starts,
   which is available for structured tasks but harder for free-form generation.

3. **Task-specific evaluation** may be necessary for high-stakes domains
   (code execution, medical QA). PPL can serve as a cheap pre-filter, but
   the final tournament should use task-specific metrics.

4. **The tension is fundamental, not an artifact of micro scale.** PPL
   measures distributional fit across the full output; accuracy measures
   functional correctness of specific outputs. These are different objectives
   and there is no mathematical guarantee they align.

5. **Practical recommendation:** Use PPL as a necessary-but-not-sufficient
   condition. An expert that catastrophically degrades PPL (>50% worse) is
   probably broken. But among experts with reasonable PPL, task-specific
   metrics should drive the tournament.

## Micro-Scale Limitations

- Character-level tokenizer (V=42) creates sharper distributions than
  subword tokenization (V=32K+), potentially exaggerating the divergence.
- Synthetic structured tasks have deterministic answers; real-world tasks
  have distributional correct answers, which may align better with PPL.
- 5 domains is a small sample for correlation analysis (N=5 is not enough
  for statistical significance; r_crit = 0.687 at p<0.05).
- The "prompt vs answer" decomposition is clean in structured tasks but
  messier in free-form generation where there is no clear delimiter.
- All experts were trained for the same number of epochs; optimal training
  duration may vary by domain and affect the PPL-accuracy relationship.

## What Would Kill This (if it had survived)

At micro scale:
- If increasing to 20+ domains showed r > 0.5 (N=5 may be too noisy)
- If answer-only PPL showed strong correlation (then full-sequence PPL
  is the wrong metric, but "some form of PPL" still works)

At macro scale:
- If Qwen2.5-7B experts on real benchmarks (HumanEval, GSM8K, MedMCQA)
  show PPL-accuracy correlation that micro-scale didn't capture
- If subword tokenization smooths the distribution enough to align PPL
  with accuracy
- If PPL measured on the SAME distribution the expert was trained on
  (rather than held-out from the base distribution) correlates better

## What Was Learned

Even though the hypothesis was killed, the result is highly informative:

1. **All experts improve accuracy** (0-21pp improvement), confirming that
   domain-specific LoRA training works for structured tasks.
2. **PPL is not useless** -- it correctly identifies repeat (already at
   ceiling) and catches that reverse/sort have distribution shift.
3. **The decomposition insight** (prompt PPL vs answer PPL) suggests a
   concrete fix: compute evolution metrics on task-relevant tokens only.
4. **This validates the reviewer's attack.** The adversarial review was
   correct: "you only measure perplexity" is a real vulnerability. The
   shadow scoring mechanism needs augmentation with task-specific signals.
