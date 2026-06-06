# Answer-Conditioned Scoring: Research Digest

## Hypothesis

Answer-only perplexity (computed over answer tokens after the task delimiter)
correlates with task accuracy (Pearson r >= 0.5) where full-sequence PPL fails
(predecessor showed r = 0.084).

**Falsifiable:** If answer-only PPL still shows r < 0.5 with accuracy, or if
answer-only PPL ranking matches the full-sequence PPL ranking (no improvement),
the hypothesis is killed.

## What This Experiment Is

A direct fix to the killed ppl_vs_task_performance experiment. The predecessor
proved that full-sequence perplexity is a misleading quality signal for expert
evaluation: reverse and sort experts showed WORSE full-sequence PPL (-27%, -7%)
but BETTER accuracy (+9.5pp, +14pp). The root cause was identified in the
predecessor's own analysis: full-sequence PPL averages over prompt and answer
tokens, and prompt degradation masks answer improvement.

The fix is surgical: instead of computing PPL over the entire sequence
"prompt + delimiter + answer", compute it only over the answer tokens (after
the delimiter). This preserves the full conditioning context (the model still
sees the prompt) but restricts the metric to the tokens that actually
determine task correctness.

Five synthetic domains with structured tasks (same as predecessor):

| Domain | Format | Delimiter | Task |
|--------|--------|-----------|------|
| arithmetic | "2+3=5" | = | Solve addition |
| reverse | "abc>cba" | > | Reverse strings |
| repeat | "ab*3=ababab" | = | Repeat patterns |
| sort | "bca>abc" | > | Sort characters |
| parity | "1011>even" | > | Determine bit parity |

## Lineage in the Arena

```
macro/lora_moe_benchmark (4-domain MoE, PPL only)
  |
  +-- micro/ppl_vs_task_performance (KILLED: r=0.084)
  |   (5 domains, full-seq PPL vs accuracy, no correlation)
  |
  +-- THIS: micro/answer_conditioned_scoring (SURVIVES: r=0.811)
      (same 5 domains, answer-only PPL vs accuracy, strong correlation)
```

## Key References

- ppl_vs_task_performance (predecessor): showed the problem (r=0.084) and
  identified the root cause (prompt token dilution in full-sequence PPL).
- Shadow scoring in VISION.md: uses per-token perplexity as the quality
  signal for clone-and-compete evolution; this experiment validates that
  answer-conditioned PPL is the right formulation.
- Hoffmann et al. 2022 (Chinchilla): scaling laws assume PPL predicts
  downstream quality; our result refines this: only answer-conditioned
  PPL reliably predicts task accuracy.

## Empirical Results

### Per-Domain Results (Seed 42, representative)

| Domain | FullPPL Improv | AnsPPL Improv | Acc Improv |
|--------|---------------|---------------|------------|
| arithmetic | +9.1% | +20.7% | +12.5pp |
| reverse | **-1.8%** | **+58.5%** | **+40.0pp** |
| repeat | +5.3% | +3.0% | +2.0pp |
| sort | **-26.5%** | **+26.4%** | **+27.5pp** |
| parity | +3.5% | +5.1% | +16.0pp |

The reverse and sort experts show the critical divergence: full-sequence PPL
gets WORSE (prompt distribution shift), but answer-only PPL gets BETTER
(expert learns correct answer patterns). Full-sequence PPL would PENALIZE
these experts; answer-only PPL correctly REWARDS them.

### Correlation Across Seeds

| Seed | r(Full, Acc) | r(Answer, Acc) | r(Prompt, Acc) | K1 | K2 |
|------|-------------|---------------|---------------|----|----|
| 42 | -0.502 | **0.907** | -0.941 | PASS | PASS |
| 123 | -0.310 | **0.943** | -0.821 | PASS | PASS |
| 7 | -0.111 | **0.582** | -0.459 | PASS | PASS |
| **Mean** | **-0.308** | **0.811 +/- 0.16** | **-0.740** | **3/3** | **3/3** |

Additional metrics (seeds 42, 123):
- Spearman rho(Answer, Acc) = 0.90 (p=0.037), statistically significant
- Answer PPL rank agreement with accuracy ranking: 83-90% vs full-seq: 20-60%

### Kill Criteria Assessment

**K1: Answer-only Pearson r >= 0.5?**
Mean r = 0.811 +/- 0.16. All three seeds pass (0.58, 0.91, 0.94).
**PASSES.**

**K2: Rankings differ from full-sequence PPL?**
Full-sequence and answer-only PPL rankings differ in all 3 seeds.
Answer-only ranking is much closer to accuracy ranking (83-90% agreement
vs 20-60% for full-sequence).
**PASSES.**

**Overall verdict: SURVIVES.** Answer-only PPL is a dramatically better
proxy for task accuracy than full-sequence PPL, improving from r = -0.31
(anti-correlated!) to r = 0.81 (strongly correlated).

## The Mechanism

The decomposition is mathematically clean:

    log PPL_full = (T_p/T) * log PPL_prompt + (T_a/T) * log PPL_answer

When a domain expert specializes:
1. **PPL_answer improves** (expert learns correct answer patterns)
2. **PPL_prompt degrades** (expert loses general prompt modeling quality)
3. **PPL_full = weighted average**, potentially dominated by prompt degradation

The prompt degradation is especially severe for reverse and sort domains
because the answer character patterns (reversed or sorted) are very different
from the prompt patterns (random order). The expert must shift its internal
representations to produce these patterns, which hurts prompt modeling.

Answer-only PPL isolates the signal that matters: how well the model
predicts the correct answer tokens, given the full prompt context.

Prompt-only PPL is actually ANTI-correlated with accuracy (r = -0.74),
confirming that the prompt distribution shift is the confounding factor.

## Implications for SOLE Architecture

1. **Shadow scoring should use answer-conditioned PPL.** For any task with
   a clear prompt/completion boundary (which includes all instruction-following),
   PPL should be computed only on the completion tokens.

2. **For free-form generation** (no clear delimiter), the boundary can be
   approximated by: (a) the last user turn in a conversation, (b) the
   system-prompted instruction boundary, (c) any structural delimiter
   (newline after a question, code after a docstring, etc.).

3. **The fix is cheap.** Computing answer-only PPL requires knowing the
   delimiter position, which is available in any structured prompt. The
   computation cost is identical to full-sequence PPL (same forward pass,
   different token mask for the average).

4. **Shadow scoring becomes viable.** The predecessor concluded that
   "shadow scoring needs task-specific evaluation." This experiment shows
   that answer-conditioned PPL achieves r=0.81 correlation with accuracy,
   making it a viable quality signal for clone-and-compete without requiring
   domain-specific evaluation infrastructure.

5. **PPL_prompt as a complementary signal.** Prompt PPL degradation (r=-0.74
   with accuracy) could be used as a SAFETY signal: large prompt PPL
   degradation indicates the expert has shifted its distribution significantly,
   which warrants monitoring even if answer quality improved.

## Micro-Scale Limitations

- **Character-level tokenizer (V=42)** creates sharper distributions than
  subword tokenization (V=32K+). At larger V, the PPL differences may be
  smaller in magnitude but the directional finding should hold.

- **Synthetic structured tasks** have deterministic correct answers and clear
  delimiters. Real-world tasks (summarization, creative writing) have
  distributional answers and fuzzier prompt/completion boundaries.

- **N=5 domains** is small for correlation analysis (r_crit=0.687 at p<0.05).
  2 of 3 seeds exceed this threshold; all 3 exceed the kill criterion of 0.5.

- **Small model (d=32, L=2, ~29K params)** vs predecessor's (d=64, L=4, ~206K).
  The phenomenon is robust across both architectures, suggesting it is
  metric-level rather than model-level.

- **Full-rank expert delta** (not LoRA). The predecessor used rank-8 LoRA.
  The PPL decomposition mechanism does not depend on the parameterization
  of the expert delta.

- **autograd-based training** (numpy autodiff) vs predecessor's PyTorch.
  Different optimizer dynamics may shift absolute PPL values but the
  relative decomposition structure is preserved.

## What Would Kill This

At micro scale:
- If domains with T_p << T_a (very short prompts, very long answers) show
  full-sequence PPL correlating better with accuracy than answer-only PPL
  (because prompt dilution is minimal when T_p is small).
- If increasing to 20+ domains shows answer-only r dropping below 0.5 due
  to domains where the prompt-answer boundary is ambiguous.

At macro scale:
- If subword tokenization (V=32K+) smooths the per-token distributions enough
  that full-sequence and answer-only PPL converge (reducing the signal).
- If real-world tasks with distributional answers (not deterministic) show
  weaker answer-PPL/accuracy correlation because PPL rewards distributional
  breadth while accuracy rewards mode sharpness.
- If the delimiter position is ambiguous or unavailable for free-form generation
  tasks, making the metric inapplicable.

## What Was Learned

1. **Answer-conditioned PPL works.** r = 0.81 (mean across 3 seeds), a massive
   improvement over the predecessor's r = 0.08 for full-sequence PPL.

2. **The decomposition insight from the predecessor was correct.** The PAPER.md
   of ppl_vs_task_performance explicitly predicted this: "answer-conditioned PPL
   (computing loss only after the task delimiter) would be a better proxy."

3. **Full-sequence PPL is actively misleading.** Not just uncorrelated (r~0) but
   anti-correlated (r = -0.31) in this experiment. Using it for shadow scoring
   would systematically prune the BEST experts.

4. **Prompt PPL degradation is diagnostic.** The strong anti-correlation
   (r = -0.74) between prompt PPL improvement and accuracy improvement
   explains why: the experts that improve accuracy most are those that shift
   their internal representations most aggressively toward the domain, which
   hurts prompt modeling but helps answer generation.

5. **Shadow scoring is viable for evolution.** With r=0.81, answer-conditioned
   PPL is a sufficiently strong signal for clone-and-compete tournaments.
   No domain-specific evaluation infrastructure is needed for the basic
   evolution loop.
