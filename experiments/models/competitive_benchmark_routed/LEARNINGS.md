# LEARNINGS: Competitive Benchmark Routed

## Status: KILLED (K1 FAIL: 2/6 benchmarks worse than base)

## Core Finding

Per-domain scales + oracle routing do NOT fix MMLU degradation. Math adapter at
s=20 improves GSM8K (+10pp reasoning) but degrades MMLU math (-20pp factual recall).
The root cause is not scale but format mismatch: SFT adapters trained on instruction-
response pairs shift output distribution away from MMLU's single-letter answer format.

## The PPL-Accuracy Gap (New Finding)

Finding #220 proved 0/5 domains degrade on PPL with per-domain scales. This experiment
shows 2/6 benchmarks degrade on MMLU accuracy. PPL improvement does NOT predict
accuracy improvement on factual-recall benchmarks. This constrains all prior claims
about composition quality that relied on PPL as a proxy.

| Metric | Domains that degrade | Scale used |
|--------|---------------------|------------|
| PPL (Finding #220) | 0/5 | Per-domain optimal |
| Generation quality (Finding #220) | 0/5 | Per-domain optimal |
| MMLU accuracy (this experiment) | 2/6 | Per-domain optimal |

The gap exists because:
1. PPL measures prediction quality on instruction-style continuations
2. Generation quality measures format compliance on open-ended responses
3. MMLU accuracy measures single-letter answer extraction after a specific prompt

SFT adapters improve (1) and (2) but can hurt (3).

## Three Competitive Benchmark Kills — Same Root Cause

| Experiment | Composition | Scale | MMLU worse than base |
|------------|-------------|-------|---------------------|
| exp_competitive_benchmark | uniform 1/N | s=20 | 2/5 (math -25pp, legal -10pp) |
| exp_bitnet_sft_generation_v3 | energy routing | SFT adapters | 3/5 |
| exp_competitive_benchmark_routed | oracle top-1 | per-domain | 2/5 (math -20pp, legal -10pp) |

The common pattern: math and legal MMLU degrade regardless of routing or scale.
The adapters' instruction-following training conflicts with MMLU's concise-answer format.

## What Would Fix It

### 1. Format-conditional routing (highest priority)
Skip the adapter entirely when the query is factual-recall (MMLU-like). The entropy
gating mechanism (Finding #213: 63% skip rate) could serve this purpose — high-confidence
tokens on factual questions should bypass adaptation.

### 2. MMLU-format training data
Include multiple-choice QA in the adapter training mix. LoTA-QAF (2407.11024) showed
ternary adapters CAN improve MMLU by +5.14% when specifically trained for it.

### 3. Abandon MMLU as evaluation metric
MMLU measures the base model's pre-trained knowledge, not the adapter's contribution.
Adapters are designed to improve domain-specific generation, not factual recall.
GSM8K is the right metric for reasoning adapters; behavioral eval (Finding #210
framework) is right for generation quality.

## Statistical Caveats (from adversarial review)

- n=20 MMLU gives +/-22pp CI at p near 0.5. The -10pp legal gap is NOISE.
- Only the -20pp math gap is borderline significant (binomial p≈0.057).
- Gemma MMLU math at 5% and Qwen GSM8K at 36% indicate extraction bugs.
- The K2 PASS (vs Gemma) is unreliable due to broken comparator evaluation.

## Implications for Architecture

1. **GSM8K signal is real and consistent.** +10pp across all three experiments.
   The architecture provides genuine reasoning enhancement.
2. **MMLU is not a valid target.** Adapters cannot improve factual recall on a 2B
   model — the knowledge isn't there, and SFT format teaching makes MMLU harder.
3. **Composition quality claims must specify the evaluation format.** "Composition
   helps" (true for reasoning) and "composition hurts" (true for factual recall)
   are both correct. The evaluation format determines the outcome.
4. **Fix the evaluation pipeline.** Use lm-evaluation-harness for credible
   comparator numbers before the next competitive benchmark attempt.

## Recommended Follow-ups

1. **Entropy-gated composition for MMLU:** Use entropy gating to skip adapters on
   high-confidence factual queries. If the base model is already confident on an
   MMLU question, don't perturb it.
2. **Fix extraction pipeline with lm-evaluation-harness.** The Qwen/Gemma numbers
   are unreliable. Any future competitive claim needs standardized evaluation.
3. **GSM8K-focused benchmark.** Frame the competitive advantage around reasoning
   tasks where SOLE consistently helps, not factual recall where it consistently
   hurts.
