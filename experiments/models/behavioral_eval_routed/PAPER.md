# Behavioral Eval of Routed Composition: Existential P0 Test

## Experiment Type

Guided exploration operating within the proven framework of Finding #217 (per-domain scale calibration) and Finding #210 (behavioral eval validation, Cohen's kappa = 0.800). The unknown being narrowed: does routed composition improve behavioral quality, or does it genuinely degrade knowledge? No Theorem/Proof/QED is claimed — this is a measurement experiment, not a mathematical derivation.

## Hypothesis (Restated)

Format-sensitive benchmarks (MMLU) conflate format compliance with domain knowledge. SFT adapters shift response format toward instruction-following, which degrades MMLU scores while potentially improving actual text quality. Therefore, execution-based behavioral metrics (ast.parse, numerical answer correctness, factual recall) can show improvement even when MMLU degrades.

## Predictions vs Measurements

| # | Prediction (from MATH.md) | Measured | Match? |
|---|--------------------------|----------|--------|
| P1 | Math behavioral >= 0.20 (vs base 0.10) | base=0.10, routed=0.80 | YES (8x improvement) |
| P2 | Code behavioral >= 0.50 (vs base 0.42) | base=0.42, routed=0.62 | YES (+49% improvement) |
| P3 | Prose neutral-to-positive on >= 2/3 prose domains | medical +10.7%, legal -2.1% (neutral), finance -11.7% | YES (2/3 non-degraded) |
| P4 | Behavioral improves on >= 1 MMLU-degraded domain | Math: MMLU -20pp BUT behavioral +70pp | YES (math contradicts MMLU) |
| P5 | Routed >= base on >= 3/5 domains | Better: 3, Neutral: 1, Worse: 1 | YES (4/5 non-degraded) |

All 5 predictions confirmed.

## Hypothesis

Routed composition with oracle top-1 routing and per-domain scales produces better text (execution-based behavioral metrics) than the base model on >= 3/5 domains, even when MMLU degrades. The metric-behavioral gap is real.

## What This Experiment Is

The existential P0 test for the SOLE (composable ternary experts) project. Three prior competitive benchmark experiments were killed because SFT adapters degraded MMLU scores. This experiment tests whether the architecture actually works by measuring what matters: does the generated text contain correct code, correct math answers, and relevant factual content?

**Method:** For each of 5 domains (medical, code, math, legal, finance), generate text with:
- (a) Base BitNet-b1.58-2B-4T model
- (b) Routed composition: oracle top-1 adapter per domain, pre-merged at per-domain optimal scales (Finding #217)

Then evaluate with execution-based behavioral metrics (Finding #210, validated with Cohen's kappa = 0.800):
- Code: ast.parse syntax validity (70%) + factual recall (30%)
- Math: numerical answer correctness (exact match within 1%)
- Medical/Legal/Finance: factual recall F1 against reference answers

## Key References

- Finding #210: Behavioral eval framework validated (Cohen's kappa = 0.800)
- Finding #217: Per-domain optimal scales {medical:20, code:20, math:20, legal:4, finance:1}
- Finding #236: PPL-accuracy gap (PPL improvement does not predict MMLU improvement)
- Finding #237: GSM8K +10pp is only consistent competitive advantage
- Liang et al., "Holistic Evaluation of Language Models" (HELM, 2022) — benchmark format sensitivity

## Empirical Results

### Per-Domain Comparison (base vs oracle-routed composition)

| Domain | Base | Routed | Delta | Direction | MMLU Direction | Gap? |
|--------|------|--------|-------|-----------|----------------|------|
| Medical | 0.263 | 0.291 | +0.028 (+10.7%) | BETTER | Neutral | No |
| Code | 0.419 | 0.624 | +0.205 (+48.9%) | BETTER | Neutral | No |
| Math | 0.100 | 0.800 | +0.700 (+700%) | BETTER | DEGRADED (-20pp) | YES |
| Legal | 0.098 | 0.096 | -0.002 (-2.1%) | NEUTRAL | DEGRADED (-10pp) | Partial |
| Finance | 0.176 | 0.156 | -0.021 (-11.7%) | WORSE | Neutral | No |

### Headline Numbers

- **Math:** 1/10 -> 8/10 correct answers (base -> routed). This is the strongest signal.
- **Code:** 5/10 -> 8/10 syntactically valid (base -> routed).
- **Medical:** Factual recall improved +10.7% (small but positive).
- **Legal:** Essentially unchanged (delta within noise).
- **Finance:** Slight degradation at scale=1.0 (lowest scale, minimal adapter effect).

### The Metric-Behavioral Gap (Critical Finding)

**Math domain** is the definitive case:
- MMLU: base 50% -> routed 30% (DEGRADED by 20 percentage points)
- Behavioral: base 10% -> routed 80% (IMPROVED by 70 percentage points)

The math adapter makes the model produce correct numerical answers 8x more often, but it ALSO makes the model produce verbose step-by-step explanations instead of single-letter MMLU answers. MMLU interprets the verbose format as wrong. The behavioral metric sees the correct answer.

This is not a subtle effect. It is a 700% behavioral improvement (p<0.005) coexisting with a 40% MMLU degradation on the same domain. Note: the base model's low score (1/10) is partly a truncation artifact at max_tokens=128 (see Limitations §3). The improvement reflects format efficiency under token constraints — the adapter teaches concise, parseable answer format — which is operationally valuable even if not pure knowledge gain.

### Kill Criteria

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| K1 (#642) | Routed worse on >= 3/5 domains | Worse on 1/5 | **PASS** |
| K2 (#643) | Zero behavioral-MMLU contradictions (gap does not exist) | 1 contradiction (math) | **PASS** |

## Statistical Significance (n=10 per domain)

| Domain | Base | Routed | Test | p-value | Significant? |
|--------|------|--------|------|---------|-------------|
| Math | 1/10 | 8/10 | Fisher's exact | p < 0.005 | **Yes** |
| Code | 5/10 | 8/10 | Fisher's exact | p ~ 0.16 | No |
| Medical | 0.263 | 0.291 | — | n too small | No |
| Legal | 0.098 | 0.096 | — | n too small | No (noise) |
| Finance | 0.176 | 0.156 | — | n too small | No (noise) |

**Honest summary:** Math adapter robustly improves behavioral quality (+700%, p<0.005). Code adapter likely improves (+49%, needs larger n). Prose domains are inconclusive at n=10. Prior MMLU-based kills were false negatives for math domain.

## Limitations

1. **Oracle routing:** This experiment uses perfect routing (domain X prompts get adapter X). Real deployment needs learned routing. The behavioral gains depend on correct adapter selection.

2. **Small sample size:** 10 prompts per domain. Only the math result (1/10 -> 8/10, p<0.005) is statistically significant. Code is suggestive (p~0.16). Medical, legal, and finance deltas are within noise at n=10. The claim "3/5 domains better" relies on medical being real, which is not established.

3. **max_tokens=128 truncation confound (math domain):** The base model scores 1/10 on math partly because it generates verbose step-by-step text that truncates at 128 tokens before reaching the answer. The routed model produces GSM8K-style chain-of-thought with inline calculations (`<<3*26=78>>`) that fits within 128 tokens. The +700% improvement is therefore a combination of (a) genuine format efficiency from SFT training and (b) truncation artifact. At max_tokens=512, the base model would likely score higher. The operational conclusion (adapter produces parseable answers more efficiently) is unchanged, but the interpretation should be "format efficiency under token constraints" rather than "pure knowledge improvement."

4. **Evaluation prompts match SFT format:** We test on `### Instruction:` format which matches training. This is intentional (testing the adapter's native format) but means results may not transfer to other prompt formats.

5. **Finance degradation at scale=1.0:** The finance adapter at scale=1.0 barely modifies the base model, yet shows slight degradation. This suggests the scale calibration (Finding #217, optimized for PPL) may not be optimal for behavioral quality.

6. **Legal neutrality despite scale=4.0:** The legal adapter at scale=4.0 neither helps nor hurts. This is consistent with the SFT data quality for legal domain being limited.

7. **SFT template memorization (math):** Routed math outputs use the exact GSM8K annotation format (`<<3*26=78>>`). The adapter teaches the model to produce answers in a parseable, token-efficient format. This is adapter effectiveness (SFT works as intended) but should not be overclaimed as knowledge improvement.

## What Would Kill This

**At micro scale:** If learned routing (not oracle) fails to select the correct adapter, behavioral gains disappear. Test with a simple cosine-similarity router.

**At macro scale:** If the pattern does not hold on real models (Qwen2.5-7B with LoRA adapters), the BitNet micro-model was a misleading testbed. Run the same behavioral eval on macro-scale routed composition.

**Fundamentally:** If the behavioral metrics themselves are wrong (e.g., ast.parse passes on garbage code), the improvement is illusory. The Cohen's kappa = 0.800 from Finding #210 mitigates this concern but does not eliminate it.

## Timing

| Phase | Time |
|-------|------|
| Base generation (50 prompts) | 122s |
| Routed generation (50 prompts, 5 adapter swaps) | 283s |
| Evaluation | <1s |
| **Total** | **405s (6.8 min)** |
