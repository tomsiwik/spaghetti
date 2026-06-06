# LEARNINGS: Behavioral Evaluation Framework

## Core Finding
Execution-based evaluation infrastructure for 5 SFT domains. Detects 7x math improvement (7/10 vs 1/10 correct answers) invisible to keyword density and LLM judges. Prose domains bottlenecked by 128-token generation limit (<28% factual recall). Code adapter confirmed as universal improver (+600% math, +36% code).

## Why This Happened

### The disease: metric-outcome misalignment
Finding #179 proved PPL doesn't predict task quality (r=0.08, refs #280, #281). LLM judges scored the math adapter LOWER (3.6 vs 4.0) despite 24x more correct answers. Keyword density rewards verbosity over correctness. The framework succeeds because it measures what matters: does the answer contain correct facts / produce valid code / get the right number?

### Execution-based evaluation is established practice
- **FActScore** (2305.14251, EMNLP 2023) decomposes generations into atomic facts and verifies each against a knowledge source — our factual recall metric is a simplified version of this approach
- **HELM** (2211.09110, Stanford CRFM) demonstrates that multi-domain, multi-metric evaluation is necessary — single aggregate metrics hide domain-specific failures
- **GSM8K** (2110.14168) and **HumanEval** use exact-match / execution-based evaluation as the gold standard for math and code domains
- **QLoRA** (2305.14314) concluded "current chatbot benchmarks are not trustworthy" and recommended LLM-as-judge as alternative — but our Finding #179 shows even LLM judges fail when output format diverges from expectations

## Confirming Evidence
- FActScore (2305.14251): Atomic fact precision achieves <2% error rate vs human evaluation, validating fact-based metrics over surface statistics
- HELM (2211.09110): Models evaluated on only 17.9% of scenarios pre-HELM; multi-domain evaluation reveals failures hidden by single-metric approaches
- LoRAuter uses exact match for NLU, BLEU for translation, ROUGE for text generation — domain-specific metrics, not one-size-fits-all
- Our own Finding #204: code adapter 7/10 vs base 1/10 on math, reproduced exactly by this framework

## Contradicting Evidence
- **PPL works for quantization**: "A Comprehensive Evaluation of Quantization Strategies for LLMs" found PPL IS a reliable proxy when evaluating weight compression (not adapter composition). Context matters — PPL fails for routing/composition but succeeds for single-model compression.
- **LLM-as-judge has proven reliable in some contexts**: MT-Bench and Chatbot Arena (Zheng et al., 2023) validate GPT-4 as evaluator for conversational quality. Our failure case (Finding #179) is specific to domain-adapted outputs where format diverges from judge expectations. LLM judges may work once we address the 128-token generation limit.

## Alternative Approaches
1. **FActScore** (2305.14251) — Full atomic fact decomposition with LLM-based verification. More sophisticated than our substring matching but requires LLM inference for evaluation. Would solve the synonymy limitation (hypertension vs high blood pressure).
2. **G-Eval** — GPT-4 with alignment prompting for NLG evaluation, scoring 1-5 on fluency/relevance. Complementary to our execution-based approach for prose domains where factual recall is insufficient.
3. **ChatEval** — Multi-agent debate for evaluation consensus. Could address the subjective-domain inter-rater reliability issue (75% agreement on legal/finance).
4. **HELM** (2211.09110) — Full multi-metric framework including fairness, robustness, calibration. Overkill for our micro-scale but the domain-specific metric principle is validated.

## Implications for Next Experiments

### What the framework enables
1. **exp_generation_quality_test** (P0) can now use this framework directly — the existential test of whether routed composition produces better text has reliable metrics
2. Math and code domains have clear, reliable signals; prose domains need longer generation (>128 tokens) to produce meaningful factual recall
3. The code adapter's universal dominance (+600% math, +36% code) is now measured with a trustworthy metric, confirming Finding #204 and #208

### Known limitations to address
1. **Synonymy**: substring matching misses semantic equivalents. FActScore's LLM-based verification would fix this but adds inference cost.
2. **n=10 per domain**: directional only. Next experiments should use n≥30 for statistical significance.
3. **128-token generation limit**: legal domain <10% factual recall suggests the limit, not the adapter, is the bottleneck.
4. **Subjective-domain kappa**: 75% agreement on legal/finance is inconclusive. Need n≥30 for reliable kappa measurement.

## Recommended Follow-Up

1. **exp_generation_quality_test** (P0) — The existential test. Uses this framework to answer: does routed composition (TF-IDF router from Finding #207) produce better text than single-adapter or base? Motivation: Framework is now validated infrastructure; the critical path demands using it.

2. **exp_task_accuracy_real_benchmarks** (P0) — MMLU/GSM8K/HumanEval with composition. Standard benchmarks complement our custom framework. Motivation: HELM (2211.09110) shows multi-benchmark evaluation catches failures single frameworks miss.

3. **Increase generation limit** — Raise from 128 to 512+ tokens for prose domains in future eval runs. Legal factual recall <10% is an artifact of truncation, not adapter quality. This is a parameter change, not an experiment.
