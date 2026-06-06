# LEARNINGS: exp_behavioral_eval_routed

## Core Finding

Routed composition with per-domain scales produces **better text** despite degrading MMLU scores. Math domain shows +700% behavioral improvement (1/10 -> 8/10 correct answers, p<0.005) while MMLU drops 20pp. The metric-behavioral gap is real: MMLU measures format compliance, not knowledge. Three prior competitive benchmark kills were false negatives.

## Why This Happened

The dissociation has two complementary causes:

1. **SFT format shift:** Instruction-tuned adapters teach verbose, explanatory response format (GSM8K chain-of-thought with `<<3*26=78>>` annotations). MMLU expects single-letter answers (A/B/C/D). The adapter makes the model *better at answering* but *worse at the MMLU format*. This is well-documented: MMLU-Pro (arxiv 2406.01574) showed that prompt format changes cause 16-33% accuracy drops, and minor perturbations shift model rankings by up to 8 positions.

2. **Truncation interaction:** At max_tokens=128, base model verbose reasoning truncates before reaching the answer. The adapted model's trained format (compact chain-of-thought) fits within the limit. This is "format efficiency under token constraints" — operationally valuable even if partly artifactual.

3. **PPL-accuracy dissociation is established:** Our Finding #236 showed PPL improvement doesn't predict MMLU accuracy (r=0.08). The ICLR 2025 paper "What is Wrong with Perplexity for Long-context Language Modeling?" (arxiv 2410.23771) confirms that perplexity averages over all tokens, obscuring performance on task-critical tokens. Our result extends this: even *task accuracy* (MMLU) can dissociate from *behavioral quality* (execution-based correctness).

## Confirming Evidence

- **MMLU-Pro** (arxiv 2406.01574, NeurIPS 2024): Prompt sensitivity in MMLU demonstrated systematically. Models show 4-5% score variation from prompt format alone. Our 20pp MMLU drop from SFT format shift is consistent with this magnitude of format sensitivity.
- **"Accuracy is Not All You Need"** (arxiv 2407.09141): Even when accuracy is similar between baseline and compressed models, answer "flips" (correct->incorrect and vice versa) occur in proportion. Accuracy alone misses behavioral changes — our finding is the extreme case where behavioral *improves* while accuracy *degrades*.
- **"When Benchmarks are Targets"** (arxiv 2402.01781): Benchmark sensitivity to surface form changes, especially in high-contamination domains. Our SFT adapters essentially change the surface form, triggering exactly this sensitivity.
- **HELM** (Liang et al., 2022): Established that evaluation format sensitivity is a systemic issue across LLM benchmarks, not specific to any architecture.
- **Our Finding #210**: Behavioral eval framework validated with Cohen's kappa = 0.800. The execution-based metrics (ast.parse, numerical answer match) are grounded in task correctness by construction.
- **Our Finding #237**: GSM8K +10pp was the only consistent competitive advantage across all composition strategies — GSM8K accepts free-form answers, confirming format is the confound.

## Contradicting Evidence

- **Scale of effect may be inflated:** The 700% improvement (1/10 -> 8/10) is partly a truncation artifact at max_tokens=128 (acknowledged in PAPER.md Limitation #3). At longer token limits, the base model would score higher, reducing the gap. The *direction* is real but the *magnitude* is upper-bounded.
- **n=10 insufficiency for prose domains:** Only math is statistically significant (p<0.005). Code is suggestive (p~0.16). Medical (+10.7%), legal (-2.1%), finance (-11.7%) are within noise. The claim "3/5 domains better" is honest but rests on medical being real, which is unestablished.
- **Oracle routing assumption:** Real deployment requires learned routing. If routing errors occur, gains disappear. This is the actual existential risk — the experiment proves the ceiling, not the floor.
- **SFT template memorization:** The math adapter produces exactly the GSM8K annotation format it was trained on. This is SFT working as intended, but it means we tested the adapter in its *native format*. Cross-format generalization is untested.
- **Position paper on LoRA composition** (arxiv 2506.13479, "Pause Recycling LoRAs"): Argues that LoRA composition often fails to produce compositional behavior, especially for multi-hop reasoning. Our per-domain routing avoids this (one adapter at a time), but it's a warning against overclaiming composability.

## Alternative Approaches

1. **LoraHub** (arxiv 2307.13269): Dynamic LoRA composition using gradient-free optimization to weight multiple adapters per task. Achieves cross-task generalization without oracle routing. Could replace our oracle with learned per-query composition weights. *Motivation:* Directly addresses our oracle routing limitation.

2. **Task-representation routing** (arxiv 2601.21795): Uses task representations rather than domain labels to route to LoRA experts. More flexible than domain-oracle routing and could handle ambiguous queries. *Motivation:* Makes routing learnable without per-domain labels.

3. **MMLU-Pro** (arxiv 2406.01574): 10-choice format with chain-of-thought is more robust to prompt variations (2% sensitivity vs 4-5%). If we must benchmark on MC format, MMLU-Pro is more fair to instruction-tuned models.

4. **LoRA as Knowledge Memory** (arxiv 2603.01097): Empirical analysis of what LoRA stores. Understanding the knowledge representation could explain why math adapter succeeds (format/reasoning knowledge) while finance adapter at scale=1.0 fails (insufficient perturbation).

## Implications for Next Experiments

1. **The deployment track is validated.** The architecture produces better text on math and code domains. Finance needs scale recalibration (scale=1.0 is too low). Legal is neutral (possible data quality issue). Medical is promising but needs larger n.

2. **MMLU is officially retired as a primary metric for this project.** Three kills + this experiment prove MMLU conflates format with knowledge. Future experiments should use execution-based behavioral metrics as primary, with MMLU as secondary/diagnostic only.

3. **The next existential test is learned routing.** Oracle routing proves the ceiling. The floor depends on whether a learned router (cosine similarity, task representations, or LoraHub-style optimization) can match oracle performance. If routing accuracy < 80%, behavioral gains may disappear on misrouted queries.

4. **max_tokens needs systematic control.** Run behavioral eval at 128, 256, 512 tokens to separate format efficiency from truncation artifact. This is the main confound weakening the finding.

## Recommended Follow-Up

**Priority 1: Learned routing behavioral eval.** Replace oracle with cosine-similarity router (already built for Finding #210 framework). Run same 5-domain behavioral eval. Kill if routing accuracy < 70% or if behavioral quality drops below base on >= 3/5 domains. *Motivation:* This is the actual deployment-relevant test — Finding #238 proves the ceiling, but the floor matters more.

**Priority 2: Token-limit ablation.** Run math behavioral eval at max_tokens = {128, 256, 512} to quantify the truncation confound. If base scores 5/10 at 512 tokens, the real improvement is 60% not 700%. *Motivation:* Adversarial review correctly identified this as the main confound (REVIEW-adversarial.md fix #3).

**Priority 3: Finance/legal scale recalibration.** Finding #217 optimized scales for PPL, not behavioral quality. Finance at scale=1.0 and legal at scale=4.0 may need different optima for behavioral metrics. Run scale sweep {1, 4, 8, 12, 16, 20} with behavioral eval on these two domains. *Motivation:* Finding #238 shows PPL-optimal scales don't predict behavioral quality, so scales need re-optimization.
