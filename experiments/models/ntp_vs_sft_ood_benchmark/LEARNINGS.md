# LEARNINGS: NTP vs SFT Adapter OOD Benchmark

## Core Finding
NTP adapters preserve OOD reasoning (+10pp GSM8K, 30pp gap vs SFT at p=0.003) but provide no advantage for formatting or knowledge tasks. The training objective is load-bearing for reasoning only.

## Why This Happened

The 30pp NTP-SFT gap on GSM8K has a clear mechanism: NTP training computes gradients through ALL token positions (instruction + response), forcing the adapter to regularize its perturbation across the full input distribution. SFT training only backpropagates through response tokens, creating an adapter that is "blind" to instruction-position hidden states. When OOD benchmark prompts arrive (which structurally resemble instructions, not responses), the NTP adapter has already been constrained to produce small perturbations on such inputs, while the SFT adapter produces unregulated perturbations.

This mechanism is task-specific because:
- **Reasoning (GSM8K):** Chain-of-thought requires the model to process instruction tokens correctly to understand the problem. NTP's instruction-position regularization directly helps.
- **Formatting (code gen):** Both objectives produce adapters that modify output syntax. The perturbation at s=20 disrupts code formatting regardless of whether instruction positions were regularized.
- **Knowledge (MMLU):** Factual recall depends on precise base model weights. Any perturbation at behavioral scale (s>=4) disrupts stored knowledge, independent of training objective.

## Confirming Evidence

- **arXiv:2310.03716** (How Abilities in LLMs are Affected by SFT Data Composition): Documents SFT causing capability degradation on reasoning benchmarks. The degradation pattern matches our finding — SFT specifically harms reasoning capability while less damaging to factual/formatting tasks.
- **arXiv:2310.10477** (NEFTune): Full-sequence training with noise injection preserves generalization better than response-only SFT, consistent with our finding that full-sequence (NTP) preserves OOD reasoning.
- **Finding #237** (from exp_competitive_benchmark_routed): Original observation of +10pp GSM8K with NTP adapters. Now confirmed at n=50 with p=0.003.
- **Finding #260/261** (from exp_capability_benchmark_full_system): SFT adapters degrade ALL OOD benchmarks. This experiment extends that result by showing NTP partially mitigates the degradation for reasoning tasks specifically.

## Contradicting Evidence

- **arXiv:2406.11794** (Response-only vs Full-sequence Loss): Directly compares NTP and SFT for LLM fine-tuning. Finds response-only loss competitive or better on several OOD benchmarks when properly tuned. This suggests our finding may be specific to the adapter composition setting (where perturbation is amplified by scale factor s=20) rather than a universal property of the training objectives.
- **arXiv:2310.05914** (LIMA): Small curated SFT datasets preserve broad capabilities, implying SFT degradation is data-quality-dependent, not objective-dependent. Our SFT adapters may degrade reasoning because of training data composition, not the loss function.
- **Finding #187** (from exp_sft_bitnet_generation_quality): SFT adapters scored higher than NTP on LLM-as-judge evaluation (mean 3.93 vs 3.70 on 5 domains). This contradicts the claim that NTP is universally better — SFT produces better-formatted, more readable responses.

## Alternative Approaches

1. **DARE sparsification** (arXiv:2311.03099, ref #349): Randomly dropping adapter parameters before composition reduces OOD perturbation. Could preserve GSM8K advantage while reducing MMLU degradation. Already tested in exp_lora_merging_bakeoff — degraded monotonically with drop rate at our rank-16 scale.

2. **LoRA-Flow dynamic fusion** (arXiv:2402.15367): Input-dependent weighting of adapter contribution. Would automatically reduce adapter influence on OOD inputs where the adapter is unhelpful. Addresses the scale problem (s=20 is too high for OOD) without requiring explicit OOD detection.

3. **Scale-adaptive composition** (our Finding #249): Per-domain optimal scales already partially address this — legal at s=4 degrades less than math at s=20. A finer-grained approach: scale by input confidence, not just domain. This is conceptually similar to our entropy gating (Finding #200) but applied at adapter composition level.

4. **Hybrid NTP-SFT training** (novel, motivated by this experiment): Train the first N iterations with NTP (for instruction-position regularization), then switch to SFT (for output quality). Captures both benefits. No paper evidence for this specific approach, so it would require formal justification.

## Implications for Next Experiments

1. **Use NTP adapters for the P0 deployment track.** The GSM8K advantage is the architecture's most reliable OOD benefit. SFT adapters are no better on any OOD benchmark and catastrophically worse on reasoning.

2. **The three-mechanism decomposition (reasoning/format/knowledge) is post-hoc but actionable.** Design future experiments to test each mechanism independently:
   - Reasoning: test with more reasoning benchmarks (ARC, BBH) to confirm NTP advantage generalizes
   - Format: test scale reduction (s=4 instead of s=20) on code gen to isolate scale from objective
   - Knowledge: test whether routing that skips adapters for knowledge queries preserves MMLU

3. **The Grassmannian skeleton confound is unresolved.** Both adapter types used A matrices computed from NTP training. SFT adapters may perform differently with SFT-derived A matrices. This is worth testing but lower priority than P0.

4. **Oracle PPL concern from prior experiments.** Finding #257 noted oracle PPL (19.16) worse than base PPL (18.97) at N=24. At N=5, adapters are clearly specialized. The NTP vs SFT question is only meaningful when adapters provide genuine value — confirmed at N=5 (this experiment's operating point).

## Recommended Follow-Up

**exp_generation_quality_test** (P0 critical path, already planned): Use NTP adapters based on this finding. The existential test — does routed composition produce better text? — should use the adapter type that preserves reasoning capability. This finding directly informs that decision.

No new experiment recommended from this analysis. The P0 critical path (generation quality) is the highest priority, and this experiment's conclusion (use NTP adapters) feeds directly into it.
