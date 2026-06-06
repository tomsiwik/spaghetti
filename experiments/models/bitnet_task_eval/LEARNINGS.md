# Learnings: bitnet_task_eval

## Core Finding

NTP-trained LoRA adapters on a non-instruction-tuned 2B ternary base model fail to improve task performance despite reducing perplexity — the kill reflects training objective mismatch, not a fundamental composition failure. The one domain where training format matched eval format (medical QA) showed genuine improvement, confirming the diagnosis.

## Why This Happened (Literature-Grounded)

Three independent mechanisms explain the kill:

### 1. NTP training does not produce task-capable adapters

Pezeshkpour et al. (2025, arxiv:2501.17840) show that continual pre-training with LoRA on raw domain documents has only **marginal effect** on domain-specific task performance — the model learns surface statistics, not task completion. Biderman et al. (2024, arxiv:2405.09673) confirm that LoRA specifically underperforms full fine-tuning for continued pretraining (NTP objective), while performing well for instruction fine-tuning. The distinction is the training objective, not the adapter method.

The mechanistic explanation comes from Li et al. (2023, arxiv:2310.00492): instruction tuning causes a fundamental internal shift enabling models to **recognize instruction parts of prompts** and condition response generation on them. Pure NTP models lack this mechanism — they continue text rather than complete tasks. Our adapters taught the model domain vocabulary, not domain reasoning.

### 2. 1/N² attenuation destroys task signal even when present

With N=5 averaged-factor composition, each adapter's diagonal contribution scales to ~4%. While PPL (an aggregate distributional metric) tolerates this attenuation, task performance requires specific knowledge at accessible strength. This is confirmed by multiple lines of evidence:

- Lotfi et al. (2026, arxiv:2603.03535) empirically show: **routing > non-uniform merging > uniform averaging** for multi-adapter composition. The uniform averaging we use is the worst strategy for task-relevant signal.
- MoLoRA (2026, arxiv:2603.15965) demonstrates per-token routing across LoRA adapters can make a 1.7B model exceed an 8B model (+14% GSM8K), proving that the adapter knowledge exists but averaging destroys it.
- Naive LoRA Summation (2025, arxiv:2508.11985) confirms that RMS cosine similarity between deltas correlates linearly with perplexity degradation as N grows.

### 3. 2B ternary base has a reasoning floor

Multiple 2025 studies converge on ~3B parameters as the minimum for autonomous multi-step reasoning (arxiv:2603.07091, arxiv:2510.13935). BitNet-2B-4T scores 58.4% on GSM8K (few-shot) vs 73.2% for Qwen2.5-1.5B-Instruct — the ternary quantization imposes a real accuracy cost on reasoning. Our base model's 5% accuracy on MATH-500 reflects this floor: the model cannot perform the task, so adapter improvements have nothing to amplify.

However, Schaeffer et al. (NeurIPS 2023, arxiv:2304.15004) show that "emergent abilities" are largely metric artifacts — performance scales continuously, not as phase transitions. And TinyGSM (Liu et al., 2023, arxiv:2312.09241) demonstrated that a 1.3B model can reach 81.5% GSM8K with synthetic data + verifier. The floor is real but **surmountable with the right training data**.

## Confirming Evidence

| Paper | Finding | Relation |
|-------|---------|----------|
| Pezeshkpour et al. (2025, 2501.17840) | CPT with LoRA on raw docs → marginal task improvement | **CONFIRMS**: NTP adapters don't transfer to tasks |
| Biderman et al. (2024, 2405.09673) | LoRA underperforms for NTP CPT, works for instruction FT | **CONFIRMS**: training objective matters more than method |
| Li et al. (2023, 2310.00492) | Instruction tuning enables instruction recognition mechanism | **CONFIRMS**: NTP models lack task completion ability |
| LoRA Soups (Prabhakar et al., 2024, 2410.13025) | Composed adapters must each be instruction-trained | **CONFIRMS**: our NTP adapters were wrong training recipe |
| Lotfi et al. (2026, 2603.03535) | Routing > merging > uniform averaging | **CONFIRMS**: 1/N averaging is worst composition strategy |
| EasyMath (2025, 2505.14852) | Standard math benchmarks create floor effects for sub-3B models | **CONFIRMS**: MATH-500 inappropriate for 2B eval |
| Our own prior: PPL-task r=0.08 (ppl_vs_task_performance/) | PPL does not predict task accuracy | **CONFIRMS**: at BitNet-2B scale too |

## Contradicting Evidence

| Paper | Finding | Discrepancy |
|-------|---------|-------------|
| BitNet 2B4T Technical Report (2025, 2504.12285) | BitNet-2B-4T scores 58.4% GSM8K (few-shot) | Our base model scored 5% on MATH-500. The discrepancy is: (a) GSM8K is easier than MATH-500, (b) few-shot prompting vs our 0-shot, (c) the technical report uses proper evaluation infrastructure. Our eval setup may understate the base model's true capability. |
| TinyGSM (Liu et al., 2023) | 1.3B model → 81.5% GSM8K with synthetic data | Contradicts the "2B is too small" narrative. With instruction-formatted synthetic training data, even 1.3B models can do grade-school math. The limit is training data quality, not model size. |
| Adapter Merging (2026, 2601.18350) | Merging DAPT LoRA + SFT LoRA causes interference via reactivated reasoning traces | Partially contradicts simple "instruction tuning fixes everything" — composing NTP adapters WITH instruction adapters creates new interference. Pure instruction adapters per-domain may be needed. |

## Alternative Approaches (What We Could Try Instead)

### 1. Instruction-format training (highest priority, planned)
Train adapters on instruction-formatted domain data instead of raw NTP. LoRA Soups explicitly shows this is required for composed adapter task performance. Already planned as `exp_bitnet_instruction_tuned_task_eval`.

### 2. Per-token routing instead of uniform averaging
MoLoRA (arxiv:2603.15965) shows per-token routing makes 1.7B+adapters exceed 8B. This eliminates the 1/N² attenuation entirely. MoTE (arxiv:2506.14435) demonstrates this works specifically with ternary experts, directly applicable to BitNet-SOLE.

### 3. Learned composition weights (LoraHub-style)
LoraHub (arxiv:2307.13269) learns per-task coefficients over adapter libraries using few-shot examples. No gradient at deployment. This is intermediate: better than uniform 1/N but simpler than per-token routing.

### 4. Interference reduction before merging
TIES-Merging (Yadav et al., 2023, arxiv:2306.01708) and DARE (Yu et al., 2023, arxiv:2311.03099) can reduce cross-adapter interference as a preprocessing step. DARE drops 90-99% of redundant delta parameters before merging. Could be combined with 1/N scaling.

### 5. Orthogonal subspace constraints at training time
OSRM (2025, arxiv:2505.22934) and LoRI (2025, arxiv:2504.07448) constrain LoRA subspaces to be orthogonal before fine-tuning. This is compatible with our Grassmannian initialization work and would make merging lossless by construction.

### 6. Synthetic instruction data generation
TinyGSM shows 1.3B models can reach 81.5% GSM8K when trained on synthetic data from larger models. Generate instruction-formatted training data using a capable model, then distill into ternary LoRA adapters.

## Implications for Next Experiments

1. **exp_bitnet_instruction_tuned_task_eval is the correct next step** — confirmed by 5+ papers showing instruction training is necessary for task performance in composed adapters.

2. **1/N uniform averaging should be reconsidered for task eval.** PPL-based evaluation can continue using 1/N (it works), but task evaluation should test routing or learned weighting. The Lotfi et al. (2026) paper provides strong evidence that routing is strictly better.

3. **Eval setup needs revision.** Our 0-shot eval on a base (non-instruction-tuned) model underestimates capability. Future task evals should use few-shot prompting at minimum, or instruction-formatted prompts that match the adapter training format.

4. **The medical exception is the key signal.** Medical adapters trained on QA flashcards (instruction-like format) improved task performance even under 1/N attenuation. This validates that format-matched training transfers. Double down on this insight.

5. **Consider DARE/TIES preprocessing for composition.** Even with instruction-trained adapters, the 1/N² attenuation at scale (N=25 target) will be severe. Sparsifying deltas before merging (DARE: drop 90-99% of redundant params) could preserve task signal at higher N.

6. **The 2B reasoning floor is real but not fatal.** Focus task evaluation on domains where 2B models have baseline capability (factual QA, classification, creative writing) rather than multi-step reasoning (MATH-500, complex code). The EasyMath benchmark (arxiv:2505.14852) may be more appropriate.

## New References to Add

| Paper | arxiv | Relevance |
|-------|-------|-----------|
| Learning Beyond the Surface | 2501.17840 | CPT with LoRA → marginal task improvement |
| LoRA Learns Less and Forgets Less | 2405.09673 | LoRA NTP vs instruction training objective |
| From Language Modeling to Instruction Following | 2310.00492 | Mechanistic explanation of instruction tuning |
| Trade-offs in Ensembling, Merging and Routing | 2603.03535 | Routing > merging > averaging empirical evidence |
| MoLoRA | 2603.15965 | Per-token adapter routing, 1.7B > 8B result |
| MoTE (Mixture of Ternary Experts) | 2506.14435 | Ternary experts + routing, directly applicable |
| DARE (Language Models are Super Mario) | 2311.03099 | Delta sparsification before merging |
| TIES-Merging | 2306.01708 | Sign-conflict resolution in adapter merging |
| OSRM | 2505.22934 | Orthogonal subspace constraints for lossless merging |
| Are Emergent Abilities a Mirage? | 2304.15004 | Emergence as metric artifact, continuous scaling |
| EasyMath | 2505.14852 | Appropriate math benchmark for sub-3B models |
| TinyGSM | 2312.09241 | 1.3B model → 81.5% GSM8K with synthetic data |
| Adapter Merging Reactivates Reasoning Traces | 2601.18350 | NTP+SFT adapter interference mechanism |
