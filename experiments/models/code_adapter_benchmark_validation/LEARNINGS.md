# LEARNINGS: Code Adapter Benchmark Validation (KILLED)

## Core Finding

Code SFT adapter at LoRA scale=20 degrades standardized benchmarks (GSM8K -18pp, HumanEval -15pp) while improving MMLU +8pp, revealing that Finding #208's "universal code adapter superiority" was measuring format compliance, not capability. This is a specific instance of the **alignment tax** (2405.13432): SFT instruction tuning degrades knowledge/reasoning benchmarks when format learning outweighs capability preservation.

## Why This Happened

### The Alignment Tax (2405.13432)

Wu et al. formally define the alignment tax: when SFT data assimilation of dataset biases outweighs instruction-following capacity growth, world knowledge and reasoning degrade. Our experiment is a textbook case — the code adapter learned the `### Instruction / ### Response` format so well that it dominated evaluation, masking reasoning degradation on benchmarks that use different formats.

### LoRA Scale as the Amplifier

NotebookLM confirmed that LoRA scale directly controls perturbation magnitude relative to the frozen base. At scale=20.0, the adapter delta dominates base model representations. This is consistent with the "logit-scale mismatch" identified in our own prior work: FP16 LoRA deltas at high scale can overwrite base model knowledge. The MMLU improvement (+8pp) is the format component; the GSM8K/HumanEval degradation (-18pp/-15pp) is the capability destruction component.

### Format-Capability Decoupling

LIMA (2305.11206) predicted SFT teaches format, not knowledge. Our experiment confirms this but reveals a critical nuance: SFT can also DESTROY pre-trained knowledge when the perturbation is too large. The base model's strong performance (58% GSM8K, 60% HumanEval) was being actively degraded, not preserved.

## Confirming Evidence

1. **Disperse-Then-Merge (2405.13432):** Directly names and measures the alignment tax. Found that dispersed training across subsets reduces the tax. Our degradation pattern (format improves, reasoning degrades) matches their characterization exactly.

2. **LoRA Learns Less and Forgets Less (2405.09673):** Biderman et al. show LoRA underperforms full fine-tuning on target tasks but better preserves source-domain performance. At commonly used low-rank settings, LoRA substantially underperforms full FT while requiring longer training. This means our rank-16 LoRA was BOTH undertrained for capability transfer AND over-scaled for knowledge preservation.

3. **Catastrophic forgetting in math SFT (NotebookLM):** Research on Flan-T5-Base found math-only SFT caused NLI accuracy to collapse from 81.0% to 16.5% in 1,000 steps. Our -18pp GSM8K degradation is a milder version of the same catastrophic forgetting mechanism.

4. **Our own prior finding (Finding #208 → #209):** The universal_adapter_ablation already showed routing value = -8.8%. This experiment explains WHY: the code adapter's "dominance" was format compliance measured in the training format, not genuine capability advantage.

## Contradicting Evidence

1. **SFT Doesn't Always Hurt (2509.20758):** Domain-specific SFT CAN preserve general capabilities under certain conditions: careful learning rate, mixed data, and moderate scale. This means our kill is specific to the scale=20.0 regime, not a fundamental impossibility of code SFT.

2. **Code Reasoning Transfer (2405.20535):** Aryabumi et al. showed code training improves cross-domain reasoning. However, their work used full fine-tuning at standard scales on larger models, not rank-16 LoRA at scale=20 on a 2B ternary model. The gap between their conditions and ours explains the divergent results.

3. **LoRI (2504.07448):** Achieves HumanEval 63.2% with modified LoRA by reducing cross-task interference. Demonstrates that the degradation is fixable with architectural changes to LoRA, not inherent to adapter-based fine-tuning.

## Alternative Approaches

1. **LoRA scale ablation (immediate priority):** Test scale=1.0, 2.0, 5.0, 10.0 to find the sweet spot where format benefits are retained without capability degradation. The alignment tax literature (2405.13432) suggests there IS a regime where both improve.

2. **Mixed training data (DMT strategy):** NotebookLM confirms that mixing as little as 6.2% general data during SFT eliminates catastrophic forgetting. Current adapters were trained on pure domain data (codeparrot/github-code-clean). Adding general instruction data could preserve reasoning.

3. **LoRI (2504.07448):** Modified LoRA with reduced cross-task interference. Outperforms standard LoRA by +24.4pp on HumanEval in continual learning. Could replace standard LoRA in our adapter training.

4. **Disperse-Then-Merge (2405.13432):** Train on dispersed subsets of instruction data, then merge. Directly targets alignment tax reduction. Applicable to our multi-domain adapter setting.

5. **Pre-merge at lower scale:** Our architecture supports pre-merging adapters at any scale. Testing pre-merge at scale=1.0-5.0 would test whether the base model's strong capabilities (58% GSM8K, 60% HumanEval) can be preserved while gaining format benefits.

## Implications for Next Experiments

1. **lora_scale=20.0 is the primary confound in ALL prior adapter experiments.** Every experiment that used this scale (Finding #208, #209, #210, #211) is potentially confounded. The alignment tax suggests a lower scale would yield different results.

2. **Base model strength changes the project narrative.** BitNet-2B-4T at 58% GSM8K and 60% HumanEval WITHOUT adapters means the composition thesis must prove adapters ADD value beyond what the base already provides. The bar is much higher than assumed.

3. **Custom eval framework (Finding #210) is validated for RELATIVE ranking but not absolute capability measurement.** The behavioral eval framework correctly identifies that code adapter > base on instruction-following tasks. But this doesn't mean code adapter > base on actual reasoning/coding capability.

4. **The TWO-WORLD problem is deeper than format.** It's not just that adapters work in their training format. It's that high-scale SFT actively DESTROYS base model capabilities. This is the alignment tax in action.

## Recommended Follow-Up

1. **exp_lora_scale_sweep_generation (P0):** Sweep scale=1.0, 2.0, 5.0, 10.0, 20.0 on both custom evals AND standardized benchmarks. Motivation: alignment tax (2405.13432) predicts a regime where format improves without capability degradation. This is the single most important experiment to run next.

2. **exp_mixed_training_adapters:** Retrain adapters with 6-10% general instruction data mixed in. Motivation: catastrophic forgetting literature shows mixed training eliminates the alignment tax at minimal cost to specialization.

3. **exp_generation_quality_test (P0, existing):** The existential test for composition. BUT must be run at multiple LoRA scales, not just scale=20.0, in light of this finding.
