# LEARNINGS: Universal Adapter Ablation

## Core Finding

Routing provides **negative value (-8.8%)** with current SFT adapters. A single code adapter outperforms domain-specific routing on 4/5 domains (alpha 0.97-1.25) because SFT primarily teaches instruction-following format, not domain knowledge. The code adapter's universality is an artifact of under-specialized adapters, not proof that routing is fundamentally useless.

## Why This Happened

Three converging mechanisms explain the result:

1. **Superficial Alignment Hypothesis (LIMA, 2305.11206):** Almost all knowledge is learned during pretraining. SFT teaches the model to surface existing knowledge in structured format. A code adapter excels at this because code data is the most structured instruction-following training data.

2. **Code activates general reasoning (2405.20535, 2509.21499):** Code instruction-tuning enhances reasoning across ALL domains, not just code. The structured, logical nature of code activates latent reasoning capacities in the base model. The code adapter is not "domain-specific" -- it is a reasoning enhancer that transfers broadly.

3. **SFT rotates knowledge, doesn't inject it (2310.00492):** Instruction tuning rotates pre-trained knowledge toward user-oriented tasks via three mechanisms: (a) recognizing instruction parts of prompts, (b) altering self-attention to focus on instruction verbs, (c) reorienting feed-forward networks. No new domain knowledge is added. Domain-specific adapters trained for only 300 steps at lora_scale=20 learn format but destroy the base model's general prose capability without compensating domain depth.

## Confirming Evidence

- **LoRA Learns Less and Forgets Less (2405.09673):** LoRA preserves base model performance on out-of-domain tasks better than full fine-tuning. This explains why domain-specific adapters degrade prose -- they modify enough to disrupt base capabilities but not enough to inject real domain knowledge.
- **MoE routing collapse literature (2408.15664, 2412.14711):** Unbalanced expert load leads to routing collapse where one expert dominates. Our code adapter absorbing 42/50 queries is a textbook case.
- **Finding #203:** Routing errors cost only ~13% at PPL level -- adapters are not differentiated enough for routing to matter.
- **Finding #205:** Energy gap routing collapses to code adapter selection, independently confirming code adapter dominance.

## Contradicting Evidence

The literature **strongly contradicts** the conclusion that routing is fundamentally negative-value:

- **Specialized Generalists MoE-LoRA (2601.07935):** Domain-specific routed MoE-LoRA achieves 59.5 vs single LoRA's 53.8 on medical benchmarks (+5.7pp). Single LoRA causes GSM8K to drop 3.6% from parameter contention.
- **SMoRA (2501.15103):** Rank-wise routing outperforms vanilla LoRA by +1.73% (Llama2-7b) and +6.13% vs standard Top-1 LoRA MoE.
- **LoRAuter (2601.21795):** Routing among 1,500+ adapters matches oracle performance (101.2%) when task-aligned adapters exist.
- **LoRA-Mixer (2507.00029):** Routed experts outperform baselines by +3.79% GSM8K, +2.90% CoLA, +3.95% ARC-C.
- **MoLoRA (2603.15965):** Per-token routing of specialized adapters allows 1.7B to exceed 8B model performance.

**Key distinction:** These papers use genuinely specialized adapters trained with diversity constraints or on truly distinct tasks. Our negative routing value reflects under-specialized adapters, not a fundamental limitation of routing.

## Alternative Approaches

Methods that could create genuinely specialized adapters:

1. **Contrastive LoRA training (LoRACLR, 2412.09622):** Contrastive objective in LoRA weight space -- positive pairs attracted, negative pairs repelled. Directly prevents cross-adapter convergence.
2. **Orthogonal gradient projection (Ortho-LoRA, 2601.09684):** Projects conflicting task gradients onto orthogonal complements during training. Mathematically guarantees adapters cannot converge to same solution.
3. **Localized balancing (LoRAMoE, 2312.09979):** Forces experts into specialized groups via balancing constraint. Without it, all experts converge to same behavior (our exact problem).
4. **Contrastive decoding (CoLD, 2505.14620):** Scores tokens based on divergence between adapter and base model. Diagnostic: if no divergence, adapter learned nothing domain-specific.
5. **Contrastive regularization (MSLoRA-CR, 2508.11673):** Pulls related task parameters closer, pushes different modality parameters apart. +1.88% over unconstrained training.

## Implications for Next Experiments

1. **Routing is NOT dead -- adapter quality is the bottleneck.** The -8.8% routing value is an artifact of under-specialized adapters, not proof routing is useless. Literature shows +1.7% to +55% gains with properly specialized adapters.

2. **The "code adapter universality" is explained by LIMA + code-reasoning transfer.** Code data teaches structured reasoning that transfers everywhere. This is a feature, not a bug -- but it means current domain adapters add nothing the code adapter doesn't already provide.

3. **Three prerequisites before routing becomes valuable:**
   - Longer training (>300 steps) with lower lora_scale
   - Diversity-enforcing objectives (contrastive loss, orthogonal constraints)
   - Evaluation on tasks requiring genuine domain knowledge (not just format compliance)

4. **The stability-plasticity dilemma (2601.07935) is the root cause.** Domain-specific tuning in a shared low-rank subspace causes conflicting gradients. Without orthogonal constraints, adapters converge to the same solution.

## Recommended Follow-Up

1. **exp_generation_quality_test (P0):** Does routed composition produce better TEXT? Still the existential test. Use code adapter as strong baseline. (Motivation: Finding #208 establishes the baseline to beat.)

2. **exp_contrastive_adapter_training:** Train adapters with orthogonal/contrastive constraints (Ortho-LoRA 2601.09684 or LoRACLR 2412.09622) to force genuine specialization. Then re-test routing value. (Motivation: literature shows +5.7pp from specialized MoE-LoRA vs single LoRA.)

3. **exp_lora_scale_ablation:** Sweep lora_scale (1, 5, 10, 20) to test whether domain adapters improve at lower scales. (Motivation: reviewer concern that lora_scale=20 disproportionately harms domain adapters with higher training loss.)
