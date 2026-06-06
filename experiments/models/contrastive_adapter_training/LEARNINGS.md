# LEARNINGS: Contrastive Adapter Training

## Core Finding

Contrastive Orthogonality Loss achieves 99.6% weight-space decorrelation (cosine 0.97 -> 0.004) but produces only 0.3-5.7% PPL differentiation across domains. Weight-space orthogonality is a weak predictor of behavioral specialization, confirming 2510.03262. The secondary finding -- LoRA scale=2.0 preserving base capability where scale=20.0 destroyed it -- may be more actionable than the contrastive mechanism itself.

## Why This Happened

**The geometry-behavior gap is fundamental, not incidental.** Rethinking Inter-LoRA Orthogonality (2510.03262) proved that guaranteed weight-space orthogonality does not ensure semantic compositionality. Our experiment provides the first quantification of this gap in a ternary LLM setting: 270x more decorrelation in weight space (99.6%) produces only 10-100x less differentiation in data space (0.3-5.7%).

The root cause is that **SFT adapters at 200 steps learn format, not knowledge** (LIMA, 2305.11206). The contrastive loss successfully suppresses the shared format direction, but what remains (domain content components) is too small to create meaningful behavioral differences. The adapters become orthogonal copies of "almost nothing domain-specific."

**Why NeuroLoRA (2603.12378) succeeded where we did not:** NeuroLoRA combines COL with a context-aware neuromodulation gate that routes inputs to the right expert at inference time. The COL creates separable subspaces; the gate exploits them. Our experiment tested COL alone without routing, which creates orthogonal subspaces that are never selectively activated. NeuroLoRA also evaluates on MMLU/GSM8K/ScienceQA -- behavioral benchmarks, not just PPL.

**Why LoRACLR (2412.09622) succeeded:** LoRACLR operates on diffusion models where each LoRA represents a visually distinct concept (object vs style). The concepts are inherently separable in data space. Our SFT domains share format structure that dominates content, making data-space separation much harder.

## Confirming Evidence

- **2510.03262** (Rethinking Inter-LoRA Orthogonality): Directly proves weight-space orthogonality is insufficient for semantic disentanglement. Our 99.6% vs 0.3-5.7% gap is a concrete instance of their theoretical finding.
- **2305.11206** (LIMA): SFT teaches format alignment, not domain knowledge. Our baseline inter-adapter cosine of 0.97 (97% weight overlap across 5 "different" domain adapters) is the strongest empirical confirmation of LIMA in a multi-adapter setting.
- **Finding #212** (our own): Code adapter at scale=20 degrades GSM8K -18pp, HumanEval -15pp. Format compliance improves while capability is destroyed. Confirms the alignment tax at high LoRA scale.
- **2405.13432** (Alignment Tax): SFT at high scale causes logit-scale mismatch that overwrites pre-trained knowledge.

## Contradicting Evidence

- **NeuroLoRA (2603.12378)**: COL + routing gate outperforms baselines on MMLU, GSM8K, ScienceQA in multi-task adaptation. However, their success requires the routing gate -- COL alone is not evaluated in isolation, making direct comparison difficult.
- **LoRACLR (2412.09622)**: Contrastive merging produces high-quality multi-concept image synthesis in diffusion models. Key difference: their concepts (objects, styles) are inherently data-separable; our SFT domains are not.
- **InfLoRA (2404.00228)**: Orthogonal projection into complement of prior task subspaces achieves SOTA continual learning on ImageNet-R, CIFAR100, DomainNet. Key difference: InfLoRA operates in gradient subspace (not weight space) and uses sequential projection (not joint contrastive penalty).

## Alternative Approaches

1. **Data-level specialization** (LIMA, 2305.11206): If SFT teaches format not knowledge, the fix is training data that forces genuine domain expertise. Curated domain-specific instruction data with verifiable correctness (not just format compliance). This is the LIMA insight applied in reverse: if 1000 curated examples teach format, then domain-specialized data must include domain-specific REASONING, not just domain-specific text.

2. **Gradient-space orthogonality** (InfLoRA, 2404.00228): Project new adapter parameters into the orthogonal complement of prior adapters' gradient subspaces, not weight subspaces. This targets functional interference directly rather than geometric interference.

3. **Context-aware routing + COL** (NeuroLoRA, 2603.12378): Combine weight decorrelation with a learned gate that activates the right expert per input. COL creates separable subspaces; routing exploits them. Our Finding #186 (energy gap top-k routing, supported) already shows routing can work with N=5.

4. **LoRA scale sweep** (motivated by Finding #215): Scale=2.0 preserves base capability where scale=20.0 destroys it. A systematic sweep (0.5, 1.0, 2.0, 5.0, 10.0) with behavioral benchmarks would establish the optimal operating point.

5. **Mixed training with general data** (2509.20758, ref #305): Including 6.2% general-domain data during SFT eliminates catastrophic forgetting. Could preserve base capability while training specialized adapters.

## Implications for Next Experiments

1. **Weight-space metrics are unreliable proxies for composition quality.** Our Grassmannian initialization (17x decorrelation) should be evaluated on behavioral benchmarks, not just cosine similarity. The decorrelation may be geometrically impressive but behaviorally irrelevant, just like COL's 99.6%.

2. **LoRA scale is the most actionable lever right now.** Scale=2.0 preserves base capability (all 5 domains improve vs base). Scale=20.0 destroys it. The LoRA scale sweep experiment (exp_lora_scale_sweep_generation) is P0.

3. **The LIMA confirmation (baseline cosine=0.97) explains why routing struggled at N=24.** If all adapters are 97% identical, no router can meaningfully distinguish them. The 7 routing kills (Findings #189-198) may have been fundamentally about adapter quality, not router architecture -- exactly what Finding #198 hypothesized.

4. **Contrastive training alone is insufficient but not useless.** It fixed the medical adapter (+14.1pp vs baseline at same scale) and broke code universality. Combined with routing (NeuroLoRA-style) or better training data, it may have a role.

## Recommended Follow-Up

1. **exp_lora_scale_sweep_generation** (P0, already open): Sweep LoRA scale (0.5-10.0) with behavioral benchmarks (MMLU, GSM8K, HumanEval). Motivated by Finding #215 (scale=2.0 preserves capability) and Finding #212 (scale=20.0 destroys it). The scale finding is more actionable than contrastive training.

2. **exp_generation_quality_test** (P0, already open): The existential test -- does routed composition produce better text? Must use behavioral evaluation (Finding #210 framework), not just PPL. This experiment should use scale=2.0 adapters based on Finding #215.

3. **exp_grassmannian_behavioral_eval** (new, P1): Evaluate Grassmannian-initialized adapters on behavioral benchmarks (MMLU, GSM8K) rather than cosine metrics. Motivated by the core finding: 99.6% weight decorrelation produced only 0.3-5.7% behavioral differentiation. Our Grassmannian 17x decorrelation (Finding #170) may face the same geometry-behavior gap. Literature: 2510.03262.
