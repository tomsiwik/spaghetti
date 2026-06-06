# Learnings: bitnet_cosine_convergence

## Core Finding
LoRA adapter pairwise cosine similarity on BitNet-2B-4T plateaus at |cos|=0.00125 over 2000 training steps (40x below the 0.05 kill threshold), proving that the low orthogonality measured at 400 steps was NOT an under-training artifact. Composition PPL remains stable throughout.

## Why This Happened (Literature-Grounded)

The plateau behavior is best explained by **concentration of measure in high dimensions** combined with **discrete base weight routing**.

1. **High-dimensional near-orthogonality is the dominant force.** With D~20.5M adapter parameters, random vectors have E[|cos|] ~ 1.76e-4. Our measured 0.00125 is only 7.1x this baseline. Cao et al. (2508.11985, "Efficient Modular Learning through Naive LoRA Summation") independently confirm this: LoRA modules trained on disjoint domains on GPT-2 exhibit low pairwise cosine, and RMS cosine correlates linearly with composition PPL degradation. Their work validates that near-orthogonality from high dimensionality alone enables naive additive composition.

2. **Ternary base weights suppress shared subspace convergence.** In FP16 bases, the gradient ∂L/∂(BA) has a shared component from W's learned directions that pulls all adapters toward common subspaces. Ternary weights {-1,0,1} act as discrete routing masks with no magnitude information, reducing this shared gradient component. This is consistent with the "intruder dimension" theory from Jang et al. (2410.21228, "LoRA vs Full Fine-tuning"): LoRA on FP16 bases develops intruder dimensions that encode shared adaptation patterns. On ternary bases, the simpler gradient landscape may produce fewer intruder dimensions.

3. **The plateau is expected from subspace geometry.** Steele (2603.02224, "Subspace Geometry Governs Catastrophic Forgetting") shows forgetting follows F = α(1 - cos²θ_min) + β, where θ_min is the minimum principal angle between task gradient subspaces. When natural orthogonality is high (high principal angles), further training doesn't reduce angles — the system is already in the "high-angle regime" where geometry is stable. Our plateau at step 800-2000 is consistent with this: once adapters find their domain-specific subspaces, continued training refines within those subspaces rather than expanding into shared directions.

**However, the causal role of ternary weights is unproven.** The adversarial review correctly flags that the 114x gap vs FP16 Qwen (0.00125 vs 0.142) compares across different models, dimensions, data, and hyperparameters. High dimensionality alone may explain most of the near-orthogonality, with ternary weights providing an additional but potentially small contribution.

## Confirming Evidence

- **Cao et al., "Efficient Modular Learning through Naive LoRA Summation" (arxiv 2508.11985):** Independently trained LoRA modules on GPT-2 (117M, rank-4) show low pairwise cosine on disjoint domains. Math+Medicine additive composition improves PPL by 9.1% vs merged-data training. RMS cosine correlates linearly with composition degradation. **CONFIRMS** that near-orthogonality enables naive additive composition, and that domain disjointness is a key driver.

- **Steele, "Subspace Geometry Governs Catastrophic Forgetting" (arxiv 2603.02224):** Principal angles between task gradient subspaces predict interference via F = α(1 - cos²θ_min) + β. When subspace angles are naturally high, rank doesn't affect forgetting (CV~0.8%). **CONFIRMS** that naturally orthogonal adapters remain stable — orthogonal methods like O-LoRA provide minimal benefit when natural orthogonality is already high.

- **LoTA-QAF (arxiv 2505.18724, NeurIPS 2025):** Ternary adaptation enables lossless merging because ternary weight updates stay on the quantization grid. Surpasses 16-bit LoRA by up to 5.14% on MMLU after merging. **CONFIRMS** that ternary weight structure supports better adapter composition than FP16.

- **Jang et al., "LoRA vs Full Fine-tuning" (arxiv 2410.21228):** LoRA introduces "intruder dimensions" — novel singular vectors with near-zero cosine to all pre-trained directions. These intruder dimensions cause forgetting and can be scaled down to recover pre-training performance. **CONFIRMS** the mechanism by which FP16 LoRA develops shared structure that ternary bases may resist.

## Contradicting Evidence

- **Cao et al. (2508.11985) also show failure modes:** Math+Finance composition degrades PPL by 4.5%, and Finance+Medicine by 27.6%, despite low cosine. This suggests **cosine similarity is necessary but not sufficient** for good composition. Domain overlap in feature space (not just parameter space) matters. Our PPL results show no degradation, but we use genuinely disjoint domains; overlapping domains (e.g., medical+biology) might fail even with low cosine.

- **"Pause Recycling LoRAs" (arxiv 2506.13479):** Explicitly warns that orthogonality alone is insufficient for semantic composability. Two adapters can be orthogonal in parameter space but produce conflicting outputs in function space. Our experiment measures parameter-space cosine but does not verify functional composability beyond PPL.

- **No direct evidence that ternary bases cause lower cosine than FP16 at matched scale.** Every comparison in our project is cross-model (BitNet-2B vs Qwen-7B). The adversarial review's simpler explanation — high dimensionality alone — remains unrefuted. A controlled experiment training FP16 LoRA on a ~2B FP16 model with identical data/hyperparameters would be needed.

## Alternative Approaches (What We Could Try Instead)

1. **Per-token routing instead of additive composition (MoLoRA, arxiv 2603.15965):** Rather than summing all adapter deltas (relying on orthogonality), route each token to the most relevant adapter. MoLoRA enables Qwen3-1.7B to beat Qwen3-8B on reasoning tasks. This sidesteps the orthogonality requirement entirely — adapters don't need to be orthogonal if only one is active per token. **Relevant for SOLE**: our 1/N additive composition could be replaced or augmented by learned per-token routing.

2. **Effective delta cosine instead of (A,B) cosine:** Measure cos(vec(B·A), vec(B'·A')) rather than cos([A,B], [A',B']). The REVIEW correctly notes these can diverge. The effective delta captures the actual weight perturbation, not the factored representation. This is a cheap diagnostic to add.

3. **Principal angle analysis (Steele, 2603.02224):** Instead of a single cosine number, compute the full spectrum of principal angles between adapter subspaces. This gives a richer picture of interference geometry. The minimum principal angle θ_min predicts forgetting via F = α(1-cos²θ_min) + β.

4. **Contrastive adapter training (CLoRA, arxiv 2403.19776):** Explicitly optimize adapters to be composable during training using contrastive losses, rather than relying on natural orthogonality. May be unnecessary given our 40x margin, but relevant if scaling to N>100.

5. **LoRAtorio intrinsic composition (arxiv 2508.11624):** Dynamic module selection at inference using patch-level similarity. Avoids the additive assumption entirely.

## Implications for Next Experiments

1. **The orthogonality foundation is solid.** The convergence trajectory + Cao et al.'s independent confirmation means we can proceed with scaling experiments (N=25+) without worrying about training-length-induced cosine inflation.

2. **The 114x FP16 comparison should be deprioritized.** Running a controlled FP16 comparison is scientifically important but NOT on the critical path. The absolute value (|cos|=0.00125, 40x below kill) matters more than the relative comparison to FP16. This is a paper-quality improvement, not a blocking dependency.

3. **Consider adding effective-delta cosine as a secondary metric.** It's cheap to compute vec(B·A) per adapter and would address the REVIEW's concern about (A,B) vs BA measurement. Could be added as a diagnostic to future experiments.

4. **MoLoRA-style routing is a future architectural option.** If additive 1/N composition shows degradation at N>50, per-token routing is the natural escape hatch. Worth tracking but not needed yet.

5. **The "Pause Recycling LoRAs" warning about semantic composability is the real frontier.** Our PPL metrics show composition works, but task-level evaluation at 2B scale was killed (NTP doesn't produce task-capable adapters). The functional composability question remains open and becomes critical at larger scale.

## New References to Add

| Paper | ArXiv ID | Relevance |
|-------|----------|-----------|
| Efficient Modular Learning through Naive LoRA Summation | 2508.11985 | Directly confirms orthogonality-based additive composition; shows cosine-PPL linear correlation |
| Subspace Geometry Governs Catastrophic Forgetting | 2603.02224 | Principal angle theory explains our plateau behavior; F = α(1-cos²θ_min) + β |
| SC-LoRA: Subspace-Constrained LoRA | 2505.23724 | Balances efficient fine-tuning and knowledge preservation via subspace constraints |
| LoRAtorio: Intrinsic LoRA Skill Composition | 2508.11624 | Alternative composition via dynamic module selection, not additive merging |
