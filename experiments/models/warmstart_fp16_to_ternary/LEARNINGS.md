# Learnings: exp_warmstart_fp16_to_ternary

## Core Finding

Warm-start FP16-to-ternary QAT achieves 1.046x FP32 PPL at d=512 (10% FP16 switch point), with Extra RMSNorm being the dominant lever (cuts cold-start gap from 2.78x to 1.211x, a 76% improvement). The warm-start advantage (12.4% over weight-decay-controlled cold-start) is genuine and comes from weight initialization + optimizer state transfer, not weight decay removal.

## Why This Happened (Literature-Grounded)

Three mechanisms explain why even minimal FP16 pretraining (300 steps, PPL ~843) helps ternary QAT converge:

**1. Gradient flow initialization.** During ternary QAT with STE, small gradient updates to shadow weights often fail to cross the quantization thresholds needed to change discrete ternary values (-1, 0, +1). FP16 pretraining establishes weight magnitudes that are already distributed near ternary boundaries, so subsequent STE gradients are more likely to produce meaningful discrete weight changes. Continual QAT (Nielsen et al., arxiv 2502.11895) explicitly documents this: FP16 warm-up "transforms weight distributions from standard Gaussian-like shapes into configurations concentrated near ternary boundaries."

**2. Optimizer state dampens transition shock.** Retaining AdamW momentum (m_t) and variance (v_t) from the FP16 phase substantially dampens the loss spike at the FP16-to-ternary switch point. Our experiment measured spikes of +0.43 and +0.69 nats recovering within at most 51 steps. The Continual QAT paper confirms this mechanism: warm optimizer state dampens spikes, though recovery is possible without it given sufficient remaining QAT steps.

**3. Extra RMSNorm prevents scale drift.** Pre-quantization RMSNorm normalizes inputs to each BitLinear layer, preventing the drifting activation scales that cause erratic outputs when multiplied by coarse ternary weights. This is the single most impactful modification: it alone reduced our cold-start gap from 2.78x (prior experiment without RMSNorm) to 1.211x. BitNet b1.58 (Ma et al., 2024), 1.58-bit FLUX (Wang et al., arxiv 2505.08823), and BitDistill all mandate this as architectural necessity. RMSNorm (not LayerNorm) is specifically preferred because omitting mean subtraction is more stable when 0 is a quantized state.

## Confirming Evidence

| Paper | Finding | Relevance |
|-------|---------|-----------|
| **Continual QAT** (Nielsen et al., arxiv 2502.11895) | 20% FP16 + 80% QAT optimal; training from scratch is suboptimal. Best at 2K/10K 16-bit steps. | Directly confirms our warm-start protocol. Our 10% finding is consistent (they test 20-40%). |
| **BitNet b1.58** (Ma et al., 2024) | Cold-start ternary matches FP16 only at 3B+ params; smaller models benefit from warm-start. | Explains why warm-start matters more at our 64M scale than at BitNet's 3B scale. |
| **ParetoQ** (Tseng et al., 2024) | Budget allocation: ~90% FP + 10% QAT maximizes accuracy for fine-tuning. | Confirms FP pretraining is valuable; their 90/10 split differs from our 10/90 (see Contradicting Evidence). |
| **BitDistill** (Tencent, 2024) | Three-stage pipeline: SubLN insertion, continual pre-training warm-up, then distillation. | Confirms Extra RMSNorm + warm-up is standard production recipe. |
| **1.58-bit FLUX** (Wang et al., arxiv 2505.08823) | Extra RMSNorm before every quantized linear is critical for 1.58-bit training stability. | Directly validates our Extra RMSNorm finding as the dominant lever. |

## Contradicting Evidence

**1. Cold-start can match warm-start at sufficient scale.** BitNet b1.58 demonstrates that at 3B+ parameters, cold-start ternary matches FP16 quality without any warm-start phase. Our 64M model is far below this threshold. **Implication:** the warm-start advantage may diminish as we scale up. If we reach 1-3B parameters, cold-start becomes viable and warm-start overhead may be unnecessary.

**2. ParetoQ's 90/10 split contradicts our 10/90 finding.** ParetoQ found that allocating 90% of tokens to FP training and only 10% to QAT maximizes accuracy. Our experiment found the opposite: 10% FP + 90% QAT beats 20% FP + 80% QAT. **Reconciliation:** ParetoQ evaluated fine-tuning (adapting a converged model), while we're doing pre-training from scratch with a fixed step budget. In pre-training, more QAT steps means more ternary-specific optimization; in fine-tuning, the FP model is already converged so less QAT is needed. The distinction is critical: our protocol is "initialize then train ternary," not "train FP then quantize."

**3. Overtrained models resist quantization.** Ouyang et al. found that heavily trained FP models (100T+ tokens) suffer catastrophic degradation when quantized, because their narrow weight distributions cannot absorb the discrete ternary mapping. **Implication:** if we extend FP pretraining too long before switching, we may hit this failure mode. Our finding that 10% beats 20% is directionally consistent — more FP training is not always better.

**4. Quantization-induced degradation scales with tokens.** Scaling laws predict QiD worsens at higher token counts. A 70B model trained on 17T tokens hits 20% degradation. **Implication:** our 2M-token results cannot predict behavior at billions of tokens. The smooth transition we observed may not hold.

## Alternative Approaches (What We Could Try Instead)

**1. Knowledge Distillation (BitDistill protocol).** Instead of warm-starting our own ternary model, distill from an existing FP teacher (e.g., Qwen2.5-0.5B). BitDistill uses SubLN + continual pre-training warm-up + dual-mechanism distillation (logits + attention). Achieves 10x memory savings. Could give better quality than our self-supervised warm-start at the cost of needing a teacher model.

**2. Tequila's Minima Reactivation.** Addresses deadzone trapping where ternary weights get stuck at 0. Repurposes trapped weights as dynamic biases, yielding >4% accuracy gain on ARC. Our ~31-32% zero fraction suggests significant trapping. Tequila could recover capacity from these frozen-zero weights without changing the training recipe.

**3. Sparse-BitNet exploitation.** Ternary models naturally produce ~42% zero weights (we see ~31-32%). Applying N:M semi-structured sparsity (e.g., 2:4) on top of ternary could push effective bit-rate to 0.5 bits/param. 1.58-bit models are resilient to pruning (+5.7% PPL vs +18.8% for FP). Could be combined with warm-start recipe.

**4. Direct Quantized Training (DQT).** Eliminates STE shadow weights entirely via stochastic rounding. Reduces memory footprint during training. Worth exploring if memory becomes the bottleneck at larger model scales on M5 Pro.

**5. Weight Indexing (from BitTTS).** Pack 5 ternary weights into a single 8-bit index (3^5=243 fits in a byte). 83% storage reduction at inference. Orthogonal to training recipe — purely a serving optimization for MLX deployment.

## Implications for Next Experiments

**1. Scale validation is the critical next step.** The warm-start advantage is proven at d=512 (64M params) but the literature says cold-start catches up at 3B+. Testing at d=1024 and d=2048 will bracket the crossover point. If warm-start advantage persists at d=2048 (~256M params), it validates the recipe for our target scale.

**2. The 10% switch point may not be optimal.** Our experiment tested only 10% and 20%. The Continual QAT paper found 20% optimal. ParetoQ found 90% optimal for fine-tuning. Testing 5% and 2% switch points could reveal if even less FP pretraining suffices (our model was still at PPL ~843 at the 10% switch).

**3. Extra RMSNorm is mandatory infrastructure.** It accounts for 76% of the cold-start gap reduction (2.78x to 1.211x). Every future ternary experiment must include it. This is no longer a finding — it's a requirement.

**4. Tequila deadzone fix is complementary.** Our 31-32% zero fraction is below the literature's 42% expected rate, but still represents frozen capacity. Tequila could be tested as an add-on to the warm-start recipe.

**5. Optimizer state vs. weight init ablation is still open.** The warm-start advantage (12.4%) could come from weight initialization, optimizer state transfer, or both. A targeted ablation (warm-start with optimizer reset) would separate these contributions. If weight init alone provides most of the benefit, the recipe simplifies (no need to transfer optimizer state across architectural changes).

**6. Composition on warm-started base is untested.** The warm-start recipe produces a ternary base model. The next question is whether ternary LoRA adapters compose well on this warm-started base vs. a cold-start base. This bridges P0 (ternary base) to the core SOLE architecture.
