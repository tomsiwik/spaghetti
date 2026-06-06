# Learnings: exp_attnres_depth_composition

## Core Finding

Depth-wise softmax attention (AttnRes) learns non-uniform depth weights (entropy ratio 0.775) without hurting base quality (0.984x PPL), but composition improvement is negligible at L=4 (0.39%, 3/3 seeds consistent direction but within noise). The mechanism is real; the effect requires deeper models to manifest.

## Why This Happened (Literature-Grounded)

**PreNorm dilution scales quadratically with depth, not linearly.** The Kimi AttnRes paper (arXiv 2603.15031) demonstrates benefits at L=48 where each layer's relative contribution drops to ~2%. At L=4, each layer contributes ~25% — there is simply not enough dilution to fix. The 1.25x compute-equivalent improvement reported by Kimi was measured on their K2 model (48B params, 48 layers), a 12x deeper architecture than our micro model.

**MoDA (arXiv 2603.15619) confirms:** "informative features formed in shallow layers are gradually diluted by repeated residual updates, making them harder to recover in deeper layers." The key word is "gradually" — at L=4, the degradation has barely begun.

**Depth utilization research (Csordás et al., OpenReview 2025)** shows that current LLMs underutilize the second half of their layers, spreading computation rather than increasing complexity. This suggests the benefit of depth attention is in the deeper half where utilization is worst — exactly the regime our L=4 model cannot access.

**The 0.39% consistent direction** is explained by our S3 finding: AttnRes redistributes adapter norms more uniformly across depth (vs standard residuals where adapter norms increase with depth: [1.30, 1.86, 2.29, 3.36]). This redistribution is a real gradient-landscape effect, but the magnitude is small because L=4 has minimal norm imbalance to correct.

## Confirming Evidence

1. **Kimi AttnRes (arXiv 2603.15031):** Block AttnRes matches 1.25x compute baseline. Output magnitudes bounded, gradient norms more uniform across depth. Our K3 result (entropy 0.775, non-uniform depth weights) directly confirms their mechanism claim at micro scale.

2. **Value Residual Learning / ResFormer (arXiv 2410.17897):** Standard hidden-state residuals fail to preserve initial token-level information as depth increases. ResFormer achieves same validation loss with 16.1% fewer parameters and 20.3% less training data. Different mechanism (value residuals vs depth attention) but same diagnosis: standard residuals are suboptimal at depth.

3. **MoDA (arXiv 2603.15619):** Unifies sequence and depth attention, +2.11% at 1.5B scale. Confirms that depth-wise attention is a productive research direction, but again at scale (1.5B, many layers).

4. **SiameseNorm (arXiv 2602.08064) and Post-LayerNorm revival (arXiv 2601.19895):** Both address the PreNorm dilution problem from the normalization side. Confirms the problem is real and widely recognized, with multiple independent solution approaches.

## Contradicting Evidence

1. **No direct contradictions found** for the mechanism (depth attention learning non-uniform weights). This is well-established.

2. **Depth attention ≠ composition improvement** is the gap. No paper tests AttnRes specifically for LoRA adapter composition. Our experiment is novel in this regard, and the null result at L=4 is informative: the mechanism helps base quality but doesn't automatically translate to composition benefits at shallow depth.

3. **Orthogonality limitations (arXiv 2510.03262, Orthogonal Monte Carlo Dropout):** Weight-space orthogonality does not guarantee semantic disentanglement or function-space independence. This is relevant because our untested Grassmannian interaction may face the same issue — AttnRes changes gradient flow through softmax normalization, which could break the interference bounds that Grassmannian A matrices rely on.

4. **OPLoRA (arXiv 2510.13003):** Orthogonal projection prevents catastrophic forgetting during PEFT. Localizes conflicts to final 6-10 layers. This is relevant: if adapter interference is concentrated in deep layers, AttnRes's ability to selectively weight those layers could help — but only if the model is deep enough to have a "deep layer" regime.

## Alternative Approaches (What We Could Try Instead)

### For the dilution problem specifically:
1. **Activation-Guided Consensus Merging (ACM):** Training-free framework that determines layer-specific merging coefficients based on mutual information between pre-trained and fine-tuned activations. No architectural change needed — pure post-hoc reweighting. Could achieve what AttnRes does without adding parameters.

2. **Sub-MoE (Subspace Expert Merging):** Joint SVD on concatenated expert weights, identifies shared U-matrix, merges only functionally distinct V-matrices. Attacks the composition problem directly rather than through residual connections.

3. **ReXMoE (cross-layer expert reuse):** Allows routers to reuse experts across adjacent layers with Progressive Scaling Routing. Changes depth accumulation without modifying residual connections — potentially more compatible with Grassmannian skeleton.

### For the composition quality problem:
4. **PPL-Probe weighting** (already validated in SOLE: +9.34pp over equal-weight, r=0.990 with oracle). This is already our best composition method — AttnRes would need to beat it to be worth integrating.

5. **MoLoRA per-token routing (arXiv 2603.15965):** Token-level expert selection. Orthogonal to AttnRes — could potentially combine depth attention with per-token routing for compounding benefit, but complexity increases significantly.

6. **Layerwise Recurrent Router (RMoE):** GRU-based router that passes routing decisions forward across layers. Addresses the same depth-wise optimization as AttnRes but through routing rather than residual modification.

## Implications for Next Experiments

1. **AttnRes at L=16+ is the decisive test.** The mechanism validates (K3 PASS), but practical benefit requires depth. The next experiment should use at least L=16 with harder data (not character-level patterns where base PPL is already ~1.01 on domain data). This would be a significant compute investment on M5 Pro.

2. **Grassmannian interaction is the critical unknown.** AttnRes introduces softmax normalization into the gradient path. Grassmannian A matrices maintain interference bounds under standard additive residuals. Under AttnRes, the effective gradient through frozen A is scaled by depth attention weights — this could either help (by focusing gradient on the most relevant layers) or hurt (by starving low-attention layers of gradient signal, breaking the orthogonality contract). This MUST be tested before any SOLE integration.

3. **ACM is a cheaper alternative.** If the goal is layer-specific composition weighting, Activation-Guided Consensus Merging achieves this post-hoc without architectural changes. Worth testing as a comparison baseline before committing to AttnRes integration.

4. **The adapter norm redistribution (S3) is the most interesting mechanistic finding.** Standard residuals cause adapter norms to increase with depth; AttnRes makes them more uniform. This gradient-landscape effect deserves investigation independent of composition quality — it may explain why deeper AttnRes models show larger benefits (more layers = more norm imbalance to correct).

5. **Don't test AttnRes + composition in isolation from PPL-probe weighting.** Our best composition method is already PPL-probe (r=0.990 with oracle). The right question is whether AttnRes + PPL-probe beats PPL-probe alone — testing against 1/N averaging (as we did here) sets too low a bar.
