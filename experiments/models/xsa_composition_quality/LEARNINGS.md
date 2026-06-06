# Learnings: exp_xsa_composition_quality

## Core Finding

XSA (Exclusive Self-Attention) for adapter composition is KILLED: removing the self-value component from attention outputs costs 3.1% representational capacity per head (at d_h=32) while solving a non-existent interference problem -- Grassmannian A-matrices already suppress inter-adapter cosine to ~0.011, leaving no interference channel for XSA to address.

## Why This Happened (Literature-Grounded)

The failure has two independent causes, both well-established in literature:

**1. Capacity cost dominates at small head dimension.** XSA removes one direction per head. At d_h=32 (our micro scale), this costs 3.1% of capacity; at d_h=64 or d_h=128 (parameter-golf scale, arXiv 2603.09072), the cost drops to 1.6% or 0.8%. The parameter-golf competition found XSA beneficial at d=512, 8 heads (d_h=64), applied only to last 2-3 of 11 layers. Our experiment at d_h=32 sits in the regime where the capacity tax exceeds any possible benefit. This is consistent with a general principle: zero-parameter modifications are not zero-cost -- they trade representational capacity for structural constraints.

**2. Weight-space orthogonality already solved.** The "Rethinking Inter-LoRA Orthogonality" paper (arXiv 2510.03262) proves that weight-space orthogonality does not equal semantic disentanglement due to nonlinearities. Our own findings confirm this (SOLE project): cosine orthogonality of 0.0002 at d=896 does not prevent composition catastrophe when magnitude scaling is uncalibrated. XSA targets a weight-space interference channel (self-value component of merged attention output) that is already negligible (|cos|~0.011) due to Grassmannian initialization. Even if XSA perfectly removed this channel, the actual composition bottleneck lies in logit-scale mismatch and nonlinear amplification -- mechanisms XSA cannot address.

## Confirming Evidence

1. **arXiv 2510.03262 (Rethinking Inter-LoRA Orthogonality):** Explicitly shows that enforcing strict inter-LoRA orthogonality in weight space does not yield semantic disentanglement. Cosine orthogonality only measures direction, not magnitude; nonlinearities in residual streams and LayerNorm amplify orthogonal perturbations into correlated outputs. Directly predicts our null result.

2. **SOLE composition catastrophe (our FINDINGS.md):** Equal-weight composition of perfectly orthogonal adapters (cos=0.0002) produced PPL in the trillions. The failure mode was logit-scale mismatch from magnitude-driven interference, not subspace collision. This proves structural orthogonality (whether Grassmannian or XSA-based) cannot protect against the actual composition failure mode.

3. **Output-averaging vs pre-merge (our exp_output_averaging_vs_param_merge):** Confirmed that pre-merge quality loss at high k comes from 1/k dilution, NOT cross-term interference. Grassmannian orthogonality is irrelevant to the PM vs OA gap. This reinforces that interference at the attention level is not the bottleneck.

4. **Orthogonal Monte Carlo Dropout (via BitNet+SOLE notebook):** Empirical analysis concluded that enforcing strict orthogonality does not lead to semantic disentanglement -- mathematical orthogonality in parameter space is insufficient for functional compositionality.

## Contradicting Evidence

1. **XSA at larger scale (arXiv 2603.09072, parameter-golf):** XSA consistently improves loss across 0.7B-2.7B parameter models, various sequence lengths and learning rates. The capacity cost (0.8-1.6% at d_h=64-128) is tolerable at production scale. **Discrepancy:** Our experiment used d_h=32 where the 3.1% cost is 2-4x higher. XSA may work for composition at larger head dimensions, but this is unlikely given that Grassmannian already solves interference.

2. **Head-wise routing modifications (MH-MoE):** Multi-Head Mixture-of-Experts modifies attention routing by performing head-wise routing over sub-representations, drastically reducing forgetting during continual learning. **Discrepancy:** MH-MoE modifies routing granularity (which heads see which experts), not removing capacity. XSA removes capacity. The mechanism is fundamentally different.

## Alternative Approaches (What We Could Try Instead)

Given that the actual composition bottleneck is magnitude/logit-scale mismatch (not attention interference), more promising directions include:

1. **TIES Merging (TRIM, ELECT SIGN & MERGE):** Resolves parameter interference through directional sparsity -- trims low-magnitude weights, merges via sign-aware majority consensus. Addresses the actual interference mechanism (conflicting weight signs/magnitudes) rather than attention structure.

2. **LoRA-LEGO (Rank-Wise Clustering):** Disassembles LoRA into "Minimal Semantic Units" per rank, performs rank-wise clustering to bypass parameter interference entirely. Novel composition paradigm that avoids the weight-space orthogonality trap.

3. **Sub-MoE (Subspace Expert Merging):** Joint SVD on concatenated expert weights extracts shared U-matrix, merges only expert-specific V-matrices. Aligns composition in a shared subspace rather than assuming orthogonality implies independence.

4. **Activation-Guided Consensus Merging (ACM):** Determines layer-specific merging coefficients from mutual information between pre-trained and fine-tuned activations. Data-space approach that directly measures what matters (functional similarity) rather than geometric properties.

5. **Output-space fusion (LoRAuter):** Sidesteps weight-space merging entirely by composing in output space with retrieval-weighted routing. Already partially validated by our output_averaging experiment (OA beats PM at k>=25).

6. **1/N scaling + PPL-probe routing (our own):** Already proven to resolve composition catastrophe (PPL trillions -> 2.36). The solution to composition quality is better routing, not attention modification.

## Implications for Next Experiments

1. **Close the attention-modification line for composition.** XSA is the second attention-layer intervention killed (after SwiGLU gate pruning at macro). The pattern is clear: modifying attention structure to improve composition attacks the wrong bottleneck when Grassmannian orthogonality is in place. Future composition work should focus on routing and magnitude calibration.

2. **"Zero-parameter" framing is misleading.** XSA, gating modifications, and similar tricks add zero learnable parameters but trade representational capacity. Frame all such modifications as capacity-cost interventions and require a capacity budget analysis before experimenting.

3. **Grassmannian orthogonality is sufficient for weight-space decorrelation.** Three independent experiments now confirm this: (a) macro ortho_scaling (cos=0.0002 at d=896), (b) ternary B-matrix (cos~0.000001), (c) XSA experiment (cos~0.011 unchanged with/without XSA). The orthogonality problem is solved. Future work should address the actual bottleneck: routing quality, magnitude calibration, and logit-scale alignment.

4. **Scale matters for capacity-trading modifications.** If any zero-parameter attention trick is revisited, require d_h >= 64 (capacity cost <= 1.6%) and test on real text with long sequences. The micro regime (d_h=32, T~8-24) is maximally hostile to capacity reductions.

5. **Untested variant noted but deprioritized.** Inference-only XSA (train normally, apply XSA only during composition) avoids training-time capacity cost but is unlikely to help given near-zero adapter interference. Low priority unless working with non-Grassmannian systems.
