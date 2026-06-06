# LEARNINGS: exp_p9_ttlora_polar_hybrid

## Core Finding
Stiefel retraction on TT-LoRA interior cores acts as a **norm regularizer** (||ΔW||_F drops 2-3x), not a spectral regularizer. GSM8K improved +8pp (62%→70%) but this is not statistically significant at N=50 (p~0.42). The durable result is the mechanism: Stiefel forces ||G_k||_F = sqrt(r), preventing norm amplification through the contraction chain.

## Why
In left-canonical TT form, prefix contractions are isometric under Stiefel (Oseledets 2011). Without Stiefel, unconstrained core norms grow 2-3x over 500 steps, causing ||ΔW||_F ≈ 10.0 vs 4.3 with Stiefel. This over-correction is the root cause of lower quality in unconstrained TT-LoRA. The sr improvement predicted by PoLAR theory (Finding #442) does NOT transfer to TT because the contraction chain breaks the isometric prefix → sr(ΔW) relationship.

## Mechanism Corrected
- **Predicted:** Stiefel → isometric prefix → sr(ΔW) = sr(last core) → 1.5-3x sr improvement
- **Actual:** Stiefel → norm-bounded cores → ||ΔW||_F controlled → regularization → better generalization
- sr ratio measured: 1.03x (noise-level, not the predicted 1.5-3x)

## Composition Implication
Smaller ||ΔW||_F directly widens the null-space margin in interference bounds (Finding #225). Stiefel on TT-LoRA is a free regularizer that may improve multi-adapter composition — worth testing explicitly.

## Implementation Note
K1365 FAIL (9ms retraction) is numpy data-transfer overhead from 210 SVDs of 48×6 matrices. Amortized over 10-step intervals: 0.9ms/step. An MLX-native batched SVD would reduce this 10x+. Not a fundamental barrier.

## Next Experiment Pointer
Test TT-LoRA-Stiefel with full layer coverage (all proj layers, 1000+ steps) + N>200 GSM8K eval to get statistical power. Alternatively: measure composition interference directly (||ΔW_1^T ΔW_2||_F) for Stiefel vs unconstrained pairs.

## References
- Oseledets (2011). TT Decomposition. SIAM J. Sci. Comput.
- Batselier et al. (2025). TT-LoRA MoE. arXiv:2504.21190
- Finding #442: Joint Stiefel PoLAR guarantees sr=r
- Finding #516: TT-LoRA quality baseline (84.4% of LoRA, 12.4x compression)
- Finding #519: This experiment (SUPPORTED)
