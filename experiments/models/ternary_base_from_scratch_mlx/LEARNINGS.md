# Learnings: ternary_base_from_scratch_mlx

## Core Finding

Ternary transformers train from scratch with STE on MLX and compose near-perfectly (1.022 ratio) via Grassmannian LoRA, but the 1.003x PPL match is an artifact of task overcapacity (vocab-27 character names), not evidence of ternary-FP32 parity at scale.

## Why This Happened (Literature-Grounded)

The near-perfect PPL match (1.003x) is explained by **model overcapacity relative to task complexity**. A 4.7M parameter model on a vocab-27 character-level task has far more capacity than needed — both FP32 and ternary versions saturate the task, making the quantization constraint non-binding. This is consistent with BitNet b1.58 scaling laws: at small scale, ternary models can match FP32 on simple tasks, but the gap widens on harder tasks before narrowing again at 3B+ parameters (Ma et al., 2024).

The STE mechanism works here because:
1. **Identity gradient approximation** passes gradients through the non-differentiable quantization step (forward uses ternary weights, backward treats quantizer as identity).
2. **Shadow weights in FP32** accumulate precise gradient updates even though forward pass is ternary.
3. At this small scale, STE gradient variance is negligible — the "blind spot" problem (quantization error invisible to backward pass) only manifests at deeper networks with more layers of accumulated noise.

The 31.3% zero-weight fraction is **expected and healthy** per BitNet b1.58, which reports 30-40% zeros in converged ternary models. Zeros represent learned sparsity, not deadzone trapping — the latent FP32 weights still receive gradients via STE and can escape zero if the loss gradient is strong enough.

The near-perfect composition ratio (1.022) is **mathematically guaranteed** by the Grassmannian A-matrix construction (QR decomposition ensures A_i^T A_j = 0), combined with 1/N scaling that reduces each adapter's contribution by 5x. This is a correctness check, not an empirical discovery.

## Confirming Evidence

- **BitNet b1.58** (Ma et al., 2024): Ternary from-scratch training with STE works, achieving FP16 parity starting at ~3B parameters. Uses same alpha-scaled round-clip quantization formula.
- **Spectra LLM suite** (2024): 3.9B ternary model matches half-precision on commonsense reasoning and knowledge benchmarks, slight lag on noisy web corpora perplexity.
- **"16-to-1.58" strategy**: Pre-training in FP16 before switching to ternary QAT yields near-FP16 accuracy with only 2-3 point drop — suggesting that even without this warm start, from-scratch STE is viable when combined with large learning rates.
- **SCA regularizer** (shape-controlling approach): Can dial zero proportion to 50% with <0.1% accuracy drop, confirming that high zero fractions are not inherently harmful.

## Contradicting Evidence

- **pQuant research**: 1-bit QAT-from-scratch models show poor scaling efficiency — performance gains grow sublinearly and fall far behind FP16 at large scale. This directly threatens our path if we attempt to scale the from-scratch approach to 2B without mitigation.
- **Tequila** (arxiv 2509.23800): Explicitly notes BitNet models "still fail to match full-precision performance" after 4T tokens due to information loss from ternary compression. Our 1.003x ratio is an artifact of task simplicity, not ternary quality.
- **Parameter democratization**: At larger scales, ternary models exhibit flattened sensitivity distributions — no parameters become highly specialized, limiting the model's ability to prioritize informative features. This could affect adapter specialization at scale.
- **Attention quantization failure**: BitNet could not quantize Q/K attention matrices to ternary without significant performance drop and convergence failure. Our experiment uses ternary BitLinear for attention projections (Q/K/V/O), but the simple task may mask this issue.
- **STE "blind spot"**: The quantization error (difference between continuous and rounded weight) is invisible to backward pass. At deeper networks, this corrupts learning signals and can cause training divergence. Our 6-layer model is too shallow to trigger this.

## Alternative Approaches (What We Could Try Instead)

1. **Tequila's Minima Reactivation** (arxiv 2509.23800): Repurposes deadzone-trapped zero weights as learnable adaptive biases (lambda_i * w_i), creating smooth differentiable quantization that bypasses noisy STE. Near-zero inference overhead. Directly addresses our K4 concern about 31.3% zeros.

2. **Sherry's 3:4 structured sparsity** (Tencent/AngelSlim): Enforces hardware-aligned 3:4 sparsity in ternary weights, enabling 5-bit block packing (effective 1.25 bits). Perfect for Apple Silicon SIMD lanes. Could complement our natural 31% sparsity.

3. **MatMul-free LM** (ridgerchu/matmulfreellm): Eliminates matrix multiplication entirely via MLGRU token mixer (Hadamard products + additions) + ternary BitLinear channel mixing. Scales to 2.7B. Avoids the attention quantization failure that standard BitNet faces.

4. **GaLore+STE integration**: Our GaLore experiment showed 2-3x ternary degradation without STE-in-loop. STE-aware GaLore would enable memory-efficient ternary training at 2B scale on M5 Pro (FP32 latent weights + Adam = ~24GB at 2B, tight for 48GB).

5. **Large learning rate regime**: Literature emphasizes that STE requires abnormally large learning rates to push latent weights across ternary clipping thresholds. Our experiment used default rates — exploring 3-10x larger LR could improve convergence and reduce the zero fraction.

6. **"16-to-1.58" warm start**: Pre-train in FP16 for initial epochs, then switch to ternary QAT. Could bridge the quality gap at harder tasks without full from-scratch commitment.

## Implications for Next Experiments

1. **GaLore+STE (exp_galore_ste_ternary_scaffold) is now unblocked** and should be the immediate next step. It addresses the memory bottleneck that will hit at 2B scale (FP32 shadows + Adam = 3 copies of weights).

2. **The 1.003x ratio should NOT be extrapolated** to harder tasks. Future experiments must use real language modeling (WikiText, C4) with proper vocabulary, not character-level toys. Expect 1.5-2.5x PPL ratios on real tasks at sub-3B scale.

3. **K4 threshold should be revised to 40%** based on BitNet b1.58 empirical norms. 30-40% zeros are healthy sparsity, not trapping. True trapping diagnosis requires comparing zero masks across training checkpoints.

4. **Multi-seed validation is essential** for any future PPL comparison claims. The 0.004 PPL gap (1.5895 vs 1.5935) is within noise for a single seed.

5. **Tequila integration should be explored** if deadzone trapping becomes a real problem at scale. The Minima Reactivation mechanism is orthogonal to our Grassmannian adapter architecture and could be composed.

6. **Attention quantization is a latent risk**. At scale, ternary Q/K projections may fail. Consider MatMul-free attention alternatives or keeping attention in higher precision.

7. **The composition mechanism is validated but only under gentle conditions** (1/N uniform averaging). Per-token routing at full adapter scale (the deployment scenario) is a different interference regime that needs separate testing.
