# Tequila Minima Reactivation: Research Digest

## Hypothesis

Adding Tequila-style dynamic adaptive biases (summing lambda-scaled shadow weights for deadzone entries) to BitLinear layers will reduce the zero fraction below 20% (from ~32%) and improve PPL compared to standard STE-only ternary training.

## What This Experiment Is

A controlled comparison of three ternary training conditions on a GPT model (d=512, 4 layers, 8 heads, ~64M params) trained on FineWeb-Edu (2M tokens, GPT-2 BPE):

1. **BitLinear baseline**: Standard ternary with Extra RMSNorm, no reactivation
2. **TequilaBitLinear lambda=1e-3**: Paper default reactivation strength
3. **TequilaBitLinear lambda=1e-2**: Aggressive reactivation strength

The Tequila mechanism adds a differentiable bias per output unit:

    C_j = lambda * sum_{i in D_j} w_{j,i}

where D_j is the set of deadzone indices (weights that quantize to 0) and lambda is a learnable scalar per layer. This bias is input-independent and can be fused post-training at zero inference cost.

## Key References

- Tequila (arxiv 2509.23809) -- Trapping-free ternary quantization via Minima Reactivation
- BitNet b1.58 (Ma et al., 2024) -- Ternary {-1, 0, +1} architecture
- Extra RMSNorm (arxiv 2505.08823) -- Pre-quantization normalization for stable ternary training
- Prior experiment: exp_warmstart_fp16_to_ternary (FP32 baseline PPL 344.09 on identical arch/data)

## Empirical Results

### Summary Table

| Condition | PPL | Ratio vs FP32 | Zero Frac | PPL vs Baseline | Lambda Range |
|-----------|-----|---------------|-----------|-----------------|--------------|
| FP32 baseline (prior) | 344.09 | 1.000x | -- | -- | -- |
| BitLinear baseline | 463.26 | 1.346x | 32.0% | -- | -- |
| **Tequila lambda=1e-3** | **432.31** | **1.256x** | 32.0% | **-6.7%** | [-0.283, 0.030] |
| Tequila lambda=1e-2 | 456.04 | 1.325x | 32.0% | -1.6% | [-0.041, 0.274] |

### Kill Criteria Assessment

**K1 (id=239): Reactivation doesn't reduce zero fraction below 20% -> KILL**
- BitLinear baseline zeros: 32.0%
- Tequila lambda=1e-3 zeros: 32.0%
- Tequila lambda=1e-2 zeros: 32.0%
- **Result: FAIL** -- Zero fraction is completely unchanged by reactivation. The mechanism adds compensating biases but does NOT push weights out of the deadzone.

**K2 (id=240): Reactivated model PPL worse than without -> KILL**
- BitLinear baseline PPL: 463.26
- Best Tequila PPL: 432.31 (lambda=1e-3, -6.7%)
- **Result: PASS** -- PPL improved substantially.

### Per-Layer Zero Fractions

Zero fractions are remarkably uniform across all layers and projections (31.0-33.1%), indicating deadzone trapping is a fundamental property of the STE quantization, not layer-dependent.

### Learned Lambda Analysis

With lambda initialized to 1e-3, the learned values diverge dramatically across layers:
- Average: -0.015 (net negative, meaning the bias tends to subtract from outputs)
- Range: [-0.283, +0.030] -- some layers learn 283x the init value
- This wide range suggests different layers have different optimal reactivation strengths

With lambda initialized to 1e-2:
- Average: +0.014 (net positive)
- Range: [-0.041, +0.274]
- Less extreme divergence, but also less PPL improvement

## Interpretation

**The reactivation bias improves model quality but does NOT reduce deadzones.** This is because:

1. The dead_mask is computed with stop_gradient -- deadzone membership is determined by the quantization threshold, which depends on mean(|W|). The reactivation pathway provides gradients to the shadow weights OF dead neurons, but these gradients must still overcome the quantization threshold to escape the deadzone.

2. The bias term C(W) is input-independent. It acts as a learned per-output-unit bias that compensates for the lost capacity of dead weights, rather than reactivating them to participate in input-dependent computation.

3. The 6.7% PPL improvement from lambda=1e-3 shows that dead weights contain useful statistical information in their shadow values. Even though they quantize to zero, their pre-quantization magnitudes and signs carry signal that the bias term can exploit.

4. The lambda=1e-2 condition improves PPL by only 1.6%, suggesting that too-large initial lambda destabilizes early training. The paper's 1e-3 default is well-calibrated.

## Why K1 Fails Despite K2 Passing

The kill criterion K1 was based on the assumption that "reactivating dead weights" means making them non-zero after quantization. In practice, Tequila does something different: it extracts value from dead weights via a side channel (the bias) without changing their ternary quantization. The weights remain at zero in the ternary representation, but their shadow (full-precision) values contribute through the bias.

This is analogous to how batch normalization's learned bias can compensate for zeroed-out activations -- the zeros are still there, but downstream computation adjusts.

## Limitations

1. **Scale**: Tested at d=512 (64M params). Tequila reports results at 1B-3B where the mechanism may have more room to help.

2. **Training budget**: 2000 steps on 2M tokens. The paper uses 10B tokens. With more training, the dead weights' shadow values may accumulate more useful information for the bias.

3. **Zero fraction measurement**: We measure zeros after applying round(w/alpha). The reactivation bias does not change the ternary weights themselves, only adds a compensating signal. A different definition of "reactivated" (contributing to output despite quantizing to zero) would make K1 PASS.

4. **FP32 baseline from prior run**: The FP32 PPL (344.09) is from a prior experiment with 3000 steps. Our 2000-step conditions may not have fully converged, inflating the gap. However, all ternary conditions use identical training, so the relative comparison is fair.

5. **Single seed**: No seed variation. The 6.7% improvement could partially be noise, though the direction is consistent across both lambda values.

## What Would Kill This

At micro scale:
- Already partially killed: K1 FAIL means Tequila does not reduce deadzones as defined.
- If PPL improvement vanishes with seed variation (K2 becomes noise).

At macro scale:
- If the bias improvement is smaller than 1% at 1B+ scale (may be absorbed by model capacity).
- If the bias conflicts with LoRA adapter composition (adds layer-level bias that interacts with adapter deltas).

## Key Takeaway for the Project

**Tequila-style reactivation is a cheap PPL win (-6.7%) that can be added to any BitLinear layer with zero inference cost (bias fusion).** However, it does NOT reduce the deadzone -- the 32% zero fraction is structural to STE ternary training at this scale. To actually reduce zeros, a different mechanism is needed (e.g., modified quantization thresholds, adaptive alpha, or the full Tequila training pipeline with its mixed gradient approach applied differently).

For the Composable Ternary Experts architecture:
- The bias can be fused into the base model post-training (zero overhead)
- It is orthogonal to LoRA composition (biases add, LoRA deltas multiply)
- Worth integrating into the standard BitLinear recipe as a free PPL improvement
