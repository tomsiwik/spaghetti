# Learnings: exp_ternary_base_scale_d512

## Core Finding
Ternary STE training at d=512/8L on real English text converges reliably but fails the 2x PPL bar (2.78x actual). The dominant failure mode is **overfitting** -- ternary train loss (2.48) beats FP32 (3.17) but val PPL is 2.78x worse -- indicating the coarse weight space memorizes training data but generalizes poorly at small data scales (0.026 tokens/param vs Chinchilla-optimal ~20 tokens/param).

## Why This Happened (Literature-Grounded)

Three mechanisms from the literature explain the overfitting-without-generalization pattern:

1. **Parameter democratization** (pQuant, 2411.04965): In extremely low-bit quantization, the sensitivity of all parameters becomes homogenized. The discrete {-1,0,+1} grid cannot smoothly distinguish between moderately important and highly critical features. Small models lack the width to compensate, which is why pQuant introduces a decoupled 8-bit branch for sensitive tokens.

2. **Kernel alignment requires width** (BitNet scaling analysis): Theoretical proofs on 1-bit training dynamics show that extreme quantization only aligns with smooth kernel behavior as network width increases. At d=512 (small width), the discrete step-functions of ternary weights cause jagged, poor interpolation on unseen inputs -- explaining why train loss drops (memorization of specific patterns) while val PPL stays high (poor interpolation to new text).

3. **Data scarcity amplifies the quantization penalty**: BitNet's scaling laws apply only "as long as N >= 3B" (Ma et al., 2024). The smallest BitNet model evaluated is 125M params trained on billions of tokens. At 77M params with 2M unique tokens, we are far outside the regime where ternary scaling predictions hold. The 2.78x gap measures "ternary penalty under extreme data starvation," not the fundamental ternary penalty for language modeling.

## Confirming Evidence

- **pQuant** (2411.04965): 1-bit QAT scales sublinearly -- the quality gap widens before narrowing at 3B+. Our 2.78x at 77M confirms this trajectory. The smallest pQuant model is 300M trained on 100B tokens.
- **BitNet b1.58** (Ma et al., 2024): Explicitly warns that "very small and capacity-limited networks may experience reduced performance unless width or network size is increased." Validation loss gap decreases from 0.5 to 0.09 as model size scales from 125M to 100B.
- **Spectra/TriLM** (2407.12327): 3.9B ternary lags behind FP16 on noisy web corpora (our FineWeb-Edu scenario) but performs better on cleaner datasets -- suggesting data quality matters more for ternary models.

## Contradicting Evidence

- **Trained Ternary Quantization (TTQ)** outperformed FP32 on ResNet-32/44/56 on CIFAR-10 by 0.04-0.36%. However, these are vision models (very different from language modeling) with much higher tokens-per-parameter ratios. The regularization benefit of quantization appears task-dependent.
- **QAT as implicit regularizer**: Multiple studies confirm that simulated quantization errors during QAT act as regularization akin to dropout, enhancing robustness. This contradicts our observation that ternary overfits MORE. Resolution: the regularization benefit depends on data volume. With sufficient data, quantization noise regularizes; with scarce data, the coarse weight space instead enables memorization of the small training set.
- **Lu et al. (2025)**: In certain sparse feature spaces, ternary quantization improves feature discrimination via "free denoising and signal selection." This suggests ternary could help with the RIGHT data regime -- our data scarcity prevents this benefit from manifesting.

## Alternative Approaches (What We Could Try Instead)

### Highest Priority
1. **"16-to-1.58" warm-start** (BitNet): Start with FP16 pre-training, then transition to ternary QAT. Research on 100K-48M BitNet models shows this nearly matches FP32 with only 2-3 point drops. This avoids the cold-start generalization issue entirely.

2. **Vocabulary reduction to 4K-8K BPE**: Our 50,257-token vocab puts 67% of params (51.5M) in embeddings. Fitting a tokenizer at 4K-8K and measuring "fertility" can find the optimal point. Falcon experiments show fertility stabilizes around 85% non-split tokens.

3. **Stream 10-50M unique tokens**: Moving from 0.026 to 0.5+ tokens/param should enter the regime where ternary scaling predictions start to hold.

### Medium Priority
4. **Weight decay scheduling**: Standard weight decay is harmful in ternary training because it affects latent (shadow) weight magnitudes which serve as confidence scores. Research shows **removing weight decay for the second half of training** significantly improves convergence.

5. **Tequila minima reactivation** (2509.23800): Even though our deadzone is healthy (31.4%), Tequila's differentiable reactivation (repurposing dead weights as dynamic biases) bypasses STE entirely for trapped weights, providing direct gradient signals. Could improve generalization even without deadzone issues.

6. **Mixed-precision (selective FP32 layers)**: TernaryLM (132M params) shows middle transformer layers have highest quantization compatibility. Early/late layers could remain FP32. Alternatively, BiPFT adds strategic low-rank FP32 within layers to absorb quantization error.

### Lower Priority
7. **StableQAT** (Fourier-based STE surrogate): Replaces STE with a theoretically grounded surrogate derived from discrete Fourier analysis of the rounding operator, providing smooth bounded gradients.

8. **Anti-curriculum for small models**: Falcon-H1-Tiny research reveals injecting high-quality data from the start outperforms traditional curriculum (broad then fine-tune). Small models benefit disproportionately from targeted data mixtures.

## Implications for Next Experiments

1. **The ternary mechanism works -- the regime is wrong.** STE converges, deadzones are healthy, throughput penalty is minimal. The failure is data scarcity + vocabulary mismatch, not ternary fundamentals. Future ternary-from-scratch experiments MUST address data volume.

2. **Warm-start is the pragmatic path.** Cold-start ternary at sub-100M requires solving data scarcity, vocabulary sizing, and regularization simultaneously. The "16-to-1.58" warm-start sidesteps all three issues. This should be tested next.

3. **Weight decay needs special handling for ternary.** The standard recipe (constant weight decay throughout training) is wrong for ternary. Schedule it: use weight decay for the first half, remove it for the second half to let latent weight magnitudes stabilize as confidence scores.

4. **The overfitting finding is a positive signal.** A model that can memorize training data has learned *something*. The 2.48 train loss (below FP32's 3.17) means the ternary weight space is expressive enough for the task. The problem is purely generalization, which data volume and regularization should fix.

5. **Don't test ternary scaling in the data-scarce regime.** Any future ternary-from-scratch experiment needs minimum 10M unique tokens (0.13 tokens/param for 77M model) to produce meaningful PPL ratios. Below this, the data scarcity confound dominates.
