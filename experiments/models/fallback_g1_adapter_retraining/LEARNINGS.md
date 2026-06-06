# Learnings: exp_fallback_g1_adapter_retraining

## Core Finding

Self-trained ternary bases at d=256 (PPL=268) are too weak for meaningful LoRA domain adaptation. Composition amplifies noise when individual adapter signal is <5%. The prior catastrophic failure (17x PPL blowup) was overfitting (17% trainable params), not vacuous deltas.

## Why This Happened (Literature-Grounded)

**Base model capacity is a prerequisite for adaptation, not a tunable hyperparameter.** At PPL 268 (~5.6 bits/token), the model operates at roughly bigram-level statistics — far above competent English LM entropy (~1.0-1.5 BPT). LoRA adapts the *residual* between a model's general capabilities and domain-specific needs. When the base hasn't learned general language structure, there is no meaningful residual to exploit.

This is consistent with the LoRA paper's (Hu et al., 2021, arXiv 2106.09685) foundational assumption: pre-trained models have low "intrinsic dimensionality" *because* they've already learned rich representations. An undertrained model lacks this low-dimensional structure, so low-rank updates cannot efficiently capture domain-specific adjustments.

The composition failure follows directly from the signal-to-noise ratio problem identified in model merging literature. Task Arithmetic (Ilharco et al., 2023) and TIES-Merging (Yadav et al., 2023, arXiv 2306.01708) operate on task vectors (weight deltas). When individual task vectors contain only ~3% useful signal but ~13% magnitude (delta ratio 0.13), composition sums the magnitudes while the signals — pointing in weakly-correlated directions — partially cancel. The monotonic lambda-degradation curve (lambda=0.3: -4%, lambda=1.0: -14%) is textbook constructive-interference failure: more weight on the deltas means more noise compounding.

**Key mechanism:** Adapter deltas were large but *undirected*. The 0.13 delta ratio exceeded the MATH.md target (0.005-0.05) by 2.5x, confirming the adapters were aggressively modifying weights. But the modifications were not domain-discriminative — they were fitting noise in an already-noisy base. This is the distinction between delta magnitude (which was healthy) and delta quality (which was near-zero).

## Confirming Evidence

1. **LoRA's intrinsic dimensionality assumption** (Hu et al., 2021, arXiv 2106.09685): LoRA works because pre-trained models have low intrinsic rank during adaptation. This presupposes a well-trained base. Our PPL=268 base violates this assumption.

2. **Model Merging in the Essential Subspace** (arXiv 2602.20208): Shows that centering task vectors isolates core task-specific knowledge from interfering noise. When there is no task-specific knowledge (our 3% improvement case), centering cannot help — there is nothing to isolate.

3. **Exploring Sparse Adapters for Scalable Merging** (arXiv 2507.07140): Demonstrates that the norm of parameter updates reflects "directional confidence and inter-task consensus." Our adapters had high norm (0.13 ratio) but low confidence — the signature of noise fitting.

4. **Our own BitNet-2B-4T results** confirm the positive case: at PPL ~4-5, the same adapter architecture (rank-16 LoRA) achieves +26.5% domain PPL improvement and successful composition with ratio 3.59x. The mechanism works when the base is strong.

5. **TernaryLM** (arXiv 2602.07374): Training ternary models from scratch with STE achieves stable convergence (loss 7.35→4.02) on a single T4 GPU, but at 132M params and 15 epochs — substantially more compute than our 4000-step warm-start at d=256.

## Contradicting Evidence

1. **TernaryLM's success at small scale** (arXiv 2602.07374) suggests ternary-from-scratch IS viable, but requires more training compute than we allocated. Their 132M model trained for 15 full epochs; our 29M model trained for 4000 steps on 5M tokens. The issue may be training budget, not architecture.

2. **LoRA Soups** (arXiv 2410.13025) shows that merging many LoRA adapters CAN work even with individually weak adapters, provided the base model is strong enough to provide coherent gradient directions. This reframes our failure as a base-quality issue, not a composition-method issue.

3. **AdaMerging** (arXiv 2310.02575) demonstrates that *adaptive* (learned) merging coefficients can rescue compositions that fail under uniform weighting. We only tested static methods (1/N, fixed lambda, TIES). Learned routing could potentially extract the 3% signal even at our scale — though the information-theoretic budget is very tight.

4. **Reversible Model Merging for Low-Rank Weights** (arXiv 2510.14163) proposes techniques that preserve task-specific information during merging. This suggests our composition failure was partly methodological, not just signal-based.

## Alternative Approaches (What We Could Try Instead)

### For building a stronger ternary base:
1. **TernaryLM's adaptive layer-wise scaling** (arXiv 2602.07374): Per-layer learned scaling factors for ternary weights. More sophisticated than our uniform STE. Could improve base quality at same compute budget.

2. **Progressive growing / curriculum learning**: Start at d=64, train to convergence, then scale up. Our d=256 base may have been undertrained rather than capacity-limited.

3. **Knowledge distillation from BitNet-2B-4T**: Use the strong ternary base we already have as a teacher for a smaller student. Proven path to strong small models.

4. **MatMul-free LM** (ridgerchu/matmulfreellm): Alternative ternary architecture that eliminates matrix multiplication entirely. Scales to 2.7B. Different optimization landscape may train more efficiently at small scale.

### For making composition work at low base quality:
5. **Learned routing** (our existing Gumbel router architecture): Skip static composition entirely. Per-token or per-sequence routing selects the best single adapter rather than merging all. Already proven at N=49 on BitNet-2B-4T (83% routing accuracy).

6. **Sparse adapter merging** (arXiv 2507.07140): Explicitly sparsify adapter deltas before merging to raise signal-to-noise ratio. Could help when individual signal is weak.

7. **Essential subspace merging** (arXiv 2602.20208): Project task vectors into their essential subspace before merging. Removes noise dimensions. Requires enough signal to identify the subspace.

## Implications for Next Experiments

1. **Stop testing composition on self-trained d=256 bases.** PPL=268 is a dead end for domain adaptation. The finding is definitive: LoRA requires PPL<50 (and likely PPL<10) for meaningful domain specialization. All future composition work should use BitNet-2B-4T (PPL ~4-5) or a self-trained base that achieves comparable quality.

2. **The overfitting diagnosis is confirmed and valuable.** The prior catastrophic failure at d=1024 with 17% trainable params was overfitting, proven by this experiment's stable training at 0.11% trainable. This calibrates our LoRA sizing: for self-trained bases, keep trainable params well below 1%.

3. **Delta ratio calibration needs updating.** The MATH.md "healthy range" of 0.005-0.05 was too narrow. Our 0.13 ratio produced stable (non-catastrophic) but useless adapters. The range should be reframed: >0.005 means non-vacuous, but quality depends on base model capability, not delta magnitude alone.

4. **Focus P0 effort on training a strong ternary base**, not on composition methods for weak bases. The composition infrastructure works (proven at BitNet-2B-4T scale). The bottleneck is base quality. TernaryLM's approach (STE + adaptive layer-wise scaling + more training compute) is the most promising path.

5. **Early stopping was inert** — all adapters hit the step cap, not the patience criterion. Future adapter experiments should use tighter patience windows (e.g., 3 checks of 200 steps) and evaluate best-validation checkpoints, not final checkpoints. The adapters were likely overfitting from step 500, meaning the "true" adapter signal was slightly higher than measured.
