# Learnings: exp_warmstart_scale_validation

## Core Finding

Warm-start ternary QAT scales from d=512 (64M params) to d=1024 (204M params) with
comparable PPL ratios (1.037x vs 1.046x), and LoRA adapters train stably on self-trained
ternary bases (+7% domain improvement). However, 1/N weight averaging of adapters with
small deltas is vacuous — averaging cancels the effects, proving safety but not utility.

## Why This Happened (Literature-Grounded)

### The PPL Ratio Improvement at Scale (1.046x → 1.037x)

BitNet b1.58 (arXiv 2402.17764) demonstrates that ternary QAT approaches FP16 parity
as model scale increases. At 3B parameters, BitNet achieves PPL of 9.91 vs LLaMA's 10.04
(ratio ~0.987). Our 1.037x at 204M is consistent with this scaling trend — the ternary
penalty shrinks with model size because larger models have more redundant parameters,
and ternary quantization acts as a regularizer (observed in OLMo 1B by "1.58 Bits Enough,"
arXiv 2411.05882, where ternary validation loss was *superior* to FP16 due to delayed
overfitting).

### Why Warm-Start Works

The warm-start mechanism succeeds for three specific reasons identified in the literature:

1. **STE accuracy at decision boundaries.** The Straight-Through Estimator approximation
   is highly inaccurate when real-valued weights are near 0 (the sign function decision
   boundary). FP16 pretraining moves weights away from 0, making STE gradients more
   stable during the ternary phase.

2. **Optimizer state retention.** Retaining AdamW momentum and variance across the
   FP16→ternary transition minimizes the switch spike. Our +0.735 spike is consistent
   with literature reports of manageable transient disruption when optimizer state is
   preserved.

3. **Pre-trained knowledge preservation.** Ternary from scratch destroys the factual
   knowledge encoded in FFN weights during early training. Warm-start buffers this by
   establishing stable representations before quantization.

### Why 1/N Averaging Produced Vacuous Composition

The composition ratio of 1.0001 is a direct consequence of small adapter deltas being
averaged. With LoRA B initialized to zero, rank 16, lr=1e-4, and only 1000 training steps,
the effective weight perturbation ||B·A|| is tiny relative to the base weights. When 3
such perturbations are averaged at 1/3 weight each, the result is indistinguishable from
zero.

This is well-documented in the model merging literature:
- **Task Arithmetic** (Ilharco et al., 2023) shows that adapter deltas must have
  sufficient magnitude for linear combination to produce measurable effects. The scaling
  coefficient λ must be calibrated per-task.
- **TIES-Merging** (Yadav et al., 2023, arXiv 2306.01708) explicitly addresses this by
  trimming low-magnitude parameters and keeping only the top-k% most significant changes.
- **DARE** (Yu et al., 2024, arXiv 2311.03099) randomly drops delta weights and rescales
  survivors by 1/(1-p), which amplifies the remaining signal.

The fundamental issue: **1/N averaging is a magnitude-reducing operation**. When N adapters
each contribute δ/N to the final weights, the signal approaches zero as N grows — unless
δ is large enough to survive the division.

## Confirming Evidence

1. **BitNet b1.58 scaling** (arXiv 2402.17764): PPL ratio improves with scale, consistent
   with our 1.037x at 204M being better than 1.046x at 64M.

2. **"1.58 Bits Enough"** (arXiv 2411.05882): Ternary regularization effect confirmed at
   OLMo 1B scale. Our observation that warm-start PPL is near-FP16 is consistent.

3. **Composition Catastrophe** (observed in our prior logit-ensemble work): Weight-space
   orthogonality does not guarantee function-space non-interference. The OSRM paper
   (arXiv 2505.22934) showed weight-space ≠ data-space orthogonality.

4. **LoRA on quantized bases** (QLoRA, Dettmers et al., arXiv 2305.14314): LoRA adapters
   are compatible with quantized base models. Our result extends this to self-trained
   ternary (not just post-hoc quantized).

## Contradicting Evidence

1. **Task-Aware LoRA Composition** (vector database retrieval routing): Achieved 70.95%
   on PIQA via Linear merging — but used relevance-weighted averaging, not equal-weight.
   The key condition: adapters must be "geometrically aligned in task space." Our 3 domain
   adapters (science/history/technology) may not meet this condition.

2. **At 3B+ scale, ternary matches FP16 exactly** (PPL ratio ~1.0). Our 1.037x at 204M
   suggests we're still in the regime where scale hasn't fully compensated for the
   quantization penalty. More tokens (we used only 54 tokens/param vs recommended 20-100x)
   would likely close this gap further.

## Alternative Approaches (What We Could Try Instead)

### For Composition (highest priority — K3 was vacuous)

1. **Task Arithmetic with scaling λ.** Instead of 1/N averaging, use λ·δ_i with λ > 1.0
   to amplify adapter effects. Typical λ range: 0.3-1.5 per task vector.

2. **TIES-Merging.** Trim bottom 80% of delta weights, elect majority sign, merge only
   top-20% most significant changes. This would prevent small deltas from being canceled.

3. **DARE + averaging.** Drop 90% of delta weights randomly, rescale survivors by 10x.
   The surviving weights retain full signal strength.

4. **Longer adapter training.** 1000 steps at lr=1e-4 with zero-init B produces tiny
   deltas. Try: 5000+ steps, lr=3e-4, or PiSSA initialization (arXiv 2404.02948) which
   uses principal SVD components instead of zero-init for B, producing larger initial
   gradients.

5. **Logit-level ensemble.** Run each adapter separately, combine output logits with
   learned or fixed mixing weights. Avoids parameter-space interference entirely.

6. **Router-based selection (MoLoRA).** Per-token routing to individual adapters avoids
   merging entirely. MoLoRA (arXiv 2603.15965) showed Qwen3-1.7B + 4 adapters > 8B.

### For Base Model Quality

7. **More tokens.** 54 tokens/param is severely undertrained. Even 200 tokens/param
   (40M tokens for our 204M model) would likely push PPL below 100.

8. **Tequila Minima Reactivation** (arXiv 2509.23800) to avoid deadzone trapping in
   ternary weights — addresses the 31.6% zero fraction we observed.

## Implications for Next Experiments

1. **Composition is the critical gap.** The warm-start mechanism is validated at d=1024.
   LoRA on self-trained ternary works. But composition was never meaningfully tested.
   The next experiment MUST produce adapters with measurably large deltas before
   attempting composition again.

2. **Increase adapter delta magnitude.** Options: longer training (5000+ steps), higher
   LR (3e-4), PiSSA initialization, or simply measuring ||B·A||/||W_base|| as a
   pre-composition diagnostic.

3. **Use smarter composition methods.** 1/N averaging is the weakest possible baseline.
   TIES or Task Arithmetic with λ > 1 should be the default going forward.

4. **The PPL ratio trend is encouraging.** 1.046x → 1.037x from d=512 to d=1024
   suggests that at d=2048+ (which our M5 Pro can handle), we may approach near-parity.
   This is consistent with BitNet scaling results.

5. **Freeze-then-attach ordering is load-bearing.** The adapter freeze bug (34.6M vs 1M
   trainable params) is a critical implementation pattern: always attach LoRA modules
   BEFORE calling model.freeze(), then selectively unfreeze lora_A/lora_B.
