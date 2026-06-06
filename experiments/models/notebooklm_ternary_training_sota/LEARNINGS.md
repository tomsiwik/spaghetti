# LEARNINGS: SOTA Ternary Training Techniques Survey

## Core Finding

The 31% dead weight fraction in ternary networks is NOT an inevitable mathematical constant — it is a consequence of per-tensor mean-based thresholding that varies dramatically by layer (35-62% per TernaryLM). Three classes of solutions exist (smooth surrogates, decoupled thresholds, gradient correction), but NONE have been validated for adapter composition. The survey's most actionable output is Tequila bias fusion (already proven, -6.7% PPL, zero cost), while its most important theoretical contribution is the three-class taxonomy of deadzone solutions.

## Why This Happened

### The alpha-coupling equilibrium is our derivation, not literature consensus

NotebookLM reveals that NO published paper frames the 31% dead weight fraction as a "mathematical equilibrium" or "fixed point." Our MATH.md derives this from erf(alpha/(2*sqrt(2)*sigma)) assuming W ~ N(0, sigma), yielding 31.1% — matching BitNet-2B-4T's empirical 31.3%. But this is our analysis, not a cited result.

The literature frames deadzone trapping differently:
- **Tequila** (arXiv:2509.23800): calls it an STE gradient failure — dead weights get noisy gradients that prevent escape
- **TernaryLM** (arXiv:2602.07374): shows sparsity is NOT uniform — 35% at embedding, 45-55% at early/late layers, 60-62% at middle layers (L5-L9)
- **TWN** (Li et al. 2016): approximates threshold as Delta ~ 0.75/m * sum(|W|) under normal/uniform prior, no fixed-point analysis

This matters because our Gaussian-equilibrium model predicts a single number (31%), but reality is layer-dependent. The reviewer's note about erf symbolic error (erf(1/(2*sqrt(pi))) not erf(1/(2*sqrt(2)))) is a symptom: the framework is our construction, not an established result to cite.

### TernaryLM std-threshold INCREASES initial zeros (38.3%), not decreases

The reviewer identified a critical prediction: for W ~ N(0, sigma), TernaryLM's threshold tau = 0.5*sigma gives P(|w| < tau) = erf(0.5/sqrt(2)) = 0.383. This is HIGHER than BitNet's 0.31. The survey claims TernaryLM helps, but the mathematical prediction says it creates MORE zeros initially, not fewer.

The counterargument (that std-coupling has "better dynamics") is unproven. No paper provides a rigorous stability analysis showing std-coupling converges to fewer zeros during training. TernaryLM's actual empirical benefit may come from the layer-wise adaptive threshold (different tau per layer), not the std-vs-mean choice itself.

### The 42% vs 31% sparsity gap reveals assumption failure

Sparse-BitNet (arXiv:2603.05168) reports 42% natural sparsity in trained ternary weights, vs our predicted 31%. Possible explanations:
1. **Per-channel alpha** (not per-tensor): different granularity changes the equilibrium
2. **Non-Gaussian weight distributions**: after training, weights are not N(0, sigma) — they develop heavy tails and multimodal structure
3. **TernaryLM's own data** shows 45-62% in middle layers — further from 31% than 42%

The Gaussian model is a starting approximation, not a law. Layer-wise analysis is essential.

## Confirming Evidence

- **Tequila bias fusion validated** (our exp_tequila_deadzone_fix, finding S1): -6.7% PPL at zero inference cost. Bias compensation works even when deadzone fraction doesn't change.
- **TernaryBERT** (Zhang et al.): knowledge distillation from FP teacher to ternary student reduces capacity loss — confirms that ternary models have recoverable missing capacity.
- **Hestia smooth surrogate** (arXiv:2601.20745): Lemma 4.2 proves convergence; reviewer confirms gradient is nonzero everywhere at finite tau.
- **FOGZO** (arXiv:2510.23926): zeroth-order correction for STE bias at quantization boundaries.
- **PT2-LLM** (arXiv:2510.03267): activation-aware grid alignment for post-training ternarization.

## Contradicting Evidence

- **TernaryLM CoLA failure**: TernaryLM achieves only 47.23% MCC on CoLA (vs 56.78% FP BERT-base). Fine-grained grammaticality judgments degrade severely with ternary quantization. This is a WARNING for adapter composition: subtle linguistic features may be lost regardless of training improvements.
- **Tequila's own "Minima Reactivation" failed before bias**: The paper tested in-place reactivation (not bias compensation) and found "only marginal accuracy gains" because STE gradients are still noisy. Bias is a workaround, not a fix.
- **Hestia requires Hessian calibration**: Uniform temperature schedule (no Hutch++) fails because LLM tensors are deeply heterogeneous. The "simplified Hestia" we recommended may not work without per-tensor sensitivity — the expensive part is the essential part.
- **Alpha-Blending** (non-STE approach): replaces STE entirely with a learned alpha interpolation between FP and quantized weights. Suggests STE-based methods may be inherently limited.
- **Hardware execution penalty**: ternary networks are currently SLOWER than 4-bit on standard GPUs due to missing hardware support. On MLX/Apple Silicon this is less relevant (we use addition-only serving), but worth noting.

## Alternative Approaches (Paper-Backed)

1. **TernaryBERT knowledge distillation**: Teacher-student distillation for ternary models. Could be applied to our adapter training — distill from FP16 adapter into ternary adapter. Proven to reduce capacity loss.
2. **SCA (Sparsity Control with Adaptive regularization)**: Adds explicit regularizer to control zero fraction. Gives fine-grained sparsity-accuracy tradeoff control, which our framework lacks.
3. **Alpha-Blending**: Non-STE training via learned alpha interpolation. Avoids STE bias entirely. Could replace STE in our BitLinear.
4. **Hyperspherical Quantization (HQ)**: Constrains weights to unit sphere before quantizing. Addresses STE bias via angular discrepancy penalties.
5. **ProxQuant/ADMM**: Iterative optimization that alternates between continuous optimization and projection onto ternary set. Mathematically cleaner than STE.

## Implications for Next Experiments

1. **Tequila bias fusion is READY for default integration.** Already validated. Zero risk. Should be standard in all adapter training.

2. **TernaryLM needs CAREFUL testing.** The std-threshold predicts MORE zeros (38.3%) not fewer. The benefit may be layer-adaptive thresholds, not the std formula itself. Test: measure per-layer zero fraction with both methods.

3. **Simplified Hestia may not work.** The Hestia paper says uniform temperature fails due to LLM heterogeneity. Per-tensor Hutch++ is the key innovation. At d=2560 with 192 layers, this adds ~576 HVPs per step — need to measure if this fits in M5 Pro memory budget.

4. **Composition impact is the unknown unknown.** No published paper tests how deadzone reduction affects adapter composition. Our architecture (ternary base + ternary LoRA) is unique — improved base training could help adapters (more capacity to perturb) or hurt them (different zero patterns create new interference).

5. **Knowledge distillation for adapters** is an unexplored angle. TernaryBERT-style distillation (FP16 adapter → ternary adapter) could yield better ternary adapters than direct STE training.

## Recommended Follow-Up

**No new experiment recommended at this time.** Rationale:

1. Tequila bias is already proven and ready for integration — this is an engineering task, not a research experiment.
2. TernaryLM threshold change needs theoretical grounding before testing (reviewer correctly identified the dynamics claim as unsupported).
3. The deployment track (exp_generation_quality_test, exp_task_accuracy_real_benchmarks) is higher priority than training method improvements.
4. If a follow-up is warranted later, it should be: **"TernaryLM + Tequila combined vs baseline on adapter composition quality"** — motivated by this survey's orthogonality argument, but only after deployment track unblocks.

## New References

- TernaryBERT (Zhang et al.): knowledge distillation for ternary quantization
- Alpha-Blending: non-STE quantization-aware training via learned interpolation
- SCA (Sparsity Control Adaptive): explicit sparsity regularization for ternary networks
- ProxQuant: proximal gradient methods for discrete quantization
