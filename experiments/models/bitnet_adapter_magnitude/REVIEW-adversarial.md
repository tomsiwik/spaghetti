# Peer Review: BitNet Adapter Magnitude Analysis

## NotebookLM Findings

Skipped -- the experiment is a straightforward diagnostic with clear kill criteria. The math is simple enough (Frobenius norms, variance, CV) that manual verification suffices.

## Mathematical Soundness

**Norms and variance computations: Correct.** The Frobenius norm of the delta, population variance across 5 adapters, CV, and max/min ratio are all implemented correctly and match the MATH.md formulas.

**Composition efficiency formula: Minor concern.** MATH.md defines the orthogonal composition norm as (1/N)*sqrt(sum ||delta_i||^2), which is correct under exact orthogonality. The "efficiency" eta > 1 indicating constructive interference is properly interpreted. No issues here.

**The "activation compression" explanation has a gap.** The paper claims ternary base produces 2x smaller activations, which reduces logit-scale mismatch during composition. The causal chain is:

1. Ternary weights are smaller in Frobenius norm (by construction -- quantization loses information)
2. Smaller weights produce smaller activations (linear algebra, correct)
3. Smaller activations mean each adapter's absolute contribution is smaller
4. Therefore composition is more stable

Step 1 deserves scrutiny. Looking at the quantization code (`bitnet_composition_stability.py`, line 183), ternary weights are stored as `W_t * alpha` where `alpha = mean(|W|)`. This means the ternary weight matrix has entries in {-alpha, 0, alpha} rather than {-1, 0, 1}. The quantization introduces two effects simultaneously:

- **Magnitude reduction**: Ternary weights have smaller Frobenius norm than FP16 (by losing fine-grained weight values)
- **Sparsity**: Weights near zero get snapped to zero, creating sparsity

The paper does not disentangle these two effects. A control experiment with FP16 weights scaled to match ternary Frobenius norm (without quantization) would distinguish magnitude-reduction from quantization-induced-sparsity. The paper acknowledges this in Limitations point 4 ("no causal analysis") but the "activation compression" framing implies more causal certainty than the evidence supports.

**The explanation for WHY norm variance increases is incomplete.** MATH.md section "Why Magnitude Bounding Fails" gives three reasons, of which reason 3 ("Ternary weights create sparser gradient signals ... which actually increases variance") is hand-waving. The gradients flow through the ternary weights during LoRA training, but the key question is whether the variance increase is a fundamental property of discrete weight spaces or an artifact of post-hoc quantization at d=64. The paper does not attempt to formalize this.

## Novelty Assessment

**This is a diagnostic experiment, not a novel contribution.** It tests and kills a specific hypothesis within the project's hypothesis graph. The finding that post-hoc quantized weights produce more variable adapter norms is not surprising -- the reduced rank of the ternary weight space means different domains "see" a more constrained base, and the adapter has to compensate differently depending on how well the ternary approximation serves each domain's data distribution.

**The activation compression observation is genuinely useful for the project** but is not novel in the broader literature. It is well-known that quantized models have smaller activation ranges (this is why quantization-aware training uses activation-aware scaling, e.g., SmoothQuant, AWQ).

**No prior art conflict.** This is internal diagnostics, not a publishable claim.

## Experimental Design

**Strengths:**

1. Three seeds with consistent results across all three -- the kill is robust.
2. Six measurement facets provide a thorough picture.
3. Code reuses the established bitnet_composition_stability infrastructure, ensuring consistency.
4. Kill criteria from HYPOTHESES.yml (K1: norm variance, K2: max/min ratio) are exactly what the code measures.

**Weaknesses:**

1. **The "activation compression" claim is correlational, not causal.** The paper acknowledges this (Limitation 4) but then proceeds to treat it as established mechanism throughout the "What This Means for SOLE" section. The language should be more hedged.

2. **Post-hoc quantization vs trained BitNet.** This is acknowledged in Limitation 2 but it is more severe than presented. Real BitNet b1.58 models learn with quantization-aware training, meaning the weight distribution is optimized for the ternary constraint. Post-hoc quantization of a trained FP16 model introduces quantization error that would not exist in a properly trained ternary model. The activation compression ratio (2x) could be entirely an artifact of quantization error, not a fundamental property of ternary weight spaces.

3. **N=5 adapters is very small for variance estimation.** Population variance with N=5 has high sampling uncertainty. The 2.6x ratio (3.37 vs 1.29) across 3 seeds is convincing for the kill, but the exact magnitude should not be over-interpreted.

4. **The per-layer analysis (FFN smaller, attention QKV larger) is interesting but not validated.** The claim that "ternary base forces adapters to redistribute learning from FFN to attention" is a post-hoc narrative that could be confounded by the specific domains, the small scale, or the quantization error pattern.

5. **Missing control: FP16 base with activation scaling.** The paper suggests "activation normalization on FP16 base might achieve the same effect" but does not test it. This would be the cheapest falsification of the activation compression hypothesis and should have been included.

## Hypothesis Graph Consistency

The experiment matches `exp_bitnet_adapter_magnitude_analysis` in HYPOTHESES.yml. Kill criteria K1 (norm variance) and K2 (max/min ratio) are exactly tested. K1 fails definitively (0/3 seeds), K2 passes (3/3 seeds). The kill on K1 is correctly applied.

The experiment does not block anything (`blocks: []`), so the kill has no downstream impact on the hypothesis graph. This is appropriate for a diagnostic.

The "activation compression" finding is noted in the evidence but correctly does not spawn a new proven node -- it remains an observation that needs causal validation.

## Macro-Scale Risks (advisory)

1. **The 2x activation compression ratio is almost certainly an artifact of post-hoc quantization.** Real BitNet b1.58 models (trained with QAT) will have different activation magnitudes. Do not plan macro experiments assuming this ratio transfers.

2. **If the activation compression mechanism is real, it can be replicated on FP16 with a scalar multiplier** -- test this before committing to ternary base as the solution to composition stability.

3. **The finding that adapter norm variance increases on ternary base is concerning for Track A (BitNet-SOLE).** If ternary experts have more variable magnitudes, 1/N composition may be less stable on ternary than on FP16, despite the activation compression benefit. These two effects (higher norm variance, lower activation scale) partially cancel, and which dominates at scale is unknown.

## Verdict

**PROCEED**

The kill of the magnitude bounding hypothesis is clean, well-evidenced, and correctly applied. Three seeds show consistent results. The kill criteria match the hypothesis graph exactly. The experiment does what it claims.

Two non-blocking notes:

1. The "activation compression" mechanism is presented with more confidence than the correlational evidence warrants. The PAPER.md language in the "What This Means for SOLE" section should use "may" and "suggests" rather than declarative statements like "It comes from activation compression" (line 95). This is a writing issue, not a scientific one -- the Limitations section correctly caveats it.

2. The post-hoc quantization confound (Limitation 2) is the most significant threat to the activation compression finding transferring to real BitNet models. Future work should note this prominently when using this result to motivate Track A decisions.

Neither issue is blocking. The primary result (magnitude bounding hypothesis killed) is sound.
