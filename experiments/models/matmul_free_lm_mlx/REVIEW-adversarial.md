# Peer Review: MatMul-Free LM (HGRNBit) on MLX

## NotebookLM Findings

Skipped -- manual deep review conducted instead, as the experiment scope is contained enough for direct analysis.

## Mathematical Soundness

### BitLinear (Section 1 of MATH.md) -- CORRECT

The STE formulation `W_ste = W + sg(W_q - W)` is standard and correctly implemented. The forward pass uses quantized weights, the backward pass flows through the latent FP32 weights. The Extra RMSNorm placement matches prior findings (warmstart experiment, arxiv 2505.08823). Implementation in code (lines 85-102) matches the math.

### HGRN Token Mixer -- SIMPLIFIED BUT HONEST

The MATH.md describes the full HGRN2 (with state expansion, outer products, rank-r decomposition) but then explicitly states "we use the simplified HGRN form." The implementation is actually a basic gated recurrence:

```
h_t = g_t * h_{t-1} + (1 - g_t) * i_t
```

This is closer to a minimal GRU variant than full HGRN2. The paper (Qin et al.) uses a more complex formulation with state expansion that enables richer long-range dependencies. The simplification is acknowledged in Limitation 4, so no deduction here, but the claim of "porting HGRN2" is an overstatement -- this is a gated linear recurrence inspired by HGRN, not HGRN2 proper.

**Concern: The `(1 - g_t)` complement gating.** The original HGRN2 uses `h_t = f_t * h_{t-1} + v_t` where the input term is NOT gated by `(1-f)`. This complement form forces a conservation constraint (information must be either retained or replaced). The actual HGRN2 decouples retention from injection. At micro scale this likely does not matter, but it means the composition-on-HGRN claim is really composition-on-GRU-variant, not composition-on-HGRN2. This distinction matters for macro scaling.

### LoRA Composition Math (Section 5) -- CORRECT BUT UNDERTESTED

The argument that element-wise gating does not create cross-adapter coupling is sound in principle: each adapter's delta `B_i @ A_i` modifies individual weight matrices, and the recurrence `g * h + (1-g) * i` applies element-wise operations on hidden states. There is no quadratic interaction between weight perturbations from different adapters (unlike attention where Q/K interaction is bilinear in the weight perturbations).

However, the argument glosses over a subtle point: the forget gate `g_t` is itself computed via `sigmoid(W_g @ x_t + lower_bound)`, where `W_g` is a BitLinear whose weights are modified by composition. A composed perturbation to `W_g` changes the gating trajectory, which then modulates how all subsequent hidden states propagate. This is a **multiplicative interaction through time** -- not cross-adapter in the same step, but cross-adapter through the recurrence dynamics. At T=32 this effect is negligible (the geometric series `prod(g_t)` decays fast). At T=2048+ with learned high-retention gates, adapter interference could amplify through the recurrence. The paper does not analyze this.

### Parameter Count -- NEEDS CORRECTION

MATH.md claims ~6.3M parameters. The results.json shows 6,333,952. Let me verify:
- Embedding: 27 * 256 = 6,912 (not 28 * 256 as in MATH.md -- vocab_size mismatch)
- Pos embed: 32 * 256 = 8,192
- Per layer: 4 BitLinear (256x256) = 4 * 65,536 = 262,144 (token mixer)
- Per layer: 3 BitLinear for GLU = 2 * 256*1024 + 1024*256 = 786,432 (channel mixer)
- Per layer total: 1,048,576
- 6 layers: 6,291,456
- RMSNorms: ~7 per layer (4 in BitLinear of token mixer + 3 in GLU BitLinear) * 256 + 2 block norms * 256 = (42+12) * 256 = 13,824
- LM head (BitLinear): 256 * 27 + 27 (RMSNorm) = 6,939
- Total: ~6,327,323

Close enough to 6,333,952 -- likely off by counting of RMSNorm parameters. The MATH.md table uses vocab_size=28 but actual code uses 27. Minor discrepancy, not a concern.

### Grassmannian A-matrix Generation -- CORRECT BUT BASIC

The QR-based generation (lines 327-342) is correct for producing orthonormal columns. This is the simplest valid approach, consistent with prior experiments. The claim that this produces "Grassmannian" bases is technically correct (any orthonormal frame is a point on the Grassmannian) but the Alternating Projection method mentioned in VISION.md is not used here -- this is plain QR. No practical difference at N=5 where n_adapters * rank = 40 << dim = 256.

## Novelty Assessment

### Prior Art

The key claim is: **Grassmannian LoRA composition is architecture-agnostic (works on non-Transformer backbones)**. This is a novel and useful finding within this project's context. No published work tests LoRA adapter composition on gated linear recurrence models specifically.

However, the broader claim that LoRA works on non-Transformer architectures is well-established:
- LoRA has been applied to diffusion models (Stable Diffusion), which are not autoregressive Transformers
- LoRA has been applied to state-space models (Mamba) by multiple groups
- The original LoRA paper (Hu et al., 2021) makes no architectural assumption beyond having linear layers

The specific novelty is the **combination**: ternary weights + gated recurrence + Grassmannian orthogonal composition. This combination has not been published.

### Delta Over Existing Work

The ternary_base_from_scratch_mlx experiment already proved composition on ternary weights with Grassmannian A-matrices (ratio 1.022x, |cos| 2.5e-7). This experiment's delta is replacing attention with HGRN recurrence. The composition ratio (1.029x vs 1.022x) is marginally worse but well within noise. The experiment successfully shows that swapping the token mixing mechanism does not break composition.

## Experimental Design

### What works well

1. **Fair baseline**: FP32 Transformer trained with identical hyperparameters (same d_model, n_layers, n_heads, steps, batch_size). This is the right comparison.
2. **Kill criteria are well-calibrated**: K1 (loss > 2.0), K2 (LoRA incompatible), K3 (composition ratio > 1.5) are all reasonable thresholds with headroom.
3. **Phase-scoped design**: Each phase creates fresh models, loads weights explicitly, cleans up. This prevents state leakage.
4. **Composition is tested correctly**: Individual adapter PPL measured by merging single adapter at scale=1.0, composed PPL by merging all at scale=1/N. This is the standard protocol from prior experiments.

### Concerns

**C1: Parameter count mismatch is not controlled.** HGRNBit has 6.3M params, Transformer has 4.7M (33% more). The paper acknowledges this but claims the extra params are ternary so storage is comparable. This is true for storage but not for representational capacity during training -- training uses FP32 latent weights via STE, so the HGRNBit model has 33% more FP32 parameters during training. The PPL parity (1.60 vs 1.60) could be partially explained by the extra capacity. A parameter-matched comparison would use fewer layers or smaller d_model for HGRNBit.

**C2: Learning rate differs between models.** The FP32 Transformer uses lr=3e-4 (line 494 comment says "Use higher LR for STE training") while HGRNBit uses lr=1e-3. This is justified by prior STE training experience, but it means the Transformer may not be at its optimum. However, since the experiment is testing kill criteria (can HGRNBit train at all? can LoRA compose?), not claiming HGRNBit is better, this is acceptable.

**C3: The orthogonality measurement is on B-matrices only.** The cosine similarity is computed on flattened B-matrix vectors (lines 866-872). This is the correct measurement for the Grassmannian skeleton claim: if A matrices are orthogonal and B matrices are approximately orthogonal, then deltas (B@A) are guaranteed near-orthogonal. The |cos|=0.0076 is slightly higher than the ternary_base_from_scratch result (2.5e-7) but still excellent. The difference is likely due to different random seeds or the recurrence creating slight correlations in gradient updates.

**C4: The "matmul-free" claim is misleading at the implementation level.** Every BitLinear still calls `x @ w_ste.T` which is a dense matmul on Metal. The "matmul-free" property is architectural (at inference with quantized weights, only additions/subtractions are needed) but requires custom kernels to realize. The PAPER.md Section S3 acknowledges this clearly, so it is not a hidden assumption. But the experiment title is somewhat misleading -- this tests "HGRN backbone + ternary weights + composition" rather than actual matmul-free execution.

**C5: Single seed.** All results are from a single random seed. The composition ratio of 1.029x is within the range seen in prior experiments (1.02-1.1x) but we cannot assess variance. Prior work showed CV=0.5% across 3 seeds on the Transformer, so this is likely stable.

**C6: Trivially separable domains.** The paper acknowledges this (Limitation 5). Alphabetic splits of names are easily separable. The composition test is lightweight -- it mostly tests that adapters do not catastrophically interfere, not that they constructively compose on overlapping domains.

## Hypothesis Graph Consistency

The experiment references kill criteria K1 (id=189), K2 (id=190), K3 (id=191) in the script docstring, but these IDs are not found in HYPOTHESES.yml. This is a bookkeeping gap -- the experiment should be registered as a hypothesis node. Not blocking for the science, but needs cleanup.

The experiment is listed in VISION.md Track A ("Port MatMul-free LM to MLX -- ternary + no matmul (exp_matmul_free_lm_mlx)"), confirming alignment with the research plan.

## Macro-Scale Risks (advisory)

1. **Recurrence amplification of composition interference.** As analyzed above, the forget gate is itself a composed weight matrix. At long sequences with high-retention gates (g near 1), perturbations to the gate weights accumulate geometrically. Test at T=512+ with N>10 adapters.

2. **Simplified HGRN vs full HGRN2.** The state expansion in HGRN2 uses outer products that introduce bilinear interactions between key and value projections. If both projections have LoRA adapters, composition could create cross-adapter terms via the outer product. This is not testable at micro scale with the simplified model.

3. **Speed is currently a dealbreaker.** 4.6x slower with no path to speedup without custom Metal kernels for the recurrence or ternary accumulation. The MLX ecosystem does not currently support custom Metal kernels from Python. This could block the entire Track A direction on Apple Silicon.

4. **Parameter efficiency.** At larger scale, 33% more parameters for the same quality is significant. If the quality gap widens (recurrent models historically underperform attention at scale), the parameter overhead becomes harder to justify.

## Verdict

**PROCEED**

The experiment is well-designed within the micro-experiment contract and answers its core question: Grassmannian LoRA composition works on non-Transformer backbones. The math is sound (with the noted caveat about simplified HGRN vs HGRN2). The kill criteria are clearly defined and the results are unambiguous (K1-K3 all pass with comfortable margin). The limitations are honestly reported.

The composition ratio of 1.029x on HGRN confirms that the Grassmannian skeleton is a property of the A-matrix construction, not the base architecture. This is a meaningful finding that justifies marking this hypothesis as SUPPORTED.

The S3 FAIL (4.6x slower) is correctly identified as an engineering limitation, not a mechanism failure. However, this should inform prioritization: the matmul-free direction requires custom kernels to deliver value on Apple Silicon, which is a substantial engineering investment with uncertain payoff.

**Minor fixes recommended (not blocking):**
1. Register the experiment in HYPOTHESES.yml with proper node IDs
2. Correct vocab_size in MATH.md table (27, not 28)
3. Note explicitly in PAPER.md that the implementation is "HGRN-inspired simplified gated recurrence" rather than "HGRN2" to avoid overstatement
4. Add a note about the gate-composition amplification risk for T >> 32 to the "What Would Kill This" section
