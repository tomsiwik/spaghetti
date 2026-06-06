# Peer Review: Minimum Viable Base Dimension (Revision)

## Context

This is a second review. The first review issued REVISE with 4 required fixes:
1. K1 kill criteria changed to "NOT TESTABLE" (geometry != PPL)
2. N_max analytical estimates qualified as extrapolations
3. Random baseline comparison added
4. "Phase transition" language replaced with "saturation"

All four fixes have been applied. Two non-blocking suggestions (cite prior work, clarify two code files) were also addressed.

## NotebookLM Findings

Skipped -- authentication not configured. Review conducted via direct file analysis.

## Mathematical Soundness

### Core theorem: |cos| ~ 1/sqrt(D_flat) -- HOLDS

The derivation is correct and well-supported:

1. **CLT argument for numerator concentration.** For independent sub-vectors across M*L modules, the dot product is a sum of M*L independent mean-zero terms. Standard deviation of the sum is O(1/sqrt(D_flat)). Correct.

2. **Denominator concentration.** For D_flat >> 1, the norm of a random vector concentrates tightly around its expectation. The ratio gives |cos| ~ 1/sqrt(D_flat). Correct.

3. **Empirical confirmation.** beta = -0.506 (R^2 = 0.997) for |cos| vs D_flat, matching the predicted -0.5. beta = -1.049 (R^2 = 0.995) for |cos| vs d, consistent with D_flat ~ d^2. Both are strong fits across 5 orders of magnitude in D_flat.

4. **Random baseline validates the mechanism.** LoRA-to-random ratio ranges 0.925 to 1.128 (mean ~1.0). This confirms the orthogonality guarantee comes from dimensionality alone, not adapter structure. This is the strongest result in the paper.

### D_flat computation -- VERIFIED

Cross-checked the code's `get_module_shapes()` against Qwen2.5 architecture specs. The GQA handling (d_kv = n_kv_heads * head_dim) is correct. D_flat values in the paper match the code output.

### sqrt(r/d) comparison -- NOW PROPERLY QUALIFIED

The MATH.md (lines 142-146) now explicitly states: "This comparison is misleading... the relevant bound is sqrt(r/D_flat), not sqrt(r/d). The large ratio is an artifact of the dimensional mismatch, not evidence of beating theory." This fully addresses the prior review's concern.

### N_max estimation -- NOW PROPERLY QUALIFIED

The paper (Experiment 6) honestly reports that validation is impossible at d=256 because the true N_max exceeds all testable values. The 100K claim is retracted. The analytical formula is flagged as unvalidated. The correct claim is stated: "N_max exceeds all empirically testable values at d>=256."

### Saturation point analysis -- APPROPRIATELY CAVEATED

BIC improvement of 3.84 for cosine is acknowledged as "marginal with only 8 data points." The slope change of 0.021 is described as "not practically meaningful." SR/ERR breakpoints are correctly identified as ceiling effects. The language is now appropriate.

### One remaining mathematical note (non-blocking)

The per-module Gram accumulation (code lines 139-228) computes signal retention as merged_energy / individual_energy. For N experts with near-zero pairwise cosine, SR ~ N (not 1.0) because ||sum(delta_i)||^2 ~ sum(||delta_i||^2) + cross-terms. The reported SR values > 1.0 at d=64 (SR=1.004) are consistent with this -- the cross-terms are small but positive due to the domain-biased B matrices. The paper does not discuss why SR > 1.0, but this is a minor presentation issue, not a mathematical error.

## Novelty Assessment

**Modest but useful novelty.** The experiment does not introduce new theory -- the 1/sqrt(D_flat) scaling is a direct consequence of concentration of measure, as the paper acknowledges. The contributions are:

1. **Quantifying D_flat for all-modules LoRA.** Computing that D_flat ~ 14*L*d^2 for the Qwen2.5 family and confirming the scaling empirically. This is a calculation, not a theoretical advance, but it is useful for the SOLE architecture.

2. **Random baseline demolishes structure claims.** The finding that LoRA-structured deltas are indistinguishable from random vectors (ratio ~1.0) is the most important result. It simplifies the entire SOLE theoretical framework: no need to analyze Stiefel frames, domain bias, or any adapter-specific structure. Only D_flat matters. This result is honestly stated and is a genuine clarification of the architecture's guarantees.

3. **Confirming all-modules > FFN-only at every dimension.** Consistent with the macro finding (FFN-only killed), providing geometric explanation.

**Prior art:** The connection to Johnson-Lindenstrauss and concentration of measure is properly cited. The intra-project comparison to structural_orthogonality_proof (beta=-0.673 vs -1.049) is now explained as a difference in module sets.

## Experimental Design

### Does this test the stated hypothesis? YES (for geometry)

The hypothesis asks about "minimum viable base size." The experiment conclusively shows there is no minimum viable size from a geometric interference standpoint -- even d=64 passes. The paper correctly reframes: the bottleneck is model quality, not geometry. K1 is honestly marked as NOT TESTABLE.

### Controls adequate? YES

- 3 seeds per configuration
- FFN-only vs attention-only vs all-modules (3-way comparison)
- Random baseline (the key control, now added)
- 8 dimensions spanning 2 orders of magnitude
- Power law fits with R^2 reporting

### Could simpler mechanisms explain results? YES -- and the paper now says so

The random baseline result proves the dominant mechanism is just high dimensionality. The paper states this clearly: "the orthogonality guarantee comes entirely from high dimensionality, not from LoRA structure." This is the right conclusion.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry (exp_minimum_viable_base, status: supported) is consistent. The evidence entry from 2026-03-16 accurately summarizes the four fixes. Kill criteria:

- K1 ("base <1.5B cannot support expert composition, PPL <5%"): Correctly marked NOT TESTABLE. The experiment provides supporting evidence (geometry is not the bottleneck) but cannot address PPL.
- K2 ("expert quality scales linearly with base size, no sweet spot"): The paper says "SURVIVES (marginally)" based on saturation points. The status "supported" (not "proven") is appropriate given that saturation points are ceiling effects, not true mechanistic transitions.

## Integration Risk

Low. This experiment provides a geometric baseline that is consistent with and reinforces existing SOLE findings. The random baseline result actually simplifies the architecture story -- no need for special adapter structure to guarantee orthogonality.

One useful integration: the finding that D_flat (not d) drives orthogonality means that the N_max = d^2/r^2 formula in VISION.md is a lower bound. The true capacity is much higher because D_flat >> d^2. The paper notes this but does not propose an updated formula. This is fine for micro; macro should revisit.

## Macro-Scale Risks (advisory)

1. **Real trained adapters on related domains.** The converged_adapter_orthogonality experiment found cos=0.142 for math-medical at d=3584 -- 4000x higher than synthetic. The random baseline result suggests this high cosine comes from domain similarity (correlated B matrices), not LoRA structure. At macro, the question is whether domain-similar experts still compose well despite higher cosine. This experiment cannot answer that.

2. **The independence assumption breaks with real training.** Modules within a layer share input activations during training. The B matrices will be correlated across modules (e.g., gate_proj and up_proj in SwiGLU always multiply). The 1/sqrt(D_flat) scaling assumes independence across modules, which real training violates. The macro pilot should measure actual inter-module B-matrix correlation.

3. **2-layer experiments are conservative.** More layers help orthogonality (D_flat scales with L), so the extrapolation direction is safe.

## Verdict

**PROCEED**

All four required fixes from the first review have been properly applied:

1. K1 is clearly marked NOT TESTABLE with honest explanation of what the experiment can and cannot show.
2. N_max analytical estimates are flagged as unvalidated extrapolations. The 100K claim is retracted.
3. Random baseline comparison is implemented and the conclusion is the strongest finding: orthogonality is from dimensionality, not structure.
4. "Phase transition" is replaced with "saturation" throughout. The one remaining use is in a negation ("this is not a phase transition").

The non-blocking suggestions (cite prior work, clarify code files) were also addressed.

The experiment provides a clean geometric baseline for the SOLE architecture. The core finding -- that geometric interference is negligible at all tested dimensions and driven entirely by high dimensionality -- is mathematically sound, empirically confirmed, and honestly stated. The limitations (synthetic adapters, no quality measurement, unvalidated N_max formula) are all acknowledged in the paper.

The experiment does not advance the minimum-viable-base question in a functional sense (it cannot say whether a 0.5B base produces useful experts), but it definitively rules out geometric interference as the bottleneck. This is a useful negative result that focuses future work on the right question: base model quality.
