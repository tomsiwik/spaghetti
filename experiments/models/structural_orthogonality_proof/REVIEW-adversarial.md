# Peer Review: Structural Orthogonality Proof

## Mathematical Soundness

### What holds

1. **Theorem 3.1 (Random Subspace Overlap).** The derivation E[tr(P_U P_V)] = r^2/d via Haar measure is standard and correct. The step E[P_U] = (r/d) I_d follows from rotational invariance, and independence of U, V justifies factoring the expectation. This is textbook Grassmannian geometry.

2. **Random subspace bound sqrt(r/d).** Correctly derived as the RMS principal angle cosine. The numerical values check out: d=64, r=8 gives 0.354; d=1024, r=8 gives 0.088.

3. **K1 criterion is mathematically clean.** Testing whether empirical cosine falls below sqrt(r/d) is a well-defined, falsifiable test. All 40 trained pairs pass with 17-69x margin. The bound is conservative because it addresses subspace principal angles while the experiment measures flattened delta vector cosine in a much higher-dimensional space D = O(d^2).

4. **Cosine measurement.** The code correctly computes the flattened delta vector as vec(A_l @ B_l) concatenated across layers, and uses standard cosine similarity. No bugs in the implementation.

### What does not hold or has hidden assumptions

1. **The "proof" is not a proof.** The title claims a "structural orthogonality proof" but delivers an empirical measurement with a power law fit. The Grassmannian bound E[cos] ~ sqrt(r/d) is a well-known result about *random* subspaces. The experiment's actual contribution is showing that *trained* adapters stay below this bound -- but that is an empirical observation, not a mathematical proof. A true proof would require bounding the gradient-alignment bias term formally (e.g., via a bound on the correlation introduced by shared base model structure). MATH.md Section 4.4 acknowledges this is "E[|cos|] = geometric_baseline + gradient_bias" but provides no bound on gradient_bias beyond empirical measurement.

2. **Power law fit is weak (R^2 = 0.637).** Five data points with R^2 = 0.637 means 36% of the variance is unexplained. The non-monotonicity at d=1024 (trained_mean = 0.00413 vs 0.00182 at d=512) is hand-waved as "different training dynamics" but could indicate a genuine reversal of the trend. The power law exponent alpha = 0.673 has no confidence interval reported. With 5 points fitting 2 parameters, there are only 3 degrees of freedom. This fit should not be used for extrapolation to d=4096 without extreme caution.

3. **Architecture varies non-uniformly across d.** This is the most serious confound:
   - d=64: 2 layers, d_ff=256, D=65,536
   - d=128: 2 layers, d_ff=512, D=262,144
   - d=256: 2 layers, d_ff=512, D=524,288 (d_ff_mult drops from 4 to 2)
   - d=512: 1 layer, d_ff=1024, D=1,048,576 (layers halved)
   - d=1024: 1 layer, d_ff=1024, D=2,097,152 (d_ff_mult drops to 1)

   The delta vector dimension D does not scale as a clean function of d. At d=1024, the model is a single-layer MLP with d_ff=d, which is architecturally very different from the 2-layer models at smaller d. The non-monotonicity at d=1024 is almost certainly an artifact of this: a 1-layer model with d_ff=d has different gradient geometry than a 2-layer model with d_ff=4d. The power law fit across these heterogeneous architectures is suspect.

4. **Training did not actually learn anything.** All losses are ~3.466, which is log(32) = 3.466 -- the random baseline for V=32. The LoRA adapters have accumulated gradient signal but have not moved the model output measurably from random prediction. This means the "trained" deltas are essentially very small random perturbations aligned with initial gradient directions, not converged domain-specific features. The experiment is measuring properties of early-gradient directions, not properties of trained expert subspaces.

5. **The "separation ratio" metric is inverted in interpretation.** The code defines separation_ratio = random_mean / trained_mean (line 300, 348). A value less than 1.0 means trained cosine is HIGHER than random. The PAPER.md correctly identifies this as K3 killed. However, the MATH.md Section 4.3 spins this as "does NOT violate structural orthogonality" and argues it is irrelevant. While the conclusion (trained cos is still far below the bound) is correct, the original K3 hypothesis ("gradient alignment pushes subspaces apart") was a specific mechanistic claim that is falsified. The paper should be clearer that the mechanism is purely geometric (concentration of measure), with gradient alignment being a mild *anti*-orthogonality force.

6. **Random baseline is not re-sampled across seeds.** The function `random_subspace_cosine` uses a hardcoded seed=42 (line 215). Both experiment seeds (42 and 137) get identical random baselines per d. When aggregating across seeds (line 335), the random cosines are simply duplicated, not independently sampled. This means the "aggregate" random statistics have artificially zero inter-seed variance. The trained/random comparison has inflated precision on the random side.

7. **Domain pairs are not truly "distinct."** The domain generation uses sequential IDs (da = p*2 + seed*100, db = p*2 + 1 + seed*100). Each pair has consecutive domain IDs, meaning their Markov transition matrices differ by only one diagonal boost (+1.5 to position (i, (i+da)%V) vs (i, (i+db)%V)) and slightly shifted high-probability character ranges. These are mildly different domains, not strongly distinct. The experiment could benefit from testing with maximally different domain constructions.

## Novelty Assessment

**The Grassmannian random subspace overlap result is not novel.** It appears in standard references (Absil et al., "Optimization Algorithms on Matrix Manifolds"). The connection to LoRA is noted in OSRM (2025), InfLoRA (2024), and related continual learning work. Those papers enforce orthogonality because they recognized the *random* guarantee is insufficient for worst-case scenarios (e.g., related tasks converging to shared subspaces -- exactly the situation acknowledged in MATH.md Section 4.5 with "projection overlap ~0.8, angles ~20 degrees" for related tasks).

**The delta over prior art is the empirical validation that trained adapters stay below the bound.** This is a useful contribution, but the framing as a "proof" or "mathematical guarantee" overstates what was achieved. The guarantee is conditional on domains being sufficiently distinct -- the experiment on orthogonality_by_domain_type already showed within-cluster pairs have 7.84x higher cosine. The structural orthogonality claim is really "structural orthogonality *for sufficiently dissimilar domains*" which is a weaker and more nuanced claim.

**Missed citation:** Aghajanyan et al. (2020), "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning," which establishes that fine-tuning operates in a low intrinsic-dimensional subspace and is directly relevant to why LoRA deltas concentrate in small subspaces of the ambient parameter space.

## Experimental Design

**Does the experiment test the stated hypothesis?** Partially. It tests K1 (bound) and identifies d_crit (K2), both successfully. K3 (gradient separation) is honestly reported as killed. The experiment design is sound for what it measures.

**Confounds:**
1. **No-learning confound.** With loss stuck at ~3.466 (random), the trained deltas reflect gradient direction from random initialization, not domain-specific learned features. At macro scale with strong learning signals, the gradient-alignment bias could be substantially larger. The paper acknowledges this (Limitation 1) but underestimates its importance -- the entire trained/random comparison could change character when models actually learn.

2. **Architecture heterogeneity across d.** Discussed above. Makes the power law fit unreliable.

3. **Sample size.** 8 pairs per d (4 pairs x 2 seeds) is small. The trained_std values are often comparable to or larger than the trained_mean (e.g., d=256: mean=0.00345, std=0.00378), indicating high variance.

**Controls:** The random LoRA-structured baseline is an appropriate control. The random subspace bound provides a theoretical upper bound. Both are correctly implemented.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry has three kill criteria:
- K1 (cos exceeds bound): PASS -- correctly assessed
- K2 (no phase transition): PASS -- correctly assessed
- K3 (no separation effect): KILLED -- correctly assessed

The status is "supported" which is appropriate given K3 was killed but K1 and K2 passed. The experiment honestly reports partial failure.

**Issue:** The kill criterion K3 was originally "gradient-aligned adapters are NOT more orthogonal than random subspaces (no separation effect)." The result is the opposite: trained adapters are LESS orthogonal (higher cosine) than random. The HYPOTHESES.yml notes say "Gradient stays near assigned subspace because domain-specific gradients have no reason to invade other slots" -- but the experiment shows gradients DO slightly invade each other's subspaces (shared base model bias). The notes should be updated to reflect this finding.

## Macro-Scale Risks (advisory)

1. **Gradient-alignment bias at scale.** The ~0.002-0.021 bias observed at micro is with zero effective learning. At macro scale where adapters genuinely specialize, this bias could be substantially larger, especially for semantically related domains. The prior finding of cos=0.85 for attention layers on math-medical pairs (from ffn_only_vs_all_modules) is a concrete example of this risk. The structural orthogonality claim may hold for FFN layers but fail for attention layers on related domains.

2. **The d_crit extrapolation (d_crit ~ 32r) is unreliable.** It is based on a power law with R^2=0.637 fitted across heterogeneous architectures where the model did not learn. At macro scale, stronger learning signals could shift d_crit upward. The theoretical d_crit = r/tau^2 = 80,000 for r=8 should not be dismissed as "800x too pessimistic" -- it may be the correct order of magnitude for strongly correlated domains.

3. **All-modules vs MLP-only.** This experiment tests MLP-only LoRA. The SOLE architecture uses all-modules LoRA (decision locked). The extension to attention layers is not tested and prior evidence (cos=0.85) suggests it does not hold. This is the single largest gap.

## Verdict

**PROCEED**

The core finding -- that trained LoRA adapter cosines are far below the random subspace bound sqrt(r/d) at all tested dimensions -- is sound and useful. K1 passes overwhelmingly. K3 is honestly killed and the revised understanding (orthogonality is geometric, not gradient-driven) is more correct than the original hypothesis. The experiment advances the SOLE foundation by quantifying the margin between empirical cosine and theoretical bound.

However, the following non-blocking issues should be addressed in the paper/MATH.md before this result is cited as a "proof":

1. **Rename from "proof" to "empirical validation."** The experiment provides empirical evidence for structural orthogonality, not a mathematical proof. A proof would require bounding the gradient-alignment bias term analytically. Call it "Structural Orthogonality Validation" or "Empirical Guarantee."

2. **Report confidence intervals on the power law exponent.** With 5 points and R^2=0.637, the uncertainty on alpha=0.673 is large. At minimum, report the standard error from the linear regression in log-log space. Do not extrapolate to d=4096 without acknowledging this uncertainty.

3. **Flag the no-learning confound prominently.** The losses at ~3.466 (random baseline) mean the adapters have not learned. State clearly: "These results characterize the geometry of gradient directions, not converged domain-specific subspaces. Macro validation with actual learning is required."

4. **Fix the random baseline re-sampling.** Use seed-dependent random baselines so that aggregate statistics reflect genuine sampling variance on both sides of the comparison.

5. **Acknowledge the attention-layer gap.** The experiment tests MLP-only LoRA. The known cos=0.85 for attention layers on related domains means the "structural orthogonality" claim does not extend to all-modules adapters without qualification. This is a critical scope limitation for SOLE.
