# Peer Review: Room Gradient Analysis

## Experiment Type

**No type / no math at all.** There is no MATH.md. PAPER.md contains a hypothesis, a heuristic prediction (|r| >= 0.3), and measurements, but no mathematical framework of any kind -- no theorem, no proof, no cited proven framework, no identified mathematical gap. This experiment is pure empirical exploration without the required mathematical grounding.

## Hack Detector

- **Fix count:** 1 (single mechanism: Sobel-like discrete differences on B-matrices). No hack stacking.
- **Is MATH.md a proof or a description?** MATH.md does not exist. PAPER.md contains a mechanism description dressed in informal language, not even equations. No proof, no QED, no theorem statement.
- **Metric used as evidence:** Pearson correlation r between gradient similarity and behavioral similarity. There is no proof that this correlation, even if high, would predict the behavioral outcome of interest (successful routing). Correlation between two proxy quantities does not establish a causal routing mechanism.
- **Kill criteria source:** The threshold |r| >= 0.3 is an arbitrary threshold. It is not derived from any proof. Why 0.3 and not 0.5 or 0.1? A "moderate correlation" threshold from social science conventions has no mathematical justification in this context.

## Self-Test Audit

MATH.md does not exist, so there is no Self-Test section. **BLOCKING: missing entirely.**

1. One-sentence impossibility property -- MISSING
2. Cited theorems -- MISSING (no theorems cited anywhere)
3. Predicted numbers -- The threshold |r| >= 0.3 appears in PAPER.md but is not derived from any theory
4. Falsification condition -- r < 0.3 is stated, but it targets an arbitrary metric threshold, not a proof prediction
5. Hyperparameter count -- Not addressed
6. Hack check -- Not addressed

## Mathematical Soundness

There is no mathematical content to verify. The experiment contains zero theorems, zero proofs, and zero formal derivations. Specific issues:

**1. No theoretical basis for the Sobel analogy.** The experiment treats B-matrices as "heightmaps" and applies discrete image-processing gradients (finite differences along rows and columns). There is no argument for why spatial adjacency in a weight matrix should carry semantic meaning. Matrix indices in a LoRA B-matrix correspond to (rank dimension, output dimension) -- rows are rank components, columns are output neurons. Adjacent rows are adjacent rank components, which have no geometric or semantic ordering guarantee. The Sobel analogy from image processing assumes spatial locality; weight matrices have no such property.

**2. Gradient similarity is a scalar reduction of dubious value.** The `gradient_similarity` function (line 64-76 of run_experiment.py) computes per-module gradient magnitudes (a single scalar per module), collects these into vectors, and computes Pearson correlation across modules. This reduces all spatial structure to a single magnitude number per module, then asks whether the profile of magnitudes across modules correlates. This is three levels of averaging away from any spatial gradient information. The claim is about "spatial gradients" but the measurement discards all spatial information.

**3. Behavioral similarity is hardcoded, not measured.** The `behavioral_similarity` function (line 78-98) returns hardcoded heuristic values from a lookup table. These are described as "based on cross-domain PPL experiments and Finding #186" but they are manual assignments (0.1, 0.2, 0.3, etc.) with no confidence intervals, no formal definition of what "behavioral similarity" means, and no measurement methodology. Correlating one set of measurements against hardcoded guesses is not a valid experimental design. The correlation r = 0.1985 is between measured gradient similarities and made-up behavioral similarities.

**4. Sample size inadequate for correlation claims.** With only n=10 pairs, Pearson r = 0.1985 has a p-value of approximately 0.58 (two-tailed). This is not statistically significant at any conventional level. The experiment cannot distinguish r = 0.1985 from r = 0 with 10 data points. The kill criterion (r < 0.3) was correctly triggered, but the experiment lacks the statistical power to make any claim about the true correlation.

## Prediction vs Measurement

PAPER.md contains a correlation matrix (Table 1) and a measured r = 0.1985 vs threshold 0.3. This is a measurement table, not a prediction-vs-measurement table. A proper table would require:
- A theoretical prediction for r (not just a threshold)
- Predicted direction and magnitude with error bounds
- Measurement with confidence intervals

The threshold |r| >= 0.3 is not a prediction from theory; it is an arbitrary bar borrowed from behavioral science conventions for "moderate" correlation.

## NotebookLM Findings

Skipped -- the experiment has no MATH.md to review, was already killed, and the issues are straightforward enough that deep review adds no value.

## Novelty Assessment

This experiment tests a novel-sounding idea (Sobel gradients on adapter weight matrices as routing signals) but:

1. **The spatial-gradient-on-weights idea has no prior art because the premise is unfounded.** Weight matrix entries do not have spatial locality -- they are indexed by abstract dimensions (rank, neurons), not by position in a physical space. Image gradient operators assume that adjacent pixels are spatially related. Adjacent weight matrix entries have no such property unless the network has been specifically structured (e.g., convolutional layers with spatial weight sharing, which LoRA B-matrices are not).

2. **The broader question (can adapter geometry inform routing?) has prior art.** Cosine similarity between flattened adapter weight vectors, SVD-based subspace angles, and direct interference measurement (Finding #186, the Grassmannian skeleton work) are all established approaches. This experiment does not cite or compare against these.

3. **The experiment does not advance the Room model.** The conclusion ("routing cannot be solved via spatial gradient analysis") is trivially derivable from first principles: weight matrices are not images, so image operators should not be expected to extract semantic information from them.

## Macro-Scale Risks (advisory)

Not applicable -- the experiment was killed and the mechanism is unsound in principle at any scale.

## Additional Concerns

1. **The experiment ran in 0.4 seconds.** This is not a compute-intensive verification -- it loaded adapter files and computed some numpy operations. The cost of the experiment was minimal, which partially mitigates the lack of mathematical grounding, but does not excuse it.

2. **Module-level insights are post-hoc.** The observation that o_proj and v_proj modules have highest gradient variance is noted as if it is a finding, but without a framework predicting this, it is pattern-matching in noise with n=5 domains.

3. **Finding #335 already records KILLED.** The experiment correctly killed itself. This review confirms the kill was appropriate but identifies that the experiment should not have been run without a MATH.md in the first place.

## Verdict

**KILL** (confirmed)

The experiment was already correctly killed based on its own criterion (r = 0.1985 < 0.3). This review confirms the kill and identifies deeper structural problems:

1. **No MATH.md exists.** This is a BLOCKING violation of the proof-first research protocol. The experiment had no mathematical framework of any type (verification, guided exploration, or frontier extension).

2. **The premise is unsound in principle.** Applying spatial gradient operators (designed for 2D images with spatial locality) to weight matrices (which have no spatial locality) has no theoretical basis. This is a category error, not a scale limitation.

3. **Behavioral similarity was hardcoded, not measured.** The experiment correlated real measurements against manually assigned heuristic values, making the entire correlation analysis circular.

4. **Statistical power was inadequate.** n=10 pairs cannot distinguish r = 0.2 from r = 0 at any meaningful significance level.

### Impossibility Structure

The fundamental issue is that LoRA B-matrices B in R^{r x d_out} have no spatial topology. The row index is a rank component, the column index is an output neuron. Neither has a natural ordering that makes adjacent entries semantically related. Discrete difference operators (Sobel, Laplacian, etc.) extract meaningful features only when adjacency in the index space corresponds to proximity in some semantic or physical space. For arbitrary dense weight matrices, this condition does not hold, making spatial gradient analysis structurally inappropriate regardless of scale, domain count, or correlation threshold.

### For Future Experiments

If the goal is finding "free" routing signals from adapter geometry, the mathematical framework must start from properties that weight matrices actually possess: spectral structure (SVD), subspace angles (Grassmannian distances), norm profiles, or interference bounds (which are already proven in this project's Grassmannian skeleton work). Any future attempt must have a MATH.md with at least a cited proven framework and an identified unknown before running code.
