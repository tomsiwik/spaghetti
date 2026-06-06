# Peer Review: Pathway Graph Construction on BitNet-2B

## Experiment Type
Verification (claimed) -- but see below.

## Hack Detector
- Fix count: 1 (co-activation graph + PH). Clean.
- Is MATH.md a proof or a description? **Description dressed in equations.** There is no Theorem/Proof/QED block. MATH.md describes a measurement procedure and cites the stability theorem, but proves nothing new. The stability theorem guarantees PH is stable under perturbation -- it does NOT predict that any particular graph will have non-trivial topology. That is an empirical question, which makes this a measurement experiment, not a verification experiment.
- Metric used as evidence: persistence count and Spearman rho. Neither is proven to predict a behavioral outcome (e.g., expert composition quality, domain task accuracy).
- Kill criteria source: Thresholds (>=10 features, rho < 0.5) are arbitrary, not derived from any proof.

## Self-Test Audit

1. **One-sentence impossibility property:** "The stability theorem guarantees PH will detect non-trivial topology if it exists." This is correct but vacuous for this experiment -- the stability theorem says PH is a *stable* detector, not that non-trivial topology *must exist*. The self-test conflates "PH works" with "the thing PH is measuring exists." **FLAG: The impossibility property is about the tool, not the claim.**

2. **Cited theorems:** Stability theorem (Cohen-Steiner et al., 2007) is real and correctly described. 1812.09764 (Neural Persistence) is real. 2506.01042 -- I cannot verify this arxiv ID. The claim that "topology-based probes outperform activation-based probes by 130%" needs verification. **FLAG: Unverified reference.**

3. **Predicted numbers:** P1 >= 10 features with persistence > 0.1, P2 rho < 0.5. These are stated but not *derived* from any theorem. They are guesses about what would constitute "interesting." The 0.1 persistence threshold is not motivated by any mathematical argument.

4. **Falsification condition:** "If co-activation graph has trivial topology or high persistence = high SV." This targets the claim, not a proof (since there is no proof). Acceptable for a measurement experiment.

5. **Hyperparameter count:** States 2 (epsilon, k). Actually there are at least 4: epsilon_fraction (0.5, different from the 0.1 stated in MATH.md), k (100), sparsification percentile (50th), and the choice of layer (15). **FLAG: MATH.md says epsilon = 0.1 * max, code uses EPSILON_FRACTION = 0.5.** This is a direct contradiction between the proof document and the implementation.

6. **Hack check:** Answered correctly -- first measurement in this track.

## Mathematical Soundness

There is no theorem to verify. MATH.md describes a procedure and makes predictions that are empirical guesses, not mathematical derivations. This is not blocking for a *measurement* experiment, but the experiment should not be labeled "Type: Verification."

### Methodological Issues (BLOCKING)

**Issue 1: Epsilon mismatch.** MATH.md section E states "epsilon = 0.1 * max|vi^T h(x)|" as the activation threshold. The code uses `EPSILON_FRACTION = 0.5`. This is a 5x discrepancy. The entire co-activation graph structure depends on this threshold. A higher epsilon (0.5) means sparser activation patterns, which mechanically creates more co-activation structure (fewer edges, more distinct clusters). The predictions in MATH.md were presumably calibrated for epsilon = 0.1. Running at epsilon = 0.5 invalidates the prediction-measurement comparison.

**Issue 2: The persistence results are likely a sparsification artifact.** Here is the critical confound:

- The co-activation graph starts nearly fully connected (the code logs confirm high average activation rate).
- After sparsification at the 50th percentile, roughly half the edges are removed.
- The distance matrix assigns `inf` to removed edges.
- Ripser then sees 100 vertices where roughly half the pairs are at infinite distance.
- This mechanically produces many connected components that merge one-by-one as the filtration threshold increases.
- All 91 features have birth = 0.0, which means every component was born at the start of the filtration. The persistence values (0.65 to 0.98) simply reflect the *order* in which the sparsified edges reconnect components.

This pattern -- all births at 0, deaths spread across [0.65, 1.0] -- is exactly what you would get from *any* graph that has been sparsified to create disconnected components, then reconnected via sublevel filtration. The high persistence values (mean 0.87) do not indicate "rich topology" -- they indicate that the sparsification threshold was set aggressively enough to fragment the graph, and the distance transform maps the remaining edge weights to a range near 1.0.

**To confirm this is an artifact:** Run the same procedure on a random graph with the same edge density and weight distribution. If you get a similar persistence diagram, the topology is trivially explained by the sparsification, not by any meaningful structure in BitNet-2B.

**Issue 3: The rank correlation test (K624) is confounded.** The test compares SV rank to *weighted degree* rank, not to persistence rank. MATH.md predicts "Rank correlation(persistence rank, SV rank) < 0.5" but the code computes Spearman rho between SV rank and *graph degree* rank. These are different quantities. Weighted degree in a co-activation graph is a measure of how often a direction co-activates with *all other* directions -- it is not a topological invariant. The claim "topology captures different structure than SVD" is not supported because degree is not topology.

**Issue 4: 62.6% energy capture is concerning.** The top-100 singular directions capture only 62.6% of the activation variance. The experiment builds the co-activation graph entirely from these 100 directions. The remaining 37.4% of variance is invisible. This is acknowledged in limitations but not addressed -- there is no argument that the missing 37.4% does not contain the domain-specific structure that would be most relevant to pathway preservation.

## Prediction vs Measurement

| Prediction | Measured | Honest Assessment |
|------------|----------|-------------------|
| P1: >= 10 features persistence > 0.1 | 91 | Likely artifact of sparsification (see Issue 2) |
| P2: Rank corr < 0.5 | rho = 0.325 | Measures degree vs SV, not persistence vs SV (see Issue 3) |
| P3: Power-law distribution | Slope -0.08, R^2 = 0.66 | Weak. Slope near 0 means nearly uniform, not power-law |
| P4: Cross-domain bridges | 100/100 are multi-domain | Unfalsifiable -- all directions are bridges, distinction collapses |

PAPER.md contains the prediction-vs-measurement table. But 0/4 predictions are cleanly confirmed when scrutinized:
- P1 is likely a sparsification artifact
- P2 measures the wrong quantity
- P3 explicitly fails (PAPER.md says "PARTIAL", but slope of -0.08 is not a power law)
- P4 is noted as "UNCLEAR" because the test cannot distinguish

## NotebookLM Findings

Skipping NotebookLM step as the issues are clear from direct analysis.

## Novelty Assessment

Neural persistence (1812.09764) computed PH on weight matrices, not activation co-occurrence graphs. The idea of building a co-activation graph from SVD projections and running PH on it is a reasonable methodological contribution. However, without the random baseline control, the results cannot be distinguished from a trivial consequence of the graph construction procedure.

The connection to pathway preservation is aspirational -- no behavioral outcome is tested. The experiment does not show that preserving high-persistence features leads to better expert composition, domain retention, or any functional property.

## Macro-Scale Risks (advisory)

- PH computation is O(n^3) in the number of vertices. At 100 vertices this is trivial, but scaling to full-model analysis (all layers, higher-dimensional homology) could become expensive.
- The sparsification sensitivity (acknowledged in limitations) is a real concern at any scale.

## Verdict

**REVISE**

### Required Fixes (ordered by priority)

1. **Add a random baseline control (BLOCKING).** Generate a random graph with the same number of vertices, same edge density, and same edge weight distribution (drawn from the empirical distribution of co-activation weights). Run the identical sparsification + PH pipeline. Report the persistence diagram. If the random baseline produces a similar diagram, the experiment's K623 result is an artifact and the finding should be killed.

2. **Fix the epsilon mismatch (BLOCKING).** MATH.md says epsilon = 0.1 * max. Code uses 0.5 * max. Either update MATH.md to match the code, or re-run with epsilon = 0.1. The prediction document and implementation must agree.

3. **Fix K624 to measure what it claims (BLOCKING).** The kill criterion says "persistence rank vs SV rank." The code measures "degree rank vs SV rank." Either: (a) change the kill criterion text to match what is actually computed, or (b) implement the actual persistence-rank comparison (map each persistence feature to the vertex that participates in the corresponding merge event, rank vertices by their max-persistence feature).

4. **Reclassify experiment type.** This is a measurement / guided exploration experiment, not a verification experiment. There is no theorem being verified. The stability theorem guarantees PH stability, not the existence of non-trivial topology in BitNet-2B. Update MATH.md accordingly.

5. **Connect to a behavioral outcome (advisory, not blocking).** The experiment shows "topology exists" but not "topology matters." A follow-up should test whether preserving high-persistence features during expert decomposition leads to better downstream performance than preserving top-k singular vectors. Without this, the finding remains purely descriptive.
