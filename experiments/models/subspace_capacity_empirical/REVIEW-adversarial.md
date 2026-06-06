# Peer Review: Subspace Capacity Empirical

## NotebookLM Findings

Skipped (tool authentication not available). Review conducted via direct analysis of MATH.md, PAPER.md, experiment.py, and results.json.

## Mathematical Soundness

**Signal retention metric is correct but trivially satisfied.**

The signal retention computation (experiment.py lines 333-373) correctly computes:

    S_i = <v_merged, v_i> / (||v_merged|| * ||v_i||) = cos(v_i, v_merged)

For orthogonal equal-norm experts, S_i = 1/sqrt(N). The retention ratio R = S_empirical * sqrt(N) measures deviation from orthogonal behavior. This is mathematically sound.

However, the retention ratio staying near 1.0 is a **near-tautology** at the tested scales. With D = 131,072 and max N = 128 experts of rank 4, each expert's delta is a rank-4 matrix flattened into R^131072. The occupied fraction of the space is N*r/D = 128*4/131072 = 0.004 (0.4%). Random vectors in R^D with D >> N have expected |cos| ~ 1/sqrt(D) ~ 0.003, which makes near-orthogonality a consequence of the concentration of measure phenomenon in high dimensions, not a property specific to trained LoRA experts. The experiment confirms concentration of measure, which has been known since at least the Johnson-Lindenstrauss lemma (1984).

**The N_max derivation has a gap in MATH.md Section 2.1.**

The text states: "E[|cos|] ~ sqrt(r/D) < tau, rearranging for the maximum N..." But the expected cosine between random rank-r subspaces in R^D does not depend on N. What depends on N is the probability that the *worst-case* pair among N(N-1)/2 pairs exceeds threshold tau. The bound should come from a union bound:

    P(max_{i<j} |cos(v_i, v_j)| > tau) <= N(N-1)/2 * P(|cos| > tau)

The formula N_max ~ D/r^2 is not derived; it is stated. The actual Grassmannian packing bound (Welch bound, Section 3.4) gives a tighter and properly derived condition: N*r <= D for zero coherence. This is the correct argument, but the paper leads with the sloppy one.

**The claim "real capacity is D/r^2 = 8Ld^2/r^2, not d^2/r^2" is not empirically validated.**

The experiment tests up to N = 80 at r=8 (125% of d^2/r^2 = 64) and N = 128 at r=4 (50% of d^2/r^2 = 256). Finding no cliff at 125% of the conservative bound does not validate that the true bound is 32x higher. To distinguish between "capacity is 1.5x of d^2/r^2" and "capacity is 32x of d^2/r^2," you would need to test at N values approaching D/r^2 = 2,048 (for r=8). The paper extrapolates from testing 4% of the claimed true capacity to asserting the bound is correct. This is a significant overreach.

**Scaling predictions to production (Section 6 of MATH.md) are therefore unsupported.**

The claim "practical capacity is on the order of millions of experts" rests on the D/r^2 formula that was not tested anywhere near its regime. The conservative d^2/r^2 bound is the only one with any empirical support from this experiment.

## Novelty Assessment

**Low novelty.** The core finding -- that random vectors in high-dimensional spaces are approximately orthogonal -- is a textbook result (concentration of measure). The application to LoRA deltas is straightforward.

The signal retention ratio metric is a reasonable way to operationalize the concept for SOLE, but the metric is cos(v_i, v_merged), which is not fundamentally different from the pairwise cosine already measured in the parent exp_collision_scaling experiment. If pairwise cosines are small, the retention ratio will be near 1.0 by linear algebra.

**Prior art:** The minimum_viable_base experiment already confirmed |cos| ~ 1/sqrt(D_flat) with R^2 = 0.997. The structural_orthogonality_proof showed cos 17-69x below sqrt(r/d). This experiment adds the retention-ratio framing and tests at higher N, but the conclusion (experts are approximately orthogonal in high-D space) was already established.

## Experimental Design

**The experiment cannot falsify its own hypothesis at the tested scale.**

The hypothesis is: "N_max = d^2/r^2 predicts the empirical capacity cliff." The experiment finds NO cliff, not at N_max and not beyond it. This means the hypothesis is not confirmed (there is no cliff to match), just unfalsified. The paper correctly notes this but then pivots to the stronger claim that "the real bound is D/r^2," which is also untested.

To actually test the hypothesis, the experiment should have pushed to N values where interference becomes detectable. With D = 131,072, even testing up to N = 2,000 (at r=8) would be computationally feasible on CPU (each expert is a small numpy operation). This is a missed opportunity.

**Quality metric is completely uninformative.**

Expert improvement over base is 0.00% (losses match random baseline at 3.466). The paper acknowledges this but frames it as "the geometric metrics are the meaningful evidence." While true, this means the experiment cannot test whether geometric orthogonality translates to functional composition quality -- which is the question that matters for SOLE production use. The geometric story is necessary but not sufficient.

**Controls are adequate for what is measured.** Three seeds, two rank settings, progressive N sweep. The experimental methodology is clean.

**Confound: domain clustering inflates cosine above random baseline.**

At N=5 (r=8), mean |cos| = 0.069, which is 25x above the 1/sqrt(D) = 0.003 random baseline. The paper notes this (Section 5 of MATH.md: "4x the random baseline") but understates it. At small N, all experts come from fewer clusters, increasing within-cluster cosine. This domain clustering effect means the "orthogonal random subspace" interpretation is inexact -- the experts are structured, just not structured enough to matter at D = 131,072. At production scale with semantically similar domains (e.g., 50 Python coding experts), the cosine could be much higher than random.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node (exp_subspace_capacity_empirical) has status "supported" with kill criteria:
- K1: empirical capacity < 10% of theoretical N_max
- K2: quality degrades smoothly (no sharp capacity cliff)

K1 is correctly evaluated as PASS. K2 is evaluated as SMOOTH, which technically matches the kill criterion wording ("quality degrades smoothly" = no sharp cliff). But K2 is satisfied vacuously: quality ratio is 1.0000 everywhere because the quality metric is uninformative. The paper should not claim K2 as meaningful evidence.

Status "supported" is appropriate given the geometric evidence, though the strength of support is overstated in the evidence claim.

## Macro-Scale Risks (advisory)

1. **Domain clustering at scale.** With 50+ experts in similar domains (e.g., multiple programming languages), pairwise cosines could be orders of magnitude above the random baseline. The experiment's synthetic Markov data produces weak clustering; real domains produce strong clustering.

2. **Quality cliff may exist even when geometric metrics are fine.** The experiment could not test this. A model could have perfect geometric orthogonality but still suffer quality degradation if the magnitude of deltas varies widely (large experts drowning small ones in the sum). The normalization in the signal retention metric hides magnitude effects.

3. **The 8L multiplier may not transfer cleanly.** The experiment uses a 4-layer MLP with separate W1/W2 per layer. Production transformers have attention (q/k/v/o) and FFN (gate/up/down) with different effective dimensionalities. The layer structure creates block-diagonal structure in the flattened delta space that does not match the i.i.d. assumption.

4. **Production adapters are full A+B trained, not frozen-A.** The experiment uses frozen-A (B-only training). Full A+B training allows subspace drift that breaks the Grassmannian packing guarantees. This is acknowledged but is the largest macro risk.

## Verdict

**PROCEED**

The experiment is methodologically sound, the math (where it is careful) is correct, and the geometric results are consistent with theory and prior SOLE experiments. The core mechanism -- additive LoRA composition preserves per-expert signal in high-dimensional parameter space -- works in principle.

However, the paper overclaims. Specific issues that should be noted but do not block PROCEED:

1. **The "D/r^2 is the real bound" claim is extrapolation, not empirical finding.** The experiment validates that d^2/r^2 is conservative (by showing no cliff at 125% of N_max), but does not validate D/r^2. The PAPER.md should soften this to: "The conservative d^2/r^2 bound is confirmed as a lower bound on capacity. The absence of a cliff at 125% of N_max suggests significant additional headroom, consistent with the D/r^2 geometric limit, but this upper bound is not empirically validated at micro scale."

2. **K2 (smooth degradation) is vacuously true** because the quality metric is uninformative. The evidence claim should note this explicitly rather than listing K2 as a meaningful finding.

3. **The production scaling predictions (millions of experts) should be flagged as projections from the geometric argument, not empirical extrapolations.** VISION.md already uses the conservative d^2/r^2 formula for its scaling table, which is the right choice.

These are presentation issues, not mechanism failures. The geometric evidence is real: trained LoRA experts at d=64 compose without detectable interference up to at least 125% of the conservative bound. This is consistent with the project's broader orthogonality story and advances the SOLE architecture incrementally. The experiment is a confirmatory data point, not a breakthrough finding.
