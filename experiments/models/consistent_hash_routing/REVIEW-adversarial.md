# Peer Review: Consistent Hash Routing

## NotebookLM Findings

Skipped due to tool unavailability. Review conducted through direct reading of MATH.md, PAPER.md, implementation code, experiment runner, tests, HYPOTHESES.yml, VISION.md, FINDINGS.md, ADVERSARIAL_REVIEW.md, and REFERENCES.yml.

## Mathematical Soundness

**Derivations are correct.** The displacement theorem (Karger et al. 1997, adapted) is stated accurately: E[displacement_top1] = 1/(N+1) and the union bound E[displacement_topk] <= k/(N+1) is a valid loose upper bound. The measured 9.1% vs theoretical 11.1% (top-1) is consistent with the explanation that some displaced tokens only change their secondary expert.

**Routing weight formula is internally consistent.** MATH.md specifies w_j = exp(1/(d_j + 1)) / sum exp(1/(d_{j'} + 1)), and the implementation (lines 209-218 of consistent_hash_routing.py) correctly computes softmax over inverse distances with numerically stable subtraction of the max.

**Minor issue -- weight formula is ad hoc.** The choice of 1/(d+1) as the pre-softmax score has no theoretical motivation. Ring distance in [0, 2^32) makes 1/(d+1) extremely small for both selected experts (distances in the hundreds of millions), so the softmax is nearly uniform over the top-k. With k=2, the weights are approximately [0.5, 0.5] in most cases. The "worked example" in MATH.md shows [0.67, 0.33], but that uses illustrative distances of 300M and 600M -- in practice, with 1200 ring entries uniformly distributed over 2^32, the average gap between adjacent entries is ~3.6M, and the distance to the nearest expert (walking clockwise past virtual nodes) is typically much smaller than 2^32, making the two distances similar and the weights near-uniform. This is not wrong, but the paper's framing of "closer experts get higher weight" overstates the discriminative power of the weighting scheme.

**Computational complexity table is correct** for the stated operations. The asymptotic comparison (O(d + log NV) vs O(Nd)) is valid, though the constant factors heavily favor the GPU-native softmax at micro scale, as the paper honestly acknowledges.

**Hidden assumption in Assumption 2.** The paper claims "similar hidden states produce similar scalar projections" from the fixed random projection. This is true in expectation (dot product preserves L2 distances in expectation), but a single random projection to R^1 destroys almost all structure. Two tokens can have very different semantics but identical projections, and vice versa. The paper partially acknowledges this ("weaker guarantee than LSH") but then uses it as justification for locality-preserving routing. At k=2 with near-uniform weights, this does not matter. At higher k or with learned projections, it could.

## Novelty Assessment

**Genuinely novel application.** The paper correctly identifies that no prior MoE work uses consistent hash ring placement for expert routing. Hash Layers (Chen et al., NeurIPS 2021) uses balanced hashing but without the consistent hashing property (no incremental add/remove guarantee). The cross-domain transfer from distributed systems caching to neural routing is a legitimate and well-motivated insight.

**The key contribution is the displacement guarantee, not routing quality.** The paper correctly positions itself: the LSH experiment showed routing strategy is irrelevant at G=8, so the differentiator is the hot-add property. This is intellectually honest and well-reasoned.

**Prior art is adequately cited.** Karger 1997, Jump Consistent Hash, Hash Layers, and the project's own LSH experiment are all referenced. The `references/consistent-hashing/` folder contains appropriate background.

## Experimental Design

**The experiment tests what it claims.** The two kill criteria (degradation > 5%, displacement > 30%) are exactly what HYPOTHESES.yml specifies, and the experiment directly measures both.

**Controls are adequate.** The softmax baseline uses the same architecture, data, training protocol, and seeds. The softmax displacement of 0% is correctly explained (random-init new router weights at 0.02 scale cannot win argmax against trained weights at 0.5-1.0 scale), and the paper draws the right conclusion: softmax stability comes at the cost of new expert liveness.

**Potential confound in quality measurement.** The new expert has random weights but immediately receives ~11% of traffic. This means ~11% of tokens now have random noise mixed into their output. The +0.20% degradation is surprisingly low. This is because (a) the new expert is weighted by the routing weight (~0.5 at most for top-2), and (b) random-init weights at scale 0.02 produce near-zero outputs relative to trained experts. The paper should note that the degradation would be larger with a badly initialized (e.g., scale 1.0) expert. However, in practice new experts would be pre-trained, so this is conservative in the right direction.

**Displacement measurement has a subtle layer-independence assumption.** The `measure_displacement` function (run_experiment.py:40-86) adds and removes the expert from each layer's ring independently, measuring displacement at each layer while keeping hidden states from the original model. This means cross-layer displacement effects (layer 0 routes differently, producing different hidden states for layer 1) are not captured. The actual displacement under simultaneous addition to all layers could be higher. This is a minor issue since the metric is defined per-layer-per-token and averaged, which is the natural measurement.

**3 seeds is acknowledged as limited.** The extremely tight per-seed variance (8.9%, 8.9%, 9.6% displacement) suggests the measurement is stable, so 3 seeds is probably sufficient for the directional conclusion.

**Missing control: uniform random routing baseline.** The LSH experiment showed all routing strategies are equivalent at G=8. A uniform random routing baseline (each token randomly picks 2 experts with equal weight) would strengthen the claim that consistent hashing's value is specifically the displacement guarantee, not routing quality. Without this, one could argue any non-learned router would pass the same kill criteria.

## Hypothesis Graph Consistency

The experiment matches `exp_consistent_hash_routing` in HYPOTHESES.yml. Kill criteria are:
- ">5% degradation when adding expert without any recalibration" -- tested, +0.20% (passes)
- "adding one expert displaces >30% of existing routing decisions" -- tested, 9.1% (passes)

The evidence is sufficient to change the node's status from `active` to `survived` at micro scale.

## Macro-Scale Risks (advisory)

1. **Python hash overhead is a deployment blocker.** The 66% throughput penalty is entirely due to Python-level hash computation. At macro scale, a C/CUDA implementation of FNV1a + binary search + ring walk is mandatory. This is engineering, not research, but it must be done before any macro benchmark.

2. **Cumulative displacement under sequential addition.** The paper correctly identifies this: adding K experts one at a time displaces up to K/(N+K) cumulatively. At N=8 adding 8 more experts: 50% cumulative displacement. At N=256 adding 1: 0.4%. The mechanism becomes more attractive at scale, but the transition from micro (N=8) to early macro (N=32) passes through a regime where cumulative displacement could be significant.

3. **Ring balance degrades with heterogeneous virtual node counts.** If different experts have different numbers of virtual nodes (e.g., to give higher-quality experts more traffic), the balance guarantee weakens. The current design assumes uniform V across experts.

4. **Fixed random projection is a single point of failure.** All routing decisions depend on a single frozen random vector p in R^d. If p happens to be poorly aligned with the data manifold (e.g., near-orthogonal to the principal component of variation), routing will have poor locality. At d=64, this is unlikely but not impossible. At d=4096 (macro), concentration of measure makes this even less likely, so the risk decreases with scale.

5. **Interaction with expert quality heterogeneity.** Consistent hashing gives each expert ~1/N of traffic regardless of expert quality. A random-init expert immediately gets 11% of traffic and produces garbage. In production, this means new experts must be at least partially pre-trained before hot-adding. The paper acknowledges this in Limitations but the macro protocol must enforce it.

## Verdict

**PROCEED**

The experiment is well-designed, mathematically sound, and honestly reported. Both kill criteria pass with comfortable margins (25x and 3.3x respectively). The mechanism is genuinely novel for MoE routing and provides a concrete property (hot-add without recalibration) that directly advances the VISION.md contribution protocol. The limitations are known, acknowledged, and appropriate for micro scale.

The routing weight near-uniformity (Section "Mathematical Soundness" above) and the missing uniform-random control are minor weaknesses that do not affect the core finding: consistent hashing enables incremental expert addition with bounded displacement and negligible quality loss.

Recommended (non-blocking) improvements for the paper:
1. Add a note that routing weights are near-uniform for k=2 with the current distance-based weighting, so the mechanism is effectively a top-2 uniform blend with deterministic selection.
2. Add a uniform random routing baseline to isolate the displacement guarantee as the specific value-add over simpler non-learned routers.
3. Note that degradation depends on new expert weight initialization scale (0.02 is conservative).
