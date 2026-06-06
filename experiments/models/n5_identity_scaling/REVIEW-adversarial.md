# Peer Review: N=5 Identity Scaling

## NotebookLM Findings
Skipped per reviewer instructions.

## Mathematical Soundness

### Derivations verified

1. **Perturbation scaling (MATH.md Section 3)**: The linear bound on perturbation magnitude with (N-1) is correct. The triangle inequality application in Section 3.3 is standard and tight for worst-case (all perturbations aligned). The paper correctly notes that uncorrelated domain signals produce sublinear scaling in practice.

2. **Expected Jaccard degradation model (Section 3.5)**: The linear-degradation-per-domain model is a first-order approximation. The assumption of constant borderline fraction alpha is reasonable at micro scale but has no validation -- it could increase with N if perturbation accumulates and shifts the margin distribution. This does not invalidate the analysis, but the "safe limit ~N=8" extrapolation should be treated as a rough estimate, not a bound.

3. **Null model (Section 5)**: The independent-death null model computation is correct: p_single=0.484, p_composed=0.576 gives E[J_null]=0.357. The observed J=0.792 >> 0.357 is strong evidence of non-independence. No errors found.

4. **Set decomposition (Section 4)**: The union-of-singles vs composed Jaccard construction is mathematically correct. The code in `decompose_n_domain` faithfully implements the offset-based re-indexing described in MATH.md. Verified by tracing through the worked example (Section 8) -- the numbers check out.

### Hidden assumptions

1. **Joint validation data for profiling composed model**: The N-sweep (Section 6.1, step 4) profiles composed models on "joint validation data" -- all domains mixed. This means each domain contributes roughly proportionally to the activation statistics. Since domain sizes are highly unequal (a-e at 32.7% vs u-z at 7.4%), the smaller domains contribute fewer tokens to the profiling. This could bias the composed dead set toward the distribution of larger domains. The paper acknowledges domain size imbalance (Section 6.3) but does not discuss this specific profiling bias. **Severity: low** -- the "composed on own-domain" control (Section 4 of the experiment, PAPER.md "N=5 Composed on Own-Domain Data" table) shows similar Jaccard values, so the bias is not driving the results.

2. **N-sweep uses a fixed domain ordering**: N=2 is always {a-e, f-j}, N=3 is always {a-e, f-j, k-o}, etc. The paper acknowledges this (Assumption 3) but the trajectory could look different with other orderings. Since the N=2 Jaccard here (0.871) is close to Exp 16's 0.895 (which used a completely different binary split), the ordering sensitivity appears small. **Severity: low**.

3. **Same seed for train and profile**: The experiment uses the same seed value for training and profiling (the `seed` parameter flows to both `train()` and `profile_activations()`). Profiling noise was validated as negligible in Exp 12 (2.6-3.8% disagreement), so this is acceptable but worth noting.

## Novelty Assessment

This experiment is a **direct scaling test** of Exp 16 (capsule_identity), not a novel mechanism. It asks: does the previously-validated identity preservation hold at N=5? This is the correct next step in the hypothesis graph.

**Prior art check**: No reference in REFERENCES.yml directly addresses the question of how neuron identity (dead/alive classification) scales with the number of composed modules. The gurbuzbalaban-neural-death reference covers death dynamics during training but not across composition. This experiment fills a gap that is specific to the project's composition protocol.

The MATH.md correctly references and extends Exp 16's framework. The code reuses existing infrastructure (`compose_relu_models`, `profile_activations`, `get_dead_set`, `jaccard_similarity`) without reinventing anything. This is good practice.

## Experimental Design

### Strengths

1. **N-sweep is well-designed**: Measuring at N=2,3,4,5 rather than jumping straight to N=5 provides trajectory information. The linear fit to the trajectory enables extrapolation (with appropriate caveats).

2. **Per-domain profiling control**: The "composed on own-domain data" analysis (profiling the N=5 composed model on single-domain validation data) is an excellent control. It disentangles the effect of composition from the effect of mixed-domain input. The similar Jaccard values confirm the dead set changes are from weight-space perturbation, not input distribution shift.

3. **3 seeds with per-domain breakdown**: Reports both aggregate and per-domain-per-seed results. Honestly flags the worst case (p-t at J=0.640, below the 0.70 threshold).

### Concerns

1. **The kill criterion tests combined Jaccard, but per-domain minimum breaches 0.70**: The combined Jaccard (0.792) passes, but one domain-seed combination hits J=0.640. The paper is transparent about this, but it raises the question of whether the kill criterion should be the combined metric or the worst-case per-domain metric. For the pre-composition pruning use case, a single domain with poor identity preservation means that domain's pruning decisions are unreliable. **This is a legitimate concern but does not invalidate the experiment** -- the kill criterion was defined in advance as "combined Jaccard < 0.70", and the experiment followed the protocol honestly.

2. **Variance in p-t domain (std=0.121) is concerning**: With only 3 seeds, a standard deviation of 0.121 on a mean of 0.776 means a 95% CI of approximately [0.474, 1.078] (using t-distribution with df=2). This interval is extremely wide, making the per-domain Jaccard estimate for p-t unreliable. The paper correctly identifies this but understates the implication: we cannot say with confidence whether p-t's true Jaccard is above or below 0.70.

3. **No permutation test on domain ordering**: The N-sweep always adds domains in alphabetical order. A permutation test (e.g., all 10 possible pairs for N=2, or all 10 triples for N=3) would strengthen the claim that the degradation rate is ~0.026/domain. Currently the trajectory could be an artifact of the specific ordering. **Severity: medium for the trajectory claim, low for the kill criterion** (the N=5 composition always uses all 5 domains regardless of order).

4. **Composed model profiled on joint data, but single-domain models profiled on own-domain data**: This is by design (you want to know if pruning decisions made from single-domain profiling transfer to the composed setting), but it introduces an asymmetry. The composed model sees all domains' data during profiling, which could activate different capsules than any single domain alone. The "composed on own-domain" control mitigates this concern.

## Hypothesis Graph Consistency

- **Node**: `exp_n5_identity_scaling`, status: proven
- **Kill criterion**: "Jaccard between single-domain and N=5 composed dead sets drops below 0.70" -- matches the experiment's primary metric (combined Jaccard = 0.792)
- **Depends on**: `exp16_capsule_identity_tracking` -- correctly identified as the predecessor
- **Evidence claim**: Accurately reflects the paper's findings including the per-domain minimum caveat

The experiment is consistent with its HYPOTHESES.yml entry.

## Macro-Scale Risks (advisory)

1. **The ~N=8 extrapolation is fragile**: It assumes constant degradation rate, which depends on constant borderline capsule fraction. At macro scale with d=4096 and P=11008, the margin distribution could be very different. The extrapolation should not be used as a design constraint without macro validation.

2. **SiLU activation floor**: As the paper notes, the entire dead/alive framework requires ReLU. Macro models typically use SiLU/SwiGLU. This is a known limitation already flagged in Exp 15.

3. **Real domain dissimilarity**: The quintary split (character-level names by first letter) produces highly similar domains. Real domains (code vs prose vs math) could have much larger perturbation magnitudes per domain, accelerating degradation.

4. **p-t variance at macro**: If per-domain variance remains high at macro scale, a production system cannot rely on combined Jaccard alone -- it needs per-domain validation, as the paper recommends.

## Verdict

**PROCEED**

The experiment is well-designed, the math is sound, the code correctly implements the protocol, and the primary kill criterion is met. The key findings are:

- Combined Jaccard = 0.792 at N=5 (above 0.70 threshold)
- Graceful degradation trajectory across N=2,3,4,5
- High overlap coefficient (0.967) confirms pre-composition pruning safety: dead capsules stay dead
- Honest reporting of per-domain worst case (J=0.640)

The per-domain minimum of 0.640 is a genuine concern for production use, but the paper acknowledges it and provides mitigation recommendations (spot-check validation, consensus pruning). The kill criterion was defined as combined Jaccard, which passes. The experiment advances the hypothesis graph correctly and provides concrete guidance for the macro transition.

Minor items that should be noted but do not block PROCEED:
1. The "safe limit ~N=8" extrapolation should be clearly labeled as an estimate, not a bound.
2. Future work should test domain ordering permutations if the degradation rate becomes a design parameter.
3. The p-t domain variance (std=0.121 on 3 seeds) means its Jaccard estimate is unreliable; more seeds would help but are not required at micro scale.
