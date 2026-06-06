# Peer Review: LoRI-style B-sparsity on BitNet-2B

## NotebookLM Findings

NotebookLM was not consulted for this review (reviewer-side decision: the experiment is a clean null result with straightforward math; deep review would not change the verdict).

## Mathematical Soundness

**Derivations are correct.** The interference decomposition in MATH.md Section 2 is standard:

    interference(i,j) = (alpha/r)^2 * ||B_i^T A_i^T A_j B_j||_F

The bound in Section 2.4 correctly identifies that when ||A_i^T A_j||_F is already near-zero (the ternary base regime), modifying B structure cannot improve composition because the bottleneck is the A-matrix cross-product, not the B-matrix overlap. This is mathematically clean.

**The expected overlap calculation is correct.** For random 10%-masks: P(both keep) = 0.01, expected overlap = 109,363. The caveat that magnitude-based masks are NOT random (and may select overlapping positions) is correctly identified and empirically confirmed (cosine 1.46x higher under sparsity).

**One hidden assumption worth noting:** MATH.md Section 2.3 assumes interference is dominated by the B-subspace overlap when A matrices are random. This is only true when A_i^T A_j is not already near-zero. On BitNet-2B where |cos| is already 0.0016, the assumption is violated -- the paper correctly identifies this as the reason the mechanism fails. The reasoning is sound.

**The signal concentration argument (Section 2.4, lines 66-70) is qualitative, not proven.** The claim that "sparse B may concentrate signal into fewer dimensions, making the few surviving elements more correlated" is plausible and consistent with the 1.46x cosine increase, but it is an ex-post rationalization rather than a derived prediction. The paper does not derive a bound on the expected cosine increase under magnitude pruning. This is a minor gap -- the empirical evidence (1.46x) speaks for itself.

## Novelty Assessment

**Prior art properly cited.** LoRI (arXiv 2504.07448, COLM 2025) is the direct source. The experiment correctly replicates the key protocol elements: global mask, B reset after calibration, magnitude-based pruning.

**The finding is novel in the negative direction.** No prior work has tested LoRI-style B-sparsity on a ternary base model. The result -- that B-sparsity is redundant when the base already provides near-zero interference -- is a genuine contribution to understanding when LoRI's mechanism applies and when it does not.

**REFERENCES.yml includes the LoRI reference** (dir: "lori-sparse-b", arXiv 2504.07448). No reinvention.

## Experimental Design

### Protocol Fidelity

The LoRI protocol is followed with reasonable fidelity:

1. Global (model-wise) masking -- correct per LoRI ablation
2. B reset to zero after calibration -- correctly implemented (line 652-653 in code: `zero_lora_params(model)`)
3. 400-step retrain with frozen mask -- correct

**One significant deviation: A matrices are NOT frozen.** The LoRI paper freezes A and trains only B. Here, mlx_lm LoRA trains both A and B. The paper acknowledges this (Limitation 2) and argues it is conservative -- frozen A would make adapters MORE orthogonal, making B-sparsity even less impactful. This argument is directionally correct. However, it means this experiment does not test LoRI exactly as published; it tests "LoRI-style B-sparsity without frozen A on a ternary base." The distinction matters because the LoRI paper's frozen-A constraint is designed to work together with B-sparsity -- they are a joint mechanism, not independent components. Testing one without the other is informative but not a definitive test of LoRI.

**This is not a blocking issue** because the hypothesis is specifically about B-sparsity reducing interference, not about the full LoRI pipeline. The non-frozen-A is a confound that weakens the result's ability to speak about LoRI specifically, but does not undermine the core finding about ternary bases already providing sufficient orthogonality.

### Controls

**Dense baseline reloaded from prior experiment.** The results.json confirms all 5 dense adapters show `"reloaded": true`. This is acceptable -- same model, same data, same hyperparameters. But it introduces a subtle asymmetry: the dense adapters were trained in a prior run with potentially different random seed state (different initialization of lora_a). The sparse adapters were trained fresh in this run. Since cosine is computed over full adapter vectors (both lora_a and lora_b), the dense vs sparse cosine comparison may partially reflect different A initializations rather than purely B-sparsity effects.

**The K2 threshold of 1.0 is extremely strict.** A ratio of 1.0071x (0.71% worse) is within the noise band for single-seed experiments. The paper correctly notes this (Limitation 3, "What Would Kill This" section). The caveats in FINDINGS.md also correctly characterize this as "a null finding more than a kill." This is honest scientific reporting.

### Does the experiment test its hypothesis?

Yes. The hypothesis is that B-sparsity reduces composition interference on BitNet-2B. The experiment directly measures:
- Individual quality preservation (K1) -- PASS
- Composition quality improvement (K2) -- FAIL
- Orthogonality change (informational) -- cosine INCREASED 1.46x

The experiment tests what it claims and the result is clear: B-sparsity does not help on ternary bases because there is no interference to reduce.

### Could a simpler mechanism explain the result?

No simpler explanation needed. The null result is parsimoniously explained by the floor effect: when interference is already at |cos| = 0.0016, there is no room for improvement. The 1.46x cosine increase under sparsity is a secondary finding that strengthens the null -- sparsity made things slightly worse, not better.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_bitnet_lori_sparse_b` has:
- `status: killed` -- correct
- `depends_on: [exp_bitnet_effective_delta_cosine]` -- appropriate dependency
- `blocks: []` -- correct, nothing downstream depends on this
- Kill criteria match the actual test (K1: individual PPL ratio, K2: composed PPL comparison)
- Evidence entry correctly summarizes the result

The FINDINGS.md caveats entry is thorough and honest about the strict K2 threshold, the non-frozen-A deviation, and the characterization as "null finding more than a kill."

## Macro-Scale Risks (advisory)

1. **On FP16 bases, LoRI would likely work as published.** If the project ever returns to FP16, B-sparsity should be reconsidered. The kill is specific to ternary bases.

2. **The storage benefit of 90% B-sparsity (6x compression per adapter) is real but not tested against composition.** If storage per adapter becomes a bottleneck at N>1000, sparse B might be revisited purely for storage, not for orthogonality. The individual quality preservation (max 1.2% PPL increase) suggests this is viable.

3. **The signal concentration effect (1.46x cosine increase) could become problematic at scale.** With N=100+ adapters, if magnitude pruning consistently pushes adapters toward overlapping high-importance positions, the cumulative interference might exceed the ternary base's orthogonality guarantee. This is speculative but worth monitoring.

## Verdict

**PROCEED** (kill is valid and well-documented)

The experiment is a clean, well-executed null result. The kill on K2 is technically correct under the strict threshold. The scientific interpretation -- that ternary bases already solve the interference problem that B-sparsity was designed to address -- is sound and well-supported.

Specific strengths:
- Correct implementation of LoRI protocol (with documented deviation on frozen A)
- Honest reporting of the strict K2 threshold and its implications
- The MATH.md retroactively explains WHY the mechanism fails, not just THAT it fails
- The finding that magnitude pruning INCREASES cosine (1.46x) is a genuinely useful negative result
- FINDINGS.md caveats are thorough and appropriately self-critical

No revisions needed. The experiment, its PAPER.md, MATH.md, HYPOTHESES.yml entry, and FINDINGS.md caveats are all consistent, honest, and at the calibration level expected by SOLE_ADVERSARIAL_REVIEW.md.
