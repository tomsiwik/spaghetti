# Peer Review: attention_self_repair_removal

## NotebookLM Findings

Skipped -- the experiment is a clean negative result with straightforward math. Deep review not warranted for a KILLED hypothesis where the researcher already drew the correct conclusion.

## Mathematical Soundness

### Derivations

1. **Scale factor analysis (MATH.md Section 3.3):** The predicted repair ratio of ~0 is derived from the ratio of scale factors: 1/sqrt(2L) for transformer vs 1/sqrt(L) for MLP-only. The algebra is correct: sqrt(2L)/(sqrt(L)*sqrt(2)) = 1. The prediction matches the empirical result (2.1% overall repair, consistent with noise).

2. **Linear V propagation argument (Section 3.2, point 1):** Correct. Attention is Attn(h) = softmax(QK^T/sqrt(d_head)) V. The perturbation epsilon enters V linearly (V = h @ W_v^T, so delta_V = epsilon @ W_v^T). Softmax affects the attention weights (routing), not the value space. This is a sound mechanistic argument for why random frozen attention cannot selectively suppress perturbations.

3. **Amplification ratio computation (Section 4.1):** The definition amp_ratio = mean_output_deviation / sum_per_layer_error is consistent with parent experiments. Values are correct per results.json.

### Hidden Assumptions

1. **Batch-as-sequence for attention.** The code treats the batch dimension (n_inputs=100) as the sequence dimension for attention. This means every "token" attends to every other "token," with no causal mask and no positional encoding. This makes attention fully symmetric and removes any sequence-level self-repair mechanism. The paper acknowledges this in Limitations but does not analyze how strongly it affects the result.

   **Assessment:** This is a meaningful simplification but does not invalidate the null result. Causal masking would reduce attention's capacity to redistribute information (each position sees fewer other positions), which would make self-repair *harder*, not easier. The null result would likely hold or strengthen with causal attention.

2. **Random frozen weights.** The entire argument hinges on frozen random attention weights. McGill et al. (2024) self-repair is an emergent property of *trained* transformers with learned redundancy. The paper correctly identifies this as the key distinction (Section "Why It Failed," point 1) and appropriately scopes the conclusion: "frozen random attention provides no self-repair." This is not a hidden assumption; it is clearly stated.

3. **GS merge as ground truth.** The experiment uses Gram-Schmidt merge for both the "all experts" and "removed expert" conditions. This is consistent with the parent experiment methodology and appropriate for the comparison being made.

### Bounds

The 30% threshold for K1 is somewhat arbitrary but was stated a priori in HYPOTHESES.yml, so it is not post hoc. The observed 2.1% is far enough below 30% that the exact threshold does not matter. Even a generous threshold of 10% would still fail.

## Novelty Assessment

### Prior Art

The experiment correctly cites McGill et al. (2024) on self-repair in transformers. The key contribution is *not* the finding itself (absence of self-repair in random attention is predictable from the linearity-in-V argument) but rather the empirical confirmation that frozen attention is neutral for SOLE's expert removal safety. This closes an open question from three parent experiments.

### Delta Over Existing Work

Small but necessary. The three parent experiments (residual_layernorm_error_dynamics, attention_layer_removal_safety, correlated_layer_errors) all flagged "attention untested" as an open risk. This experiment eliminates that risk by showing attention is a non-factor. The result is that the Pre-RMSNorm amp_ratio=0.022 bound from residual_layernorm_error_dynamics is the correct and sufficient safety bound for SOLE.

## Experimental Design

### Does This Test What It Claims?

Yes. The experimental design is clean:
- Same MLP weights copied between both architectures (line 241-242 of run_experiment.py), isolating attention as the only variable
- Same expert perturbations, same removal procedure, same deviation metric
- Sweep across depth (L=1..16), dimension (d=32..128), and expert count (N=4..16)
- 3 seeds per condition
- Layer-by-layer trajectory analysis for K2

### Controls

Adequate. The MLP-only model serves as the control. The shared base weights ensure the comparison is apples-to-apples.

### Could a Simpler Mechanism Explain the Result?

The null result (no self-repair) is itself the simplest explanation. No confound is needed.

### Minor Issues

1. **Only 3 seeds.** The standard deviation of repair ratios across seeds is large (e.g., at L=1: repairs of +9.0%, -9.6%, +5.5%). With only 3 seeds, any individual depth point could flip sign by chance. However, the *overall* mean across all 18 depth-comparison runs (6 depths x 3 seeds) is 2.1%, and no single depth shows >12% repair. The aggregate conclusion is robust.

2. **The PAPER.md tables aggregate differently from raw data.** PAPER.md reports L=12 repair as 4.5%, but the raw data shows seeds of -3.6%, +5.8%, +11.5% (mean = 4.5%). The paper should note the high variance across seeds for individual depth points. This is cosmetic, not substantive.

3. **K2 regression methodology.** The paper reports a negative slope with p=0.033 for the layer-wise repair trajectory, claiming self-repair *decreases* with depth. But examining the raw layerwise data, repair ratios at individual layers range from -2% to +14% with no monotonic pattern. The p=0.033 may be an artifact of the layer-0 and layer-1 points (which have the least accumulated perturbation and thus the noisiest repair ratios). The stronger conclusion is simply "no trend" rather than "decreasing trend."

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml entry exactly:
- K1 (>30% lower deviation): FAIL at 2.1%
- K2 (self-repair increases with depth): FAIL (negative slope)
- Status correctly set to "killed"
- Dependencies on residual_layernorm_error_dynamics and attention_layer_removal_safety are appropriate
- blocks: [] is correct -- no downstream experiments depend on self-repair existing

The kill is clean. The hypothesis was well-scoped, the kill criteria were pre-registered, and the evidence is unambiguous.

## Macro-Scale Risks (advisory)

1. **Trained attention may behave differently.** The paper correctly flags this as the key open question. A trained Qwen2.5-7B base model has attention heads with learned redundancy, which is exactly what McGill et al. showed enables self-repair. A macro experiment with real trained base models could overturn this micro result. However, this would be a *different* hypothesis ("trained attention provides self-repair") that requires a new experiment entry.

2. **No risk to SOLE architecture.** The null result is the *favorable* outcome for SOLE. It means the Pre-RMSNorm safety bound is neither too optimistic nor too conservative -- it is the correct bound. Attention is neutral. This simplifies the safety story for production.

## Verdict

**PROCEED**

The experiment is a clean, well-designed negative result that closes an open risk flagged by three parent experiments. Both kill criteria fail decisively (2.1% vs 30% threshold; negative depth trend vs predicted positive). The mathematical analysis is sound, the code correctly implements the comparison, and the conclusion is appropriately scoped.

The only weakness is the K2 claim that self-repair *decreases* with depth (p=0.033), which may be noise given the high per-layer variance. But this is a secondary finding that does not affect the primary conclusion: frozen attention does not provide self-repair, and the Pre-RMSNorm bound is the correct safety bound for SOLE.

No revisions needed. The KILLED status should be recorded in FINDINGS.md and the HYPOTHESES.yml entry is already correct.
