# Peer Review: notebooklm_apple_silicon_optimization

## Experiment Type
Guided exploration (survey within proven roofline framework)

## Hack Detector
- Fix count: 0 (this is a survey, not an implementation)
- Is MATH.md a proof or a description? **Description dressed in equations.** "Theorem 1" and "Theorem 2" have no formal proof structure -- they state calculations and label them QED. The roofline "proof" is: compute I = 8/1.58 = 5.06, note 5.06 < I_ridge, declare QED. This is arithmetic, not a theorem. A real theorem would prove that ternary matmul achieves exactly N FLOPs (it does not -- there are also activation computations, norms, attention softmax, etc.). The text acknowledges this later in Section F but the "Theorem" pretends the N-FLOPs assumption is exact.
- Metric used as evidence: tok/s and GB/s from Finding #176 (prior experiment, not new measurements)
- Kill criteria source: K1 is "research produces no actionable recommendations" -- a binary existence check, not derived from any proof

## Self-Test Audit
**MATH.md has no Self-Test section.** This is a blocking deficiency per the review protocol. However, given this is a P3 research survey (not a mechanism experiment), the missing self-test is less damaging than it would be for a verification experiment.

1. One-sentence impossibility property: ABSENT
2. Cited theorems: ABSENT (no formal citation of roofline model origin -- Williams, Waterman, Patterson 2009)
3. Predicted numbers: PARTIALLY PRESENT in Section D (tok/s predictions) but not in self-test format
4. Falsification condition: ABSENT
5. Hyperparameter count: N/A (survey)
6. Hack check: N/A

## Mathematical Soundness

**Theorem 1 (bandwidth-bound claim):** The core calculation is correct in direction but imprecise in detail.

- I_ternary = N / (N * 1.58/8) = 5.06 FLOPs/byte. The algebra is correct.
- However, claiming "approximately N FLOPs per token (additions only)" undercounts. A transformer forward pass also includes: attention score computation (Q*K^T, O(n*d_head) per head per layer), softmax (O(n*n_heads) per layer), RMSNorm (O(d) per layer), SiLU/SwiGLU gates, and the LM head (which is the largest single component at 2.53ms per Finding #176). These are NOT ternary operations -- they use FP arithmetic. The true FLOPs are higher than N, making I_ternary slightly higher than 5.06. This does not change the conclusion (still bandwidth-bound) but the "Theorem" overclaims precision.
- I_ridge is left vague: ">> 5.06 for any reasonable GPU TFLOPS figure." The worked example in Corollary 1.2 estimates ~65, which is indeed >> 5.06, so the conclusion holds with wide margin. The imprecision about M5 Pro GPU TFLOPS is honest (Apple does not publish this number).

**Theorem 2 (adapter memory):** Straightforward arithmetic, correctly computed. The 10.2% overhead from MLX allocator alignment is a reasonable empirical correction. The N_max = 813 vs measured 853 discrepancy (5%) is within allocator variance as claimed.

**Corollary 1.2 (batch crossover at ~13):** The estimate assumes I_batched = b * I_ternary, which is correct for the weight-reading component (b forward passes share one weight read). But the KV cache grows linearly with batch size, adding reads that partially offset the intensity gain. At batch=13 with sequence length 100, KV cache is ~13 * 30 * 2560 * 2 * 100 * 2 bytes = ~400MB, which is non-trivial relative to the 1.18GB model. The crossover point is therefore somewhat higher than 13 for non-trivial sequence lengths. This is not wrong, but the "~13" number is presented without this caveat.

**Section F (worked example):** The total of the itemized components sums to ~7.7ms but the measured time is 6.04ms. The text explains this as "MLX's lazy evaluation successfully overlaps weight reads with compute" with "~35% overlap." This is consistent with the sum-of-parts = 135% finding in metal_kernel_profiling. The explanation is plausible.

**Arithmetic intensity table (Section G):** The INT4 row claims AI = 4.0 FLOPs/byte with bytes/weight = 0.5. For INT4: 4 bits = 0.5 bytes/weight, FLOPs = 2*N (still multiply-accumulate for INT4), so AI = 2N / (N * 0.5) = 4.0. Correct. The FP16 row: AI = 2N / (N * 2) = 1.0. Correct. The ternary batch=16 row: AI = 5.06 * 16 = 80.96. Correct.

## Prediction vs Measurement

There is no prediction-vs-measurement table because this experiment produces no new measurements. All numbers are drawn from Finding #176 (metal_kernel_profiling). The "predictions" in each recommendation section are forward-looking estimates for future experiments, not verifiable claims.

This is structurally appropriate for a survey experiment but means the SUPPORTED status rests entirely on the quality of the analysis and recommendations, not on empirical verification.

## NotebookLM Findings

NotebookLM was used as the research tool for this experiment (the experiment title includes "notebooklm"). The output is a curated synthesis of web research rather than a traditional experiment. No separate NotebookLM deep review was conducted for this peer review.

## Novelty Assessment

**Overlap with Finding #176 (metal_kernel_profiling) is substantial.**

Finding #176 already established:
- 165.6 tok/s at 73% bandwidth utilization
- 268.6 GB/s measured peak bandwidth
- mx.compile gives 21% at seq=1
- Sum-of-parts overhead breakdown (layers, LM head, unpacking)
- System is at the bandwidth limit

This survey adds:
1. Roofline model formalization (I_ternary = 5.06, ridge point analysis)
2. Batch crossover estimate (~13)
3. Memory budget model for N adapters
4. Seven prioritized recommendations with paper citations
5. M5 Pro Neural Accelerator analysis (irrelevant for batch=1)
6. Negative findings compilation (N1-N4)

Items 1-3 are genuinely new analysis. Items 4-7 are literature curation. The negative findings (N1-N4) are pointers to prior experiments, not new analysis.

**Prior art concern:** The roofline model is a well-established framework (Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," CACM 2009). The application to LLM inference is standard -- see e.g., Pope et al. "Efficiently Scaling Transformer Inference" (MLSys 2023). The MATH.md does not cite these foundational references.

**Redundancy with Finding #176:** Approximately 60% of MATH.md restates or repackages findings from metal_kernel_profiling. The overhead breakdown table (Section C) reproduces the same numbers. The worked example (Section F) uses the same measurements. The value-add is the roofline framework wrapper and the forward-looking recommendations.

## Macro-Scale Risks (advisory)

1. Memory wiring recommendation (R1) is trivially testable and should have been tested rather than predicted. A one-line sysctl command is lower effort than writing this survey.
2. The vllm-mlx recommendation (R7) requires significant custom integration for SOLE routing + adapter composition. The prediction of "~500+ tok/s for 4 users" assumes linear scaling from batching, which is optimistic given the routing overhead per token.
3. The fused BitLinear kernel recommendation (R4) is the highest-value item but also the highest-risk. The pseudo-code in PAPER.md is illustrative, not functional -- actual Metal kernel development for ternary unpacking + matmul fusion is a multi-week effort.

## Additional Issues

1. **Citation accuracy:** arxiv 2601.19139 (vllm-mlx) -- could not verify this specific arxiv ID exists (the 2601 prefix implies January 2026; plausible but unverifiable from this review). The Orion paper (arxiv 2603.06728) and Sparse-BitNet (arxiv 2603.05168) are March 2026 papers, consistent with the research date.

2. **Recommendation 5 (ANE offload) honesty is commendable.** The paper explicitly states "<1% throughput impact" and frames it as architectural preparation. This is the right way to present a micro-optimization.

3. **Recommendation 3 is already done.** The paper marks mx.compile as "P0 (done)" and Finding #176 already measured it. Including it as a "recommendation" inflates the count from 6 genuine recommendations to 7.

4. **Kill criterion is trivially satisfiable.** K1 ("research produces no actionable recommendations") can only fail if the researcher produces zero output. Any survey that writes recommendations will pass. This is not a meaningful kill criterion for testing whether the analysis is correct or valuable.

## Verdict

**PROCEED** (with caveats)

This is a P3 research survey, not a mechanism experiment. Judged as a survey, it provides useful roofline formalization and a well-prioritized recommendation list. The mathematical framework is directionally correct despite imprecision in the "Theorem" labels. The negative findings compilation (N1-N4) adds organizational value.

The experiment does not meet proof-first standards (no self-test, theorems are calculations not proofs, no new measurements, trivially satisfiable kill criterion). However, these deficiencies are structural to the experiment type (literature survey) rather than evidence of flawed reasoning.

Conditions for PROCEED:
1. **Do not claim SUPPORTED based on this alone.** The finding should be tagged as **provisional** (literature synthesis, no new measurements).
2. **Acknowledge redundancy with Finding #176** -- approximately 60% overlap. The delta is the roofline framework and recommendation prioritization.
3. **R1 (memory wiring) should be tested, not predicted.** It is a one-line command. Running it would convert one prediction into a measurement and strengthen the finding.
4. **Cite the roofline model properly** (Williams et al. 2009).
5. **Remove R3 (mx.compile) from the recommendation count** -- it is already implemented and measured. The honest count is 6 new recommendations, not 7.
