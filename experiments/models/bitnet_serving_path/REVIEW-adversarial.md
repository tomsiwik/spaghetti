# Peer Review: bitnet_serving_path

## NotebookLM Findings

Skipped (NotebookLM not invoked due to the experiment being a negative result with clear kill justification; the mathematical analysis is straightforward enough to verify directly).

## Mathematical Soundness

**The core math is correct and well-presented.** Stepping through:

1. **Scale analysis.** The LoRA delta magnitude computation (alpha_A * alpha_B * lora_scale = 0.012 * 0.006 * 20.0 = 0.00144) is arithmetically correct. The reported mean |DeltaW| = 0.004 is consistent with the integer matrix B_int^T @ A_int^T having typical element magnitude ~2.8 (for rank-16 with ~50% nonzero entries), yielding 2.8 * 0.00144 ~ 0.004. This checks out.

2. **Grid flip threshold.** The condition |DeltaW_ij| > alpha'/2 for a ternary state to change is correct for the round() operation. At alpha' ~ 0.926, the threshold is 0.463. The 116x gap (0.004 / 0.463 ~ 1/116) is verified: zero ternary states changed across all layers and all domains (0.00% in results.json). The empirical result perfectly matches the theoretical prediction.

3. **bfloat16 ULP analysis.** For values near 0.926, the exponent is 2^(-1) (since 0.5 <= 0.926 < 1.0), giving ULP = 2^(-1) * 2^(-7) = 0.00391. The delta of 0.004 being ~1 ULP is correct. The claim of "~50% information loss" is a reasonable approximation: values at the boundary round stochastically, and the expected fraction of deltas that survive bfloat16 addition depends on the alignment of the delta with the ULP grid -- 50% is the right order of magnitude.

4. **Float32 analysis.** ULP near 0.926 in float32 is 2^(-1) * 2^(-23) = 5.96e-8, and 0.004 / 5.96e-8 ~ 67,000 ULP. The statement that float32 merge would be lossless is correct.

**One minor mathematical note:** The MATH.md states the condition for flipping is |DeltaW_ij| > alpha'/2, but this is only exact for flipping between adjacent states (0 to +/-1 or +/-1 to 0). For a flip from -1 to +1, the threshold is alpha'. Since the delta is 116x too small for even the easier flip, this doesn't change the conclusion.

**Verdict: Math is sound. No errors found.**

## Novelty Assessment

**This is a well-executed negative result, not a novelty claim.** The experiment correctly tests whether LoTA-QAF's principle applies to SOLE's adapter training regime. The answer is definitively no.

**Critical distinction the paper correctly identifies:** LoTA-QAF (arxiv 2505.18724) assumes adapters are trained with t-SignSGD to produce truly ternary A_T, B_T matrices where the product DeltaW is integer-valued in [-r, r]. The SOLE adapters use standard Adam + STE ternary quantization, which produces latent FP32 weights with very small magnitudes (alpha_A ~ 0.012). These are fundamentally different training regimes. The paper acknowledges this in Limitation #2 and the "What Would Kill This" section.

**Prior art check:** The BITNET_SOLE_RESEARCH.md already notes LoTA-QAF's gap: "No multi-adapter composition discussed." The experiment fills this gap by testing the merge principle empirically, even though the adapters were not trained with LoTA-QAF's native recipe. This is a legitimate contribution to the project's knowledge base.

**What would have been stronger:** A direct comparison using LoTA-QAF's t-SignSGD training alongside the Adam+STE training on the same data. This would disambiguate "LoTA-QAF merge does not work at all on BitNet-2B" from "LoTA-QAF merge does not work with our specific adapter training recipe." The paper correctly identifies this as future work but does not flag how critical this distinction is for the project.

## Experimental Design

**Strengths:**

1. **Clean experimental structure.** Four phases (base PPL, single-adapter merge, multi-adapter scaling, latency/memory) with proper weight resets between conditions. The code correctly resets model weights to base before each merge attempt.

2. **Two merge strategies compared.** Float merge (no requantization) vs QAF merge (with requantization) provides useful diagnostic information. The float merge results show that the adapters DO contain useful information -- it's the requantization that destroys it.

3. **Kill criteria are well-defined and applied honestly.** K1 (>5% benefit loss) is clearly violated by 2476%. The researcher did not try to save the result.

4. **Supplementary float merge analysis is valuable.** The finding that float merge at N=5 (PPL 8.52) beats the reference composed PPL (9.55) is an important positive result that suggests float merge (in fp32) could be a viable serving path. This is not buried.

**Weaknesses:**

1. **K2 and K3 are measured on the QAF-merged model, which is functionally the base model.** Since 0% of ternary states changed, the QAF-merged model IS the base model (up to floating-point rounding in the requantization step). Measuring latency and memory of a model that is identical to the base is trivially uninformative. K2 PASS and K3 PASS tell us nothing about serving composed adapters -- they tell us the base model fits in memory and runs fast, which was already known. The PAPER.md correctly notes "Both configurations run at ~13 tok/s ... because the model architecture is unchanged" but still reports these as "PASS" results. They should be labeled "TRIVIALLY SATISFIED (merge erased)" or similar.

2. **N=15 memory test is vacuous.** Merging 5 adapters 3x is not equivalent to merging 15 distinct adapters in terms of memory. Moreover, since the QAF merge erases all information, the "N=15" model is still just the base model. The 0 MB delta is expected and meaningless.

3. **The experiment title and hypothesis frame LoTA-QAF as a testable approach, but the researcher did not use LoTA-QAF's training recipe.** This is like testing "does RLHF improve alignment?" by training with supervised fine-tuning and then claiming RLHF doesn't work. The experiment correctly tests "do our existing adapters survive requantization?" (answer: no), but it does not actually test LoTA-QAF as published. The Limitations section acknowledges this (point #2), but the title and hypothesis should be more precise: "LoTA-QAF merge with STE-trained adapters" rather than "LoTA-QAF merge."

4. **No STE applied during merge.** The `compute_lora_deltas` function has an `apply_ste` parameter but it is never called with `apply_ste=True` in the merge path. During training, the forward pass computes `x @ Q(A) @ Q(B) * s` where Q is STE ternary quantization. The merge path computes `B.T @ A.T * lora_scale` using the raw latent FP32 weights, not the quantized values. This means the delta being merged is slightly different from the delta applied during training. At the scale of these deltas (0.004) this difference is unlikely to change the conclusion, but it is a methodological inconsistency. The float merge PPLs at N=1 being worse than reference routed PPLs may partially be explained by this (in addition to bfloat16 truncation).

5. **p95 calculation with n_runs=5.** `latencies[int(0.95 * 5)]` = `latencies[4]` = max value. With only 5 runs, the "p95" is actually the maximum. This is disclosed implicitly (you can see p95=max in the results) but not called out. Not a serious issue given the large margin (8.33s vs 10s threshold).

## Hypothesis Graph Consistency

The HYPOTHESES.yml kill criteria match the experiment:
- K1: "lossless ternary merge loses >5% of adapter PPL benefit at N>1" -- tested, FAIL (2476%)
- K2: "p95 latency >10s for 100-token generation on Apple Silicon" -- tested, PASS (8.33s)
- K3: "memory exceeds 8GB with 15 pre-merged adapters" -- tested, PASS (1.69GB)

**K1 FAIL is sufficient to kill.** The experiment is correctly killed.

**However:** K1's framing assumes LoTA-QAF is the only merge strategy worth testing. The float merge results suggest a viable alternative that was not in the original hypothesis. The experiment should have proposed a follow-up hypothesis node for float merge (fp32) as a serving path.

## Macro-Scale Risks (advisory)

1. **The conclusion "runtime LoRA is the only viable serving path" may be premature.** Float merge in fp32 showed PPL 8.52 vs 10.64 base (20% improvement). If the fp32 memory cost (estimated ~8 GB) is acceptable, this could be a viable serving path without the overhead of runtime adapter application. A follow-up experiment should test fp32 merge quality more carefully.

2. **The 13 tok/s throughput on Apple Silicon (unpacked bf16) is relevant to production.** This is 2-3x slower than what bitnet.cpp achieves with packed ternary. If the project needs the packed serving path, it must investigate LoTA-QAF's native t-SignSGD training recipe. This is correctly identified as future work.

3. **The finding that float merge N=5 beats reference composed N=5 (8.52 vs 9.55) is architecturally important.** It suggests that pre-merge may be superior to runtime 1/N composition for static expert sets. This deserves a dedicated follow-up experiment.

## Verdict

**PROCEED** (kill is justified, findings are sound, no revisions needed)

The experiment correctly identifies that LoTA-QAF merge with STE-trained LoRA adapters fails catastrophically on BitNet-2B-4T. The mathematical analysis is rigorous and the empirical results perfectly match the theoretical predictions. The kill is justified: 0% ternary state changes with a 116x scale gap leaves no room for ambiguity.

**Minor recommendations (non-blocking):**

1. Add a note in the PAPER.md that K2 PASS and K3 PASS are trivially satisfied because the QAF merge produces the base model (0% weight changes). These should not be cited as evidence that "merged serving works" in future documents.

2. The `compute_lora_deltas` function should have been called with `apply_ste=True` for methodological consistency with the training forward pass. This is unlikely to change the result but would be correct practice.

3. Consider adding a HYPOTHESES.yml node for "float merge (fp32) as serving path" based on the supplementary finding that float merge at N=5 outperforms runtime 1/N composition.

4. The framing should be sharpened: the experiment killed "STE-trained adapter merge via requantization," not "LoTA-QAF" per se. LoTA-QAF with its native t-SignSGD training remains untested and potentially viable.
