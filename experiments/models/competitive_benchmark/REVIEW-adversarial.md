# Peer Review: Competitive Benchmark

## NotebookLM Findings

Skipped -- this is a post-mortem review of an already-killed experiment. The artifacts are self-contained and the kill is clear-cut. Deep review resources are better spent on live experiments.

## Mathematical Soundness

### Pre-merge composition formula: CORRECT

The composition formula `W_new = W + (1/N) * alpha * sum(B_i^T @ A_i^T)` is correctly implemented in `premerge_adapters_into_model()`. The code matches the math: `w * LORA_SCALE * (b_mx.T @ a_mx.T)` where `w = 1/N = 0.2` and `LORA_SCALE = 20.0`, giving effective per-adapter scale of 4.0. This is consistent with the LoRA scaling convention (alpha/r * r cancels in the standard formulation, but here alpha=20 is used directly as a multiplier, which means the effective scaling depends on what alpha was during training).

### Memory analysis in MATH.md: INCORRECT (pre-experiment predictions were wrong)

MATH.md predicted ~1.7GB inference memory for BitNet+SOLE, based on the assumption that pre-merged weights would be the same size as base. Actual result: 10.98GB peak. The paper correctly identifies this as merge-time overhead (loading all adapter A and B matrices simultaneously), but the MATH.md prediction of "should PASS K3" was badly calibrated. The researcher was honest about this in the paper (Limitation 4), but the pre-experiment math should have caught that loading 5 adapters' worth of Grassmannian skeleton (numpy arrays) plus B matrices simultaneously on top of the already-unpacked bf16 model would spike memory.

Specific issue: the skeleton is loaded as a single .npz containing ALL domain x layer x key matrices. At 26 layers x 7 target keys x 5 domains = 910 matrices, each (2560, 16) = ~40K elements in float32 from numpy. That is ~910 * 40960 * 4 bytes = ~150MB for A matrices alone. The B matrices add similar. The real memory spike comes from the bf16 unpacking of the base model itself (700MB packed -> ~5.35GB unpacked), not the adapters. The 10.98GB peak likely includes both the unpacked base AND temporary merge computation buffers.

**Key correction:** MATH.md's "~1.7GB" was the steady-state estimate. The actual 10.98GB is peak. The paper acknowledges steady-state is ~5.35GB (same as base). The K3 comparison should arguably be against steady-state, not peak. Even so, 5.35GB > 2.45GB Qwen, so K3 still fails. This is a fair kill but the margin is less extreme than 4.5x suggests.

### Parameter efficiency argument: MISLEADING

MATH.md claims adapters add "~10KB total." This is technically true for the ternary-packed B matrices on disk, but irrelevant to the memory comparison because the adapters are merged into base weights. The meaningful comparison is the total inference footprint, which is ~5.35GB (post-merge steady-state) vs 2.45GB (Qwen 4-bit). The adapters themselves are not the memory problem; the bf16 unpacking is.

## Novelty Assessment

This is not a novelty experiment -- it is a system-level benchmark. No novel mechanisms are introduced. The experiment correctly positions itself as a gate test for the BitNet-SOLE architecture.

### Prior art context
The comparison methodology is standard (head-to-head on GSM8K + MMLU with greedy decoding). Using model-native prompt formats is the right choice for ecological validity, though it introduces a confound (acknowledged in Limitation 3).

## Experimental Design

### What it tests correctly
- Four systems on identical benchmark splits with identical evaluation code
- Model-native prompt formatting (not artificially handicapping instruction-tuned models)
- Greedy decoding for reproducibility
- Memory measurement via MLX peak tracking
- Both a size-matched control (Gemma-2-2B) and an aspirational target (Qwen2.5-3B)

### Serious concerns

**1. Sample sizes are critically small for the claims made.**

20 MMLU questions per domain gives a 95% confidence interval of roughly +/-20pp (for proportions near 0.5, the Wald interval is +/- 1.96 * sqrt(0.5*0.5/20) = +/-21.9pp). This means:

- SOLE 45% vs Qwen 50% on legal (5pp gap): completely within noise
- SOLE 45% vs Qwen 70% on medical (25pp gap): likely real but imprecise
- SOLE 25% vs base 50% on math (25pp gap): likely real

The K1 verdict of "4/6 worse" is fragile. With 20-question samples, at least 2 of those 4 comparisons (math -10pp Qwen advantage, legal -5pp) could flip with different random samples. The kill is still justified because the pattern is consistent, but the paper should not report precise percentage-point differences as if they are reliable.

Similarly, GSM8K at n=50 gives a +/-14pp confidence interval. The +12pp SOLE advantage over Qwen (48% vs 36%) is within the margin of error. This is the paper's strongest positive claim and it is NOT statistically significant at p<0.05.

**2. Qwen2.5-3B scores are suspiciously low.**

The paper acknowledges this (Limitation 5): published GSM8K for Qwen2.5-3B is 65-70%, but the experiment got 36%. The paper attributes this to 4-bit quantization and/or prompt format. But a nearly 2x degradation from quantization alone is implausible -- 4-bit Qwen models typically lose 1-3pp on benchmarks, not 30pp. This strongly suggests a prompt format or answer extraction bug specific to Qwen.

Possible causes:
- The ChatML format may not include the `#### ` hint that the GSM8K extraction regex relies on. Qwen may produce answers in a different format ("The answer is 42" instead of "#### 42"), and while the extractor has fallbacks, they are fragile (e.g., matching the last number in any `= $X` pattern).
- The `max_tokens=256` limit may be too short for Qwen's verbose chain-of-thought, cutting off before the answer.
- The MMLU 70% on medical/code suggests Qwen IS working correctly on some tasks, so the issue is GSM8K-specific answer extraction, not a loading/inference bug.

If Qwen's true GSM8K score with correct extraction would be ~60-65%, then the +12pp SOLE advantage evaporates and becomes a -15pp deficit. This is a serious methodological concern that undermines the paper's brightest finding.

**3. The Gemma math score of 5% (1/20) is a red flag.**

Gemma-2-2B scoring 5% on math MMLU (below the 25% random baseline for 4-choice questions) suggests either a prompt/extraction issue or that this particular 20-question sample is adversarial for Gemma. Published Gemma-2-2B scores are typically 40-50% on MMLU math subjects. This warrants investigation -- if Gemma's evaluation is broken, the "SOLE beats Gemma 6/6" claim is suspect.

**4. Peak memory vs steady-state conflation.**

The K3 comparison uses peak memory (10.98GB) which includes the merge-time overhead. The paper mentions this in Limitation 4 but the kill criteria table and analysis code use peak. If the kill criterion is about deployment memory (which is what matters for "runs on Apple Silicon"), steady-state post-merge (~5.35GB) is the right number. The kill still holds (5.35 > 2.45) but the 4.5x ratio is inflated. The honest ratio is ~2.2x.

**5. K1 threshold interpretation.**

K1 says "worse on >60% of benchmarks." With 6 benchmarks, >60% means >3.6, so 4 or more. SOLE is worse on exactly 4. The PAPER reports mmlu_math as one of the "worse" benchmarks, but SOLE scores 25% vs Qwen 35% -- that is a 10pp Qwen advantage, correctly counted. However, the K1 evidence string says "mmlu_math (+10pp Qwen wins but both low)" which is confusing notation. The code correctly implements `sole.get(b, 0) < qwen.get(b, 0)` which counts 25% < 35% as a loss. Verdict: K1 is correctly triggered.

## Kill Verdict Assessment

**K1: CORRECTLY KILLED.** SOLE loses 4/6 vs Qwen. Even accounting for sample noise on 2 borderline cases, the pattern is clear: Qwen dominates factual MMLU by large margins (25pp on medical/code). However, the statistical rigor is low and the GSM8K "win" may be an extraction artifact.

**K2: CORRECTLY KILLED, but the most important finding.** Math MMLU drops from 50% to 25% under uniform composition. This is a 5-question swing (10 correct -> 5 correct on n=20), which could be noise, but it replicates the prior finding from exp_task_accuracy_real_benchmarks that uniform composition hurts some MMLU domains. The legal drop (55% -> 45%, 2 questions) is within noise. K2 as "ANY benchmark worse" is a strict criterion -- even noise would trigger it. The criterion may be too strict for n=20 samples.

**K3: CORRECTLY KILLED, but the ratio is overstated.** Peak 10.98GB vs 2.45GB is 4.5x. Steady-state ~5.35GB vs 2.45GB is 2.2x. Either way, the ternary->bf16 unpacking penalty is real and K3 fails. The root cause (no native ternary inference kernel in MLX) is correctly identified as an engineering problem.

**Overall kill: JUSTIFIED.** The experiment asked "is BitNet-SOLE competitive with Qwen2.5-3B as a system?" and the answer is clearly no. All three kill criteria triggered for defensible reasons, even if the margins are noisier than the paper suggests.

## Extractable Learnings

The paper does a good job identifying what to learn from this kill. I would sharpen:

1. **The Qwen GSM8K anomaly must be investigated.** If Qwen's true score is 60%+, the "SOLE beats Qwen on reasoning" narrative collapses. Before citing this result in future work, run Qwen on GSM8K with proper answer extraction (or use lm-evaluation-harness).

2. **The Gemma comparison is the honest one.** Against a size-matched 2.6B model, SOLE wins or ties everywhere. Future framing should use Gemma as the primary comparison and Qwen as a stretch target.

3. **Uniform composition is harmful for some domains.** The K2 failure (math -25pp, legal -10pp) confirms that routing (not uniform mixing) is needed for production. This is consistent with the macro finding that "equal-weight composition is fragile."

4. **Memory efficiency requires native ternary kernels.** The bf16 unpacking penalty makes the architecture non-competitive on memory against any quantized dense model. This is Track D's critical problem.

5. **n=20 MMLU and n=50 GSM8K are insufficient for reliable benchmark comparisons.** Future competitive benchmarks should use at least n=100 per domain for MMLU and n=200+ for GSM8K to get confidence intervals under +/-10pp.

## Macro-Scale Risks (advisory)

Not applicable -- this experiment was a macro-readiness gate and it failed. The identified risks (base model quality, memory, uniform composition) are the correct blockers for Track A/D priority.

## Verdict

**PROCEED** (with the kill)

The kill is correctly triggered on all three criteria. The experimental methodology has real weaknesses (small n, possible Qwen extraction bug, peak vs steady-state memory conflation), but none of these would flip the overall verdict. The honest summary is:

- SOLE is NOT competitive with Qwen2.5-3B (correct kill)
- SOLE IS competitive with Gemma-2-2B (genuine positive finding)
- The GSM8K advantage over Qwen needs verification before being cited elsewhere
- The architecture's value proposition should be reframed around size-matched comparisons and modularity, not competing with instruction-tuned models 30% larger

No revisions needed -- the experiment is done. The learnings should be recorded in FINDINGS.md with appropriate caveats about statistical power.
