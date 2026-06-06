# Peer Review: Batched LoRA Latency

## Mathematical Soundness

**FLOP Analysis: Correct.**
The theoretical overhead calculation is straightforward and sound:
- F_lora = k * L * T * 2 * r * d = k * 24 * 7 * 2 * 16 * 896 = k * 4,816,896
- F_base = 2 * 494M = 988M
- Overhead at k=2 = 0.98%

This is standard LoRA arithmetic and checks out.

**Base FLOP estimate: Approximate but reasonable.**
Using 2*P as the FLOP count is the standard "2 FLOPs per parameter per token" heuristic for transformer forward passes. The actual number varies with sequence length and attention patterns, but for a rough denominator this is fine.

**Memory analysis: Correct.**
18.4 MB per expert at fp32 for pre-extracted matrices. Arithmetic verified: 24 * 7 * 2 * 16 * 896 * 4 = 19,267,584 bytes = 18.375 MB. Matches results.json.

**No mathematical errors found.** The MATH.md derivations are clean.

## Experimental Design

**Strengths:**

1. **Clean controlled comparison.** Five approaches tested on the same model, same input, same hardware, with GPU synchronization for timing. This is good benchmarking discipline.

2. **Numerical equivalence check included.** The experiment doesn't just measure speed -- it verifies that the faster approaches produce equivalent outputs. Direct copy achieves exact match (0.0 diff). Stacked/persistent achieve near-exact (3.3e-05 from fp32 accumulation order).

3. **Batch size scaling measured.** Testing at batch=1,4,8 gives confidence the results aren't an artifact of a single configuration.

4. **Warmup included.** 10 warmup iterations before timing, 50 timing iterations. Adequate for stable GPU measurements.

**Weaknesses and Issues:**

1. **Hook-based approach (method D) has a max abs diff of 3.45 -- this is NOT a rounding error.** The PAPER.md hand-waves this as "hooks add delta on top of PEFT zero-weight path," but a max absolute difference of 3.45 in logit space is substantial. This means the hook-based approach (the one claimed as the fastest at 61% overhead) produces materially different outputs from the sequential reference. The paper buries this and instead highlights the 61% overhead number prominently. The numerically correct approaches (direct copy at 87%, stacked at 79%) should be the ones headlined.

2. **The "best measured overhead" of 61% is from a numerically INCORRECT method.** The paper's headline claim -- "61% overhead at k=2" -- comes from forward_direct_matmul, which has 3.45 max abs diff. The numerically correct alternatives achieve 71-87% overhead at k=2. This is still a significant improvement over 256% sequential, but the headline number is misleading.

3. **Stacked approach anomaly unexplained.** The stacked approach (method E) at k=1 shows 28% overhead despite running the base model forward with batch_size=1 (same as monolithic). The hooks shouldn't add delta when there's only one expert slice with zeroed LoRA weights already applied by PEFT. This suggests 28% of overhead comes purely from hook registration/removal per call (168 hooks). The persistent variant confirms this: it's at 31% for k=1, which is roughly the same. This means even the "persistent" hooks still add ~31% just from the 168 Python function invocations during the forward pass, before any LoRA computation.

4. **No variance reported.** The timing results are single mean values with no standard deviation or confidence intervals. With 50 iterations, the standard error should be reportable. Without it, we can't assess whether the differences between methods (e.g., stacked at 79% vs persistent at 71% at k=2) are statistically significant.

5. **Sequence length of 31 tokens is extremely short.** This biases toward exposing Python/framework overhead because the actual computation per forward pass is minimal. At longer sequence lengths (512, 1024, 2048), the base forward would dominate more and the percentage overhead from hooks would shrink. The paper acknowledges this in limitations but doesn't quantify the effect. Testing at 256 or 512 tokens would strengthen the case.

6. **Single GPU, single run.** Only tested on RTX A5000. No repeat of the full experiment. Acceptable for a directional result but the paper's tone is more definitive than warranted.

## Novelty Assessment

**This is an engineering benchmark, not a research contribution.** The paper correctly identifies this: "an engineering task, not a research problem." The core insight (LoRA deltas are additive low-rank perturbations that can be applied without full weight swapping) is well-known in the serving literature:

- **S-LoRA (Sheng et al., 2023):** Introduced batched LoRA serving with custom CUDA kernels, achieving near-zero overhead for multi-adapter serving. The paper cites this.
- **Punica (Chen et al., 2023):** GPU kernel for batched LoRA computation (BGMV kernel).
- **vLLM:** Production implementation of batched LoRA serving.

The experiment confirms what these systems already demonstrated: the overhead is implementation-bound, not architecture-bound. The value is in providing a concrete measurement for the project's specific architecture (Qwen2.5-0.5B with 7 LoRA targets), not in the mechanism itself.

**Delta over prior art: Low.** The result "Python hooks are slow, fused CUDA kernels are needed" is exactly what S-LoRA already showed. The experiment's contribution is confirming this for the project's specific setup.

## Hypothesis Graph Consistency

This experiment does not appear in HYPOTHESES.yml as a named node, which is appropriate -- it's a macro-scale engineering validation, not a hypothesis test. It successfully addresses the latency concern from the lora_moe_benchmark and correctly identifies the next step (fused CUDA kernels or torch.compile).

The falsification criterion stated in PAPER.md is clear and reasonable: "if batched LoRA application cannot achieve <100% overhead at k=2, the overhead is partially architectural." The numerically correct methods (direct copy at 87%, stacked at 79%, persistent at 71%) all pass this criterion.

## Key Concern: The 61% Headline

The paper's abstract claim is "hook-based approach achieves 61% overhead at k=2." But:

| Method | k=2 Overhead | Numerically Correct? |
|--------|-------------|---------------------|
| Hook-based (forward_direct_matmul) | 61% | NO (max diff = 3.45) |
| Persistent hooks | 71% | YES (max diff = 3.3e-05) |
| Stacked | 79% | YES (max diff = 3.3e-05) |
| Direct copy | 87% | YES (exact match) |

The honest headline should be: "Persistent hooks achieve 71% overhead at k=2 with near-exact numerical equivalence, a 3.6x reduction from the 256% sequential baseline."

The 61% number from the hook-based method is lower because it applies deltas on a zeroed-LoRA forward pass, meaning the PEFT LoRA path computes zeros while the hook adds the real delta. This creates a different computational graph than the sequential reference and produces different outputs. It's not just a "different accumulation order" -- it's computing a different function.

## Macro-Scale Risks (advisory)

1. **fp16/bf16 precision.** The experiment ran entirely in fp32. Production inference uses half precision. The additive delta approach may accumulate errors differently at lower precision. The paper correctly flags this as a kill criterion.

2. **KV cache interaction.** The stacked/persistent approaches run k copies through the model. With KV caching enabled (autoregressive generation), this multiplies memory by k. Not tested here.

3. **Attention masking with stacked batches.** Stacking k copies of the input creates a (k*batch, seq) tensor. Self-attention within each copy is correct, but if the model has any cross-batch attention patterns (unlikely in standard transformers but possible in some architectures), this could produce incorrect results.

4. **Router overhead not measured.** The benchmark times only the expert application, not the full pipeline including routing decision. At serving time, the router adds latency too.

## Verdict

**PROCEED** with revisions to the paper.

The experiment successfully demonstrates that the 256-489% sequential LoRA overhead is reducible to 71-87% with pure Python optimizations, confirming the overhead is implementation-bound. The methodology is sound and the measurements are properly GPU-synchronized.

However, the paper needs corrections before the results should be cited in downstream work:

1. **Retract the 61% headline.** The hook-based method (forward_direct_matmul) has a 3.45 max absolute logit difference from the reference -- this is a functional error, not a rounding artifact. The correct headline is 71% (persistent hooks) or 79% (stacked), both with 3.3e-05 max diff (true fp32 rounding).

2. **Add timing variance.** Report standard deviation or confidence intervals for all timing measurements. 50 iterations is enough to compute these.

3. **Test at seq_len >= 256.** The current 31-token input exaggerates framework overhead relative to computation. A single additional run at 256 tokens would significantly strengthen the argument that overhead decreases with more realistic workloads.

4. **Clarify the "theoretical 0.98%" is a lower bound that no Python implementation can approach.** The paper already says this but the juxtaposition of 0.98% and 61% in the same table invites misinterpretation. Consider separating the theoretical row or adding a footnote.
