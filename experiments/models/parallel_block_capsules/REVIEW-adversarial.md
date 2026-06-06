# Peer Review: Parallel Block Capsule Composition

## NotebookLM Findings

Skipped -- manual deep review performed instead. The experiment is straightforward enough (A/B architecture comparison with shared protocol) that the mathematical and experimental analysis below is sufficient.

## Mathematical Soundness

### Interference chain analysis: mostly correct, one overstatement

The MATH.md argument that sequential blocks create a "depth-2 interference chain per layer" while parallel blocks create "depth-1" is directionally correct but imprecise.

**What holds:**
- The parallel block formulation `x_{l+1} = x_l + Attn(Norm(x_l)) + CapsulePool(Norm(x_l))` is correctly stated and matches the code (`ParallelCapsuleBlock.__call__`).
- The sequential formulation `h_l = x_l + Attn(Norm1(x_l)); x_{l+1} = h_l + CapsulePool(Norm2(h_l))` is correct and matches `SequentialCapsuleBlock.__call__`.
- The claim that in sequential blocks, capsule pool sees post-attention features (and thus composition interference propagates Attn -> CapsulePool within each layer) is correct.
- The claim that in parallel blocks, capsule pool is independent of attention *at the same layer* is correct.

**What is overstated:**
- The "2L vs L" interference chain depth characterization is misleading. In sequential blocks, the interference at layer l's capsule pool is not independent of the interference at layer l's attention -- they share the same residual stream from layer l-1. The correct framing is that sequential blocks have a *within-layer serial dependency* (attention output feeds into capsule input), while parallel blocks eliminate this dependency. But the *cross-layer* propagation is identical in both cases: both architectures propagate through the residual stream. The paper correctly notes this in Assumption 4 ("Shared attention remains the composition bottleneck in both architectures") but the "2L vs L" framing in the main analysis overstates the difference.

**Hidden assumption acknowledged:**
- The paper correctly acknowledges (point 1 in "The effect may be small because") that the dominant interference is from shared attention weights, which is unchanged. This is intellectually honest.

### Parameter count: verified correct

The MATH.md claim that parameter counts are identical is correct. RMSNorm in this codebase has zero learnable parameters (verified: it uses only the `__call__` computation `x * rsqrt(mean(x^2) + eps)` with no scale/bias). The code's `test_param_count_comparable` test confirms this.

### Computational cost analysis: minor issue

The claim of "~30-40% higher fine-tuning throughput" is attributed to "simpler computation graph (one norm call instead of two sequential ones)." This explanation is weak. Saving one RMSNorm call (O(d) per token) out of attention + capsule pool compute (O(d^2) per token) should yield <1% speedup from pure FLOP reduction, as the paper itself notes. The actual throughput improvement likely comes from MLX graph optimization (parallel branches can be scheduled differently than sequential ones), which is an implementation artifact, not an architectural property. The paper should be clearer that this speedup is MLX-specific and may not transfer to other frameworks.

## Novelty Assessment

### Prior art: well-grounded

- Parallel transformer blocks are from Cohere's Tiny Aya architecture (correctly cited).
- The reference `references/tiny-aya-parallel/` exists and is appropriately acknowledged.
- The `references/LLMs-from-scratch/ch05/15_tiny-aya/` from-scratch implementation exists as a reference baseline. The researcher built a custom implementation rather than reusing the reference code, but this is appropriate since the reference implements a full model while the experiment needs capsule-injected variants.

### Delta over existing work

The delta is narrow but valid: no one has tested whether parallel block architecture affects composition quality for independently-trained adapter modules. This is a legitimate micro-experiment question.

### Not a reinvention

The experiment does not reinvent existing code. It builds new `ParallelCapsuleBlock` and `SequentialCapsuleBlock` classes and reuses the existing `CapsulePool`, `CapsuleGroup`, `CausalSelfAttention`, and `RMSNorm` from the codebase. The composition protocol (`run_composition_experiment.py`) follows the established pattern from previous experiments.

## Experimental Design

### Does it test what it claims? Mostly yes, with caveats.

**Strengths:**
1. Clean A/B comparison: both conditions share the same composition protocol, hyperparameters, seeds, and data splits.
2. The sequential control is architecturally faithful to the existing capsule_moe (verified by code comparison with `capsule_moe.py`).
3. The composition protocol matches established practice: pretrain base -> freeze attention -> fine-tune capsules per domain -> compose by concatenation -> calibrate router -> evaluate.
4. Three seeds provide basic variance estimation.

**Weaknesses:**

1. **The effect size is smaller than the noise.** Mean improvement is -0.39pp, but per-seed standard deviation is ~1pp. The per-seed results tell the real story:
   - Seed 42: parallel is *worse* (+0.10pp)
   - Seed 123: parallel is *worse* (+0.48pp)
   - Seed 777: parallel is *better* (-1.74pp)

   Seed 777 is a massive outlier that drives the entire "parallel is better" narrative. Without seed 777, parallel is consistently (slightly) worse. **The paper reports mean and median as if they tell a coherent story, but the per-seed data shows no consistent direction.** The median (-1.33pp) is driven entirely by seed 777 being one of two data points below the median position. With n=3, the median is simply the middle value, which happens to be the parallel-favorable seed 123's gap of +0.02%.

   Wait -- re-examining the per-seed table more carefully:

   | Seed | Seq gap | Par gap | Difference |
   |------|---------|---------|------------|
   | 42   | +1.44%  | +1.54%  | +0.10pp    |
   | 123  | -0.46%  | +0.02%  | +0.48pp    |
   | 777  | +1.35%  | -0.39%  | -1.74pp    |

   The "Difference" column (par_gap - seq_gap) shows parallel is worse on 2/3 seeds and better on 1/3. The mean difference is (-1.74 + 0.48 + 0.10) / 3 = -0.39pp -- but this is not "parallel consistently better." It is "one seed had a large parallel advantage that happened to outweigh two seeds of small parallel disadvantage."

   **This is not evidence of a mechanism. It is noise.** The paper's claim of "-0.39pp mean improvement" is technically correct but misleading. A fair summary is: "no detectable difference between parallel and sequential composition quality at 3 seeds."

2. **No statistical test.** With 3 paired observations, a paired t-test or Wilcoxon signed-rank test would immediately show p >> 0.05 for the mean difference. The paper avoids reporting this.

3. **The kill criterion is very generous.** A 5% degradation threshold is easy to pass when both conditions produce similar results. The real question is whether parallel blocks are *better* (the theoretical prediction), and the evidence does not support that claim.

4. **No per-layer interference measurement.** The paper acknowledges this limitation. The theoretical argument (shorter interference chain) is the motivating hypothesis, but no diagnostic measures interference directly. Without measuring the mechanism, we only know the outcome (no degradation), not the cause. The outcome could equally be "parallel blocks don't matter" rather than "parallel blocks help by shortening the interference chain."

### Throughput measurement

The throughput comparison (115K vs 153K tok/s) is a secondary finding but is reported with only a single measurement per condition per seed (or possibly averaged). No variance is reported for throughput, and it is unclear whether the measurement accounts for warmup, compilation overhead, or other MLX-specific factors. This is a minor concern since throughput is advisory, not the primary hypothesis.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_parallel_block_capsules` has:
- Kill criterion: "parallel block composition degrades >5% vs sequential block composition"
- Status: proven
- Evidence: correctly reports the findings

The kill criterion is met (not triggered, so PASS is correct). The status "proven" is appropriate for a compatibility test -- the experiment proves parallel blocks are composition-compatible, not that they are composition-superior.

## Macro-Scale Risks (advisory)

1. **Gradient magnitude asymmetry.** In parallel blocks, the capsule pool gradient flows through `d(loss)/d(capsule_pool(norm(x)))` without any post-attention normalization gate. At macro scale with larger activations, this could create gradient scale mismatches between the attention and capsule branches. The sequential block's norm2 acts as an implicit gradient scaler.

2. **Tiny Aya's actual architecture includes features not tested here.** Tiny Aya uses CohereLayerNorm (mean-centering with learnable scale), not RMSNorm. It also dropped QK-Norm (noted in the reference README). The interaction between these design choices and composition has not been tested.

3. **The throughput advantage is MLX-specific.** On CUDA with PyTorch, the parallel block throughput advantage depends on whether the framework can actually execute the two branches concurrently. With modern PyTorch this is likely via CUDA graphs, but it is not guaranteed.

4. **The "shorter interference chain" argument becomes stronger at depth.** With L=4, the difference between 8 and 4 interference steps is modest. At L=36, the difference (72 vs 36) is more meaningful. This is a potential *upside* risk: parallel blocks may show larger composition benefits at scale than at micro.

## Verdict

**PROCEED**

The experiment cleanly tests its stated hypothesis (parallel blocks do not degrade composition quality >5%) and passes its kill criterion. The code is well-structured, the control is faithful, and the composition protocol matches established practice.

However, the following findings should be noted for intellectual honesty:

1. The claim of "-0.39pp mean improvement" overstates the evidence. The per-seed data shows 2/3 seeds where parallel is slightly worse and 1/3 where it is substantially better. The honest summary is "no detectable difference" rather than "parallel is better." The paper already hedges this in limitations ("per-seed variance ~1pp is comparable to the mean effect size"), but the main results section and HYPOTHESES.yml evidence present it as a positive finding.

2. The throughput advantage (~30%) is a real practical benefit but is MLX-specific and should not be generalized to other frameworks without testing.

3. The mechanistic explanation (shorter interference chain) remains untested. The experiment demonstrates outcome equivalence, not mechanistic advantage. This is fine for a compatibility test but should not be cited as evidence that parallel blocks *help* composition.

The experiment succeeds at what it set out to do: establish that parallel blocks are a viable architectural choice for capsule composition. It does not establish that they are superior. Proceeding to macro with the understanding that the architecture choice is a throughput/simplicity preference, not a quality advantage.
