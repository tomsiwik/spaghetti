# Peer Review: BitLoRA Ternary Adapter Composition

## NotebookLM Findings

Skipped -- NotebookLM not authenticated in this environment. Review proceeds from direct analysis of MATH.md, PAPER.md, code, and raw results.

## Mathematical Soundness

### What holds

1. **STE implementation is correct.** The `W + (W_q_np - W_np)` trick is a standard and correct STE implementation for autograd. The forward pass uses quantized values; the backward pass gives identity gradients through W. The paper honestly documents the initial silent failure of naive `np.round`/`np.clip` in autograd, which is a genuinely useful finding.

2. **BitNet absmean quantization is correctly implemented.** The `ternary_quantize_weight` function correctly computes `alpha = mean(|W|)` then rounds `W/alpha` to `{-1, 0, 1}`. This matches the BitNet b1.58 specification.

3. **INT4 quantization is correct.** Symmetric quantization with `scale = max(|W|) / 7`, clamping to `[-8, 7]`, yielding 16 levels. Standard approach.

4. **Kill criteria evaluation is arithmetically correct.** Verified K1 (individual ratio), K2 (composed PPL comparison), K3 (convergence ratio) against raw JSON data. The aggregation logic correctly computes per-seed results then means/stds across seeds.

### What does not hold or is misleading

5. **MATH.md Section 3.1 (Magnitude Boundedness) is wrong, and the paper acknowledges this.** The prediction that ternary adapters would have lower delta norm CV was falsified: ternary CV = 0.253 vs FP16 CV = 0.199, a 27% increase. The paper correctly identifies this and pivots the explanation to "directional decorrelation." This is honest, but the MATH.md section should be annotated as falsified rather than left standing as theoretical motivation.

6. **The "Johnson-Lindenstrauss analogy" (MATH.md Section 3.2) is hand-waving, not a derivation.** The claim that "discrete inner products concentrate more tightly around zero for random ternary vectors than continuous ones" is stated without proof. The JL lemma applies to random projections that preserve distances in target dimensionality. Trained LoRA adapters are not random projections -- they are optimized for specific objectives. The analogy is suggestive but provides no actual bound on pairwise cosine similarity under QAT. The observed 19% cosine reduction could be an artifact of the optimization landscape at d=64, r=4 rather than a fundamental property of ternary discretization.

7. **Section 3.3 (Composition PPL Bound) is incomplete.** The claim "the numerator's growth with N is bounded because integer matrices concentrate" has no derivation or citation. What concentration inequality applies here? The integer entries are in `[-r, r]` but they are not independent (they are products of correlated ternary vectors from QAT). This is a placeholder, not a bound.

8. **The worked example (Section 5) contradicts the main claim.** The MATH.md itself computes that ternary deltas could be LARGER than FP16 (3.0 vs 0.5), and notes "magnitude bounding is NOT automatic." This is correct and honest, but it means the entire theoretical motivation in Sections 3.1-3.3 is undermined from within the document.

### Statistical concerns

9. **The 4.4% composition improvement is not statistically tested.** The key result is ternary composed PPL = 4.159 (std=0.202) vs FP16 composed PPL = 4.348 (std=0.138). The difference is 0.190 PPL. With 3 seeds, a paired t-test gives t = 0.190 / sqrt((0.202^2 + 0.138^2)/3) approximately 0.190 / 0.141 = 1.35, p approximately 0.31. **This does not reach significance at p < 0.05.** The 95% CI for the difference includes zero. The paper reports this as a consistent finding ("consistent across all 3 seeds") which is true -- all 3 seeds show ternary better -- but 3/3 agreement with a small effect can easily arise by chance. This is the most important weakness in the paper.

10. **The cosine similarity reduction (19.3%) has similarly wide confidence intervals.** FP16 mean |cos| = 0.275 (std=0.028), ternary = 0.222 (std=0.032). With 3 seeds this is also not significant. The mechanism claim rests on statistically underpowered evidence.

11. **K1 is close to threshold.** The individual ratio is 1.026 +/- 0.009 vs threshold 1.05. The margin is 0.024, which is 2.7 standard deviations from threshold. This is adequately powered and passes cleanly.

## Novelty Assessment

### Prior art

The paper cites three relevant references:
- **BitNet b1.58** (2402.17764): Ternary base model architecture. Used correctly as the base quantization scheme.
- **LoTA-QAF** (2505.18724): Lossless ternary adaptation for quantized models. This is directly relevant prior art that the experiment acknowledges but does not compare against. LoTA-QAF uses a different training strategy (not STE-based QAT). The delta over LoTA-QAF is unclear.
- **MoTE** (2506.14435): Mixture of ternary experts. MoTE uses frozen shared base + ternary routed experts with top-k gating, which is exactly where the SOLE architecture is heading (per `exp_mote_sole_architecture`). MoTE already demonstrated ternary expert composition at real scale. The novelty delta of this micro experiment over MoTE is small -- the new finding is the "directional decorrelation" mechanism, but this was not the original hypothesis.

### What is genuinely novel

The observation that ternary QAT reduces pairwise cosine similarity between adapters (if it holds at scale) is a useful finding. The mechanism -- quantization noise as implicit directional regularization -- is plausible and has not been explicitly characterized in prior work. However, the evidence is statistically underpowered (see item 9).

The STE implementation note (autograd silent failure) is a genuinely useful pedagogical contribution for the autograd-based micro framework.

## Experimental Design

### Strengths

- **Three conditions (FP16, ternary, INT4) on the same base model.** This is a clean comparison. All conditions share the same ternary base, same data, same LoRA init seeds, same training hyperparameters. The only variable is adapter quantization during training.
- **Three seeds.** The minimum for variance estimation. Adequate for micro.
- **INT4 as intermediate condition.** Smart design choice. Allows testing whether the effect is monotonic with quantization aggressiveness. It is: INT4 is between FP16 and ternary on all metrics.
- **Pre-registered kill criteria.** K1, K2, K3 are stated upfront with clear thresholds.
- **Honest falsification of magnitude bounding hypothesis.** The paper does not hide the failed prediction.

### Weaknesses

12. **No train/test split for domains.** Training data and eval data are both generated by the same domain generators (lines 631-633). While different random samples, the distributions are identical. This is fine for PPL comparison across conditions (same contamination level), but it means the absolute PPL values are meaningless as held-out metrics. The paper does not claim otherwise, so this is non-blocking.

13. **All conditions use identical LoRA initialization per domain.** The seed `lora_seed = seed * 100 + di` is deterministic per domain index and master seed. This means all three conditions start from the same LoRA parameters and diverge only through the quantization during training. This is correct experimental design (controls for initialization), but it means the cosine similarity between conditions' adapters for the same domain is an artifact of shared initialization. The paper measures cosine between domains, not between conditions, so this is fine.

14. **The ternary base model is post-quantized, not natively trained.** The paper acknowledges this in Limitations. A natively-trained BitNet model (e.g., microsoft/bitnet-b1.58-2B-4T) would have different weight distributions and potentially different adapter dynamics. The post-quantization introduces information loss that may artificially inflate the "headroom" for adapters to recover quality, which is exactly what the prior experiment (bitnet_composition_stability) found: the composition benefit was partly "quantization recovery." The paper acknowledges this but does not control for it.

15. **Equal-weight 1/N composition is the weakest composition strategy.** At macro scale, this was KILLED (CV=112%, PPL in trillions). Testing ternary vs FP16 under a strategy that does not work at scale limits the transferability of the finding. The relevant question is whether ternary adapters compose better under PPL-probe weighted or top-k routing, not under equal-weight averaging.

16. **The "adapter_bits" calculation counts FP16 as 16 bits per parameter, not 32.** Looking at the code (line 590): `total_params * 16`. But the LoRA weights are stored as FP32 (`onp.float32` at line 285-286). The "FP16" label refers to the intended deployment precision, not the experimental precision. This is a naming inconsistency. The 9.7x compression ratio is computed as FP16 vs ternary, which is the correct comparison for deployment, but the experiment actually trains and evaluates in FP32 for all conditions. Non-blocking, but the PAPER.md should clarify.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node (`exp_bitnet_ternary_adapter_composition`) correctly:
- **K1** (individual quality < 1.05x): tested and passes.
- **K2** (ternary composition PPL < FP16 composition PPL): tested and passes, but without statistical significance testing.
- **K3** (convergence ratio < 2.0): tested and passes with wide margin.

The status is "supported" which is appropriate for a micro-scale result. The node correctly blocks `exp_mote_sole_architecture`.

The evidence statement in HYPOTHESES.yml is accurate and self-critical. No concerns here.

## Integration with VISION.md

The experiment directly advances Track A (BitNet-SOLE). The finding that ternary adapters compose better (if it holds) supports the thesis that ternary weight spaces are better suited for additive composition. This integrates cleanly with the architecture roadmap.

However, there is a tension: VISION.md's Track A motivates ternary bases as preventing "logit-scale mismatch that causes composition catastrophe." This experiment does not test logit-scale mismatch at all. It tests adapter-level cosine similarity and PPL. The connection between reduced pairwise cosine and reduced logit-scale mismatch is assumed, not demonstrated.

## Macro-Scale Risks (advisory)

1. **STE may not scale.** The mean quantization error for ternary is 57% (!). At r=4 with d=64, the model has few enough parameters that STE can correct for this. At r=16 with d=4096, the quantization error budget is tighter and STE may not provide sufficient gradient signal. The paper acknowledges this.

2. **The cosine reduction may be a small-d artifact.** At d=4096, random FP16 vectors are already near-orthogonal (cos approximately 1/sqrt(d) = 0.016). There may be no room for ternary quantization to further reduce cosine at high dimensionality. The decorrelation benefit may vanish where it is least needed.

3. **Equal-weight composition is dead at macro.** The finding needs to be replicated under PPL-probe routing or top-k selection. The relative advantage of ternary over FP16 under routing is unknown.

4. **MoTE already exists at scale.** The production architecture (exp_mote_sole_architecture) is directly inspired by MoTE. If MoTE's results transfer, this micro experiment may be redundant.

## Verdict

**PROCEED** -- with caveats

The experiment is well-designed within micro constraints, the code is correct, the kill criteria are clearly defined and honestly evaluated, and the paper is transparent about falsified predictions. The main results are directionally interesting: ternary QAT adapters show lower pairwise cosine similarity and modestly better composed PPL.

However, the statistical evidence is weak (3 seeds, p approximately 0.3 for the headline result), and the theoretical framework is incomplete (no actual concentration bound, JL analogy is hand-waving). The "4.4% better composed PPL" should be reported with its confidence interval, not as a point estimate.

The experiment earns PROCEED rather than REVISE because: (a) the status is already "supported" not "proven," (b) the micro constraints make 3 seeds reasonable, (c) the direction is consistent across all seeds even if not significant, and (d) the next experiment (exp_mote_sole_architecture) will test the architecture at a more relevant level.

**Non-blocking recommendations:**

1. Add a statistical test (paired t-test or Wilcoxon signed-rank) to the PAPER.md for the K2 result. Report the p-value and 95% CI. If the CI includes zero, state this explicitly.
2. Annotate MATH.md Sections 3.1 and 3.3 as "falsified" or "not validated" rather than leaving them as standing theoretical predictions.
3. Clarify in PAPER.md that the 9.7x compression ratio compares FP16 deployment precision to ternary, not the FP32 experimental precision.
4. The 57% mean quantization error for ternary adapters deserves more analysis. Is this error concentrated in A or B matrices? Does it correlate with which domains show better/worse composition? This diagnostic is already collected but not analyzed.
