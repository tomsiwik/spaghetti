# Peer Review: wanda_structured_macro

## NotebookLM Findings

Skipped -- the experiment is already killed and the evidence is unambiguous. A deep review would not change the verdict.

## Mathematical Soundness

**Derivations are correct.** The structured Wanda adaptation (Section 2.4 of MATH.md) is a natural extension: replace per-weight magnitude with per-neuron L2 norm, replace per-input activation with mean activation across calibration data. The math for why this degenerates when weight norms are uniform (Section 3.2) is sound:

If `||W_j||_2 ~ C` for all j, then `S_wanda(j) = C * S_act(j)`, which preserves the activation-only ranking. The empirical Spearman rho of 0.974 confirms this algebraically.

**One subtle issue the paper gets right:** The non-monotonic calibration sweep (more data = worse results) is correctly diagnosed. When the scoring signal is an anti-signal, more accurate estimation of that signal makes the anti-signal stronger. This is a genuinely useful insight for future pruning work.

**No hidden assumptions found.** The experiment is straightforward -- compute scores, rank, prune, measure perplexity. The Spearman correlation implementation (manual, without scipy) is correct: it uses the standard Pearson correlation of ranks.

## Novelty Assessment

**Low novelty, but that is acceptable.** This is a hypothesis test, not a novel method. The question was: "Does adding weight norms fix the specialist problem?" The answer is no. The contribution is the negative result plus the root cause analysis (weight norm uniformity in Qwen2.5-0.5B, CV ~6%).

**Prior art is properly cited.** Sun et al. (2023) Wanda paper is referenced. The key distinction -- unstructured (per-weight) vs structured (per-neuron) -- is clearly stated and is the central reason for failure.

**One gap in prior art discussion:** The paper does not reference SparseGPT (Frantar & Alistarh, 2023), which also does structured pruning and could provide context for why structured approaches struggle with SwiGLU. However, this is minor for a killed experiment.

## Experimental Design

**Strengths:**

1. Clean four-way comparison (Wanda, activation-only, weight-only, random) at identical neuron counts. This is good experimental design.
2. The random baseline uses 3 seeds with mean and std, providing a credible control.
3. Multi-level pruning sweep (5% to 30%) shows the degradation curve.
4. Calibration sweep directly tests kill criterion 2.
5. Rank correlation analysis provides mechanistic explanation, not just outcome measurement.

**Weaknesses:**

1. **Missing random comparison at 5% pruning.** At 5%, Wanda achieves PPL 32.63, which is only +53% above baseline. Without knowing what random achieves at 5%, we cannot assess whether Wanda is better or worse than random at low pruning fractions. The PAPER.md notes this is "much closer to random" but provides no data. This is a missed opportunity -- if Wanda beats random at low fractions but loses at high fractions, that would indicate a threshold effect worth investigating.

2. **Per-layer pruning distribution is highly non-uniform.** The results.json shows Wanda pruning 3453 neurons from layer 0, 2882 from layer 1, but 0 from layers 19-23. This extreme layer imbalance (70% of pruning concentrated in layers 0-1 and 8-15) is a confound. A per-layer-balanced random baseline would better isolate whether the scoring signal or the layer concentration is the problem. The paper does not discuss this.

3. **No per-layer random baseline.** The random pruning selects uniformly across all 116,736 neurons, distributing ~768 per layer on expectation. Wanda concentrates pruning in early layers. These are testing fundamentally different pruning geometries. The comparison conflates "which neurons to prune" with "which layers to prune from."

4. **KC2 threshold is arbitrary.** The kill criterion for calibration efficiency uses 20% stability threshold, which the paper acknowledges is "somewhat arbitrary." This is acceptable since KC1 already kills the experiment decisively.

## Macro-Scale Risks (advisory)

Not applicable -- the experiment is killed. For future reference:

1. If a variant (max-based, variance-based) is attempted, it should include per-layer-balanced random as an additional control.
2. Weight norm uniformity may be specific to Qwen2.5-0.5B. Other model families (Llama, Mistral) may have higher weight norm variance, making Wanda viable. This should be checked before generalizing the negative result.
3. The 15.8% pruning fraction is aggressive for zero-shot structured pruning. Lower fractions (2-5%) with random baselines would better isolate the scoring signal from catastrophic capacity loss.

## Verdict

**PROCEED** (with the kill)

The experiment is correctly killed. Both kill criteria are clearly violated:
- KC1: Elevation ratio 8.78 vs threshold 0.5 (17.6x worse than needed)
- KC2: Calibration instability 36.2% vs threshold 20%, and non-monotonic

The root cause analysis is thorough and mechanistically sound. Weight norm uniformity (CV ~6%) in Qwen2.5-0.5B causes Wanda to degenerate to activation-only scoring (rho = 0.974). The non-monotonic calibration sweep provides additional evidence that this is a fundamental signal problem, not a noise/sample-size issue.

**Two non-blocking suggestions for the paper:**

1. Add random pruning results at 5% and 10% fractions to complete the multi-level comparison. The current gap (Wanda PPL at 5%/10%/15%/20%/30% but random only at 15.8%) weakens the multi-level analysis.
2. Discuss the per-layer pruning concentration effect. Wanda concentrates 34% of pruning in layers 0-1 while random distributes uniformly. This confound does not change the kill verdict but matters for understanding what a better structured pruning approach would need to handle.
