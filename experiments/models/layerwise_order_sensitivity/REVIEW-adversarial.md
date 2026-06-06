# Peer Review: layerwise_order_sensitivity

## NotebookLM Findings

Skipped (experiment is a clean kill with no ambiguity requiring deep review).

## Mathematical Soundness

**The math is correct and well-structured.** Step-by-step verification:

1. **Synthetic expert construction (Sec. 1, code lines 60-154).** The alpha/beta decomposition `v = sqrt(cos)*shared + sqrt(1-cos)*unique` with unique orthogonalized against shared produces vectors with the target pairwise cosine. The construction is standard and correct. Each sublayer gets its own independent shared direction, which is appropriate for the stated goal.

2. **Weighted-average cosine formula (Sec. 4).** The claim that `cos_flat ~ (n_attn * cos_attn + n_ffn * cos_ffn) / (n_attn + n_ffn)` holds exactly is verified numerically (0.763). This is correct *because* the sublayers have equal dimension and the shared directions are independent across sublayers. The paper correctly notes this assumption but does not flag that the formula would break if sublayer dimensions differed (e.g., GQA). Minor gap, acknowledged in Sec. 8.2.

3. **Per-sublayer scaling law (Sec. 3).** The argument that `variation ~ f(N, d_s) * cos` has slope depending on (N, d_s) but not layer type is correct when N and d_s are held constant, which they are in this experiment (N=8, d_s=256 for all sublayers). This is the core theoretical contribution and it is sound.

4. **R-squared values of 0.882/0.884.** These are adequate for a 9-point sweep but noticeably lower than the parent experiment's ~0.99. The paper attributes this to per-sublayer independence reducing effective dimensionality. This explanation is plausible but the lower R-squared also means the linear model `variation = slope * cos` (no intercept) is a less perfect fit at the per-sublayer level. A nonlinear term or intercept might improve fit. Not blocking, but worth noting.

5. **Phase 2 sweep design (code lines 380-464).** Each cosine value is tested with the *other* layer type at cos=0.01 (near-orthogonal baseline). This isolates the effect of one layer type's cosine on its own order sensitivity, preventing crosstalk. Good experimental design.

**One hidden assumption:** The use of `variation_pct = (1 - cos_min) * 100` as the variation metric (rather than CV of norms or Frobenius variation) means the experiment measures worst-case directional disagreement across orderings. This is a defensible but specific choice. The parent experiment used the same metric, so the comparison is consistent.

**No errors found.**

## Novelty Assessment

This is an **incremental extension** of the parent `merge_order_dependence` experiment, which itself established the `variation ~ 80*cos` law. The contribution here is the per-sublayer decomposition showing that the slope is universal across layer types. This is a useful negative result that closes off a plausible hypothesis (layer-specific composition strategies).

No prior art was found that specifically decomposes GS order sensitivity by transformer layer type in the LoRA composition context. The finding is modest but novel within the project's scope.

The experiment correctly cites Golub & Van Loan for GS order dependence and the project's own `ffn_only_vs_all_modules` for the cosine measurements that motivated the hypothesis.

## Experimental Design

**Strengths:**

1. Three-phase design is well-structured: Phase 1 tests the main hypothesis, Phase 2 maps the scaling law, Phase 3 tests the flattening-masks-effects concern. Each phase addresses a distinct question.

2. Multi-seed (3 seeds) for Phase 1 with aggregate statistics. The ratio is 1.04 +/- 0.01, which is a tight enough confidence interval to support the kill.

3. Kill criteria are pre-registered in HYPOTHESES.yml and match what is tested. K2 is the relevant one (identical scaling), and the 1.01x slope ratio clearly triggers it.

**Weaknesses:**

1. **Phase 2 uses only seed=42.** The cosine sweep that produces the 61.9 vs 61.6 slope comparison has no multi-seed validation. Given the lower R-squared (~0.88), the slope estimates have meaningful uncertainty. However, the Phase 1 multi-seed result (ratio 1.04 +/- 0.01) already confirms identical scaling independently, so this is not blocking.

2. **Same ordering seed for sensitivity measurement.** `measure_order_sensitivity` uses the same `seed` for generating random orderings across all conditions. This means the *same 30 orderings* are tested for attention and FFN in Phase 1. This is actually a strength for paired comparison (reduces variance) but could be misleading if there is an ordering-specific artifact. With 30 orderings this is unlikely to matter.

3. **N_ORDERINGS=30 is modest.** The worst-case (min cosine) across 30 orderings may not capture the true worst case. However, 30 is sufficient for the linear fit and the ratio comparison, which are the actual kill criteria.

**Could a simpler mechanism explain the result?** Yes -- the result is explained by dimensional analysis alone. When N and d_s are identical for all sublayers, the GS process is geometrically identical up to the input cosine. The experiment confirms this obvious prediction, which is exactly why it is a clean kill. The value is in confirming the prediction rather than discovering something unexpected.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry for `exp_layerwise_order_sensitivity` correctly:
- Sets status to `killed`
- Records K2 as the kill criterion with the 1.01x ratio
- Notes that K1 passes (attention IS sensitive, just not differently from FFN)
- Lists evidence with date and source

The downstream node `exp_symmetric_gs_cost_benefit` correctly notes it depends on this experiment and is deprioritized by the kill.

No inconsistencies found.

## Macro-Scale Risks (advisory)

1. **Nonlinear amplification.** The paper correctly identifies this as the surviving question: softmax attention creates quadratic Q*K interactions that could amplify small attention-layer perturbations more than ReLU/SiLU amplifies FFN perturbations. Since SOLE uses simple averaging (not GS), this is moot for the current architecture, but would matter if GS were ever reintroduced.

2. **Heterogeneous sublayer dimensions.** Real architectures with GQA have smaller K/V dimensions than Q/O. The slope f(N, d_s) depends on d_s, so K/V sublayers would have slightly different slopes. At production cosines (<0.001), this is irrelevant.

3. **Correlated shared directions.** Real LoRA deltas likely have correlated structure across sublayers (e.g., attention heads learning related features to FFN gating). This would not change the per-sublayer scaling law but would change the flattened-vs-layerwise comparison. Not relevant since GS is not used.

None of these risks are blocking. The experiment's conclusion -- that layer-wise composition strategies are not motivated by order sensitivity -- holds under all plausible scaling scenarios.

## Verdict

**PROCEED**

This is a clean, well-executed negative result. The hypothesis was well-motivated (attention cosines ARE higher than FFN cosines at production scale), the experiment was properly designed to test it (controlled synthetic experts with per-layer-type cosines, cosine sweep, multi-seed), and the kill is unambiguous (slope ratio 1.01x, threshold was "identical" = within 5% of 1.0).

The experiment adds a useful datapoint to the project: order sensitivity is a function of cosine similarity alone, independent of layer type. This simplifies the SOLE architecture by eliminating the need for layer-wise composition strategies, which was a plausible design fork that can now be pruned.

No revisions needed. The kill is correctly recorded in HYPOTHESES.yml and FINDINGS.md.
