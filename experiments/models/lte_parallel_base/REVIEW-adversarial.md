# Peer Review: LTE Parallel Base Construction

## NotebookLM Findings

Skipped (experiment review performed by direct source analysis rather than NotebookLM pipeline, as the materials are all local markdown/code and the mathematical content is tractable by direct inspection).

## Mathematical Soundness

### What holds

1. **LoRA merge scaling is correct.** The parallel merge computes `(alpha/r) * (A @ B).T` and adds it to `fc.linear.weight`. The sequential merge calls `get_delta()` which returns `(alpha/r) * A @ B`, then transposes before adding. Both are consistent with the forward pass `x @ W.T + (alpha/r) * x @ A @ B`. Verified against the LoRALinear implementation in `lora_procrustes.py`.

2. **Rank bound (Section 2.2) is correct.** The average of K rank-r matrices has rank at most min(K*r, min(d_in, d_out)). Standard linear algebra.

3. **Cosine similarity computation is correct.** Flattened expert deltas, standard cosine. The random baseline expectation `sqrt(2/(pi*D))` at D=131,072 is correct.

4. **Effective rank via Shannon entropy of singular values is standard.** Implementation checks out.

5. **Kill criteria evaluation is correct.** par_base_ratio = par_val / cont_val, comparing against 1.20 threshold. The code matches the stated criteria.

### Issues found

1. **Compute fairness is materially violated (not just rounding).** MATH.md Section 2.5 states: "Total gradient steps are matched: Parallel: M * K * T = 2 * 4 * 50 = 400" vs "Sequential: total_steps = 500." This is a 20% difference, not an integer-rounding artifact. The paper frames it as "Note: not perfectly matched due to integer division" -- but 20% is substantive. Sequential gets 25% more gradient steps than parallel. This *favors sequential*, so the finding that parallel is "equivalent" is actually conservative (parallel achieves parity with fewer steps). This bias direction makes the equivalence claim stronger, not weaker, so it is **non-blocking** but should be stated honestly.

2. **MATH.md Section 2.2 has a notation inconsistency.** The document states `W' = W + (alpha/r) * (A @ B)^T` with `A in R^{d_in x r}, B in R^{r x d_out}`, which means `A @ B` is `(d_in, d_out)` and `(A @ B)^T` is `(d_out, d_in)`. But then the delta is written as `dW_k = (alpha/r) * B_k^T @ A_k^T` which equals `(d_out, d_in)`. This is consistent internally but departs from the standard LoRA convention (Hu et al. 2021) where `W' = W + BA` with `B in R^{d_out x r}, A in R^{r x d_in}`. The code uses `(A @ B).T` which is correct given the local convention, but the mismatch with standard notation is confusing. Non-blocking.

3. **Bootstrap CI with n=3 is unreliable.** The `ci()` function does 5000 bootstrap resamples from 3 data points. With only 3 values, bootstrap CIs collapse to the range of observed values (any resample is a combination of the 3 points). The stated CI of [0.80, 2.20] for par_vs_seq_cos is just the range of the 3 seed values. This is not a proper confidence interval -- it should be presented as "range across seeds" rather than "95% CI." Non-blocking for verdict, but the paper should not call these confidence intervals.

## Novelty Assessment

**Prior art:** The experiment directly builds on two published methods -- LTE (Hyeon-Woo et al. 2024) for parallel merging and ReLoRA (Lialin et al. 2023) for sequential merging. The novelty is not in the methods themselves but in the specific comparison: does the *base adaptation method* affect downstream *expert composition quality*? This particular question -- substrate equivalence for LoRA expert composition -- has not been asked in the literature.

**Delta over existing work:** Modest but well-targeted. The LTE paper shows parallel merging works for pretraining from scratch. The ReLoRA paper shows sequential merging works for pretraining. This experiment shows that for the SOLE-specific use case (adapting a pretrained base as a substrate for expert LoRA composition), the choice between them is an engineering decision, not a quality decision. This is a useful result within the SOLE research program.

**Reinvention check:** The researcher correctly uses the existing LoRALinear/LoRAGPT from `lora_procrustes.py` rather than reimplementing. The merge logic is custom but necessarily so (the upstream references do not provide MLX implementations).

**Important deviation from LTE paper:** The experiment uses reset-after-merge only, while the original LTE paper's main mode is no-reset with correction terms. The researcher documents that no-reset diverged at micro scale (forward-pass double-counting). This is a meaningful limitation honestly stated. The experiment tests a restricted variant of LTE, which is appropriate for the stated hypothesis but should not be generalized to claims about LTE itself.

## Experimental Design

### Strengths

1. **Clean 3-way design.** Parallel, sequential, and continued-conventional all start from the same pretrained base. The isolation of the single variable (parallel vs sequential branching) is well-done.

2. **Expert composition is the measured outcome, not just base quality.** The experiment goes beyond just comparing adapted bases -- it trains 4 domain experts on each substrate and measures cosine similarity and expert validation loss. This is directly relevant to the SOLE architecture.

3. **3 seeds with per-seed results.** All raw data is available. Results are consistent across seeds.

4. **Kill criteria are reasonable and well-calibrated.** K1 (>20% worse base), K2 (>2x compute), K3 (>5x worse cosine) are generous enough to avoid false kills at micro scale while still meaningful.

### Weaknesses

1. **Data overlap between heads is not true sharding.** The parallel heads use different RNG streams (`random.Random(seed + k * 7919)`) but draw from the same dataset pool. In the original LTE paper, data is explicitly partitioned across heads. With a small character-level names dataset, the random batches across heads will have substantial overlap. This weakens the "diverse subspace exploration" mechanism that is the theoretical advantage of parallel merging. At micro scale this is acceptable, but the paper should note this distinction.

2. **MLP-only LoRA.** Both adaptation phases and expert training use LoRA on `fc1` and `fc2` only (MLP layers). The SOLE architecture uses all-modules adapters (q/k/v/o/gate/up/down). The decision to use MLP-only LoRA was locked in from the `lora_procrustes` base implementation, not from experimental design. This is consistent with other micro experiments but means the result about substrate equivalence applies to MLP-only composition, not all-modules composition. Given that the core question is about merge protocol (parallel vs sequential), and attention weights would undergo the same merge mechanics, this is acceptable but should be flagged.

3. **Stale `results_seed_42.json`.** The standalone file shows `n_par_heads: 2, adapt_steps: 200, expert_steps: 100` while the aggregate JSON contains seed 42 data with `n_par_heads: 4, adapt_steps: 500, expert_steps: 300`. The standalone file was not overwritten by the final run. This is a data hygiene issue -- anyone reading the per-seed files directly would get results from a different experimental configuration than what is reported.

4. **"Continued conventional" control uses full-parameter training, not LoRA.** Phase 1c trains the base with full-parameter SGD for 500 steps, while Phase 1a/1b use LoRA (rank-8 out of d=64). This is a valid control (it asks "is LoRA adaptation worse than full fine-tuning as a substrate?") but it means the LoRA-vs-full-parameter variable is confounded with the parallel-vs-sequential variable. The continued condition cannot distinguish between "LoRA is a good regularizer" and "the merge protocol doesn't matter." For the stated hypothesis (parallel vs sequential equivalence), the parallel-vs-sequential head-to-head is the authoritative comparison, and the continued control is supplementary. The paper correctly focuses on the head-to-head.

### Does it test the hypothesis?

Yes. The hypothesis is that parallel merging produces "at least as good" a composition substrate as sequential. The evidence shows par_vs_seq_base = 1.007 (parallel 0.7% worse), par_vs_seq_cos = 1.46 (parallel has 46% higher cosines, but CI spans 1.0), par_vs_seq_loss = 1.006 (parallel experts 0.6% worse). All differences are within noise. The conclusion "equivalent substrate" is well-supported.

### Could a simpler mechanism explain the result?

Yes -- the most parsimonious explanation is that at d=64/r=8, the weight space is so small that any reasonable amount of LoRA training covers similar subspaces regardless of merge protocol. The paper acknowledges this explicitly (Limitation 1: "d=64 is too small for rank effects to matter"). This is not a flaw; it is the expected micro-scale behavior that motivates macro validation.

## Hypothesis Graph Consistency

The experiment matches `exp_lte_parallel_base_construction` in HYPOTHESES.yml. The three kill criteria (K1: >20% worse base, K2: >2x compute, K3: parallel interference) are exactly what the code tests. All three SURVIVE across all 3 seeds. The "proven" status is appropriate for the micro-scale claim.

Dependencies: depends on `exp_adapter_taxonomy_wild` (proven). No circular dependencies.

## Macro-Scale Risks (advisory)

1. **Data shard diversity may amplify differences.** With real domain data on K=8 or K=16 GPUs, parallel heads would see genuinely different distributions. This could either help (diverse subspaces) or hurt (conflicting gradients that average to near-zero). The micro experiment cannot distinguish these outcomes.

2. **No-reset mode is untested.** The theoretical advantage of LTE (continuous training with correction terms) was not testable at micro scale. At macro scale (d=4096, alpha/r << |W|), the double-counting issue may resolve itself and no-reset could outperform reset. This is the main open question for macro validation.

3. **Merge frequency scaling.** At 500 steps with K=4 heads, only 2 merge intervals are tested. Production training at 50K+ steps with K=8 heads would have hundreds of merge intervals. Stability over many merges is untested.

4. **Communication overhead.** The parallel "simulation" runs heads sequentially on one device. True multi-GPU LTE has synchronization costs (gathering K sets of LoRA parameters, averaging, broadcasting back). This may negate the wall-time advantage depending on network bandwidth.

## Verdict

**PROCEED**

The experiment is well-designed, the code is correct, the kill criteria are appropriate, and the conclusion (parallel and sequential produce equivalent composition substrates at micro scale) is well-supported by the evidence. The result is directionally useful for the SOLE architecture: the choice between LTE and ReLoRA for base construction is an engineering decision driven by GPU availability, not a quality decision.

Two non-blocking fixes recommended:

1. **Fix stale `results_seed_42.json`.** The standalone file contains results from a different experimental configuration (n_par_heads=2, adapt_steps=200) than the final run (n_par_heads=4, adapt_steps=500). Either regenerate the file from the correct run or delete it and direct readers to the aggregate JSON.

2. **Reframe "95% CI" as "range across 3 seeds".** Bootstrap CI with n=3 degenerates to the observed range. Calling it a confidence interval overstates the statistical precision. Replace "95% CI" with "3-seed range" throughout PAPER.md and MATH.md.
