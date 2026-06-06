# Peer Review: TTT Expert Selection

## NotebookLM Findings

Skipped -- the experiment is self-killed and the review focuses on verifying that kill and assessing whether the positive findings are correctly stated.

## Mathematical Soundness

### MATH.md Derivations

**Strategy 1 (Exhaustive Probe):** Correct. Score s_i = L_0 - L_i is a valid loss-reduction ranking. The cost of N+1 forward passes is correctly stated.

**Strategy 2 (Arrow Projection):** The formulation r_i = ||A_i^T h_bar||^2 / ||h_bar||^2 is mathematically valid as a projection energy measure. However, the theoretical justification is weak:

- The claim that "random-uniform A-matrix init means subspaces DO vary across adapters, providing some discriminative signal" is hand-waving. Random d-by-r matrices with d=2560, r=16 will have near-identical projection energies for any fixed input vector -- the Johnson-Lindenstrauss lemma tells us these projections concentrate. The experiment confirms this: 49% accuracy (near random at top-2/49, which is ~8% for exact match but ~51% for at-least-one-correct-in-top-2 by combinatorics on 49 items). This is roughly the expected concentration baseline, not "some discriminative signal."
- The MATH.md correctly identifies this weakness ("Projection relevance != loss reduction") but then dismisses it. The empirical result (49% accuracy) vindicates the concern.

**Strategy 3 (Hierarchical Probe):** The optimization C* = sqrt(2N) is correct. The conclusion that it cannot meet K3 for N=49 is valid.

**Strategy 4 (Hybrid):** Cost accounting is correct: 1 + 1 + m forward passes (hidden state extraction + base loss + m probes).

**Cosine Centroid:** Not in MATH.md but implemented in code. The method is straightforward nearest-centroid classification. No mathematical issues.

**Composition formula:** theta' = theta + (1/k) * sum_{i in S} Delta_i. The 1/k averaging is consistent with the N=50 experiment's routed composition protocol.

**Overhead analysis (K1):** The MATH.md analysis is somewhat confused -- it gives contradictory overhead calculations (0.2%, then 50%, then 25%). The empirical timing (0.22% for cosine centroid from Phase 6) is what matters and is directly measured.

### Issue: Forward pass count for hybrids

The hybrid strategies count "1 hidden-state + 1 base loss + m probes" but the hidden-state extraction and base loss computation each require a separate forward pass through the full model. The code confirms this: `arrow_projection_scoring` calls `model.model(x)` for hidden states, then `hybrid_arrow_probe` calls `compute_loss_on_tokens` for base loss (another `model(x)` call), then m more `compute_loss_on_tokens` calls. So hybrid arrow m=5 is 1+1+5=7, hybrid arrow m=3 is 1+1+3=5. This matches what's reported. Correct.

## Novelty Assessment

**Prior art:** The experiment correctly identifies MBC/Arrow-style projection scoring and L2R Gumbel-sigmoid routing as prior art. The cosine centroid approach is essentially a nearest-centroid classifier, which is textbook. The hybrid approach (cheap filter + expensive probe) is a standard cascade pattern.

**What's novel:** Testing these strategies on the specific ternary LoRA adapter setup with N=49 adapters on BitNet-2B-4T. This is an engineering evaluation, not a novel method. The finding that exhaustive loss-probing beats the learned router is a useful diagnostic.

**Delta over existing work:** The experiment does not claim methodological novelty, only empirical findings about what works for this specific architecture. This is appropriate.

## Experimental Design

### Critical Flaw 1: Single-sample evaluation per domain

The paper acknowledges this in Limitations (item 1): "Selection evaluated on first validation text only, not averaged over multiple samples. The learned router was evaluated on 10 samples per domain."

This is a significant design flaw, not just a limitation. With 1 sample per domain:
- The accuracy numbers have high variance (a single outlier text can flip a domain's correctness)
- The PPL comparison is between TTT strategies evaluated on 1 probe text selecting for up to 20 validation texts (in compute_ppl with max_batches=20) vs the router evaluated on 10 samples
- The "93.9% accuracy" for exhaustive probe means 46/49 domains had the self-adapter in the top-2. With 9 PPL=1.0 domains (guaranteed correct for any strategy), the real-data accuracy is 37/40 = 92.5%. Still high but noisier than it appears.

### Critical Flaw 2: Accuracy metric is questionable

"Correct" is defined as `name in selected` -- does the domain's own adapter appear in the top-2 selection? This measures self-selection, which is a proxy for quality. But the loss-probe strategy selects the top-2 by *loss reduction on the probe text*, not by identity matching. An adapter from a related domain (e.g., "science_qa" for "chemistry") could provide equal or better loss reduction. The accuracy metric penalizes potentially correct cross-domain selections.

Looking at the exhaustive probe selections: for "code", it selects ["coding_style", "code"]. For "math", it selects ["math", "cooking"]. The "cooking" selection for math is suspicious -- this is likely a noise artifact from single-sample evaluation. The actual PPL (3.68) is close to the router's (3.57), suggesting the second adapter matters little.

### Critical Flaw 3: PPL comparison is apples-to-oranges

The exhaustive probe's reported PPL (14.91) vs the router's (15.07) must be scrutinized. I verified the comparison:

- The exhaustive probe selects adapters using a 32-token prefix from the *first* validation text
- It then evaluates PPL on the validation data (up to 20 batches)
- The N=50 router selects using hidden states from evaluation texts (10 samples, up to 128 tokens each)

The probe's selection is based on the same text it's evaluated on (the first validation text is both the probe and part of the eval set). This is *not* a fair comparison -- the probe has seen the test distribution at selection time. The router generalizes from training-time hidden states. The 14.91 vs 15.07 comparison is therefore confounded by this data leakage.

**However:** The PPL is computed on `valid.jsonl` with up to 20 batches. The probe only uses the first 32 tokens of the first text. So the leakage is partial (32 tokens out of potentially thousands of evaluation tokens). The direction of the finding (loss-probing is strong) is likely robust, but the exact magnitude (14.91 vs 15.07) is unreliable.

### Design Strength: Comprehensive strategy comparison

Testing 7 strategies with consistent evaluation protocol is good experimental design. The ablation structure (arrow vs cosine, standalone vs hybrid, m=3 vs m=5) systematically explores the design space.

### Design Strength: The kill is honest

The experiment correctly applies its K2 criterion: best K3-valid strategy (15.27) fails to match the learned router (15.07). The self-kill is appropriate.

## Kill Criteria Assessment

### K1 (overhead <= 50%): PASS -- Correctly assessed

The cosine centroid at 0.22% overhead trivially passes. Even the hybrid strategies at ~1.2% overhead are well within bounds.

### K2 (avg PPL <= 15.07): FAIL -- Correctly assessed, with caveats

The 15.07 reference is the mean of the N=50 routed_ppls. I verified by summing the per-domain routed PPLs from the N=50 results.json: the values are consistent with a ~15.07 mean.

The 1.3% gap (15.27 vs 15.07) is genuinely marginal. But the comparison has confounds:
1. The probe uses 32 tokens vs the router's access to full sequences during training
2. Single-sample evaluation inflates variance on both sides
3. The probe's eval data partially overlaps with selection data (as noted above)

A more generous reading: the cosine centroid at 15.49 (2.8% gap) with zero training is the more meaningful comparison. This is a useful engineering finding even if it fails K2.

### K3 (<= 10 forward passes): PASS -- Correctly assessed

Hybrid cosine m=3 uses 5 forward passes. The counting is correct per the code analysis above.

### Missing hypothesis node

The experiment references kill criteria IDs 198-200 but these do not exist in HYPOTHESES.yml. The experiment has no formal hypothesis node in the graph. This is a process gap -- the experiment should have been registered before execution.

## Specific Findings Assessment

### "Loss-probing produces BETTER selections than the learned router" -- PARTIALLY VALID

The 14.91 vs 15.07 comparison is confounded by partial data leakage (probe text = first eval text). However, the 93.9% accuracy vs 86.3% is a cleaner comparison. The direction is likely correct: exhaustive loss-probing with O(N) forward passes should outperform a 2-layer router with 668K params, because it directly measures what the router only approximates. This is expected and not surprising.

### "Cosine centroid achieves 97% of router quality with zero training" -- VALID

15.49 / 15.07 = 1.028, so cosine centroid gets 97.2% of router quality (in terms of PPL ratio to some baseline). This is correctly stated. It's a nearest-centroid classifier using the same features (mean-pooled hidden states) the router trains on -- of course it works reasonably well for easily separable domains.

### "Arrow projection fails" -- VALID

49% accuracy at N=49 with top-2 selection is near the combinatorial baseline. The explanation (random A-init concentrates projection energies) is correct per JL-lemma arguments. This is a genuine negative result.

### "Routing at N=49 is fundamentally easy" -- OVERSTATED

The claim is overstated. 65% accuracy from cosine similarity does NOT mean routing is "fundamentally easy." It means the majority of domains are well-separated in hidden-state space (especially the 9 PPL=1.0 domains that are trivially classified). Removing those, accuracy drops to approximately 55-60% on real data. The 4/49 domains with 0% router accuracy in the N=50 experiment (chemistry, wikitext, dialogue, debate) show that some domains are genuinely hard to route.

## Macro-Scale Risks (advisory)

1. **N=500+ scaling:** At larger N, the cosine centroid's 65% accuracy will degrade as domain overlap increases. The hybrid strategy's O(m) probes are constant in N, but the cosine shortlisting becomes the bottleneck.

2. **Disk I/O dominance:** The Arrow strategy's 2.1s is dominated by adapter loading, not compute. At N=500, this makes per-adapter loading impractical. Pre-loading A-matrices or using memory-mapped files would be required.

3. **Prefix length sensitivity:** The 32-token prefix is short. At macro scale, ambiguous domains may need longer context windows, increasing probe cost.

4. **Composition interactions:** The single-adapter probing ignores pairwise interactions. At larger k (top-4 or top-8), these interactions become first-order effects.

## Process Issues

1. **No HYPOTHESES.yml node:** The experiment references IDs 198-200 that don't exist.
2. **Single sample per domain:** This is below the minimum for reliable accuracy estimates. Even 5 samples per domain would substantially reduce variance.
3. **The experiment is well-labeled as KILLED** with clear reasoning.

## Verdict

**PROCEED** (as a killed experiment with valid findings)

The self-kill on K2 is correct and honest. The experiment was well-executed within its constraints:

1. The exhaustive probe as oracle upper bound is a sound experimental design choice.
2. The strategy comparison is systematic and comprehensive.
3. The kill criteria are applied correctly.
4. The positive findings (cosine centroid as 97% quality with zero training, loss-probe as quality ceiling) are directionally valid even if the exact numbers have variance from single-sample evaluation.

The LEARNINGS.md correctly captures the practical implications without overclaiming.

**Non-blocking issues to document (not requiring re-execution):**

1. The "14.91 beats 15.07" claim has a partial data leakage confound (probe text overlaps eval text). Reword to acknowledge this. The accuracy comparison (93.9% vs 86.3%) is cleaner and should be the primary evidence.
2. "Routing at N=49 is fundamentally easy" should be softened to "routing at N=49 is easy for well-separated domains" with explicit acknowledgment that 9 PPL=1.0 domains inflate accuracy.
3. Register the hypothesis node in HYPOTHESES.yml with killed status and evidence, or explicitly note in LEARNINGS.md that the node was never registered.
4. The MATH.md overhead analysis section (lines 95-113) gives contradictory percentages -- clean up or note that empirical Phase 6 timing supersedes the theoretical estimates.
