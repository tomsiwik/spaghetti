# Peer Review: adapter_pruning_lifecycle

## NotebookLM Findings

Skipped -- NotebookLM automation not configured in this environment. Review proceeds with manual deep analysis.

## Mathematical Soundness

### 1. LOO Approximation Is Tautological (Major)

The LOO "delta" computed in Phase 6 is not measuring quality impact of removing an adapter. It is measuring how far each adapter's oracle PPL deviates from the pool mean. The formula:

```
LOO(i) = mean(oracle_ppls for j != i) - mean(oracle_ppls for all j)
```

This simplifies algebraically to:

```
LOO(i) = (full_avg - oracle_ppl(i)) / (N - 1)
```

Proof: Let S = sum of all oracle PPLs, N = 24. Then full_avg = S/N. LOO_avg(i) = (S - oracle_ppl(i)) / (N-1). Delta = (S - oracle_ppl(i))/(N-1) - S/N = (N*S - N*oracle_ppl(i) - (N-1)*S) / (N*(N-1)) = (S - N*oracle_ppl(i)) / (N*(N-1)) = (full_avg - oracle_ppl(i)) / (N-1).

This means the LOO ranking is purely determined by oracle PPL magnitude. High-PPL adapters (legal at 14.66, finance at 14.01) get large negative deltas because their PPL is far above the mean. Low-PPL adapters (math at 2.38, music at 2.34) get positive deltas.

**The LOO strategy is therefore identical to "remove the 5 worst-performing adapters by oracle PPL."** This is fine as a pruning heuristic, but it is not Leave-One-Out in any meaningful sense -- it does not measure the impact of removal on the remaining adapters' quality. It measures nothing that a simple sort of oracle PPLs would not tell you. The PAPER.md and MATH.md present this as if it captures "end-to-end quality impact" -- it does not.

The actual LOO experiment would require: for each adapter i, build a composition of the remaining N-1 adapters, and evaluate all N-1 remaining domains under that composition. The key question is whether removing adapter i changes the composed PPL of the other domains (interference effects, routing redistribution). The approximation used here assumes zero cross-adapter interaction, which is exactly the conclusion it then claims to demonstrate.

### 2. K1 Threshold Is Circular (Moderate)

K1 asks: "Does any single removal cause >5% degradation?" But with 24 adapters, the maximum LOO delta is bounded by (max_ppl - mean_ppl) / (23 * mean_ppl). With oracle PPLs ranging 2.34 to 14.66 and mean 6.29:

max positive delta = (6.29 - 2.34) / 23 = 0.172 -> 2.73% of mean

This can never exceed 5% for any reasonable distribution of oracle PPLs at N=24. The K1 threshold is essentially unfalsifiable at this pool size. You would need a single adapter with oracle PPL of approximately -67 (impossible) for the positive delta to reach 5%. K1 PASS is guaranteed by construction.

### 3. Frobenius Norm Proof Is Correct (Sound)

The proof that ||BA^T||_F = ||B||_F when A has orthonormal columns is correct:
||BA^T||_F^2 = tr(AB^TBA^T) = tr(B^TBA^TA) = tr(B^TB) = ||B||_F^2.

### 4. Cross-Similarity Metric Mismatch (Minor)

MATH.md Section 4 derives the effective delta cosine similarity involving the A_i^T A_j cross-term. But the code (Phase 4) computes raw B-vector cosine similarity, ignoring the A-matrix cross-terms entirely. Since A_i^T A_j is near-zero by Grassmannian design, the true effective delta cosine would be even closer to zero. The B-matrix cosine is actually the more informative metric (it asks whether B-matrices learned similar patterns despite orthogonal projections), but the MATH.md presentation is misleading -- it derives one formula and the code implements something different.

## S1 Evaluation Methodology

### The Bug Fix Is Correct (Sound)

The original bug compared average oracle PPL of 19 remaining domains against average of all 24 -- trivially "improving" by removing high-PPL domains. The fix correctly:

1. Computes full-pool composed PPL for all 24 domains (Phase 9)
2. Computes pruned-pool composed PPL for the 19 remaining domains (Phase 8)
3. Compares the same 19 domains in both conditions

Code at line 1010-1016:
```python
remaining_domains = [d for d in ALL_DOMAINS if d not in prune_sets["loo_delta"]]
same_domain_full = [full_comp_ppls[d] for d in remaining_domains if d in full_comp_ppls]
same_domain_pruned = [loo_validated[d] for d in remaining_domains if d in loo_validated]
```

This is methodologically correct. The +0.43% delta on the same 19 domains is a fair comparison.

### Missing Statistical Rigor (Moderate)

The +0.43% average delta has no confidence interval or significance test. With 19 domains, individual deltas could vary widely. Looking at the validated PPLs, some domains degrade more than others:

- economics: 9.5455 -> 9.6647 (+1.25%)
- agriculture: 8.8321 -> 8.9349 (+1.16%)

While the average is +0.43%, individual domain degradation reaches 1.25%. No per-domain threshold is specified or tested. A paired t-test or bootstrap CI on per-domain deltas would strengthen the claim.

### Only LOO Strategy Is Validated (Acknowledged)

Only the LOO strategy receives actual composition validation. The other three strategies have only oracle-PPL-based estimates. This is acknowledged in the Limitations section, so it is not a blocking issue, but it means S1 PASS applies only to the LOO strategy.

## Experimental Design

### Uniform 1/N Composition Limits Generalizability (Major)

The `MultiAdapterLoRALinear.__call__` method (line 205) uses `self.scale / self.n_experts` -- uniform 1/N weighting. Under routed composition (where a softmax router selects one or few adapters per token), removing an adapter has a fundamentally different impact:

- **Uniform**: each adapter contributes 1/24 of the delta. Removing one shifts others from 1/24 to 1/23 (+4.3% each).
- **Routed**: the router selects one adapter per token. Removing an adapter means tokens from that domain get routed to the nearest neighbor. Quality impact depends entirely on how good the nearest neighbor is for those tokens.

The experiment claims (PAPER.md line 89-101) that the softmax router "handles absence" and "redistributes queries." But this is never tested. The validated composition uses uniform weighting throughout. The conclusion "pruning is safe" is supported only under uniform composition, which is not the deployment scenario described in VISION.md.

### Confounds in K2 (Minor)

K2 tests whether strategies select different adapters. They do (max Jaccard = 0.25). But given the tautological nature of the LOO metric (Section 1 above), the actual information content is: "sorting by oracle PPL, routing frequency, B-norm, and max-cosine produce different rankings." This is unsurprising because these are four different numbers. K2 PASS is trivially expected and provides minimal insight.

The more interesting K2 question would be: "Do different strategies lead to different quality outcomes when applied?" Only the LOO strategy is validated with actual composition, so this cannot be assessed.

### Cross-Similarity Strategy Has a Design Flaw (Minor)

The greedy cross-similarity pruning (lines 750-782) finds the adapter with highest max_cos to any remaining adapter and removes the one with lower delta magnitude from that pair. But since all cosines are below 0.09 (decorrelation holds), the strategy is essentially removing adapters based on delta magnitude with a noise perturbation from the cosine ranking. It is not meaningfully capturing "redundancy" when no adapter pair is redundant.

## Novelty Assessment

The experiment applies standard pruning concepts (LOO, magnitude pruning, routing frequency) to the specific context of Grassmannian-initialized LoRA adapter pools. The most novel finding is that B-matrix decorrelation holds (mean |cos| = 0.024), extending the Grassmannian guarantee from A-matrices to effective deltas. This is a genuine contribution.

However, the LOO metric is a known technique applied in a way that reduces to trivial oracle-PPL ranking. The LoRA-Hub and Unchosen Experts citations are appropriate but the experiment does not engage deeply with their methods -- LoRA-Hub's gradient-free composition weight optimization would be a more rigorous approach to identifying dispensable adapters than uniform composition.

## Macro-Scale Risks (advisory)

1. **Routed composition changes everything.** Under top-1 routing, removing an adapter means its domain's tokens go to the nearest neighbor. The quality impact depends on inter-domain transfer quality, which this experiment does not measure. Macro must validate pruning under routed composition.

2. **PPL vs. task accuracy.** The experiment acknowledges (PAPER.md Limitation 3) that PPL-benign pruning may degrade task performance. Macro should include downstream task evaluation.

3. **The 5 removed domains lose all coverage.** In production, a user querying about legal topics gets base-model-only quality. The experiment frames this as "remaining domains are unaffected" but the overall system quality drops for 5/24 of the domain space.

4. **Scale of adapter pool.** At 853 adapters (the memory budget limit), LOO via oracle-PPL-average would rank 853 adapters by oracle PPL and prune the worst. This is trivial and does not need the LOO framing. The more important question at scale is whether removing adapter i degrades adapter j's quality through composition interference -- which this experiment does not test.

## Verdict

**REVISE**

The core finding -- that 5/24 adapters can be removed with minimal impact on remaining domains under uniform composition -- is real and the S1 same-domain comparison fix is correct. The B-matrix decorrelation finding is genuinely valuable. However, several issues need addressing before the results can be trusted for downstream decisions:

### Required Fixes

1. **Reframe LOO as oracle-PPL ranking.** The current LOO framing implies it measures removal impact on remaining adapters. It does not -- it is algebraically equivalent to sorting by oracle PPL. Either (a) rename the strategy to "worst-oracle-PPL pruning" for honesty, or (b) implement true LOO with N-1 composition evaluation (expensive but what the math actually describes). The MATH.md Section 1 defines LOO correctly as PPL(P\{i}) - PPL(P), but the implementation approximates this with per-adapter oracle PPLs, collapsing it to a ranking by individual adapter quality.

2. **Acknowledge K1 is unfalsifiable.** Add a note that at N=24 with bounded PPLs, the 5% threshold cannot be reached by the positive LOO delta. K1 PASS is guaranteed by the pool size, not by the data. Either lower the threshold to something that could actually fail (e.g., 3%) or redefine K1 in terms of per-domain validated composition degradation.

3. **Add per-domain degradation bounds to S1.** Report not just average delta (+0.43%) but max per-domain delta and a bootstrap confidence interval. The current S1 could hide a domain with >2% degradation behind a favorable average.

4. **Add a brief note on uniform vs. routed generalizability.** The Limitations section mentions this but the paper's "Why It Works" section (line 99) claims "Router handles absence" without any evidence from this experiment. Either remove that claim or add a sentence noting it is hypothesized, not tested.

5. **Fix MATH.md Section 4 cross-similarity formula.** The derivation shows the formula with A_i^T A_j terms, but the code computes pure B-matrix cosine. State clearly that the code intentionally measures B-matrix similarity (which is the more informative metric given Grassmannian A-matrices), not the derived effective-delta cosine.
