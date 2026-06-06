# Peer Review: Union-Find Expert Merging

## NotebookLM Findings

Skipped -- the experiment is self-killed with clear reasoning. Deep review is unnecessary for a negative result that the authors have already thoroughly diagnosed.

## Mathematical Soundness

**The math is correct and honestly presented.**

1. The union-find data structure implementation (Tarjan 1975) is textbook-correct: path compression in `find()`, union by rank in `union()`. The complexity claims (amortized inverse-Ackermann) are standard.

2. The similarity metrics (Jaccard, output correlation) are inherited from behavioral_dedup and correctly defined.

3. The merging rule (a-average, b-sum) is inherited from capsule_dedup and its rationale is sound: b-sum preserves the total output contribution when capsules fire on similar inputs.

4. The key mathematical insight is correctly identified: approximate similarity is NOT an equivalence relation. The non-transitivity argument is rigorous:
   - sim(A,B) > tau AND sim(B,C) > tau does NOT imply sim(A,C) > tau
   - This is a well-known property of metric spaces (triangle inequality gives a weaker bound than transitivity requires)
   - Union-find assumes the "similar enough to merge" relation is an equivalence relation, which requires transitivity

5. **One minor imprecision**: MATH.md states "Union-find implements single-linkage clustering" -- this is correct only when the union-find is built by iterating over all pairs above threshold, which is what the code does. Worth noting that the equivalence to single-linkage is specific to the construction algorithm, not to union-find itself.

**No hidden assumptions or errors found.** The assumptions are explicitly listed and honestly evaluated (one falsified, one partially true, one true-but-irrelevant).

## Novelty Assessment

**Low novelty, but that was known and accepted.**

This experiment tests whether a standard data structure (union-find) applied to a known problem (expert deduplication) with known metrics (behavioral Jaccard from behavioral_dedup) produces better results than greedy pairing. The answer is no, and the reason is a well-known clustering theory result (single-linkage chaining).

**Prior art**: The chaining problem of single-linkage clustering has been known since at least Jardine and Sibson (1971). The paper does not cite this specific lineage but does correctly identify the phenomenon and its cause.

**Delta over behavioral_dedup**: Negative. This experiment confirms that behavioral_dedup's greedy pairing is the correct abstraction. That confirmation has value as evidence in the hypothesis graph.

**No reinvention detected**: The code correctly reuses `profile_behavioral`, `compute_jaccard_matrix`, `compute_output_correlation_matrix` from behavioral_dedup and `merge_capsules` from capsule_dedup. Good engineering practice.

## Experimental Design

**The experiment tests exactly what it claims, with adequate controls.**

Strengths:

1. **Direct comparison at identical thresholds**: The `compare_uf_vs_greedy` function runs both methods on the same model with the same thresholds, isolating the clustering strategy as the only variable. This is a clean ablation.

2. **Multi-seed evaluation**: 3 seeds (42, 123, 7) with aggregated results. Sufficient for a micro-scale negative result.

3. **Threshold sweep**: 5 threshold settings tested, showing the failure is robust across the parameter space -- not a threshold tuning issue.

4. **Root cause analysis**: The per-layer cluster size data (Layer 0: 512 -> 33-255 capsules, max cluster 239-476) provides a mechanistic explanation, not just aggregate numbers. This is above average for micro experiments.

5. **Kill criteria are well-defined and honestly applied**: Both criteria (>3% quality loss, <20% compression without quality) are clearly exceeded.

Minor concerns:

1. **The kill criteria have a logical issue in PAPER.md's verdict table.** The text says "No threshold achieves both kill criteria simultaneously" but then says "Both kill criteria exceeded simultaneously." What is meant: no threshold achieves both PASS criteria (>=20% compression AND <=3% quality loss). The wording in the verdict table footer is confusing but the conclusion is correct.

2. **Profiling on validation data**: In `test_union_find_merge_pipeline`, the profiling uses `all_val` rather than training data. The full experiment in `run_full_experiment` correctly uses `all_train_ds` for profiling and `all_val_ds` for evaluation. The unit test has this inconsistency, but the reported results come from the full experiment, so this does not affect conclusions.

3. **The recovery directions are sensible but untested.** Cluster size caps and average-linkage are mentioned as potential fixes. These would be separate experiments; listing them here is appropriate.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry `exp_union_find_expert_merging` has:
- `status: killed` -- correct
- Kill criteria match PAPER.md: ">3% worse" and "<20% compression"
- Evidence entries accurately summarize the findings
- No downstream experiments are blocked by this kill (blocks: [])

The experiment cleanly removes a branch from the hypothesis tree and strengthens the case for greedy pairing (behavioral_dedup).

## Macro-Scale Risks (advisory)

Not applicable -- experiment is killed. However, the insight transfers to macro: any expert merging scheme at macro scale should avoid single-linkage / transitive closure. Use greedy pairing, complete-linkage, or average-linkage instead. Layer 0's high baseline similarity is likely a general phenomenon (early layers learn low-level features shared across domains), so layer-specific merge strategies may be needed at macro scale.

## Verdict

**PROCEED** (as a completed, killed experiment)

The experiment is correctly designed, correctly executed, and correctly killed. The negative result is informative: it eliminates transitive closure as a merging strategy and validates behavioral_dedup's greedy pairing as the correct abstraction. The math is sound, the code reuses existing components properly, the controls are adequate, and the root cause analysis (Layer 0 cluster explosion due to non-transitive similarity) is mechanistically clear.

No revisions needed. This is a well-executed negative result that advances the hypothesis graph.

One minor recommendation (not blocking): fix the ambiguous wording in PAPER.md line 81 where "No threshold achieves both kill criteria simultaneously" could be misread. The intended meaning is "no threshold satisfies both PASS conditions," but the sentence uses "kill criteria" where it means "success criteria."
