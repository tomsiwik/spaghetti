# Peer Review: Mixed-Rank Grassmannian Capacity

## NotebookLM Findings

Skipped (CLI not authenticated; analysis performed manually from source materials).

## Mathematical Soundness

### What holds

1. **Cross-rank coherence normalization.** The choice to normalize by `sqrt(min(r_i, r_j))` is mathematically well-motivated. The number of principal angles between Gr(r_i, d) and Gr(r_j, d) is `min(r_i, r_j)`, so the Frobenius norm of U_i^T U_j is bounded by `sqrt(min(r_i, r_j))`. The normalization correctly maps coherence to [0, 1].

2. **Block Gram matrix construction.** The implementation correctly builds the variable-block-size Gram matrix, enforces identity diagonal blocks, and applies structural + spectral projections. The code matches the math in MATH.md.

3. **Spectral projection.** Clipping to top-d non-negative eigenvalues and rescaling trace to `sum(r_i)` is correct for ensuring the Gram matrix remains realizable in R^d.

4. **Chordal distance formula.** `d_c^2(i,j) = min(r_i, r_j) - ||G_{ij}||_F^2` is the standard squared chordal distance generalized to mixed ranks. The normalization by `sqrt(min(r_i, r_j))` maps to [0, 1].

### What does not hold

**Issue 1: mu_factor mismatch between mixed and uniform baselines (BLOCKING).**

The mixed-rank AP uses `mu_factor=1.5` (line 424), while the uniform baselines use `mu_factor=1.2` (line 657). This means the mixed-rank AP has a 25% more permissive coherence target. The structural projection caps coherence at `mu_factor * welch_estimate`, so a higher mu_factor produces better-looking min_dist values (less aggressive capping allows the spectral projection more room to find good configurations).

This invalidates the K2 degradation comparison. The mixed-rank system is being given an easier target than the baseline it is compared against. A fair comparison requires identical mu_factor for both, or at minimum the paper must acknowledge and justify the difference.

**Issue 2: K2 baseline selection is biased (BLOCKING).**

The K2 degradation is computed by comparing each mixed-rank configuration against the uniform baseline at the "dominant rank" (line 771: `dominant_rank = max(rank_counts, key=rank_counts.get)`). For `mixed_heavy_low` (70% rank-4), this compares against uniform r=4, which has min_dist near 0.98-0.99. The resulting degradation appears as 18.5%.

But this comparison is misleading. The question is whether mixed ranks degrade packing quality relative to the best uniform alternative for the same total rank budget. The correct baseline is the uniform r_max=16 system, because that represents the constraint the skeleton actually faces. Using uniform r=4 as the baseline inflates the denominator, shrinking the apparent degradation.

A corrected comparison at N=20: mixed_equal min_dist=0.819 vs uniform r=16 min_dist=0.889 gives degradation = 1 - 0.819/0.889 = 7.9%. This happens to still pass K2, but the methodology is wrong in principle and could hide real problems at other operating points.

**Issue 3: Welch bound estimate is approximate, acknowledged but under-discussed.**

The formula `mu_welch_est = sqrt(r_max * (N*r_max - d) / (d * (N*r_max - d)))` substitutes `r_max` for all ranks. MATH.md correctly notes this overestimates the true bound, but the magnitude of overestimation is never quantified. For `mixed_heavy_low` with 70% rank-4, the overestimation could be substantial, making the mu_target unnecessarily loose.

### Hidden assumptions

- **Independence of AP convergence from rank distribution.** The convergence theory of Dhillon et al. (2008) applies to uniform-rank packings on a single Grassmannian. The mixed-rank AP operates on a product of Grassmannians. Empirical convergence is observed, but there is no theoretical guarantee that the alternating projection converges to a good packing (or converges at all) in the mixed-rank case. The paper acknowledges this in Limitation 3 but the gap is deeper than suggested: the structural projection operates on blocks of different sizes with different thresholds, breaking the symmetry that Dhillon et al. exploit.

## Novelty Assessment

**Prior art.** Mixed-rank subspace packing on Grassmannians is a natural extension of the uniform-rank problem. The Dhillon et al. (2008) paper focuses on uniform rank. I am not aware of published work specifically on AP for mixed-rank packings, so this is a reasonable micro-scale exploration.

**Delta over existing work.** The experiment extends the parent `grassmannian_expert_init` by generalizing the block Gram matrix to variable block sizes and adapting the structural projection threshold. This is an incremental but necessary step for the SOLE architecture.

**No reinvention detected.** The code builds naturally on the parent experiment's AP implementation.

## Experimental Design

### Does this test what it claims?

**Partially.** The experiment tests whether AP can produce geometrically well-separated mixed-rank subspaces. The kill criteria are reasonable and well-defined. However:

1. **The "5x conservative bound" framing is misleading.** The conservative bound d^2/r_max^2 = 16 assumes all 80 experts are rank 16, consuming 80*16 = 1280 dimensions in d=64 space. Of course having mostly rank-4 experts (using 4 dimensions each) allows packing many more. This is not evidence that the skeleton has higher capacity; it is evidence that the conservative bound is conservative. The paper acknowledges this ("because it assumes ALL experts have the maximum rank") but still headlines the 5x figure.

    A more informative metric would be: what fraction of the total dimensional budget `sum(r_i)` can be packed before degradation? At N=80, mixed_equal has sum(r_i) ~ 740 in d=64, which is 11.6x the ambient dimension. This is the actual achievement worth highlighting.

2. **The capacity is not actually found.** The sweep stops at N=80 and all distributions still pass. The effective capacity is reported as ">= 80" but the true N_max is unknown. The experiment cannot distinguish between N_max=81 and N_max=1000. To properly test K1, the sweep should continue until min_dist actually falls below the threshold.

3. **Only 3 seeds.** For a geometric experiment where random initialization matters, 3 seeds provides limited statistical power. The std_min_dist values (e.g., 0.006 for N=80) suggest the results are stable, but this should be explicitly tested.

### Controls adequate?

The uniform baselines at r=4, r=8, r=16 are good controls. The crowding analysis is well-designed (incrementally adding large-rank experts). However, the mu_factor mismatch (Issue 1) undermines the controlled comparison.

### Simpler explanation?

The min_dist plateau around 0.80 for N=10 to N=80 is suspicious. It suggests AP is converging to a configuration dominated by the r=16 pairs (which have min_dist ~ 0.83 at N=20 in the uniform case), and the small-rank experts are just living in orthogonal complement space without interfering. This is a simpler explanation than "small experts fit in gaps" -- they may simply be invisible to each other at d=64 with r=4 (probability of random 4-d subspace overlapping in 64-d space is very low). The interesting regime would be when N is large enough that even the rank-4 experts start interfering with each other, which at d=64 would be around N=256 for rank-4 alone.

## Macro-Scale Risks (advisory)

1. **Computational cost.** At d=4096 with N=500 mixed-rank experts, the Gram matrix size T = sum(r_i). For mixed_equal with average rank ~9.3, T ~ 4650. The O(T^3) spectral projection = O(10^11), taking minutes. The paper acknowledges this. For N > 1000, this becomes a real bottleneck.

2. **B-matrix dynamics dominate post-training.** The minimax packing experiment was killed because B-matrix training dynamics, not A-matrix geometry, control post-training coherence. This same finding likely applies here: the mixed-rank geometric advantage may be irrelevant after training.

3. **The normalization choice matters more at scale.** Normalizing by `min(r_i, r_j)` treats a rank-4 expert overlapping with a rank-16 expert the same as two rank-4 experts overlapping. But the rank-16 expert's larger B-matrix amplifies the interference in practice. The geometric analysis here cannot capture this asymmetry.

## Verdict

**REVISE**

The mechanism works in principle -- AP converges for mixed-rank systems and produces reasonable packings. The crowding analysis is insightful. But two methodological issues must be fixed before SUPPORTED status is warranted.

### Required fixes

1. **Fix mu_factor mismatch.** Either run the uniform baselines with `mu_factor=1.5` (same as mixed) or rerun mixed-rank AP with `mu_factor=1.2` (same as uniform). Both systems must use identical AP hyperparameters for the K2 degradation comparison to be valid. Recompute K2 degradation with the corrected baselines.

2. **Fix K2 baseline selection.** Compare mixed-rank packing quality against the uniform r_max=16 baseline at the same N, not the "dominant rank" baseline. The relevant question is whether mixing ranks degrades packing compared to the worst-case uniform component (r_max), not compared to the best-case component (r_min). The current methodology makes mixed_heavy_low look better than it is by comparing against uniform r=4.

### Non-blocking recommendations

3. **Extend the N sweep beyond 80** until at least one distribution shows min_dist falling below 0.3 (or to N=256, the theoretical limit for uniform r=4). The current ">= 80" effective capacity is a lower bound, not a measurement. This would make the capacity claim much stronger.

4. **Reframe the "5x conservative bound" claim.** The conservative bound is not a meaningful comparison target because it assumes all experts have r_max. Report instead: (a) the actual effective N_max (requires fix 3), and (b) the ratio of total rank budget sum(r_i) to d at the capacity limit.

5. **Add a note in PAPER.md** that the crowding analysis shows 10r4+10r16 (min_dist=0.802) is actually worse than all_r16 (min_dist=0.834). This means mixing ranks can degrade the r_max component's packing, not just be neutral. The paper currently frames mixing as uniformly beneficial ("small experts fit in gaps") without noting this cost.
