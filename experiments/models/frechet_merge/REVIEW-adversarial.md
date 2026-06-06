# Peer Review: Frechet Merge

## Mathematical Soundness

### What holds

1. **Chordal Frechet mean derivation is correct.** The optimization problem (minimize sum of squared chordal distances) reduces to maximizing tr(P_mu * P_avg), which is solved by the top-r eigenspace of the averaged projection matrix. The proof in MATH.md Section 3.2 is textbook-correct (e.g., Absil et al. 2008, Chapter 3).

2. **Log/Exp map implementations are correct.** The code in `grassmannian_log` and `grassmannian_exp` follows the standard formulas from Edelman, Arias, Smith (1998). The test suite verifies the roundtrip property (Exp(Log(Y)) = Y), which passes at 1e-8 tolerance.

3. **Karcher flow with chordal warm-start is sound.** Initializing the geodesic mean with the chordal solution is a well-known technique that accelerates convergence. The convergence behavior in results.json (1-15 iterations) is reasonable.

4. **The chordal distance formula is correct:** d_chord^2 = r - ||U^T V||_F^2. The relationship to principal angles (sin^2 vs theta^2 for geodesic) is correctly stated.

### What requires scrutiny

5. **The subspace preservation metric is tautological for the chordal mean.** The metric is ||U_merged^T U_i||_F^2 / r, which equals tr(P_merged * P_i) / r. The chordal Frechet mean is *defined* as the subspace that maximizes sum tr(P_merged * P_i). Therefore, reporting that chordal beats naive on this metric is **mathematically guaranteed, not an empirical finding.** The PAPER.md acknowledges this partially in the K3 discussion ("chordal OUTPERFORMS geodesic on our preservation metric because it directly optimizes the chordal overlap we measure") but does not flag that the *same tautology* applies to the chordal-vs-naive comparison. The chordal mean will always beat any other subspace on this metric by definition. The "+5% to +34%" headline claim is just measuring how far the naive SVD-extracted subspace deviates from the optimum of the chordal objective -- it is not evidence that the merge produces better downstream models.

6. **Projection preservation tells a different story.** The code computes a second metric: ||P_merged @ delta_expert_i||_F / ||delta_expert_i||_F, which measures how much of each expert's full weight delta (not just A subspace) is captured. In the first result (d=64, N=2, random), naive projection preservation (0.816) *beats* chordal (0.799). This is because the naive method preserves B-weighted information, while the chordal method preserves A subspace geometry. The PAPER.md omits this finding at low N. The PAPER's Table for "Projection Preservation" only shows N=10 (where chordal does win), hiding the fact that the picture is mixed at N=2.

7. **Chordal distance formula has a minor notation inconsistency.** MATH.md Section 3.2 writes d_chord^2 = r - tr(P_S P_T), but Section 4.3 writes d_chord^2 = r - ||U^T V||_F^2. These are equivalent (since tr(P_S P_T) = ||U_S^T U_T||_F^2 for orthonormal frames), but the intermediate identity in the proof (d_chord^2 = r - ||P_S - P_T||_F^2 / 2) is actually wrong. Expanding ||P_S - P_T||_F^2 = tr(P_S) + tr(P_T) - 2 tr(P_S P_T) = 2r - 2 tr(P_S P_T), so r - ||P_S - P_T||_F^2 / 2 = r - r + tr(P_S P_T) = tr(P_S P_T), not the chordal distance squared. The correct chordal distance squared is r - tr(P_S P_T), and the proof's final step (maximizing tr(P_mu P_avg)) is correct regardless. The intermediate step is just a typo that does not affect the conclusion.

8. **The "advantage grows with N" claim needs qualification.** The advantage percentage (chordal - naive) / naive grows with N partly because the denominator (naive preservation) shrinks faster than the numerator gap. In absolute terms, the improvement shrinks: at d=256, the absolute preservation gain goes from 0.032 (N=2) to 0.023 (N=50). The relative gain looks impressive (+6% to +34%) but absolute quality degrades for both methods. At N=50, d=256, chordal preservation is 0.090 vs naive 0.067 -- both are close to the random baseline of r/d = 0.031. The merged subspace captures roughly 9% of each expert's information, which may be insufficient for downstream quality regardless of merge method.

### Hidden assumptions

9. **Random B matrices.** Correctly acknowledged in MATH.md Section 8. This is the key assumption -- if B matrices are correlated with A (as they would be in trained LoRA adapters), the naive addition's implicit B-weighting might be *desirable*, not a defect. The experiment cannot distinguish beneficial from harmful B-weighting.

10. **Square weight matrices (d_out = d).** The code sets d_out = d for simplicity. Real LoRA has d_out != d for many modules (e.g., A is (d, r) and B is (r, d_ff) for up/down projections). This does not affect the subspace analysis (which operates only on A), but it affects the B-projection reconstruction step and the projection preservation metric.

## Novelty Assessment

**The chordal Frechet mean on Grassmannian is well-established mathematics.** The core algorithm (average projection matrices, take top-r eigenvectors) appears in:
- Turaga et al. (2011), "Statistical Computations on Grassmann and Stiefel Manifolds for Image and Video-Based Recognition"
- Marrinan et al. (2014), "Finding the subspace mean or median to fit your need"
- Multiple computer vision papers on subspace averaging for face recognition

**What is novel here is the application to LoRA expert composition.** The connection between LoRA A matrices and points on the Grassmannian, and the use of Frechet mean as an alternative to naive delta summation, does not appear in the LoRA literature (LoRA Soups, Task Arithmetic, TIES-Merging, etc.). This is a genuine contribution.

**The B-projection reconstruction step (MATH.md Section 3.4) is the truly novel part** and is under-analyzed. The alignment step (merged^T @ U_i) @ B_i determines how much of each expert's output information is preserved. This is where the actual downstream impact lives, and it receives only 4 lines in MATH.md.

## Experimental Design

### Strengths

- The sweep across 5 dimensions and 5 expert counts is thorough for a micro experiment.
- Two expert regimes (random, AP-packed) properly test different interference levels.
- Two seeds provide minimal replication.
- The test suite verifies core mathematical properties (Log/Exp roundtrip, identical-subspace mean).
- Including the geodesic Karcher mean as a reference implementation is good practice.

### Weaknesses

11. **The primary metric is tautological** (see point 5). The experiment measures how well the chordal mean optimizes its own objective, not whether that objective correlates with downstream quality. A proper test would be: compose experts with both methods, then measure NTP loss on held-out domain data. Even at micro scale, this is feasible with the existing toy-data infrastructure.

12. **No downstream evaluation.** The experiment never measures whether the merged model actually works better. Subspace preservation and projection preservation are proxies. The PAPER.md acknowledges this as a limitation but the title claim ("merges experts better") implies downstream improvement, not geometric proxy improvement.

13. **The AP-packed regime is incomplete.** Large (d, N) configurations are skipped for runtime reasons (lines 592-596 in the code). The PAPER's Table 1 shows data for random regime only. AP-packed results are mentioned nowhere in PAPER.md despite being collected. Were they omitted because they showed no advantage? At AP-packed regime when N*r <= d, all experts are already orthogonal and all methods should produce identical results. This should be stated explicitly.

14. **Only 2 seeds.** For a purely numerical experiment with no training stochasticity, 2 seeds gives minimal confidence intervals. The results look stable, but this is partly because the metric is tautological -- the chordal mean is always optimal for its own objective regardless of seed.

15. **K2 is trivially satisfied.** The kill criterion asks whether Frechet merge adds >5% latency at *serving* time. Since both methods pre-merge to the same weight format, per-token cost is identical by construction. This kill criterion does not test anything meaningful -- it should have tested whether the one-time merge cost is acceptable (e.g., <1% of training time, or <10 seconds at production scale). As stated, K2 can never be killed.

16. **K3 reinterpretation is suspect.** The original kill criterion was "chordal approximation diverges significantly from geodesic exact mean." This was clearly intended to test whether the chordal approximation is a reliable proxy for the "true" Riemannian mean. The results show it diverges (up to 91% of max distance). The PAPER reinterprets this as "the chordal mean is better because it optimizes our metric" -- but this is circular reasoning. The chordal mean is better on the chordal metric by definition. The geodesic mean would be better on the geodesic metric by definition. The real question (which metric correlates with downstream quality?) is unanswered.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry (`exp_frechet_merge_vs_arithmetic`) lists three kill criteria:
- K1: quality within 1% of naive -- **properly tested, survives**
- K2: >5% serving latency -- **trivially survives (not a real test)**
- K3: chordal diverges from geodesic -- **killed in code, reinterpreted as positive in paper**

The results.json verdict is "MIXED" (K3 killed). The PAPER.md overrides this to "PROVEN" via reinterpretation. This is a judgment call. The reinterpretation has merit (the two methods optimize different objectives), but the original K3 criterion was meaningful and should be honestly registered as killed rather than retroactively reinterpreted.

Status should be "supported" not "proven" given: (a) primary metric is tautological, (b) K3 was killed, (c) no downstream validation.

## Macro-Scale Risks (advisory)

1. **The O(d^3) eigendecomposition at d=4096 takes ~0.5-2 seconds on GPU.** This is fine for one-time merges but may become a bottleneck if merges happen frequently (e.g., every time an expert is added to a dynamic system).

2. **Trained B matrices may eliminate the advantage.** If experts learn B matrices that are aligned with their A subspaces (likely, since gradient descent optimizes A and B jointly), the naive addition's implicit B-weighting becomes a feature. The chordal mean ignores B entirely when computing the merged subspace, which may discard useful structure.

3. **The absolute preservation at scale is concerning.** At N=50, d=256, preservation is 0.090. At N=500, d=4096, r=16, the expected preservation per expert would be roughly 16/4096 = 0.004 (close to random). At this point, any rank-r merged subspace captures negligible information from each expert. The Frechet mean is still optimal among rank-r subspaces, but the optimal may be insufficient.

4. **The reconstruction step (Section 3.4) is the critical path.** The B-projection quality determines whether downstream performance improves. This step was not tested in isolation. At macro scale, compare: (a) naive sum of A_i @ B_i, (b) Frechet-merged A with projected B, measuring downstream PPL on real tasks.

## Verdict

**REVISE**

The mathematical framework is sound and the application to LoRA composition is novel. However, the primary evidence is tautological (measuring a metric that the proposed method optimizes by definition), and the paper overclaims by calling the result "PROVEN" when K3 was killed and no downstream validation exists.

### Required fixes

1. **Acknowledge the tautology explicitly.** Add a section in PAPER.md stating: "The subspace preservation metric is the chordal objective itself. The chordal Frechet mean is optimal for this metric by construction. The empirical finding is not that chordal is better, but rather *how much* better than the naive heuristic, and whether the absolute preservation level is sufficient." This is honest and still supports the contribution.

2. **Report projection preservation results for all N, not just N=10.** The current PAPER.md Table shows projection preservation only at N=10 where chordal wins. At N=2, naive wins on this metric. Show the full picture or explain the discrepancy.

3. **Change status from "PROVEN" to "SUPPORTED".** K3 was killed. The reinterpretation is valid but does not change the fact that a pre-registered kill criterion fired. Changing a kill criterion after seeing results, even with justification, is post-hoc. Register the chordal-is-better-for-our-metric finding as a new observation, not a K3 override.

4. **Rewrite K2 or acknowledge it was trivial.** K2 (serving latency) can never be killed because pre-merged weights are identical by construction. Either acknowledge this was a vacuous criterion or reformulate it to test one-time merge cost (e.g., "merge computation exceeds 10% of single-expert training time at production scale").

5. **Add a micro-scale downstream test.** Even with the existing toy-data infrastructure, run a simple comparison: (a) compose 10 random experts via naive addition, measure NTP loss on held-out data; (b) compose via chordal Frechet merge, measure NTP loss. If the improvement is within noise, the geometric advantage does not translate to quality and the mechanism is purely theoretical. If it does translate, this becomes genuine evidence. This is feasible within the micro budget (one additional experiment, minutes of compute).

6. **State the AP-packed regime findings.** The code collects AP-packed results but PAPER.md ignores them. Either report them (expected: no advantage when N*r <= d, small advantage when N*r > d) or explain why they are omitted.
