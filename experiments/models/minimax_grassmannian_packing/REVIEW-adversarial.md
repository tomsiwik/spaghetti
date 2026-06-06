# Peer Review: Minimax Grassmannian Packing

## NotebookLM Findings

Skipped. The experiment is a self-kill with clean results. Deep review via NotebookLM would not change the verdict.

## Mathematical Soundness

### What holds

1. **Equidistribution claim is empirically verified.** The results.json data confirms max/mean ratios of 1.00x to float32 precision across all three dimensions and both seeds. For example, d=64: max=0.60302275, mean=0.60302269 (delta ~6e-8). This is genuine equidistribution, not approximate.

2. **Zero acceptance rate is real.** All six seed-dimension combinations show exactly 0 accepted moves out of 500. The minimax cosine arrays are bit-identical to the AP arrays (verified by comparing every element in results.json). This is the strongest possible null result.

3. **Post-training tail analysis is correct.** The decomposition of interference into skeleton (A_i^T A_j) and training (B_i, B_j) components in MATH.md Section 3 is algebraically sound. The bound ||delta_W_i^T delta_W_j||_F = (alpha/r)^2 * ||B_i^T (A_i^T A_j) B_j||_F correctly identifies B-matrix alignment as the uncontrolled variable.

4. **The Tropp et al. citation is appropriate.** AP convergence to equidistributed configurations for symmetric constraint sets is a known result. The claim that the AP fixed point is simultaneously mean-optimal and max-optimal for equidistributed configurations follows directly from mu_max = mu_mean = constant.

### What does not hold or needs qualification

1. **"Saddle-free local minimum" claim in MATH.md Section 4.2 is overstated.** The paper claims "the Hessian is positive definite in the tangent space at this point" but provides no derivation or citation for this. Zero acceptance in 500 random perturbation trials is evidence of a local minimum but does not establish positive definiteness of the Hessian. A flat region (zero Hessian) would also produce 0% acceptance for small perturbations. The distinction matters: a saddle-free minimum cannot be escaped by any local method, while a flat region could be escaped by larger perturbations or different search strategies.

2. **The claim "AP fixed point IS the minimax optimum" needs the qualifier "locally."** The paper acknowledges this in Limitations (Section 7.3) but the principal findings in the PAPER.md omit the qualifier in several places (e.g., "AP is already minimax-optimal" at line 139). At N=500+, AP may not achieve equidistribution, and the global optimum on the Grassmannian could differ.

3. **The Run 1 results in PAPER.md have no corresponding code.** The experiment code (`minimax_grassmannian_packing.py`) only implements Run 2 (stochastic refinement). The PAPER.md Table for Run 1 (modified AP structural projection, lines 47-60) reports specific numerical results (e.g., d=64: 0.035263 vs 0.041067, -16.5%) that are not reproducible from the provided code or results.json. Either the Run 1 code was deleted after execution or these results came from a separate script. This is a reproducibility gap -- not a fatal flaw, but the Run 1 results should be treated as unreproducible claims.

4. **d=256 post-training max/mean ratio discrepancy.** MATH.md Section 6 reports max/mean=3.83x for the worked example. PAPER.md reports 9.36x as the anomaly. The aggregate results.json shows 9.36x at d=256. However, seed 42 alone shows max/mean=3.75x while seed 137 shows 8.40x. The 9.36x figure appears to be the aggregate max (0.02162, from seed 137) divided by the aggregate mean (0.00231). The discrepancy between seeds (3.75x vs 8.40x) at d=256 means the tail anomaly is dominated by a single outlier pair in seed 137 (cosine=0.02162). With only 2 seeds and 28 pairs per seed, the max/mean ratio is driven by a single data point. The "9.36x anomaly" is real but fragile.

## Novelty Assessment

**Prior art:** The core finding -- that AP for Grassmannian packing produces equidistributed configurations -- is a known result from Tropp et al. (2005) and Dhillon et al. (2008). The experiment confirms this known property rather than discovering it.

**Delta over existing work:** The novel contribution is the diagnosis that the d=256 tail anomaly from the parent experiment (grassmannian_expert_init) is caused by B-matrix training dynamics, not skeleton geometry. This is a useful negative result that correctly redirects future work. The MATH.md Section 3.2-3.3 analysis of how B-matrix alignment amplifies skeleton overlap is a clean original derivation, even if the conclusion is "the skeleton is fine, look elsewhere."

**No reinvention detected.** The experiment correctly builds on the parent's code (imports from grassmannian_expert_init) rather than reimplementing AP.

## Experimental Design

### Strengths

1. **Clean null result.** The 0% acceptance rate across 6 conditions (3 dimensions x 2 seeds) is unambiguous. There is no p-hacking risk with a literal zero.

2. **Three-way comparison.** AP-standard, AP+minimax, and random-orthonormal baselines allow attributing effects correctly.

3. **Kill criteria are well-defined and honestly applied.** K1 (minimax max < AP max) fails trivially because the outputs are identical. The researcher correctly kills the hypothesis.

### Weaknesses

1. **Only greedy local search tested.** The paper acknowledges this (Section 7.3) but does not attempt simulated annealing, gradient-based Riemannian optimization (e.g., Manopt), or basin-hopping. While the equidistribution finding makes these unlikely to help, the claim "no method can improve on AP" is stronger than what was tested. The experiment tested "random local perturbations cannot improve on AP," which is a weaker but still useful statement.

2. **Same AP seed produces same skeleton across both seeds for same d.** Looking at the code (line 198): `rng_ap = np.random.RandomState(seed)`. Seeds 42 and 137 produce different AP skeletons. But the pre-training max coherences are identical across seeds for d=128 (0.3244429...) and d=256 (0.2264554...), suggesting the AP algorithm converges to the same fixed point regardless of initialization for these (N, r, d) parameters. This is actually evidence FOR the equidistribution claim but means the "2 seeds" provide less independent evidence than claimed for the pre-training results.

3. **The statistical test (Wilcoxon) is vacuous.** Since minimax and AP arrays are bit-identical, Wilcoxon returns NaN. The paper reports this honestly but could have omitted the test entirely and simply stated the arrays are identical. Including NaN p-values in results.json is mildly confusing.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry for `exp_minimax_grassmannian_packing` is correctly marked as `killed` with status date 2026-03-14. Kill criteria match:
- K1 (minimax max|cos| not lower than AP max|cos|): correctly assessed as KILLED
- K2 (compute within 2x): correctly assessed as PASS

The evidence strings accurately summarize the findings. The notes correctly identify the B-matrix alignment as the source of the tail anomaly and suggest training-time intervention rather than initialization optimization.

The experiment cleanly closes the open question "Minimax Grassmannian packing" listed in VISION.md (line 352).

## Macro-Scale Risks (advisory)

1. **Equidistribution may not hold at N=500+.** At micro N=12-40, AP achieves perfect equidistribution. At production N=500 with d=4096, r=16, the Nr/d ratio changes significantly. AP may produce non-equidistributed configurations where minimax refinement could find improvements. This is acknowledged in Limitations but bears repeating as a macro-scale risk.

2. **B-matrix tail control is the real open problem.** The experiment correctly identifies this but offers only candidate solutions (Section 5.3) without testing them. At macro scale with real domain data and converged training, the B-matrix overlap could be much larger than the micro toy-data results suggest. This should be the next experiment, not TAAP or other skeleton refinement.

3. **The 2.8x gap to Welch bound remains unexplained.** The experiment rules out minimax refinement as a way to close it. TAAP (exp_taap_convergence, still open) is the remaining candidate. At macro scale, this gap determines whether N_max capacity estimates are tight or conservative by ~8x (since capacity scales as 1/mu^2).

## Verdict

**PROCEED** (as a kill)

The kill is clean, well-evidenced, and correctly applied. The experiment achieves its purpose: definitively ruling out skeleton-level optimization as a solution to the d=256 tail anomaly. The principal finding (AP is locally minimax-optimal because it produces equidistributed configurations) is mathematically sound and empirically verified. The secondary finding (post-training tails come from B-matrix alignment, not skeleton geometry) is a valuable redirection.

Two non-blocking issues:

1. **Run 1 reproducibility.** The modified AP structural projection results (PAPER.md lines 47-60) should either have their code included in the experiment directory or be explicitly labeled as "from a prior iteration, code not preserved." Currently the reader cannot reproduce the Run 1 table.

2. **Hessian claim.** MATH.md Section 4.2 should either provide a derivation/citation for the positive-definite Hessian claim or weaken the statement to "the 0% acceptance rate across 500 perturbations is consistent with a local minimum of the L_inf objective."

Neither issue changes the verdict. The kill stands.
