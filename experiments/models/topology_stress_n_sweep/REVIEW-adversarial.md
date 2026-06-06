# Peer Review: Topology Stress N-Sweep

## Experiment Type
Guided exploration (Type 2) -- stated and correct. The proven framework is the Algebraic Stability Theorem (Cohen-Steiner et al., 2007). The unknown is empirical: at what N does perturbation norm cross the vulnerability threshold? The experiment narrows this unknown by showing it does not cross even at N=50.

## Hack Detector
- Fix count: 0. Single measurement sweep. Clean.
- Is MATH.md a proof or a description? Mixed. Theorem 1 is a trivial restatement of Cohen-Steiner Theorem 5.2 (one-line application). Theorem 2 is a one-step corollary. These are valid but not novel proofs -- they are the stability theorem restated for this notation. This is acceptable for Type 2 (guided exploration within a proven framework).
- Metric used as evidence: bottleneck distance, perturbation norms, lost feature count. These are direct topological measurements, not proxies. Appropriate.
- Kill criteria source: K634 derived from the stability theorem (feature loss when perturbation exceeds persistence). K635 is a monotonicity check on scaling behavior. Both are well-grounded.

## Self-Test Audit
1. One-sentence impossibility property: "topology is safe whenever perturbation norm stays below half the median persistence." -- Correct, single property.
2. Cited theorems: Cohen-Steiner et al. (2007) Theorem 5.2 is real and correctly applied. The precondition (pointwise correspondence between point clouds) is satisfied because base and composed weight matrices have identical row indices. PASS.
3. Predicted numbers: "Under 1/N averaging, perturbation norm DECREASES with N, so NO features should be lost at any N. Under additive composition, critical N ~ 1400." -- Specific and falsifiable. PASS.
4. Falsification condition: "If d_B exceeds max||delta_i|| (stability theorem violation = implementation bug)." -- This targets the proof. Also correctly notes the proof "cannot be falsified; it can only be vacuously true." Honest. PASS.
5. Hyperparameter count: 0 new. Uses existing adapter scale, rank, subsample. PASS.
6. Hack check: "Single measurement sweep, no fixes stacked." PASS.

## Mathematical Soundness

**Theorem 1:** Direct application of Cohen-Steiner et al. Theorem 5.2. The application is correct: the point cloud is W's rows, the perturbed point cloud is (W + Delta)'s rows, and the row-wise correspondence gives the Lipschitz bound. Valid.

**Theorem 2:** Setting the threshold at p_med/2 so that 2*delta = p_med is the standard vulnerability calculation. Valid.

**Scaling analysis (Cases 1-3):** The incoherent/coherent decomposition is standard. The 1/sqrt(N) decay under averaging with incoherent adapters follows from: ||sum_i X_i|| ~ sqrt(N)*||X_i|| for uncorrelated vectors, divided by N gives sqrt(N)/N = 1/sqrt(N). Correct.

**Critical N estimate:** N_crit = 5 * (30/(2*0.4))^2 ~ 5 * 56.25 ~ 281 for averaging. Wait -- let me recheck. MATH.md says "N_crit = 5 * (p_med / (2 * max||delta_5||))^2 ~ 5 * (30/4)^2 ~ 281". The formula uses max||delta_5|| from the N=5 data. But the actual measured max_delta at N=5 (averaging) is 0.88, not 0.4. And 2*0.88 = 1.76. So 30/(2*0.88) = 17, and 5 * 17^2 = 1445. There is a discrepancy: MATH.md uses "0.3-2.0" for max||delta_i|| and then computes with "30/4" which implies max||delta|| = 2.0. This is sloppy but directionally correct -- the conclusion (N_crit >> 50) holds regardless of which value is used.

**Minor issue:** The worked example (Step F) uses max_delta_norm = 0.41 for layer_0 q_proj, but the actual measured value at N=5 averaging is different (the per-module table shows 0.88 mean across all modules). This suggests the worked example was written before running the experiment using estimates from prior experiments. Not a serious problem since it is illustrative.

**Bottleneck distance computation:** The code uses `persim.bottleneck()` which implements the standard bottleneck distance algorithm. The `count_lost_features` function uses a greedy persistence matching with 50% tolerance -- this is an approximation, not the exact bottleneck matching. However, since the result is 0 losses everywhere, even an approximate detector finding 0 is meaningful.

**Concern: The stability bound is applied in the wrong direction for the conclusion.** The stability theorem says d_B <= max||delta||. The experiment confirms d_B << max||delta|| (bound is vacuously loose). PAPER.md then concludes "topology is radically robust." But the stability theorem only gives an UPPER bound. The fact that d_B is small does not prove topology is preserved -- it shows the bound is loose. However, the direct measurement of 0 lost features IS evidence of preservation, independent of the bound. The conclusion is valid but the reasoning chain through the stability theorem is unnecessary -- the direct measurement is what matters.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table with 5 predictions. Assessment:

| # | Prediction | Match | Comment |
|---|-----------|-------|---------|
| P1 | Norm decreases under averaging | YES | Measured: 0.88 -> 0.42. Consistent with 1/sqrt(N) |
| P2 | Additive grows ~sqrt(N) | PARTIAL | 4.4 -> 21.1 is 4.8x vs predicted 3.16x. Superlinear, indicating partial coherence |
| P3 | No features lost (averaging) | YES | 0 lost across all N |
| P4 | Features enter vulnerability window (additive) | YES | 191 features at N>=24 |
| P5 | Features in window may be lost | NO | 0 lost even with window 2.7x median persistence |

This is a well-constructed table. P5's failure is the key finding: the bound is vacuously loose. The experiment correctly identifies this as the main result rather than hiding it.

## NotebookLM Findings
Skipped -- the documents are straightforward enough that a deep review is not needed. The mathematical framework is a well-known theorem applied directly.

## Novelty Assessment

**Low novelty, high utility.** The Algebraic Stability Theorem is well-known (2007). Applying it to LoRA adapter composition is modestly novel but straightforward. The real value is the empirical finding: the bound is vacuously loose for low-rank perturbations, so topological preservation is "free" and does not need an active protection mechanism.

**Prior art:** arXiv:2410.11042 (persistent topological features in LLMs) is cited. The experiment correctly builds on 3 prior findings in the same track (#225, #228, #230).

**Impact on the project:** This is a NEGATIVE result in the best sense -- it proves that an entire research direction (pathway preservation) is solving a non-problem. This saves future compute by closing the track.

## Experimental Design Concerns

1. **Synthetic adapters (N>5).** This is the biggest weakness. The 45 synthetic adapters are sampled from a Gaussian matching mean/std of the 5 real adapters. Real adapters trained on genuinely different domains might have higher coherence (if domains share structure) or adversarial alignment. The Gaussian assumption likely UNDERESTIMATES coherence, making the experiment optimistic. However, even under additive composition (a much more extreme stress test than realistic composition), no features are lost, which provides a large safety margin.

2. **Row subsampling (500/2560).** Standard concern from parent experiments. The subsample is ~20% of rows, taken as a uniform grid. This could miss localized effects but is unlikely to systematically bias the result.

3. **Limitation 3 is the real gap.** PAPER.md acknowledges: "we have not shown that topological preservation implies behavioral quality preservation." This is critical. The entire pathway preservation track measures weight-space topology, but the connection to model behavior is assumed, not proven. The experiment could show 0 topological features lost while model quality degrades, if the features that matter for behavior are not the high-persistence ones being tracked.

4. **Feature loss detector.** The greedy matching algorithm (lines 272-313) uses a 50% tolerance threshold and sorted persistence values. This is reasonable but not the same as the formal bottleneck matching. Since the result is 0 losses, this does not affect the conclusion.

## Macro-Scale Risks (advisory)

1. The conclusion "averaging is self-stabilizing" holds under the assumption that adapters are roughly incoherent. At macro scale with hundreds of adapters, some subset might be highly correlated (e.g., multiple medical sub-specialties), creating localized coherence that the Gaussian synthetic model does not capture.

2. The stability theorem bound being 10x loose for low-rank perturbations suggests a tighter bound exploiting the rank-r structure could be derived. This would be a genuine theoretical contribution but is not needed for the practical conclusion.

3. The bigger risk: topology is the wrong metric entirely. If behavioral degradation occurs via mechanisms not captured by persistent homology of weight row point clouds (e.g., spectral changes, activation pattern shifts), then topological preservation guarantees nothing. This is acknowledged in Limitation 3 but should be front-and-center in any macro-scale planning.

## Verdict

**PROCEED**

Justification:

1. The experiment is cleanly designed as Type 2 (guided exploration within the Algebraic Stability Theorem framework).
2. The Self-Test is complete and honest.
3. The mathematical framework is correctly applied (trivially so -- it is a direct application of a known theorem).
4. The prediction-vs-measurement table is clear and shows 3/5 full matches, 1 partial, 1 informative failure.
5. The key conclusion -- averaging composition is self-stabilizing and topological feature loss is not a practical concern -- follows from the direct measurements (0 losses across 60 PH computations), not just from the bound.
6. The experiment correctly identifies itself as closing the pathway preservation research track, which is an efficient use of research resources.

Minor issues that do not block PROCEED:
- N_crit calculation in MATH.md uses inconsistent values for max||delta|| (Section C says "0.3-2.0", computation uses 2.0, actual measurement is 0.88). This is sloppy but does not affect the conclusion since all estimates give N_crit >> 50.
- The worked example (Step F) uses pre-experiment estimates that do not match actual measurements. Label it as "pre-experiment estimate" or update it.
- Limitation 3 (no behavioral validation of topology) is the real open question for the broader research program. The track closure is valid given the topology measurements, but the assumption that topology matters for behavior remains unproven.
