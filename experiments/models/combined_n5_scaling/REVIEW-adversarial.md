# Peer Review: combined_n5_scaling

## NotebookLM Findings

Skipped (offline review). Manual deep review conducted below.

## Mathematical Soundness

**Derivations verified step-by-step:**

1. **Composition gap formula**: gap% = (L_composed - L_joint) / L_joint * 100. Correctly applied. From results.json: (0.5055 - 0.4893) / 0.4893 * 100 = 3.31%, matching the reported +3.32% (rounding from per-seed aggregation). Verified.

2. **Cross-architecture degradation**: (test_comp_mean - baseline_comp_mean) / baseline_comp_mean * 100 = (0.5055 - 0.4996) / 0.4996 * 100 = 1.19%. Verified in both code and results.json.

3. **Router capacity argument (D << d)**: 20 groups in R^64. The claim that 20 vectors in 64-dimensional space are "well-separated" is correct in the linear algebra sense -- 20 < 64 so the vectors can be linearly independent. However, this is a necessary condition, not sufficient; it depends on whether the learned representations actually cluster into separable regions. The empirical result (+3.32% gap, no catastrophic failures) validates this assumption *at micro scale*.

4. **Interference scaling sqrt(k)**: The claim that MLP output magnitude scales as sqrt(10/4) ~ 1.58x assumes uncorrelated group outputs. This is a reasonable first-order estimate. The paper acknowledges RMSNorm partially controls this. No hidden assumption issues here.

5. **Calibration budget scaling**: 200 steps round-robin over 5 domains = 40 batches/domain vs 50 at N=2. The paper correctly identifies this as slightly reduced but cites the prune_compose_n5 precedent (100 steps at N=5 sufficed). Sound reasoning.

**One minor issue in MATH.md**: The "Expected Gap at N=5" section estimates ~+3.0%, which turned out to be close to the actual +3.32%. However, the derivation chain has a gap: "baseline_N5_gap for seq_hybrid at N=5 is the primary unknown" and then it assumes seq_hybrid degrades at the same rate as standard capsule_moe. This is a reasonable heuristic, not a derivation. The paper does not claim it is rigorous, so this is acceptable.

**No mathematical errors found.**

## Novelty Assessment

This is a **scaling validation experiment**, not a novelty claim. The architecture (parallel blocks + all-linear GatedDeltaNet attention + capsule MoE) was validated at N=2 in the parent experiment (parallel_pure_linear_combined). This experiment tests whether the composition gap stays manageable at N=5.

**Prior art check**: No published work specifically tests composition gap scaling of parallel-block + all-linear-attention MoE architectures across domain counts. The closest is standard MoE scaling literature (Switch Transformers, Mixtral), but those use jointly-trained experts. The independent-composition protocol is the project's unique contribution, and scaling it to N=5 is a necessary validation step.

**References directory check**: The `references/` folder contains relevant works (mixtral-8x7b, switch-transformers, soft-moe, deepseek-v3) but none address post-hoc composition of independently-trained parallel-block experts. No reinvention detected.

**Delta over prior**: The experiment's contribution is empirical: it shows the architectural penalty (par_pure_linear vs seq_hybrid) is +1.19% at N=5, *smaller* than +1.48% at N=2. This falsifies the reviewer concern from the N=2 experiment that routing competition would amplify the interaction. This is a meaningful empirical finding.

## Experimental Design

**Does this test what it claims?** Yes. The hypothesis is "composition gap stays under 8% at N=5." The experiment measures exactly this with an appropriate baseline (seq_hybrid) and comparison to N=2 results.

**Controls:**
- The seq_hybrid baseline provides a within-experiment control for N=5-specific effects (both degrade ~2.5-3pp from N=2, confirming the degradation is protocol-driven, not architecture-driven).
- Joint training baseline per condition ensures gap is measured against the right reference. Each architecture type has its own joint baseline, preventing confounds from architecture quality differences. This is correct.

**Potential confounds and weaknesses:**

1. **3 seeds is thin.** The par_pure_linear gap std is 1.41% with a max of 4.83%. With only 3 seeds, the sample mean (3.32%) has substantial uncertainty. A rough 95% CI using t-distribution with 2 df: 3.32 +/- 4.303 * 1.41 / sqrt(3) = 3.32 +/- 3.50, giving [-0.18%, 6.82%]. The upper bound (6.82%) is still under 8%, so the kill criterion is robust even accounting for statistical uncertainty. However, the claim "architectural penalty does NOT amplify with N" rests on comparing +1.19% (N=5) vs +1.48% (N=2) -- a -0.29pp difference that is well within noise. The correct statement is "no evidence of amplification," not "amplification falsified." The PAPER.md uses "empirically falsified" which slightly overstates the evidence.

2. **Domain data splits are not independent.** All 5 domains are character-level name subsets split by first letter. The a-e domain (32.7% of data) vs u-z (7.4%) creates a size confound. The u-z gap of +5.09% may reflect calibration data starvation rather than an architectural issue. The paper acknowledges this in the limitations section, which is appropriate.

3. **Round-robin calibration with unequal domains.** The calibration uses step % n_ds for domain selection (line 141 of run_experiment.py). This gives exactly equal exposure to each domain regardless of domain size. For balanced calibration this is defensible, but it means the largest domain (a-e) is underrepresented relative to its data volume during calibration. This could mask a scenario where proportional calibration would perform differently. Not a flaw per se, but a design choice worth noting.

4. **Joint training also uses round-robin (line 178).** The joint baseline trains with step % n_ds, giving equal exposure per domain. This is a fair comparison since both joint and composed use the same domain balancing. If anything, the balanced approach might slightly disadvantage both conditions on the small u-z domain vs proportional sampling, but the comparison is internally consistent.

5. **No independent model registration.** The `CombinedN5ScalingGPT` is literally `pass` -- it inherits everything from `ParallelPureLinearCapsuleMoEGPT`. This is fine for a scaling validation (no architecture change expected) but means the model registry name is cosmetic.

**The experiment correctly tests its stated hypothesis.** The controls are adequate for the micro-scale context.

## Hypothesis Graph Consistency

- **Node**: `exp_combined_n5_scaling` in HYPOTHESES.yml
- **Kill criterion**: "N=5 composition gap >8% for parallel+pure-linear architecture"
- **Measured**: +3.32% mean gap
- **Status**: proven
- **Dependency**: `exp_parallel_pure_linear_combined` (proven). Dependency satisfied.
- **Evidence correctly recorded** with specific numbers and source reference.

The kill criterion is generous (8%) but justified by the parent experiment's margins and the experiment's exploratory nature. The PAPER.md correctly notes the 2.4x margin. Consistent.

## Macro-Scale Risks (advisory)

1. **Variance scaling.** The par_pure_linear gap std is 1.41% at N=5 (4x higher than seq_hybrid's 0.33%). If this variance grows with N, the tail risk at N=10+ could breach thresholds. The paper notes this. At macro scale with real diverse domains, variance could increase or decrease depending on domain separation quality.

2. **GatedDeltaNet state saturation.** At d_h=256 with 24+ all-linear layers, the recurrent state S in R^{d_h x d_h} might saturate. The hybrid 3:1 pattern provides periodic "state refresh" via full attention. Removing this at macro scale is the primary risk of the pure-linear choice. The paper correctly identifies this.

3. **Per-domain gap dispersion.** The u-z domain at +5.09% shows calibration sensitivity to domain size. Real-world domains with limited data (e.g., a niche programming language) could show similar or worse gaps. Proportional calibration sampling might help.

4. **The cross-architecture degradation shrinking from +1.48% to +1.19% should not be interpreted as a trend.** It is a single data point with wide confidence intervals. Do not extrapolate that the penalty *decreases* with N.

## Verdict

**PROCEED**

The experiment is well-designed, the math is sound, and the results are clear. The composition gap of +3.32% is well within the 8% threshold with robust margin even after accounting for statistical uncertainty (upper 95% CI ~6.8%). The experimental code correctly implements the stated protocol. The kill criterion in HYPOTHESES.yml matches what was actually tested. The limitations are honestly acknowledged.

Minor items that do not block PROCEED:

1. The PAPER.md claim that the reviewer's concern about routing competition is "empirically falsified" (Finding 2) should be softened to "no evidence of amplification observed" given the -0.29pp difference is within noise on 3 seeds. This is a presentation issue, not a scientific error.

2. Future experiments testing N=8+ should use more seeds (5+) and consider proportional calibration sampling to reduce variance in small-domain gaps.
