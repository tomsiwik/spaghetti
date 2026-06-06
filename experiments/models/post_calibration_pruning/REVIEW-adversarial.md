# Peer Review: Post-Calibration Pruning Safety

## NotebookLM Findings

Skipped -- the experiment is sufficiently contained (incremental validation of pipeline ordering) that the math and code review alone are adequate.

## Mathematical Soundness

### What holds

1. **Revival rate definition (Section 3.1)** is clean and correct: R(S_cal) = |D_0 \ D_{S_cal}| / |D_0|. Set difference over pre-calibration dead set. No hidden assumptions.

2. **Exact pruning theorem (Section 4.1)** is inherited from dead_capsule_pruning and is mathematically trivial for f=0 capsules under ReLU: relu(a_i^T x) = 0 for all profiled x implies the capsule contributes exactly zero. Removing it is lossless.

3. **Symmetric difference argument (Section 3.3)** correctly identifies that if R(S_cal) is small, then D_0 and D_{S_cal} differ by few elements, so profiling before or after calibration yields approximately the same pruning set.

4. **New death tracking (Section 3.4)** is a good addition. The paper correctly notes that Pipeline B captures newly dead capsules that Pipeline C misses, and vice versa for revived capsules. The net change formula (Delta_alive = revivals - new_deaths) is correct.

### What is weak but not wrong

1. **The "approximately sqrt(S_cal) scaling" claim (PAPER.md Finding 3)** is stated without derivation. Three data points (2.2%, 3.3%, 4.6% at S=50, 100, 200) are fitted to sqrt. With only 3 points, any monotone sublinear function would fit (log, power law, etc.). The paper does not claim this is a validated model, just an approximation, so this is an overstatement of pattern confidence rather than an error. Not blocking.

2. **The pipeline equivalence argument (Section 4.2)** conflates two distinct claims: (a) pruning the same dead set is lossless, and (b) pruning slightly different dead sets is approximately lossless. Claim (a) is exact. Claim (b) relies on the empirical observation that marginal capsules (those near the dead/alive boundary) contribute negligibly. This is true at micro scale where f_i transitions from 0 to small positive values, but the argument lacks a bound on the quality impact of erroneously pruning a capsule with f_i = epsilon > 0. The paper acknowledges this implicitly but does not formalize the risk.

3. **Assumption 3 (Section 8)** notes the revival_under_composition measurement used a different learning rate schedule (full fine-tune LR vs calibration at 0.1x LR). The paper correctly predicts this should reduce revival but then measures 3.3% vs 2.9% -- slightly higher, not lower. The discrepancy is within profiling noise (2.6-3.8% from Exp 12), so it is not alarming, but it does mean the LR assumption failed directionally. The paper waves this off with "does not significantly [differ]" but the honest interpretation is: the LR effect on revival is negligible or counteracted by other factors.

## Novelty Assessment

### Prior art

This experiment is purely incremental within the project's own lineage. It validates a third pipeline ordering for an already-validated mechanism (dead capsule pruning). The contribution is operational (flexibility in when to prune) rather than scientific (new mechanism).

In the broader literature, post-training pruning order is a well-studied problem in structured pruning (e.g., "when to prune" discussions in lottery ticket hypothesis work, Frankle & Carlin 2019). However, the specific context here -- pruning dead ReLU capsules in composed multi-domain models after calibration -- is sufficiently niche that no direct prior art applies. The finding that pruning order does not matter for truly dead (f=0) neurons is trivially expected from the exact pruning theorem and is not novel per se, but the empirical confirmation under composition + calibration dynamics is useful.

No references in `references/` directory address this specific question. No reinvention detected.

### Delta

The delta is: confirming that pipeline ordering is irrelevant for dead capsule pruning. This closes a practical question but does not advance the project's core research frontier. The paper correctly frames this as "closing the pruning pipeline chapter."

## Experimental Design

### Does this test what it claims?

Yes. The experiment directly compares four pipeline orderings (A, B, C, D) with shared pretraining and fine-tuning steps, diverging only at the pruning/calibration ordering. The kill criteria are tested exactly as stated.

### Controls

Adequate. Pipeline D (no pruning) serves as a sanity check confirming the exact pruning theorem (pruned and unpruned produce identical loss). Pipeline A is the validated baseline from prune_before_compose. Pipeline C is the original dead_capsule_pruning pipeline.

### Potential confounds

1. **Double profiling in Pipeline B (lines 188-199 + 217).** Pipeline B profiles the composed model twice: once before calibration (for revival measurement) and once after (for actual pruning). The first profiling pass involves a forward pass through the full validation set (20 batches x 32 samples). This does not change weights (no backprop), so it should not affect results. Confirmed correct.

2. **Seed matching.** All pipelines share the same pretrained base and fine-tuned domain models per seed. The pipelines diverge only at composition time. This is correct -- it isolates the pipeline ordering effect.

3. **Could a simpler explanation account for the result?** Yes, trivially. The exact pruning theorem guarantees zero quality change for f=0 capsules regardless of when you identify them. The only question is whether calibration changes which capsules are at f=0. The 3.3% revival rate shows it changes very few. This is the expected result from revival_under_composition. The experiment confirms rather than discovers.

### HYPOTHESES.yml consistency

The node `exp_post_calibration_pruning` has:
- Kill criteria: "post-calibration pruning degrades quality >2% vs pre-calibration pruning" and "revival rate after 100-step calibration >5%"
- These match exactly what the code tests (lines 450-456)
- Status is correctly set to `proven`
- Evidence claim accurately summarizes results

No issues.

## Macro-Scale Risks (advisory)

1. **SiLU/GELU base models have no dead capsules.** The macro_match experiment confirmed 0% dead capsules with SiLU. This entire pruning pipeline is ReLU-specific. At macro scale, if the base model uses SiLU (as most production models do), this work does not apply. The paper acknowledges this (Limitation 6) but it bears repeating: the pruning pipeline chapter is closed for ReLU only.

2. **200-step calibration already shows one seed at 5.9% revival.** At macro scale with longer calibration, higher learning rates, or more domains, revival could exceed 5% mean. The sub-linear scaling claim provides some comfort but is based on 3 data points at micro scale.

3. **N=2 domains only.** With N=20+ domains, the calibration gradient signals are more diverse, potentially increasing revival. The paper notes this (Limitation 1). The dimensional dilution argument (2x capsule dimension suppresses revival) actually weakens at higher N because capsule dimension grows as N*P while individual capsule contributions remain fixed -- but the ratio of "coupling effects" to "total dimension" decreases, which should continue to suppress revival. This needs macro verification.

## Verdict

**PROCEED**

This is a clean, well-designed incremental validation experiment. The math is sound, the experimental design properly isolates the variable of interest, the kill criteria are tested exactly as stated, and both pass with large margins (0.01% vs 2% threshold; 3.3% vs 5% threshold). The code correctly implements all four pipelines and the revival measurement.

The experiment does not advance the research frontier but closes a practical question definitively: pruning order does not matter for dead ReLU capsules under composition and calibration. The paper makes appropriately scoped claims and acknowledges limitations honestly.

The only note of caution: the "pruning pipeline chapter closed" framing should be understood as "closed for ReLU at N=2 micro scale." The macro applicability is limited by the SiLU reality (0% dead capsules in production architectures). This is acknowledged but worth emphasizing when updating FINDINGS.md.
