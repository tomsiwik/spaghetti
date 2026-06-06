# Peer Review: Channel Capacity Bound (v2 -- Post-Revision)

## NotebookLM Findings

Skipped -- this is a purely analytical experiment (curve fitting + validation) with no trained model architecture. The math and experimental design are fully tractable by direct inspection. The revision already addresses all issues raised in the original REVIEW-adversarial.md.

## Mathematical Soundness

### What holds

1. **The MAC channel analogy remains sound in principle.** The residual stream is additive. N expert MLPs do add outputs to a shared d-dimensional vector. The Gaussian MAC framework is a defensible conceptual model.

2. **The Shannon capacity formula is applied correctly.** C(N) = (1/2) log(1 + SNR_eff(N)) with SNR_eff(N) = SNR_0 / (1 + (N-1)*alpha) is standard AWGN MAC theory. No mathematical errors in the derivation.

3. **The validation methodology is correct.** Fitting on N=2,5,8 and validating on N=3,4,6,7 is the right approach. R-squared computed correctly. The comparison against linear and power-law baselines is appropriate.

4. **The self-diagnosis is accurate.** The paper correctly identifies that 3 params / 3 points = zero degrees of freedom, that the rate-distortion interpretation is tautological, and that the non-monotonic gap pattern falsifies the constant-interference assumption.

### What does NOT hold

**The fundamental assumption is falsified: gap is not a function of N alone.**

The data shows gap(N=3)=5.23% > gap(N=5)=1.60%, which violates the monotonic degradation assumption that is structurally embedded in the Shannon model (more transmitters = more interference = higher gap). This is not a matter of parameter tuning -- no reparameterization of the Shannon model can produce non-monotonic behavior because log(1 + SNR/(1+(N-1)*alpha)) is monotonically decreasing in N for alpha > 0.

The paper correctly identifies the confound: different N values require different domain-splitting methods (binary, ternary, quaternary, etc.), and split quality dominates the interference effect. This is a clean falsification.

**No remaining mathematical issues.** The revision has removed the overclaims, added proper caveats, and the derivations themselves were never wrong -- the model just does not describe the data.

## Novelty Assessment

The MAC channel framing for expert composition remains a novel conceptual contribution not found in the MoE literature (checked: Switch Transformers, Mixture-of-Depths, Union of Experts, Soft MoE, DeepSeek-V3 references in the project). However, since the model fails empirically, the novelty is of limited practical value at present.

The recovery path suggested (compose subsets of a fixed 8-domain partition to isolate N from split quality) is the right experimental design for a future attempt. If someone revisits this, that controlled setup would be the correct starting point.

## Experimental Design

The revision addresses all 6 fixes from the original review:

| Original Fix | Status |
|---|---|
| 1. Collect held-out data (N=3,4,6,7) | Done, 3 seeds each |
| 2. Compare against baseline models | Done (linear, power-law) |
| 3. Downgrade from "proven" to "consistent" | Done, then further to "killed" |
| 4. Caveat rate-distortion interpretation | Done (Section 7 marked descriptive only) |
| 5. Sensitivity analysis | Done (perturbation +/-1%) |
| 6. Fix dead code | Noted but not critical given kill verdict |

The experimental design of the revision is sound. The kill criteria (R-squared >= 0.5 on held-out data) are appropriate and were correctly applied. R-squared = -53.2 is an unambiguous failure.

One observation: the training data itself may be contaminated by the same confound. The N=2 gap of -0.20% (negative -- composition HELPS) and the N=5 gap of +1.60% were collected with different split methods. The apparent monotonic trend in {-0.20, 1.60, 5.71} at N={2,5,8} was coincidental. The held-out data at N={3,4,6,7} revealed that the true gap landscape is noisy and non-monotonic, dominated by split quality rather than expert count.

## Hypothesis Graph Consistency

HYPOTHESES.yml entry `exp_channel_capacity_bound` is correctly marked `status: killed` with appropriate evidence text. The kill criterion "model fails to predict composition gap trend on held-out data (r-squared < 0.5)" was triggered with R-squared = -53.2. The evidence string correctly identifies the recovery path (control for split quality).

No inconsistencies found.

## What Was Learned (Positive Contributions)

Despite the kill, this experiment produced valuable findings for the broader project:

1. **Composition gap depends more on domain-split quality than on N.** This is actionable: future scaling experiments must control for split method as a confound.

2. **3 params / 3 points is not validation.** This methodological lesson applies to all future micro-experiments with small data.

3. **The MAC channel framework is not wrong, it is confounded.** The residual stream really is an additive channel. The framework could be salvaged with proper experimental controls (fixed partition, varying subsets). This is a "not yet" rather than a "never."

## Macro-Scale Risks (advisory)

Not applicable -- experiment is killed. If the controlled-partition version is ever attempted at macro scale:

1. With BPE tokenization and natural domain boundaries (Python vs JavaScript), domain splits would be structurally cleaner, potentially removing the split-quality confound.
2. Sparse routing (top-k < N) would change the interference model fundamentally -- only k experts contribute per token, not N.
3. d=4096 would dramatically change SNR_0, requiring complete refitting.

## Verdict

**KILL**

The researcher has already correctly killed this experiment. The held-out validation (R-squared = -53.2) is an unambiguous falsification. The non-monotonic gap pattern (N=5 lower than N=3,4) breaks the core structural assumption of the channel model. All three candidate models (Shannon, linear, power-law) fail equally, confirming that gap(N) is not a smooth function of N alone.

The revision process was exemplary: every fix from the original review was addressed, the validation was conducted properly, and the conclusions are honest and well-calibrated. The "What Was Learned" section identifies a genuine insight (split quality dominates N) and a concrete recovery path (fixed-partition experiment) that could be pursued in a future experiment if warranted.

No further action needed. The HYPOTHESES.yml entry is correctly marked as killed.
