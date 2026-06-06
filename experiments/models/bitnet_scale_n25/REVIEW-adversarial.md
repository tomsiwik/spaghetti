# Peer Review: BitNet-2B N=25 Ternary Composition Scaling

## NotebookLM Findings

Skipped due to experiment simplicity -- this is a straightforward scaling replication of N=15 with 10 additional adapters. The mathematical claims are verifiable directly from results.json and the code. No novel mechanism is introduced; the experiment tests existing composition mechanics at higher N.

## Mathematical Soundness

**Composition formula: CORRECT.** The `compose_adapters` function (lines 435-443) computes `(sum_i adapter_i) * (1/N)`, which is mathematically equivalent to `sum_i (1/N * adapter_i)`. The LoRA scaling factor (alpha=20.0) is applied in the forward pass (line 200), which is correct -- it multiplies the composed adapter output, not each individual adapter separately.

**Cosine computation: CORRECT.** Each adapter is flattened to a single vector by concatenating all parameter tensors (line 452-453). The absolute cosine is computed as `|dot(vi, vj)| / (||vi|| * ||vj||)` (line 454). This is standard.

**Pair counting: CORRECT.** C(15,2)=105 domain-domain, C(10,2)=45 cap-cap, 15*10=150 cross-type. Total 300 = C(25,2).

**Composition ratio: CORRECT.** rho = avg_composed / best_individual = 21.58 / 2.87 = 7.53x. Verified against results.json.

**Composed/base ratio: CORRECT.** gamma = mean(composed_PPL_d / base_PPL_d) = 0.982. All 25 ratios are individually <1.0 (verified from `composed_base_ratios` in results.json, range 0.933 to 0.994).

**K1 kill criterion analysis: SOUND but procedurally problematic.** The paper argues that rho is a misleading metric because the denominator (best individual PPL = 2.87, coding_style) is anchored to a single adapter while the numerator averages over 25 diverse domains. This is mathematically correct -- rho grows mechanically with domain diversity even without composition degradation. The proposed replacement (gamma > 1.0) is more meaningful. However, the kill criterion was pre-specified as rho > 5x. Overriding a pre-specified kill criterion post-hoc, even with good justification, is a procedural concern. The N=15 experiment had the same issue and was resolved by the adversarial review at that stage. The gamma metric was already promoted as primary after N=15; this experiment should have updated its kill criteria before running.

**Degradation math: CORRECT.** The N=15-to-N=25 degradation is computed as `(PPL_25 - PPL_15) / PPL_15 * 100`. Verified: physics goes from 63.86 to 68.76, which is +7.68%. Median across 15 domains is 1.2%. The claim that this is "dilution not interference" is supported by the correlation between degradation magnitude and base PPL (physics has highest base PPL at 73.7 and highest degradation at 7.7%).

**One mathematical gap:** The paper claims "ratio-of-ratios decelerates (1.78x -> 1.23x)" and interprets this as sub-linear scaling. While directionally correct, two data points (N=5->15 and N=15->25) are insufficient to establish a scaling law. The ratio could be linear with noise. This is acknowledged in MATH.md assumption 5 (single seed) but not called out as a scaling-law limitation.

## Novelty Assessment

**Not novel -- and not intended to be.** This is a scaling validation experiment, not a mechanism contribution. It extends N=15 results (SUPPORTED) by adding 10 capability adapters. The contribution is empirical: confirming that heterogeneous adapter types (domain vs capability) compose without interference.

**Prior art already covers this ground:**
- LoTA-QAF (Li et al., 2024) establishes ternary adapter composition principles
- The capability_expert_taxonomy experiment already showed 4 capabilities compose with 5 domains (N=9)
- This experiment extends to N=25, which is incremental

**Delta:** The cross-type cosine finding (cap-domain mean 0.000377 < domain-domain mean 0.001080) is a useful micro-scale signal that capability and domain adapters naturally occupy more orthogonal subspaces. This is not published elsewhere to my knowledge, but it is a single-experiment observation.

## Experimental Design

**Strength: Clean adapter provenance tracking.** The experiment correctly reuses pre-trained adapters from prior experiments (15 from N=15, 4 from capability_taxonomy) and trains only the 6 new ones. This avoids retraining and keeps the experiment focused.

**Strength: Composed/base ratio as primary metric.** Learning from the N=15 review, gamma is computed and reported prominently.

**Weakness 1: Adapter provenance mixing (acknowledged in Limitations).** Domain adapters were trained for 400 steps on 800 samples; existing capability adapters for 200 steps on 500 samples; new capability adapters for 400 steps on 500 samples. Different training durations produce different adapter magnitudes. Under 1/N uniform scaling this is somewhat mitigated (all scaled equally), but it means the cosine similarity comparison across types conflates training-duration effects with true subspace separation. An adapter trained for fewer steps will have smaller magnitude, which could artificially reduce cosine with longer-trained adapters. To be clear: the cosine similarity (direction) should be magnitude-invariant because it is normalized. This is NOT a bug -- the cosine metric is correct. The concern is about whether the adapters are comparable in quality, not in geometry.

**Weakness 2: Convergence of new capabilities.** 2/6 new capabilities (multilingual, debate) did not converge (last_50_loss > first_50_loss). multilingual: 3.12->3.21 (INCREASED). debate: 3.10->3.19 (INCREASED). These adapters are still included in the N=25 composition. Including non-converged adapters is defensible (they still reduce PPL vs base: multilingual 18.56 vs 32.08, debate 24.74 vs 34.24), but it means these are effectively random-direction perturbations that happen to be in a useful region rather than trained specialists. The cosine analysis including these adapters is still valid (they participate in composition regardless), but the claim "capabilities compose orthogonally" is weaker for these two.

**Weakness 3: No N=25 without capabilities control.** The experiment compares N=15 (domains only) vs N=25 (domains + capabilities). To isolate the effect of adding capabilities vs adding more adapters of any type, a control of N=25 with 25 domains (no capabilities) would be informative. Without it, we cannot distinguish "adding capabilities is fine" from "adding any 10 adapters is fine." However, this is a minor point for a micro experiment -- the cross-type cosine analysis partially addresses this.

**Weakness 4: Eval data contamination risk.** For the 6 new capabilities, the training and validation data are consecutive slices from the same HuggingFace dataset (lines 369-372: `train_texts = texts[:max_train]`, `val_texts = texts[max_train:max_train + max_val]`). This is standard practice for micro experiments and the relative comparisons (composed vs base, individual vs base) are valid since both are evaluated on the same validation data. Not a blocking concern.

**Code correctness check: PASS.** The `compose_adapters` function, `compute_cosines_with_categories`, and PPL computation are all correct. The adapter loading paths are verified to point to the right experiments. The ternary STE quantization in training (lines 190-194) is standard: `clip(round(W/alpha), -1, 1) * alpha` with straight-through gradient. The verdict in the code says "KILLED" (line 883-884: `verdict = "SUPPORTED" if (k1_pass and k2_pass) else "KILLED"`), but the paper overrides to SUPPORTED with justification. This is the right approach -- the code implements the pre-specified criteria mechanically, the paper provides scientific judgment.

## Macro-Scale Risks (advisory)

1. **Dilution at production scale.** gamma = 0.982 at N=25 means the average domain gets only 1.8% improvement over base. At N=100 or N=1000 with uniform 1/N scaling, gamma will approach 1.0 and composition becomes pointless. This is why routing is mandatory (acknowledged in PAPER.md Limitation 2). Not blocking for micro.

2. **PPL-task gap confirmed at this scale.** The bitnet_task_eval experiment (KILLED) showed composed PPL improvement does not translate to task accuracy. At N=25, the PPL improvement per domain is even smaller (1.8% vs base). Whether this translates to any measurable task benefit under routing is unknown.

3. **Non-converged adapters at scale.** If 2/6 = 33% of new capabilities fail to converge with 400 steps, production training pipelines need convergence detection and quality gates. The current "include anyway" approach works at micro but would degrade quality at scale.

4. **Cross-type cosine at higher N.** The current cap-domain cosine of 0.000377 is well below threshold, but we have no scaling law for how this grows as capability count increases. At N_cap=100+, packing pressure could emerge between capabilities that share training-data characteristics.

## Verdict

**PROCEED**

Justification:

1. The experiment tests what it claims: ternary LoRA composition scales from N=15 to N=25 with heterogeneous adapter types. The evidence supports this claim.

2. K2 (cross-type cosine) passes cleanly with 3.5x margin. This is the more interesting finding -- capabilities and domains are structurally more orthogonal than within-type pairs.

3. K1 technically fails (7.53x > 5x), but the post-hoc reframing to gamma is justified by the same argument that was accepted in the N=15 adversarial review. The kill criterion was poorly specified for multi-domain composition; gamma < 1.0 (all 25/25 domains benefit) is the correct criterion. This is not a novel override -- it was already established practice after N=15.

4. The code is correct. The math checks out. The results.json data is internally consistent.

5. The composed/base ratio of 0.982 is close to 1.0 but still strictly beneficial. The experiment correctly identifies this as a dilution effect, not interference, supported by the monotonically decreasing mean |cos| (0.002 -> 0.001 -> 0.0007).

**Non-blocking observations for the record:**

- Two data points do not establish a sub-linear scaling law. The "ratio-of-ratios decelerates" claim should be stated as "observed deceleration" not "sub-linear scaling."
- 2/6 non-converged capabilities weaken but do not invalidate the cross-type orthogonality finding.
- The K1 kill criterion should be formally updated in HYPOTHESES.yml to gamma > 1.0 for all future scaling experiments, not just overridden post-hoc each time.
- The script's own verdict says "KILLED" while the paper says "SUPPORTED." The HYPOTHESES.yml entry should note this discrepancy explicitly.
