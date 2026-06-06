# Peer Review: FFN-only Matched Rank Validation

## NotebookLM Findings

Skipped -- this experiment is primarily infrastructure preparation (training script + analysis pipeline) with no novel mathematical claims to deep-review. The theoretical content in MATH.md is well-grounded review of prior findings, not new derivations.

## Mathematical Soundness

**Parameter counting (Section 3): CORRECT.**

FFN per layer: 3 modules (gate, up, down), each contributing r*(d + d_ff) parameters for A and B matrices combined. 3 * 16 * (3584 + 18944) = 1,081,344 per layer. 28 layers = 30,277,632. Matches code output (dims show 30,277,632 in results.json).

Attention per layer: the paper claims r * (6d + 2*d_kv) = 16 * (6*3584 + 2*512) = 360,448. Let me verify: q_proj (A: r*d, B: d*r), k_proj (A: r*d_kv_total, B: d_kv_total*r?). Actually, the test file (line 117-118) computes attention params differently: `L * (2*r*d + r*d + r*d_kv + r*d + r*d_kv + 2*r*d)`. This is inconsistent with MATH.md's formula. The test asserts 0.70 < ratio < 0.80, which is loose enough to pass despite any formula discrepancy. The results.json confirms the actual parameter counts from real adapters: 40,370,176 for all-modules, 30,277,632 for FFN-only. Ratio = 30.3M/40.4M = 0.750. **The actual adapter sizes validate the claim; the intermediate formula in MATH.md and the test formula are both approximations but converge on the same answer.**

**Random baseline calculation (Section 4.1): CORRECT.** E[|cos|] = sqrt(2/(pi*D)) with D = 5.7B gives ~1.06e-5. Standard result.

**Orthogonality prediction (Section 4.2): REASONABLE.** The prediction that independent FFN-only cosine should be LOWER than retroactive is plausible but the reasoning is speculative. The paper honestly presents the opposite case as well. No mathematical error here -- this is a stated hypothesis, not a derived result.

**Kill criterion formulation (Section 5): SOUND but with one concern.** The 5% PPL threshold is applied per-domain ("Kill if: PPL_gap > 5% for any domain"). This is strict -- a single domain exceeding 5% kills the experiment even if the mean gap is 2%. The analysis code (analyze.py line 162-164) correctly implements this. However, PPL has already been killed as a task accuracy proxy (exp_ppl_vs_task_performance). The paper acknowledges this in Assumption 3 and argues relative PPL is more reliable. This is defensible for a same-data, same-rank comparison -- we are measuring "does the model fit the training distribution equally well with fewer adapted modules," not "does lower PPL mean better task performance."

**No hidden mathematical assumptions found.** The assumptions are explicitly stated (Section 8) and appropriate.

## Novelty Assessment

**This is not a novelty-driven experiment.** It is a validation experiment to resolve a known confound (retroactive vs. independent FFN-only training). The confound was identified in the prior experiment's review and flagged in FINDINGS.md. This is exactly the right follow-up.

**Prior art check:** The Geva et al. 2021 citation is appropriate. No prior work specifically compares retroactive vs. independent FFN-only LoRA orthogonality. The experimental question is specific to this project's architecture.

**No reinvention detected.** The training script correctly uses the existing `composer/distill.py` patterns (SFTTrainer, QLoRA, same hyperparameters). The analysis script builds on the same orthogonality computation used in ffn_only_vs_all_modules.

## Experimental Design

**Strengths:**

1. The confound identification is precise and well-motivated (MATH.md Section 2). Joint training causes co-adaptation: FFN params trained with attention LoRA present see different hidden representations than FFN params trained alone. This is a real methodological concern.

2. Fair comparison is well-designed: same seed, same data, same rank, same steps, same optimizer. The `--also-train-all` flag for retraining all-modules with the identical script eliminates software version confounds.

3. Kill criteria are pre-registered and specific. The analysis code implements them exactly as stated.

4. Infrastructure is complete and tested (8/8 tests pass). The experiment is ready to execute.

**Weaknesses:**

1. **Single seed is the main risk.** The paper acknowledges this (Limitation 1) and notes 3 seeds would cost ~$0.60. At $0.20-0.40 for a single run, multi-seed is clearly affordable. A single-seed result close to the 5% threshold would be inconclusive. **Recommendation: run 3 seeds from the start.**

2. **The math-medical pair dominates all orthogonality statistics.** In results.json, the math-medical pair has |cos| = 0.70 (full), 0.59 (FFN), 0.85 (attn). All other pairs are <0.003. The "mean |cos|" is almost entirely determined by this single pair. If independent training changes the math-medical cosine by 0.03, the mean shifts by ~0.003, which is 5% of 0.0605. The 50% kill threshold is generous, but the signal is concentrated in one pair. **The analysis should report both (a) mean including math-medical and (b) mean excluding math-medical, or equivalently, the median.** As it stands, the experiment is really measuring "does independent training change the math-medical FFN cosine by more than 50%?"

3. **No convergence verification.** Assumption 2 states "300 steps is sufficient for convergence" and says "we verify by checking that training loss has plateaued." But the training script (train_ffn_only.py) only logs loss at 50-step intervals and does not perform any convergence check. The metrics saved include `train_loss` (final average) but not the loss curve. If FFN-only adapters need more steps to converge (because they must compensate for the lack of attention adaptation), the PPL comparison would be unfair. **Recommendation: log loss at every step or save the loss curve, and verify that the last 50 steps show <5% improvement.**

4. **Eval set size (50 examples) may be noisy for PPL.** With 50 examples and packing enabled (so potentially fewer effective sequences), PPL variance could be substantial. No confidence intervals are computed. The analysis script reports point estimates only.

5. **The comparison against existing `adapters/` is potentially confounded.** The paper notes this (Limitation 5) -- existing all-modules adapters may have been trained with different software versions or data ordering. The `--also-train-all` flag exists but is commented out in the run script. **Strong recommendation: uncomment and use it.** The cost is negligible ($0.20 more).

**Controls:**
- Adequate: same data, same rank, same hyperparameters
- Missing: no random baseline (untrained LoRA PPL) to bound the scale of adaptation effect. Not critical but would contextualize the results.

## Hypothesis Graph Consistency

The experiment is correctly placed in the hypothesis graph:
- Depends on `exp_ffn_only_vs_all_modules` (which provided the retroactive analysis and the confound)
- Blocks `exp_distillation_pilot_50` (the 50-expert pilot should use whichever module configuration wins)
- Kill criteria in HYPOTHESES.yml match the paper exactly
- Priority 2 is appropriate -- this is on the critical path

**Scale label concern:** HYPOTHESES.yml marks this as `scale: macro` but the experiment directory is `micro/models/ffn_only_matched_rank`. The training happens on real Qwen2.5-7B (7B params, macro model), but the analysis runs on CPU with pre-trained adapters. This is a hybrid. The "macro" label is more accurate since the training uses full-scale models. Not blocking, but the directory placement is slightly misleading.

## Macro-Scale Risks (advisory)

1. **Math-medical anomaly.** The cos=0.59 FFN overlap between math and medical domains is concerning at scale. With 50+ experts, there will be more within-cluster pairs (as shown by exp_orthogonality_by_domain_type). The FFN-only vs all-modules comparison may not generalize beyond these 5 specific domains. The 50-expert pilot will be the real test.

2. **Domain-dependent quality gaps.** Even if mean gap < 5%, specific domains (e.g., code with complex scope) might need attention adaptation. The 50-expert pilot should track per-domain quality variance, not just mean.

3. **Hyperparameter sensitivity.** FFN-only adapters might need different learning rates or more steps than all-modules. The experiment holds hyperparameters fixed, which is the right choice for a fair comparison but may not represent the best FFN-only configuration. A follow-up with FFN-only-optimized hyperparameters would be needed if the gap is 3-5%.

## Verdict

**PROCEED** -- with the following recommended improvements before execution:

1. **Use `--also-train-all` (uncomment in run_on_runpod.sh).** The fair comparison eliminates the software version confound for a negligible additional cost of ~$0.20. Without this, a positive result could be challenged as comparing against stale baselines.

2. **Run 3 seeds, not 1.** Budget permits it ($0.60 vs $0.20). A single borderline result (e.g., 4.8% gap) would be inconclusive. With 3 seeds, you get mean and standard deviation to make a confident call. Seeds 42, 123, 7 or similar.

3. **Report orthogonality with and without the math-medical pair.** The mean |cos| is dominated by one outlier pair. Add a `median_abs_cos` and `mean_abs_cos_excl_max` to the analysis output so the kill criterion is evaluated on a richer picture.

4. **Save the loss curve** (or at minimum, log at steps 50, 100, 150, 200, 250, 300) and verify convergence. If FFN-only loss is still decreasing at step 300 while all-modules has plateaued, the comparison is unfair and more steps are needed.

These are improvements, not blockers. The experimental design is sound, the math is correct, the confound is real and worth resolving, and the infrastructure is production-ready. The experiment directly advances the critical path (blocks the 50-expert pilot). Execute it.
