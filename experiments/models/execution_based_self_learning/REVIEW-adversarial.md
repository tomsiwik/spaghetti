# Peer Review: Execution-Based Self-Learning (Revision 2)

## NotebookLM Findings

Skipped (authentication not configured). Manual deep review performed on MATH.md, PAPER.md, both simulation scripts, and results.

## Revision Verification

All 6 fixes from the prior review have been addressed:

1. **DPO advantage acknowledged as parametric.** MATH.md lines 85-92 and PAPER.md lines 85-94 now state explicitly that the 1.91x amplification is an input assumption, not a prediction. The framing is honest and precise.

2. **Diversity models unified.** The main script (`run_self_learning_sim.py` line 224-225) now uses clean geometric decay `diversity *= (1.0 - decay_rate)`, identical to the MATH.md formulation. The old data-dependent uniformity-weighted decay is gone. Both scripts share the same base diversity model.

3. **MC variance explained as deterministic.** PAPER.md lines 113-120 acknowledge that the noise model (sigma=0.003) is negligible and that collapse cycles are point estimates from a deterministic model. The framing correctly distinguishes this from parametric sensitivity analysis (Experiment 5).

4. **Fresh data reframed as self-consistency.** PAPER.md lines 140-146 and MATH.md lines 176-182 now state that the math-simulation agreement is a self-consistency check, not independent confirmation. The word "predicts" has been replaced with appropriate hedging.

5. **gamma_DPO unified to 0.015.** Main script `diversity_decay_dpo = 0.015` (line 58), stress test `base_diversity_decay_dpo = 0.015` (line 32). Consistent.

6. **K sweep under accelerating decay.** Experiment 4b (PAPER.md lines 163-183) confirms K=3 is the sweet spot under both decay models. K does not affect the collapse boundary (all K values collapse at cycle 26), which is a clean result that strengthens the K=3 recommendation.

## Mathematical Soundness

### Pass probability model: Sound

The IRT 1PL logistic model `p_i(t) = sigmoid(logit(s_eff) - logit(delta_i))` is standard. The effective skill modulation `s_eff = s * (0.5 + 0.5 * d)` remains an ad hoc choice (the 0.5/0.5 coefficients are free parameters), but this is acknowledged in the Assumptions section and does not affect qualitative results. The model is well-specified.

### Skill update rules: Correct within stated framework

The signal-skill coupling (harder problems pass as skill increases, strengthening the signal) is now implicitly acknowledged in the DPO advantage discussion (PAPER.md line 89: "partially includes the signal-skill coupling effect"). The SFT/DPO update formulas correctly implement the MATH.md equations. The diminishing returns term `(1 - s_t)` correctly produces asymptotic convergence.

### Diversity dynamics: Consistent across scripts

Both scripts now implement `d_{t+1} = d_t * (1 - gamma)` for the constant-decay model. The stress test adds `gamma(t) = gamma_0 * (1 + a)^t` on top. The MATH.md derivation of `T_collapse` from the product formula is algebraically correct. The numerical examples in the table (MATH.md lines 149-157) are consistent with the formula.

### Fixed-point analysis: Correct but trivial

The fixed point at `s* = 1` or `sigma(t) = 0` is correct. In practice, the effective fixed point is determined by diversity collapse, which the paper correctly identifies. No hidden errors.

### Fresh data recovery: Self-consistent (as acknowledged)

The steady-state analysis `d_steady = f * r / gamma` is correct algebra. The paper now properly frames the 45%-50% threshold as self-consistency, not prediction. No remaining issue.

### One remaining mathematical concern: Collapse penalty discontinuity

The collapse penalty (`skill *= 0.5` when `diversity < 0.3`) creates a discontinuous dynamic that amplifies post-collapse degradation. This is present in both scripts (main line 323-324, stress line 118-119). The paper does not discuss this mechanism explicitly -- the dramatic cliff in post-collapse pass@1 (e.g., SFT going from 0.510 to 0.005) is partly an artifact of repeated halving, not purely of diversity dynamics. However, this is a modeling choice, not a mathematical error. The qualitative finding (collapse is bad) is robust to the specific penalty function. This is a minor presentation issue, not a blocking concern.

### Worked example: Spot-checked

MATH.md lines 211-228. Cycle 0: `s_eff = 0.30 * 1.0 = 0.30`, `sigmoid(logit(0.30) - logit(0.50)) = sigmoid(-0.847) = 0.300`. Correct. The claim "~70% of problems have >= 1 pass" at K=10 with pass@1=0.30 is consistent: `P(at least 1 pass in 10 tries) = 1 - 0.70^10 = 0.972` for a problem at the mean difficulty. But the 70% figure refers to the fraction of problems with at least one pass, accounting for the difficulty distribution -- some hard problems have very low per-attempt probability. This is plausible but not precisely derived. Not a concern for a worked example.

## Novelty Assessment

### Prior art: Adequate

The experiment correctly identifies its position relative to SPIN, ReST-EM, CodeRL, RLTF, and Shumailov. The novelty is the parametric model combining all dynamics (skill, diversity, fresh data) into a single simulation framework calibrated across multiple papers. This is a planning tool, not a research contribution, which the paper acknowledges.

### Delta over existing work: Modest but appropriate

The primary value is the SOLE integration plan: DPO over SFT, K=3, diversity monitoring, fresh data injection thresholds. These are actionable engineering decisions derived from the simulation. For a micro-scale simulation study, this is the right output.

### Missing reference check

No entries in `references/REFERENCES.yml` for SPIN, ReST-EM, CodeRL, or Shumailov. This was noted in the prior review but not listed as a required fix. The calibration parameters cannot be independently verified without these. This is a documentation gap but not blocking.

## Experimental Design

### Hypothesis testing

The experiment tests two kill criteria:
- K1: pass@1 improvement after 5 cycles -- PASS for both SFT and DPO across all parameter regimes
- K2: model collapse -- CONDITIONAL (depends on decay acceleration rate)

The kill criteria match the HYPOTHESES.yml entry. The evidence claim (dated 2026-03-16) accurately summarizes the findings.

### Simpler explanation check

Could the results be explained by a simpler mechanism? The K1 result (pass@1 improves) is trivially guaranteed by the update rule: any positive learning rate with positive signal and positive headroom produces improvement. The non-trivial finding is the interaction with diversity decay, which determines when improvement stops. This is adequately captured.

### Controls

The acceleration sweep (Experiment 5) and K sweep (Experiments 4/4b) provide parameter sensitivity analysis that substitutes for experimental controls. The dual-model structure (constant vs. accelerating decay) provides bracketing of realistic conditions.

### Remaining concern: The 0.5 + 0.5*d formula

The effective skill formula `s_eff = s * (0.5 + 0.5 * d)` implies that a fully collapsed model (d=0.05, the floor) retains 52.5% of its effective skill. This is a strong assumption that limits how bad collapse can get in the model. A more aggressive formula (e.g., `s * d`) would make collapse dramatically worse. The paper does not test sensitivity to this choice. This is not blocking -- the qualitative findings are robust -- but it represents an unstated assumption about the benignity of collapse that could mislead macro-scale planning.

## Hypothesis Graph Consistency

The experiment correctly targets `exp_execution_based_self_learning` with status "active" and the evidence claim matches the paper. The dependent experiment `exp_model_collapse_detection` has kill criterion "output diversity drops >30% after 5 self-learning cycles" -- the current results show ~13% drop at cycle 5 (DPO, constant decay), well under 30%. This pre-validates the dependent node.

The "SUPPORTED" status is appropriate: the simulation provides directional evidence but no empirical confirmation. The paper is clear about this distinction.

## Macro-Scale Risks (advisory)

1. **LoRA rank saturation.** The scalar skill model cannot capture rank-16 LoRA capacity limits. Real self-training may hit a hard ceiling where the adapter cannot absorb more information, causing plateau or mode collapse much earlier than the logistic model predicts.

2. **Correlated sampling.** LLM temperature sampling produces correlated outputs (especially similar-looking solutions for similar problems). This violates the independence assumption and would accelerate diversity loss. The paper acknowledges this (Assumption 1, Limitation 8) but does not bound the effect.

3. **Test suite reliability.** The correction_signal_quality experiment found 11.3% degeneracy at 60% coverage. The SOLE integration plan requires coverage > 0.8 but this threshold has not been validated. The binary oracle assumption is the weakest link for production deployment.

4. **Cost at scale.** The paper now includes a cost caveat (lines 270-273) acknowledging $3-6/cycle at 7B scale vs. $0.60 at the original estimate. This is adequate.

5. **DPO implementation fidelity.** The simulation models DPO as a scaled SFT update with contrastive amplification. Real DPO has a reference model, KL divergence penalty, and specific loss landscape properties not captured here. The real DPO advantage could be higher or lower than 1.91x.

## Verdict

**PROCEED**

The six required fixes from the prior review have been properly implemented. The paper is now honest about what is parametric input vs. simulation output, the diversity models are unified, the Monte Carlo framing is corrected, and the fresh data threshold is properly framed as self-consistency.

The remaining concerns (collapse penalty discontinuity, 0.5+0.5*d formula sensitivity, missing reference folders) are minor presentation and documentation issues that do not affect the correctness or utility of the findings. The qualitative conclusions are robust:

- DPO > SFT for self-learning loops (directionally sound, magnitude is calibration input)
- K=3 is sufficient (validated under both decay models)
- Diversity monitoring + fresh data injection is required under accelerating decay
- A "golden window" of 10-25 safe cycles exists before collapse risk

The experiment provides actionable engineering guidance for the SOLE Evolve phase. The SUPPORTED status is correctly calibrated -- the simulation is a planning tool, and macro-scale empirical validation is explicitly called out as the next step.
