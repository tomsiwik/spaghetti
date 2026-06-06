# Peer Review: SwiGLU Gate-Aware Pruning (Revision Round)

## Prior Review Summary

The initial review required 5 fixes:
1. Fix reversed floor bound inequality
2. Disclose auxiliary sparsity losses in PAPER.md
3. Add random pruning baseline
4. Report 95% CI on quality delta
5. Fix Cauchy-Schwarz citation to "independence"

## Fix Verification

### Fix 1: Floor bound inequality -- APPLIED CORRECTLY

MATH.md Section 3.2 (lines 87-103) now presents the floor reduction as a **qualitative argument**, not a formal bound. The incorrect inequality `min_i(mu_i^g * mu_i^u) <= min_i mu_i^g * min_i mu_i^u` has been removed entirely. The replacement text correctly states:

> "The gate product floor is NOT bounded by floor_g * floor_u from below -- the minimizing capsule for the product need not be the same capsule that minimizes either factor."

And:

> "This floor reduction is an empirical observation, not a provable bound."

This is honest and correct. The "quadratically smaller" claim is gone. The mechanism explanation (up-projection learns to suppress specific capsules) is sound.

### Fix 2: Auxiliary sparsity loss disclosure -- APPLIED CORRECTLY

PAPER.md now includes a "Training Regularization Disclosure" section (lines 188-212) that:
- Names both losses: adaptive L1 sparsity (target 50%, coeff 0.01) and balance loss (coeff 0.01)
- States both SiLU and SwiGLU models receive identical regularization
- Acknowledges the absolute pruning rate (66.5%) may be inflated
- Explicitly flags the macro transfer risk: production models lack these losses
- Added as Limitation #7 (line 241-243)

This is thorough and honest.

### Fix 3: Random pruning baseline -- APPLIED CORRECTLY

A separate script `run_random_baseline.py` implements the control correctly:
- Prunes the **same fraction** per seed (not a fixed 66.5%), matching the actual gate-product pruning rate
- Uses 3 random seeds per training seed (9 evaluations total)
- Results reported in PAPER.md (lines 128-153)

Results: gate-product pruning causes 2.3x less degradation than random pruning on average. At high pruning fractions (77-82%), the advantage is 1.8-2.8x. This confirms gate-product profiling provides genuine signal.

Minor note: the random baseline was run as a separate replication, and the paper discloses that the gate-product numbers differ slightly from the original run (e.g., seed 42: 77.1% vs 81.2%) due to MLX non-determinism. This is honest.

### Fix 4: 95% CI -- APPLIED CORRECTLY

PAPER.md (lines 117-122) now reports:
- 95% CI: [-2.27%, +4.72%] using t-distribution with df=2
- Explicitly acknowledges the CI includes the 3% kill threshold
- States "we cannot rule out that the true mean degradation exceeds 3% at this sample size"
- Added as Limitation #6 (lines 236-239)

The CI calculation is correct: t(0.025, df=2) = 4.303, SE = 1.41/sqrt(3) = 0.814, margin = 3.50, CI = 1.22 +/- 3.50.

### Fix 5: Cauchy-Schwarz vs independence -- APPLIED CORRECTLY

MATH.md Section 3.2 (lines 78-85) now correctly states:
- `E[|XY|] <= E[|X|] * E[|Y|]` holds "by independence" (not Cauchy-Schwarz)
- Cauchy-Schwarz gives the different bound `E[|XY|] <= sqrt(E[X^2] E[Y^2])`
- Notes that g_i and u_i share input x, so they are correlated

This is now precise.

## Remaining Issues After Fixes

### Issue A: HYPOTHESES.yml status is "active", should be "proven"

The experiment has passed both kill criteria and the review fixes have been applied. The node `exp_swiglu_gate_pruning` in HYPOTHESES.yml still shows `status: active`. This should be updated to `proven` with a reference to this review.

**Severity: Bookkeeping, non-blocking.**

### Issue B: The CI including the kill threshold is a genuine weakness

The 95% CI [-2.27%, +4.72%] includes 3%. This means the experiment cannot statistically guarantee its own kill criterion is met at the standard significance level. The paper acknowledges this honestly. With only 3 seeds, the uncertainty is inherent.

**Assessment within micro constraints**: This is acceptable. The point estimate (+1.22%) is well below 3%. All three seeds individually pass both kill criteria. The wide CI is a consequence of n=3, which is the micro-experiment standard. The paper correctly flags this for macro validation. This is not a blocking issue.

### Issue C: Seed 42 quality delta (+2.84%) is close to kill threshold

Seed 42 shows 81.2% pruning at +2.84% degradation. The random baseline replication for the same seed shows +3.27% gate-product delta, which would individually FAIL the 3% quality criterion. These are different runs (MLX non-determinism), but it suggests seed 42 is on the boundary.

**Assessment**: The kill criteria are evaluated on the mean across seeds, not per-seed. The mean (+1.22%) passes. The seed variance is disclosed. This is acceptable at micro scale.

### Issue D: Auxiliary loss inflates the absolute pruning rate -- acknowledged but unresolved

The experiment acknowledges that 66.5% pruning may be inflated by the sparsity-encouraging aux loss. Without the aux loss, the pruning rate could be substantially lower. The core mechanism (multiplicative suppression) is architectural, but the quantitative claim "66.5% prunable" is conditional on the training regime.

**Assessment**: This is an inherent limitation of the micro setup, not a fixable issue within the current experiment. The comparison (SwiGLU vs SiLU) is fair because both have the same aux losses. The paper correctly flags this for macro transfer. Acceptable.

## Mathematical Soundness

All mathematical claims in the revised MATH.md are now correct:
- The SwiGLU forward pass and notation are standard
- The multiplicative suppression argument is qualitative and sound
- The pruning error bound is correct (triangle inequality)
- The computational cost analysis is correct
- The independence/Cauchy-Schwarz distinction is now precise
- No remaining incorrect bounds or derivations

## Novelty Assessment

Unchanged from prior review. Gate-product profiling as a pruning criterion for SwiGLU MLPs is novel relative to the references in this repository. Standard structured pruning (SparseGPT, Wanda) uses weight magnitude or gradient information, not gate product statistics. The delta over SiLU-only profiling (Exp 15) is clear and the mechanism is distinct.

## Experimental Design

The design is now adequate:
- Clean A/B comparison (SiLU vs SwiGLU, same training)
- Random baseline confirms profiling adds value (2.3x better)
- Three seeds with honest reporting of variance
- Threshold sweep shows full pruning landscape
- Kill criteria are correctly evaluated

## Macro-Scale Risks (advisory)

1. **No sparsity aux loss in production models**: Remains the primary risk. The gate product distribution in Qwen/Llama trained with standard cross-entropy may not have the concentrated low-magnitude region seen here.

2. **Threshold sensitivity**: The cliff at tau=0.05 (1.4% to 66.5% prunable) means threshold selection at macro scale requires careful calibration per model.

3. **Error accumulation across 32+ layers**: The +1.22% at 4 layers may compound through deeper models.

4. **Bimodal distribution assumption**: May not hold at macro scale without the sparsity regularization that shapes it here.

## Verdict

**PROCEED**

All 5 required fixes have been applied correctly and honestly. The mathematical errors are resolved. The auxiliary loss is disclosed with appropriate caveats. The random baseline confirms the mechanism provides genuine signal. The confidence interval weakness is acknowledged transparently.

The core finding is sound: SwiGLU's multiplicative gate creates a pruning channel that SiLU alone cannot provide, and gate-product profiling identifies better pruning targets than random selection. The quantitative claims (66.5% prunable, +1.22% degradation) are conditional on the training regime and carry statistical uncertainty that the paper honestly reports.

The HYPOTHESES.yml status should be updated from "active" to "proven".
