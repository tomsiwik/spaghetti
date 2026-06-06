# Peer Review: Sequential Freeze-Graft-Calibrate (N>2)

## NotebookLM Findings

Skipped -- the experiment is a clean negative result with self-kill. The data is unambiguous and the analysis thorough. Deep review would not change the verdict.

## Mathematical Soundness

**Derivations are correct and honest.**

1. The progressive halving allocation (Section 2 of MATH.md) is the natural consequence of sequential binary splits. The capacity imbalance (D_0 gets 4 leaves, D_3 gets 1) is correctly identified as inherent to the protocol rather than a bug.

2. The degradation scaling analysis (Section 3.2) computes the exponent alpha = log(3.65)/log(3) = 1.18. The paper calls this "approximately quadratic" (alpha ~ 1.8-1.9) but the actual fit gives alpha ~ 1.18, which is superlinear but not quadratic. **This is a minor imprecision** -- the paper hedges between two claims. The 3-point fit with N=1,2,3 grafts (not N=2,3,4 domains) gives:
   - delta(1) = 3.72, delta(2) = 6.73, delta(3) = 13.58
   - Ratio delta(3)/delta(1) = 3.65
   - alpha = log(3.65)/log(3) = 1.18

   The text says "alpha ~ 1.8-1.9" which appears nowhere in the actual calculation. This does not affect the kill verdict (alpha > 1.0 is sufficient for superlinear, and the 3.65x ratio far exceeds the 2.0x threshold), but it is sloppy.

3. The calibration capacity analysis (Section 4) is sound: 66,576 -> 33,548 -> 17,164 params as more tree is frozen. The sublinear cost claim (KC2) is correct.

4. The worked example (Section 5) is internally consistent with the reported means.

**Hidden assumption worth noting:** The degradation metric is always computed relative to the *initial* domain A baseline (before any grafting). This is the right choice for measuring cumulative damage, but it means the N=4 degradation includes damage from grafts 1, 2, AND 3 -- it is not the marginal damage of graft 3 alone. The paper is clear about this, so no issue.

## Novelty Assessment

**This is a well-motivated negative result, not a novelty claim.** The experiment tests whether a protocol validated at N=2 (split_freeze_protocol) extends to N>2. It does not.

Prior art correctly cited:
- Progressive Neural Networks (Rusu et al., 2016) -- column-per-task with frozen predecessors. The paper correctly notes that "interference accumulates" is the same fundamental problem.
- PackNet (Mallya and Lazebnik, 2018) -- per-neuron freezing with increasing interference.

The experiment's contribution is narrower than these references: it shows the specific failure mode of *tree-structured* sequential freezing, where progressive halving creates capacity imbalance that compounds routing drift. This is a useful architectural finding for the project's vision.

**No reinvention detected.** The experiment builds properly on split_freeze_protocol and minimal_graft_recal.

## Experimental Design

**The experiment tests what it claims and the controls are adequate.**

Strengths:
1. Three seeds (42, 123, 777) with all showing the same direction. Per-seed variance is documented (ratio ranges 3.07x to 3.89x, all above 2.0x threshold).
2. Two calibration strategies (all-unfrozen vs selective) tested independently.
3. Extended calibration experiment (3 schedules) rules out "just needs more training" as an explanation.
4. Kill criteria are pre-specified in HYPOTHESES.yml and match what was tested.

**One design concern -- not blocking but worth noting:**

The degradation baseline for domain A is set once (after Step 0 base training on all 8 leaves). But at N=2, domain A is frozen into leaves 0-3, which is only half the capacity it was trained with. Some of the measured "degradation" at N=2 is not from graft interference but from the capacity halving itself (domain A loses access to leaves 4-7 that it originally trained on). The paper does not disentangle these two effects.

However, this concern applies equally at all N values, and the *ratio* between N=4 and N=2 degradation (which is what KC1 measures) still isolates the scaling behavior. The baseline shift is approximately constant. So this does not threaten the kill verdict.

**Confound acknowledged but not controlled:** Domain sizes are unequal (3,667 to 11,629). The smallest domain (t_z) is assigned to the last graft position with the fewest leaves (1 leaf at N=4). This means the hardest capacity constraint hits the smallest domain, which could make degradation look worse than it would with balanced domains. Again, this does not change the verdict because all three seeds show the pattern, but it is a confound.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_sequential_freeze_graft` has:
- Status: killed (correct)
- Kill criteria match exactly what was tested (KC1: degradation ratio, KC2: cost scaling)
- Evidence summary matches the paper
- Dependencies on `exp_split_freeze_protocol` and `exp_minimal_graft_recal` are correct

The evidence is sufficient to kill this node.

## Macro-Scale Risks (advisory)

This experiment is a clean kill -- there is nothing to scale. The advisory is:

1. **Tree grafting at N=2 remains viable.** The N=2 result (+3.72% degradation) is within acceptable bounds and consistent with split_freeze_protocol findings. Do not over-generalize this kill to reject all tree-based composition.

2. **The flat MoE composition protocol (cited as +1.6% at N=5) is the correct N>2 path.** This is already identified in VISION.md. No course correction needed.

3. **Deeper trees (D=5+) could theoretically rescue sequential grafting** by providing more balanced allocation, but this was correctly identified as contradicting the "one contributor at a time" sequential model. Not worth pursuing given the flat MoE alternative.

## Verdict

**PROCEED** (as a completed, killed experiment -- the negative result is clean and well-documented)

This is a textbook micro-experiment negative result. The hypothesis was falsifiable, the kill criteria were pre-specified, three seeds consistently exceed the threshold, and an extended calibration sweep rules out the obvious "just train more" objection. The paper correctly identifies the structural cause (capacity imbalance from progressive halving compounds routing drift) and correctly recommends the alternative (flat MoE for N>2).

One minor fix recommended but not blocking:

1. MATH.md Section 3.2 claims "alpha ~ 1.8-1.9" but the actual computed value is alpha ~ 1.18. Replace the spurious estimate with the calculated value. The superlinearity claim holds either way.

No revisions needed for the kill verdict or experimental conclusions.
