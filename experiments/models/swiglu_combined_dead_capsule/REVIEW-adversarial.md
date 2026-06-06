# Peer Review: swiglu_combined_dead_capsule

## NotebookLM Findings

Skipped -- this experiment is a clean negative result with straightforward math. The mechanism under test (dead capsule pruning for SwiGLU) is falsified by a property of the activation function itself. No deep review is needed to identify the core issue; it is self-evident from the SiLU floor argument.

## Mathematical Soundness

**MATH.md is correct and well-structured.** Step-by-step:

1. **Dead capsule criterion (Section 2):** The definition of fire_freq_i and the exactness claim at tau=0 are correct. If |h_i(x)| = 0 for all calibration x, removing capsule i produces zero output change. Sound.

2. **Gate-product magnitude criterion (Section 2):** The bound ||delta_y|| <= C * tau_gate * ||b_i|| is correct as stated. Minor note: C is left undefined ("accounts for the difference between mean and instance-level magnitudes"). For a rigorous bound, C should be the ratio max_x |h_i(x)| / mean_x |h_i(x)|, which is data-dependent and could be large for heavy-tailed distributions. This is a known looseness but does not affect the experiment's conclusion since the combined advantage is exactly zero regardless of bound tightness.

3. **SiLU floor analysis (Section 4):** The claim that SiLU(z) > 0 for z > 0 is correct. The formal bound |SiLU(z)| > 4.5e-5 for z > -10 is approximately correct (SiLU(-10) = -10 * sigmoid(-10) ~ -10 * 4.54e-5 ~ -4.54e-4; the minimum of SiLU is at z ~ -1.28 where SiLU ~ -0.278). Wait -- this requires clarification. The MATH.md states |SiLU(z)| > 4.5e-5 for z > -10. But SiLU(-1.28) ~ -0.278, so |SiLU(-1.28)| ~ 0.278, which is much larger than 4.5e-5. The bound is vacuously loose at moderate z. The tightest floor for |SiLU(z)| when z is in any reasonable range is actually the minimum of |SiLU(z)|, which is SiLU(0) = 0. **SiLU(0) = 0 exactly.** The claim "SiLU(z) never reaches zero for finite inputs" is wrong -- SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0.

   However, this mathematical error does not invalidate the experimental conclusion. For h_i(x) = SiLU(w_gate^T x) * (w_up^T x) to be zero, you need SiLU(w_gate^T x) = 0 (requiring w_gate^T x = 0 exactly) OR w_up^T x = 0 (requiring orthogonality). Both require exact zero inner products, which are measure-zero events in continuous distributions. The experimental observation (0% dead capsules, min fire freq 0.998) confirms this: while the SiLU floor argument in Section 4 has a mathematical error at z=0, the practical conclusion holds because exact zeros are vanishingly unlikely in trained networks with diverse data.

4. **Set overlap analysis (Section 3):** Correct and clear. The worked example (Section 6) matches experimental results.

5. **Computational cost (Section 5):** The O(N * L * P) overhead for combined profiling is correct and negligible.

**Summary:** One mathematical error (SiLU(0) = 0, not strictly positive), but it does not change the conclusion. The experiment correctly identifies that SiLU produces exact zeros only at a measure-zero point, making dead capsule pruning vacuous in practice.

## Novelty Assessment

**No novelty claimed, none needed.** This is a negative-result experiment testing whether two existing pruning criteria (dead capsule from Exp 9, gate-product from Exp 16) are complementary for SwiGLU. The answer is no. The finding -- that dead neuron pruning is ReLU-specific and does not transfer to smooth activations -- is well-known in the pruning literature (the entire "neuron death" literature, including ReDo/Klein et al. 2024, implicitly assumes ReLU-family activations).

The researcher appropriately cites Exp 15 (silu_pruning, KILLED: 0% prunable) which already demonstrated this for raw SiLU. This experiment extends to the gated SiLU (SwiGLU) case, confirming the same result. The extension is incremental but worth confirming.

No prior art was reinvented. The code reuses parent experiment infrastructure cleanly.

## Experimental Design

**The experiment correctly tests the stated hypothesis.** Specific observations:

1. **Controls are adequate.** Three independent seeds (42, 123, 7), sweep over 5 gate thresholds and 3 dead thresholds, comparison of dead-only vs gate-only vs combined. The design isolates the contribution of each criterion.

2. **The experiment was doomed from the start, but this is acceptable.** The researcher should have predicted 0% dead capsules from the silu_pruning experiment (Exp 15, KILLED) and the SiLU floor property. However, confirming for SwiGLU specifically (where the gated product introduces a second multiplicative factor) is reasonable due diligence. The gate product h = SiLU(g) * u could in principle be zero when u = 0 even if SiLU(g) is nonzero. The experiment shows this does not happen in practice.

3. **Minor design gap:** The experiment profiles on joint_val but trains on a single domain (a_m). This means the "dead" classification reflects single-domain activation patterns, not the composition scenario. However, since no capsules are dead even in this potentially lower-coverage setting, testing on the composition scenario would only increase fire rates, strengthening the conclusion.

4. **The "nearly dead" extension (dead_tau > 0) is a good addition.** Testing at tau=0.001 and tau=0.005 probes whether relaxing the exact-zero criterion helps. It does not, because even the minimum fire frequency is 0.998.

5. **Kill criteria are correctly defined and correctly evaluated.** KC1 (>5pp combined advantage) is the right threshold -- anything less would be noise. The 0.0pp result is decisive. KC2 (quality < 3% degradation) passes trivially since combined = gate-only.

## Macro-Scale Risks (advisory)

Not applicable. This experiment is killed. The finding (SwiGLU has no dead capsules) is architecture-level and confirmed at macro scale (0% dead at d=896, noted in the paper's limitations). No macro follow-up is needed for this specific hypothesis.

The parent finding (gate-product pruning for SwiGLU) was tested at macro and also killed there (+196% quality loss at tau=0.05 without aux sparsity loss), per VISION.md. This is a separate issue from the dead-capsule question.

## Verdict

**PROCEED** (as a completed, killed experiment)

This is a clean negative result. The hypothesis was reasonable (combining two complementary pruning criteria), the experimental design is sound, the execution is thorough (3 seeds, multiple threshold sweeps), and the kill is decisive (0.0pp advantage, not close to the 5pp threshold). The root cause analysis is correct and provides a useful taxonomy: frequency-based pruning is ReLU-specific, magnitude-based pruning is SwiGLU-specific.

One mathematical correction should be noted for archival accuracy:

1. **MATH.md Section 4** should acknowledge that SiLU(0) = 0 exactly. The claim "SiLU(z) never reaches zero for finite inputs" is false at z=0. The correct statement is: "SiLU(z) = 0 only at z = 0, a measure-zero event under any continuous input distribution. For all z != 0, |SiLU(z)| > 0." This does not change any conclusion but should be fixed for mathematical precision.

The experiment is correctly killed and archived. No further work needed on this branch.
