# Peer Review: SiLU Gamma Correction

## NotebookLM Findings

Skipped -- this experiment is a straightforward activation-function swap validation with clear analytical predictions. The math is simple enough that manual verification suffices.

## Mathematical Soundness

### Step 1: Curvature claim verification

The paper claims SiLU has max |sigma''| = 0.500 vs GELU max |sigma''| = 0.798. Verified:

- SiLU(x) = x * sigmoid(x). First derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)). Second derivative has max absolute value at approximately x = 0, yielding 0.5. **Correct.**
- GELU(x) uses tanh approximation. Second derivative peaks near x = 0 at approximately 0.798. **Correct.**
- The code computes curvature numerically via np.gradient (lines 306-307), which matches the analytical values. Numerical differentiation on 1000 points in [-5,5] is adequate for smooth functions. **Correct.**

### Step 2: The cancellation argument (Section 5.1 of MATH.md)

The claim that `alpha(gamma) / alpha(1) = 1 + O(sigma''(z) * (gamma - 1)^2)` is the key theoretical result. The argument is:

1. Gamma is a fixed parameter applied to all forward passes identically.
2. At first order, gamma scales both the signal path and the perturbation path equally, so the ratio (alpha) is invariant.
3. The second-order correction depends on the activation curvature.

This argument is **sound in principle** but somewhat informal. The Taylor expansion in Section 3.1 is around the operating point, which is valid when gamma*x is not too far from x. For extreme gamma (10x), the expansion is not strictly local, but the empirical results confirm the bound holds even there. The argument explains the mechanism correctly even if it is not a formal proof.

### Step 3: Correction factor definition

The "correction ratio" is defined as alpha(gamma_profile) / alpha(uniform). The code (lines 427-429, 488-489) computes this correctly as the mean amplification ratio under a gamma profile divided by the baseline amplification ratio.

### Step 4: Divergence metric

Divergence is computed as `|SiLU_ratio - GELU_ratio| / GELU_ratio * 100%` (line 649). This is a relative divergence, which is the right metric for K2. **Correct.**

### Minor issue: prediction vs reality

MATH.md Section 3.3 predicts SiLU correction ~ 1.27x based on the curvature ratio. The empirical result is 1.41x, closer to GELU's 1.43x than predicted. The prediction assumed a linear relationship between curvature and correction factor, but the actual relationship is more complex (the correction depends on the full curvature profile, not just the peak). This does not affect the conclusions since 1.41x is still well below the 2.0x threshold, but the analytical prediction is overly optimistic. The paper should acknowledge this discrepancy more clearly. The curvature argument explains the direction (SiLU <= GELU) but not the magnitude accurately.

## Novelty Assessment

This is a **validation experiment**, not a novelty claim. It validates that safety bounds derived under GELU transfer to SiLU. No prior art search is needed -- the question is specific to this project's safety bound chain.

The experiment correctly identifies the gap (flagged by the adversarial review of the parent experiment) and closes it with a well-designed side-by-side comparison.

**Housekeeping issue:** There are two HYPOTHESES.yml entries for essentially the same experiment:
- `exp_silu_gamma_correction` (status: proven, dir: micro/models/silu_gamma_correction)
- `exp_silu_vs_gelu_gamma_correction` (status: open, dir: micro/models/silu_vs_gelu_gamma_correction)

The second should be marked as superseded or merged into the first.

## Experimental Design

### Strengths

1. **Side-by-side comparison with identical seeds.** Same random base weights, same expert generation, same gamma profiles, same inputs. The only variable is the activation function. This is clean experimental design.

2. **Comprehensive gamma profiles.** Log-normal sweep (8 sigma values), 5 layer-wise profiles (including the worst-case single spike), 3 bimodal profiles, and a scale sweep. 16 total gamma configurations, 2 activations, 3 seeds each = 96 individual runs at the base scale plus additional scale sweep runs. Good coverage.

3. **Scale sweep to d=256, N=50.** Tests the K2-relevant regime where deviations converge. Shows 42-51x margin below the 5% threshold.

4. **Controls adequate.** Uniform gamma (sigma=0) serves as the identity control, correctly showing 1.00x ratio for both activations.

### Weaknesses

1. **SiLU tested, not SwiGLU.** This is acknowledged in Limitations and Assumption 5 of MATH.md. The argument that SwiGLU's gate "applies identically to signal and perturbation paths" is plausible but hand-wavy. The gate is `SiLU(W_gate @ x) * (W_up @ x)`, which is a multiplicative interaction. If gamma affects the gate and up projections differently (e.g., different learned gamma per module), the cancellation may not be exact. However, at the micro scale there is no way to test SwiGLU properly (it requires separate gate/up weight matrices with independent learned gammas). This is a known limitation, not a design flaw. It is a **macro validation item**.

2. **3 seeds.** The paper acknowledges this. Given the 5.4x margin on K2 (9.3% vs 50% threshold), 3 seeds is sufficient to establish the result directionally. More seeds would tighten confidence intervals but would not change the verdict.

3. **Random base weights.** Real pretrained weights have structure (low effective rank, correlated directions). The micro experiment cannot test whether structured correlations between gamma and perturbation direction change the picture. Again, a macro validation item that the paper correctly flags.

### Does it test what it claims?

Yes. The experiment claims that SiLU produces a similar gamma correction factor as GELU, and the side-by-side comparison with matched conditions directly tests this. Both K1 and K2 are operationalized correctly and tested against their thresholds.

### Could a simpler mechanism explain the result?

The result (SiLU and GELU behave similarly) could be explained simply by "both are smooth, monotone-ish activations with bounded curvature." This is essentially what the paper argues. There is no hidden confound here -- the similarity is expected and the experiment confirms it.

## Hypothesis Graph Consistency

- The experiment matches `exp_silu_gamma_correction` node's kill criteria exactly: K1 (2.0x threshold) and K2 (50% divergence threshold).
- The parent `exp_rmsnorm_gamma_nonuniformity` is correctly listed as the dependency.
- Evidence strings in HYPOTHESES.yml accurately reflect the results.
- **Action needed:** `exp_silu_vs_gelu_gamma_correction` (open) appears to be a duplicate hypothesis that was superseded by this experiment. It should be updated to point to this experiment or marked as closed/superseded.

## Macro-Scale Risks (advisory)

1. **SwiGLU gate interaction.** The biggest unresolved question. At macro scale, test with actual Qwen2.5 SwiGLU layers (separate gate/up projections with independent RMSNorm gammas) to confirm the cancellation argument holds through the multiplicative gate.

2. **Structured weight correlations.** Pretrained weights may have gamma-perturbation correlations that random weights do not. The parent experiment (`gamma_perturbation_correlation`) found r=0.018 with random weights, but this should be verified with real adapters.

3. **The analytical prediction was off.** SiLU correction was 1.41x vs predicted 1.27x. At macro scale, verify the empirical number rather than relying on the curvature-ratio prediction.

None of these are blocking for micro. The experiment establishes the mechanism works in principle.

## Verdict

**PROCEED**

The experiment is well-designed, the math is sound, the code is correct, and both kill criteria pass with substantial margin (K1: 29% below threshold, K2: 5.4x below threshold). The side-by-side comparison with matched conditions is clean experimental design. The key finding -- that SiLU and GELU produce nearly identical gamma correction factors due to similar bounded curvature -- is both theoretically expected and empirically confirmed.

Minor items (not blocking):
1. Acknowledge the discrepancy between predicted correction (1.27x) and observed (1.41x) more explicitly. The curvature argument predicts direction correctly but magnitude inaccurately.
2. Clean up the duplicate HYPOTHESES.yml entry (`exp_silu_vs_gelu_gamma_correction` should be marked superseded).
3. SwiGLU validation is a macro-scale item to track.
