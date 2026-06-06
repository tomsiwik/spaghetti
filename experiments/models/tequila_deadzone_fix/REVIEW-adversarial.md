# Peer Review: Tequila Deadzone Fix

## NotebookLM Findings

Skipped -- the experiment is already self-killed by the experimenter. NotebookLM deep review is not warranted for a post-mortem confirmation.

## Mathematical Soundness

**The math is correct but the mechanism was misunderstood at the hypothesis stage.**

1. **Forward pass derivation is sound.** The bias term C_j = lambda * sum_{i in D_j} w_{j,i} is correctly derived and correctly implemented. The code at line 134-138 of `run_experiment.py` matches MATH.md Step 3.

2. **Gradient derivation is correct but irrelevant to K1.** MATH.md Section 4 claims dead weights receive gradient `lambda * dL/dY` through the reactivation path. This is true -- but this gradient updates the *shadow weight*, not the *quantized weight*. For a shadow weight to escape the deadzone, it must cross the threshold `alpha/2`. The reactivation bias has no mechanism to preferentially push shadow weights *toward* the threshold. The gradient from the bias term is proportional to `dL/dY`, which is loss-driven and has no directional preference toward the deadzone boundary. This is the core theoretical flaw: the hypothesis confused "providing gradient signal to dead weights" with "pushing dead weights out of the deadzone."

3. **The `stop_gradient` on `dead_mask` is correct** (line 134). Without it, gradients would flow through the mask computation, which would be mathematically wrong (the mask is a discrete step function). However, this also means lambda has no incentive to shrink the deadzone -- it can only learn to compensate for it.

4. **Alpha coupling creates a self-reinforcing equilibrium.** The deadzone threshold is `alpha/2 = mean(|W|)/2`. If reactivation gradients cause some dead weights to grow, `alpha` increases proportionally, raising the threshold and trapping other weights. The paper does not analyze this equilibrium. In a balanced weight distribution, the zero fraction is approximately determined by the CDF of `|W|` below `mean(|W|)/2`, which for a roughly Gaussian distribution is approximately `erf(1/(2*sqrt(2))) ~ 0.31`. This explains why the zero fraction is 31-33% across all layers, all conditions, and even at midpoint vs endpoint -- it is a statistical property of the weight distribution, not a training artifact.

5. **Worked example is correct** but illustrative only. The gradient `(x[1] + 1e-3) * dL/dY[0]` is accurate. The issue is that `1e-3` is negligible compared to typical gradient magnitudes, so the reactivation path provides almost no additional signal beyond STE.

**Hidden assumption:** MATH.md Assumption 5 notes scale caveat but does not flag the alpha-coupling issue above. This is the key theoretical gap.

## Novelty Assessment

**Low novelty, correctly attributed.**

- The mechanism is from Tequila (arxiv 2509.23809). The experiment is a faithful reimplementation at micro scale. This is appropriate for a micro-experiment -- the goal was to test whether the mechanism works, not to invent something new.

- The paper correctly cites the Tequila reference and describes the delta over standard BitLinear.

- **Important negative finding:** The result that Tequila-style bias compensation does NOT reduce deadzones is a genuine finding. The original Tequila paper reports results at 1B-3B scale where the mechanism may behave differently (more training, different weight distributions). At micro scale, the 32% zero fraction is shown to be a structural property of STE with Gaussian-like weight distributions.

- No prior art in `references/` implements this mechanism. No reinvention concern.

## Experimental Design

**Well-designed controlled experiment with one significant flaw.**

1. **Controls are adequate.** Three conditions (baseline, lambda=1e-3, lambda=1e-2) with identical architecture, data, training schedule, and seed. The only variable is the reactivation mechanism.

2. **Kill criteria are well-defined and pre-registered** (K1: zeros below 20%, K2: PPL not worse). The 20% threshold for K1 is aggressive but reasonable -- the hypothesis claims "reactivation," not "modest reduction."

3. **The FP32 baseline is from a prior run** with 3000 steps vs 2000 steps here. The paper acknowledges this (Limitation 4). The relative comparison between BitLinear and Tequila conditions is fair since they share identical training. The absolute comparison to FP32 is directional only.

4. **Single seed is a weakness** but acceptable at micro scale. The 6.7% PPL improvement is large enough that noise is unlikely to fully explain it, though the exact magnitude is uncertain.

5. **Significant flaw in hypothesis formulation:** The kill criterion K1 (zero fraction below 20%) tests whether Tequila *reduces deadzones*. But reading the Tequila paper carefully, the "reactivation" is about making dead weights *contribute to the output*, not about changing their ternary quantization. The experiment correctly tests the stated hypothesis, but the stated hypothesis mischaracterizes what Tequila actually does. The experimenter identified this post-hoc (PAPER.md "Why K1 Fails Despite K2 Passing" section), which is honest and valuable.

6. **Missing condition:** A lambda=0 ablation (Tequila architecture but lambda frozen at 0) would have isolated the effect of the extra parameters vs the bias mechanism itself. The baseline uses `BitLinear` (different class), not `TequilaBitLinear(lam_init=0)`. This is a minor concern since the param count difference is 24 scalars (64134680 vs 64134656), which is negligible.

7. **Zero fraction at midpoint** is a good diagnostic. The fact that it equals the final value (0.3197 at both midpoint and end for baseline; 0.3207 vs 0.3204 for lambda=1e-3) confirms that the equilibrium is reached early and is stable.

## Hypothesis Graph Consistency

**No HYPOTHESES.yml entry exists for this experiment.** The PAPER.md references K1 (id=239) and K2 (id=240), but these IDs do not appear in HYPOTHESES.yml. The kill criteria in the code match what the paper describes, so the assessment is internally consistent, but the experiment is disconnected from the formal hypothesis graph. This is a bookkeeping gap, not a scientific one.

## Macro-Scale Risks (advisory)

1. **The -6.7% PPL win may shrink at scale.** At 1B+ params, the model has more capacity to compensate for dead weights through live weights alone. The bias fusion adds zero inference cost, so even a 1% improvement is free, but the claim should be verified.

2. **Lambda divergence across layers** (range [-0.283, +0.030]) suggests per-layer lambda may be insufficient. At scale, per-channel or per-head lambda could help, but adds complexity.

3. **Composition with LoRA adapters:** The paper claims bias is "orthogonal to LoRA composition (biases add, LoRA deltas multiply)." This is approximately true for additive LoRA composition (Y = X @ (W + sum(B_i @ A_i))^T + bias), but the bias is computed from the *base model's* dead weights. After adapter composition, the effective weight matrix changes, so the deadzone set changes. The fused bias may be slightly stale. This is unlikely to be a problem in practice but should be tested at macro scale.

4. **The 32% zero fraction finding generalizes:** If deadzones are structural to STE (as this experiment strongly suggests), then any mechanism targeting deadzone *reduction* (as opposed to compensation) must modify the quantization function itself (e.g., asymmetric thresholds, learned alpha, or non-uniform rounding).

## Verdict

**PROCEED** (kill confirmed)

The experimenter's self-assessment is correct and well-reasoned. The experiment was cleanly executed, the results are clear, and the interpretation is honest. Specific findings:

- K1 FAIL is definitive: 32% zeros are structural to STE ternary quantization at this weight distribution. No amount of gradient signal through a bias term will change the quantization threshold.
- K2 PASS (-6.7% PPL) is a genuine, cheap win. The bias fusion at zero inference cost makes this worth integrating into the standard BitLinear recipe.
- The negative result (reactivation does not reactivate) is itself valuable: it redirects future work away from bias-based approaches toward threshold-modifying approaches for deadzone reduction.

No revisions needed. The PAPER.md and FINDINGS.md entry accurately represent the results. The only bookkeeping gap is the missing HYPOTHESES.yml entry, which does not affect the scientific conclusions.
