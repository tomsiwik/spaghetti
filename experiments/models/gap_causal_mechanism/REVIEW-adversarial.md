# Peer Review: Gap Causal Mechanism

## NotebookLM Findings

Skipped (CLI authentication unavailable in this environment). Review proceeds from direct analysis of MATH.md, PAPER.md, code, and supporting context.

## Mathematical Soundness

### What holds

1. **The router gradient derivation is directionally correct.** For a softmax router with N=2 experts and top_k=2, the gradient of the loss with respect to router weight R_j depends on the term dL/dw_j, which measures how the loss changes when routing more weight to expert j. This term is proportional to the difference between expert j's output and the current weighted combination. When expert outputs are similar, this difference vanishes. The MATH.md explanation of this mechanism is sound.

2. **The discriminability argument is physically intuitive.** If f_A(x) = f_B(x), then changing the mixing weight has no effect on the output, so the gradient is zero regardless of the gap between the composed model and the joint model. This is a correct observation.

3. **The Gram-Schmidt projection preserves norms.** The parent experiment's projection protocol is well-defined and produces controlled cosine levels.

### What does not hold or is imprecise

4. **The gradient formula in MATH.md is incomplete and hand-wavy.** The document writes:

   ```
   ||g_router|| ~ E_x[||f_A(x) - f_B(x)||]    (expert discriminability)
   ```

   This is a proportionality claim without derivation. The actual gradient of CE loss with respect to router logit z_j (before softmax) for a 2-expert system is:

   ```
   dL/dz_j = w_j(1-w_j) * (f_j(x) - f_{-j}(x))^T * d(CE)/d(logits)
   ```

   The gradient depends on THREE factors: (a) the softmax saturation term w_j(1-w_j), (b) expert output difference f_j - f_{-j}, and (c) the loss gradient d(CE)/d(logits). The paper only discusses factor (b) and ignores (a) and (c). Factor (a) is particularly important: if the router starts saturated (w_j near 0 or 1), gradients vanish regardless of discriminability. Factor (c) means the gradient also depends on how far the CURRENT mixture is from the target, not just on expert difference.

   The proportionality claim elides these dependencies. It happens to be correct in the regime tested (uniform initialization, similar loss landscapes), but the derivation is incomplete.

5. **The "negative correlation" claim between CE gap and gradient needs careful interpretation.** The experiment measures CE gap = |CE(composed) - CE(joint)|, where "composed" is the task-arithmetic average (not the routed model). The router gradient is measured on the RoutedDeltaGPT model. These are computed on different models. The CE gap measures how far a naive average is from joint; the gradient measures how hard the router is working to improve the routed model. These are related but not the same quantity. The paper correctly identifies they move in opposite directions with cosine, but calling this a "negative correlation between gap and gradient" conflates the gap of a static average with the gradient of a dynamic router.

6. **r-squared on N=7 mean-curve points is statistically weak.** Computing Pearson r-squared on 7 aggregated points (mean across 3 seeds) yields unreliable estimates. With 5 free parameters (7 - 2 for linear regression), the effective sample size is tiny. The reported r-squared=0.63 for cos vs gradient and r-squared=0.75 for gradient vs quality should be treated as descriptive, not inferential. The paper does not report confidence intervals or p-values for these correlations.

7. **The non-monotonicity at cos=0.5 undermines the linear correlation claim.** The gradient norms INCREASE from cos=0.3 (0.1751) to cos=0.5 (0.1957), which is the opposite direction from the claimed monotonic decrease. The paper acknowledges this as a "phase transition" but the data is also consistent with noise in the flat regime [0.0, 0.5]. The coefficient of variation in regime A (0.090) shows 9% noise, and the cos=0.5 value is within this noise band. The "phase transition" interpretation is plausible but not uniquely supported.

## Novelty Assessment

### Prior art

The observation that MoE router gradients depend on expert output diversity is well-established in the MoE literature. Shazeer et al. (2017) noted that expert collapse (all experts producing similar outputs) causes gradient vanishing. Switch Transformers (Fedus et al. 2022) introduced auxiliary load balancing losses precisely because routers stop differentiating between similar experts. The DeepSeek-V3 paper discusses "expert homogenization" as a known failure mode.

The specific formulation -- measuring discriminability as ||f_A(x) - f_B(x)|| and correlating it with router gradient norms under controlled cosine similarity -- is a novel empirical contribution within this project's framework. It is not novel as a theoretical insight.

### Delta over existing work

The delta is: (1) the controlled sweep from cos=0.0 to cos=0.9 showing the phase transition at ~0.5-0.7, and (2) the explicit negative correlation between CE gap and gradient norm, which refines the parent experiment's "gap-as-signal" framing. The reframe from "gap drives learning" to "discriminability drives learning" is a useful clarification within the project.

## Experimental Design

### Does it test what it claims?

**Partially.** The experiment claims to test whether CE gap CAUSES router gradients. But:

1. **No causal intervention on gap.** The experiment varies cosine similarity, which affects BOTH CE gap and discriminability simultaneously. To test the causal claim "gap causes gradient," one would need to hold gap constant while varying discriminability (or vice versa). The experiment shows that cosine is a common cause of both, and that the gap-gradient correlation is weak/negative. This rules out "gap -> gradient" but does not establish "discriminability -> gradient" as causal -- it establishes a stronger correlation, which is consistent with causation but does not prove it.

2. **The N=2, top_k=2 design is acknowledged but more limiting than stated.** With two experts both always selected, the router only learns a mixing ratio. The "gradient norm" is just the sensitivity of the loss to this ratio. At N>2 with top_k<N, the gradient also reflects expert SELECTION, which is a qualitatively different optimization problem. The paper acknowledges this but marks it as a limitation rather than a potential kill condition.

3. **Controls are adequate for the scope.** Three seeds, seven cosine levels, joint model baseline, and the parent experiment's infrastructure are all reused appropriately. The alternating domain batches during calibration (odd steps = domain A, even steps = domain B) is a reasonable proxy for mixed-domain data.

### Could a simpler mechanism explain the results?

**Yes.** A simpler explanation: at high cosine, the two experts have nearly identical weights, so the composed model is nearly identical to each individual expert. The router has nothing to learn -- ANY mixing ratio produces the same output. The gradient is small because the loss surface is flat in the routing dimension, not because of anything specific about "discriminability as a signal." This is the same mechanism the paper describes, just stated more simply: collinear experts make routing irrelevant.

## Hypothesis Graph Consistency

The experiment is registered as `exp_gap_causal_mechanism` with status "proven" in HYPOTHESES.yml. The kill criteria are:

1. "router gradient magnitude does NOT correlate with gap magnitude (r-squared < 0.3)" -- KC1 partially fails (r-squared=0.24 pooled, 0.63 mean curve). The experiment reports this honestly as PARTIAL.
2. "equal gradient magnitudes at cos=0.0 and cos=0.9" -- KC2 passes decisively (15.5x ratio).

**Issue:** The HYPOTHESES.yml status is "proven" but the original hypothesis ("gap magnitude drives router gradient magnitude") was actually DISPROVEN. What was proven is a different, corrected hypothesis: "orthogonality drives discriminability drives gradient." The node should arguably be marked "disproven (with revised mechanism established)" rather than "proven." The current label is misleading -- someone reading the hypothesis graph would think "CE gap drives gradients" was confirmed.

## Macro-Scale Risks (advisory)

1. **At real scale (cos~0.0003), ALL experts are maximally discriminable.** The paper correctly notes this makes the micro finding moot at scale -- discriminability is always high when experts are naturally orthogonal. The experiment's main value is ruling out the gap-as-signal mechanism, not establishing discriminability as operationally important.

2. **The phase transition at cos~0.5 is only relevant if non-orthogonal experts appear.** At macro scale with independent training and r=16 in d=896, this never happens. The transition is interesting theoretically but has no practical consequence for the architecture.

3. **N>2 gradient dynamics are qualitatively different.** The mixing-ratio gradient measured here does not generalize to expert selection gradients at N=8 or N=64. The claim that discriminability drives router gradients should be re-tested when the router makes discrete selection decisions.

## Verdict

**PROCEED**

The experiment is well-executed within its scope, correctly identifies that the CE gap is not the gradient signal, and provides a plausible (though not causally proven) alternative mechanism in expert discriminability. The code is clean, reuses infrastructure properly, and the analysis is thorough. The three key weaknesses are:

1. The gradient derivation in MATH.md omits the softmax saturation and loss-gradient terms, making the proportionality claim hand-wavy. This does not invalidate the empirical result but weakens the theoretical story.

2. The HYPOTHESES.yml status "proven" is misleading -- the original hypothesis was disproven and a revised hypothesis was established. This should be reflected in the status.

3. The N=7 mean-curve correlations are descriptively useful but statistically underpowered. The r-squared values should not be treated as robust estimates.

None of these are blocking. The experiment successfully refines the gap-as-signal framing and correctly identifies that at real scale the distinction is moot. The finding that the parent experiment's practical conclusion (orthogonal = better) remains valid despite the mechanistic correction is the important takeaway.

**Advisory note for VISION.md:** The document still references "gap-as-signal" prominently. The reframe to "orthogonality guarantees discriminability, which guarantees composability" is more accurate and should be adopted. The "gap" language can be retained as a diagnostic shorthand but should not be presented as a mechanism.
